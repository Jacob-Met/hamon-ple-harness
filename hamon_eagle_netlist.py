"""Flat EAGLE XML schematic pin mappings and independent PCB contact checks."""
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET


def _read(path, kind):
    data = Path(path).read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Expected an XML EAGLE {kind}: {exc}") from exc
    document = root.find(f"./drawing/{kind}")
    if root.tag != "eagle" or document is None:
        raise ValueError(f"Expected an XML EAGLE {kind}")
    return document, hashlib.sha256(data).hexdigest()


def _index(rows, attributes):
    index = {}
    for row in rows:
        key = tuple(row.attrib[a] for a in attributes)
        if key in index:
            raise ValueError(f"Ambiguous {row.tag} identity {key!r}")
        index[key] = row
    return index


def _named(rows):
    return {key[0]: row for key, row in _index(rows, ("name",)).items()}


def _physical_pads(package):
    return _named([*package.findall("pad"), *package.findall("smd")])


def schematic_netlist(path):
    """Resolve explicit pinrefs and automatic power pins using the selected device variant."""
    schematic, digest = _read(path, "schematic")
    if schematic.findall("modules/module") or schematic.findall(".//moduleinst"):
        raise ValueError("Hierarchical EAGLE module scopes require instance expansion")
    libraries = _named(schematic.findall("libraries/library"))
    parts = _named(schematic.findall("parts/part"))
    sheets = schematic.findall("sheets/sheet")
    if not sheets:
        raise ValueError("Schematic has no sheets")
    # Cache the selected variant for identical components. Pin and pad names
    # remain exact, including EAGLE suffixes that distinguish stacked pins.
    variants = {}

    def mapping(part):
        selection = tuple(part.attrib[k] for k in ("library", "deviceset", "device"))
        if selection not in variants:
            lib = libraries[selection[0]]
            ds = _named(lib.findall("devicesets/deviceset"))[selection[1]]
            device = _named(ds.findall("devices/device"))[selection[2]]
            symbols = _named(lib.findall("symbols/symbol"))
            gates = _named(ds.findall("gates/gate"))
            pins = {name: _named(symbols[gate.attrib["symbol"]].findall("pin"))
                    for name, gate in gates.items()}
            package = device.get("package")
            pads = _physical_pads(_named(lib.findall("packages/package"))[package]) if package else {}
            links = {}
            for (gate, pin), row in _index(device.findall("connects/connect"), ("gate", "pin")).items():
                if gate not in pins or pin not in pins[gate]:
                    raise ValueError(f"Invalid symbol pin {(gate, pin)!r} in device {selection!r}")
                targets = row.attrib["pad"].split()
                if not targets or len(set(targets)) != len(targets) or any(pad not in pads for pad in targets):
                    raise ValueError(f"Invalid package pad mapping for {(gate, pin)!r} in {selection!r}: {targets!r}")
                links[gate, pin] = targets
            variants[selection] = pins, package, links
        return variants[selection]

    pin_nets, contacts = {}, {}
    pin_rows, net_names = [], set()

    def add_pin(part, gate, pin, net, sheet, source):
        logical = (part, gate, pin)
        if logical in pin_nets:
            if pin_nets[logical] != net:
                raise ValueError(f"Pin {logical!r} belongs to multiple nets")
            return
        try:
            pins, package, links = mapping(parts[part])
            symbol_pin = pins[gate][pin]
            pads = links[(gate, pin)] if package else []
        except KeyError as exc:
            raise ValueError(f"Unresolved EAGLE pin mapping {logical!r}: {exc}") from exc
        pin_nets[logical] = net
        net_names.add(net)
        pin_rows.append(dict(part=part, gate=gate, pin=pin, net=net, sheet=sheet,
                             source=source, direction=symbol_pin.get("direction", "io"),
                             package=package, pads=pads))
        for pad in pads:
            contact = (part, pad)
            if contact in contacts and contacts[contact] != net:
                raise ValueError(f"Physical contact {contact!r} belongs to multiple nets")
            contacts[contact] = net

    placements = {}
    for sheet_no, sheet in enumerate(sheets, 1):
        instances = _index(sheet.findall("instances/instance"), ("part", "gate"))
        for instance in instances:
            part, gate = instance
            try:
                pins, _, _ = mapping(parts[part])
                if gate not in pins:
                    raise KeyError(gate)
            except KeyError as exc:
                raise ValueError(f"Unresolved EAGLE gate instance {instance!r}: {exc}") from exc
            if instance in placements:
                raise ValueError(f"Repeated gate instance {instance!r}")
            placements[instance] = sheet_no
        for net in sheet.findall("nets/net"):
            name = net.attrib["name"]
            net_names.add(name)
            for ref in net.findall("segment/pinref"):
                part, gate, pin = (ref.attrib[key] for key in ("part", "gate", "pin"))
                if (part, gate) not in instances:
                    raise ValueError(f"No instance for pinref {(part, gate, pin)!r} on sheet {sheet_no}")
                add_pin(part, gate, pin, name, sheet_no, "pinref")

    # EAGLE's automatic supply connection applies only to unconnected Pwr
    # pins, including an uninvoked supply gate. Explicit wires take priority.
    # The @ suffix distinguishes pins inside the symbol, not the rail name.
    for part_name in sorted({part for part, gate in placements}):
        pins, package, _ = mapping(parts[part_name])
        if package:
            for gate, symbol_pins in pins.items():
                for pin, symbol_pin in symbol_pins.items():
                    if symbol_pin.get("direction") == "pwr" and (part_name, gate, pin) not in pin_nets:
                        add_pin(part_name, gate, pin, pin.split("@", 1)[0],
                                placements.get((part_name, gate)), "automatic_power")

    return dict(ok=True, kind="eagle-native-schematic-netlist", schematic_sha256=digest,
                scope="Flat schematic pinrefs and automatic power pins through selected-device pad mappings",
                sheet_count=len(sheets), logical_pins=pin_rows, net_names=sorted(net_names),
                physical_contacts=[dict(part=part, pad=pad, net=net)
                                   for (part, pad), net in sorted(contacts.items())])


def compare_board(prediction, board_path):
    """Compare schematic-only contacts to independent, explicitly named PCB signal contacts."""
    board, digest = _read(board_path, "board")
    elements = _named(board.findall("elements/element"))
    libraries = _named(board.findall("libraries/library"))
    packages = {}
    expected = {}
    for signal in board.findall("signals/signal"):
        name = signal.attrib["name"]
        for ref in signal.findall("contactref"):
            element, pad = ref.attrib["element"], ref.attrib["pad"]
            contact = (element, pad)
            try:
                part = elements[element]
                key = (part.attrib["library"], part.attrib["package"])
                if key not in packages:
                    packages[key] = _physical_pads(_named(libraries[key[0]].findall("packages/package"))[key[1]])
                packages[key][pad]
            except KeyError as exc:
                raise ValueError(f"Unresolved PCB contact {contact!r}: {exc}") from exc
            if contact in expected and expected[contact] != name:
                raise ValueError(f"PCB contact {contact!r} belongs to multiple signals")
            expected[contact] = name
    actual = {(r["part"], r["pad"]): r["net"] for r in prediction["physical_contacts"]}
    mismatches, exact_contacts = [], 0
    nets_with_contacts = set(expected.values()) | set(actual.values())
    unequal_nets = set()
    for part, pad in sorted(actual.keys() | expected.keys()):
        a, b = actual.get((part, pad)), expected.get((part, pad))
        if a == b:
            exact_contacts += 1
            continue
        unequal_nets.update(n for n in (a, b) if n is not None)
        mismatches.append(dict(part=part, pad=pad, schematic_net=a, board_net=b))
    return dict(board_sha256=digest, agreement=bool(nets_with_contacts) and not mismatches,
                exact_contacts=exact_contacts, schematic_contacts=len(actual), board_contacts=len(expected),
                nets_with_contacts=len(nets_with_contacts), exact_nets=sorted(nets_with_contacts-unequal_nets),
                mismatches=mismatches)

