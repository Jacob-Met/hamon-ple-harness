"""Independent, source-ordered copper checks for the two registered EAGLE boards."""

import argparse
from decimal import Decimal as D
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ZERO = D(0)
FIXTURES = {
    "due": {
        "board": "due-1.brd",
        "sch": "due-0.sch",
        "signal": "N$3",
        "hash": "fd7cc5f9a14623ff62a7555c826373fe8f842cca400f05ec4f7b4c5589e1780a",
        "terminals": [("RN3", "4"), ("ON", "A")],
        "membership": (605, 163),
    },
    "feather": {
        "board": "feather/Adafruit Feather RP2040 Original.brd",
        "sch": "feather/Adafruit Feather RP2040 Original.sch",
        "signal": "MISO",
        "hash": "ed5eff88e1ab8f1cc5ae8db1a1188c1932a65c35a5df9f06b8910f73ff65b0d7",
        "terminals": [("IC1", "31"), ("JP1", "4")],
        "membership": (207, 52),
    },
}


def named(rows, name):
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous source item {name}")
    return matches[0]


def source_geometry(doc, signal):
    board = doc.find("drawing/board")
    return board, named(board.findall("signals/signal"), signal)


def terminal(board, reference):
    part, pin = reference
    e = named(board.findall("elements/element"), part)
    library = named(board.findall("libraries/library"), e.attrib["library"])
    package = named(library.findall("packages/package"), e.attrib["package"])
    pad = named([node for node in package if node.tag in ("pad", "smd")], pin)
    x, y = D(pad.attrib["x"]), D(pad.attrib["y"])
    angle = e.get("rot", "R0")
    # Exact transformations used in these authentic records only.
    dx, dy = {"R0": (x, y), "R90": (-y, x), "R180": (-x, -y), "R270": (y, -x)}[angle]
    return (D(e.attrib["x"]) + dx, D(e.attrib["y"]) + dy), pad, angle


def trace(a, b, layer, wires):
    matches = []
    for i, w in enumerate(wires):
        endpoints = [(D(w.attrib[f"x{j}"]), D(w.attrib[f"y{j}"])) for j in (1, 2)]
        if (
            set(endpoints) == {a, b}
            and int(w.attrib["layer"]) == layer
            and D(w.attrib["width"]) > 0
            and D(w.get("curve", "0")) == 0
        ):
            matches.append(
                {
                    "wire": i,
                    "layer": layer,
                    "width_mm": w.get("width"),
                    "from_mm": a,
                    "to_mm": b,
                }
            )
    return matches[0] if len(matches) == 1 else None


def through_at(node, vias):
    for i, v in enumerate(vias):
        if (D(v.attrib["x"]), D(v.attrib["y"])) != node or v.get("extent") != "1-16":
            continue
        drill = D(v.get("drill"))
        if drill <= 0 or (
            "diameter" in v.attrib
            and D(v.attrib["diameter"])
            and D(v.attrib["diameter"]) <= drill
        ):
            continue
        return {"via": i, "from": 1, "to": 16, "center_mm": node, "drill_mm": drill}
    return None


def source_ordered_witness(doc, name):
    spec = FIXTURES[name]
    board, s = source_geometry(doc, spec["signal"])
    start, first, _ = terminal(board, spec["terminals"][0])
    finish, last, _ = terminal(board, spec["terminals"][1])
    if name == "due":
        mid = (D("13.2188"), D("29.3138"))
        steps = [
            trace(start, mid, 1, s.findall("wire")),
            trace(mid, finish, 1, s.findall("wire")),
        ]
        assert (
            first.tag == last.tag == "smd"
            and first.get("layer") == last.get("layer") == "1"
        )
    else:
        p = (D("32.617"), D("6.8995"))
        v = (D("32.4485"), D("6.731"))
        q = (D("32.4485"), D("5.2705"))
        r = (D("36.449"), D("1.27"))
        ws = s.findall("wire")
        bridge = through_at(v, s.findall("via"))
        steps = [
            trace(start, p, 1, ws),
            trace(p, v, 1, ws),
            bridge,
            trace(v, q, 16, ws),
            trace(q, r, 16, ws),
            trace(r, finish, 16, ws),
        ]
        assert first.tag == "smd" and first.get("layer") == "1" and last.tag == "pad"
        if bridge:
            # Both traces reach the plated hole from beyond its drill radius.
            radius = float(bridge["drill_mm"]) / 2
            assert math.dist(p, v) > radius and math.dist(q, v) > radius
        # The final two contiguous segments enter the explicit through pad annulus.
        assert math.dist(q, finish) > float(last.attrib["diameter"]) / 2
        assert math.dist(r, finish) < float(last.attrib["drill"]) / 2
        assert D(last.attrib["diameter"]) > D(last.attrib["drill"])
    return None if None in steps else {"terminals": spec["terminals"], "steps": steps}


def due_board_drill(doc):
    """Minimum full-width gap at the center of the registered diagonal trace."""
    board, signal = source_geometry(doc, "N$3")
    hole = next((h for h in board.findall("plain/hole")
                 if h.get("x") == "12.4049" and h.get("y") == "28.4999"), None)
    if hole is None:
        return None
    width = D("0.254")
    diameter = D(hole.attrib["drill"])
    center = D(hole.attrib["x"]), D(hole.attrib["y"])
    hits = []
    for w in signal.findall("wire"):
        a, b = [(D(w.get("x" + i)), D(w.get("y" + i))) for i in ("1", "2")]
        if tuple((u + v) / 2 for u, v in zip(a, b)) == center:
            assert D(w.get("width")) == width
            assert math.dist(a, b) > float(diameter)
            hits.append(w)
    assert len(hits) == 1 and diameter > width
    return {"cut_length_across_full_width_mm": (diameter**2 - width**2).sqrt()}


def cut_bound(doc, name):
    board, s = source_geometry(doc, FIXTURES[name]["signal"])
    if name == "due":
        (px, py), pad, angle = terminal(board, ("RN3", "4"))
        assert angle == "R270" and pad.tag == "smd" and pad.get("rot", "R0") == "R0"
        bottom = py - D(pad.attrib["dx"]) / 2
        wires = s.findall("wire")
        assert len(wires) == 1
        w = wires[0]
        top = max(D(w.attrib["y1"]), D(w.attrib["y2"])) + D(w.attrib["width"]) / 2
        assert bottom - top == D(".0464")
        return {
            "positive_y_gap_mm": bottom - top,
            "basis": "rotated SMD boundary minus highest remaining capped trace boundary",
        }
    (px, py), pad, _ = terminal(board, ("JP1", "4"))
    assert (
        not s.findall("via")
        and pad.tag == "pad"
        and D(pad.attrib["diameter"]) > D(pad.attrib["drill"])
    )
    upper = py + D(pad.attrib["diameter"]) / 2
    top_wires = [w for w in s.findall("wire") if w.get("layer") == "1"]
    lower = min(
        min(D(w.attrib["y1"]), D(w.attrib["y2"])) - D(w.attrib["width"]) / 2
        for w in top_wires
    )
    assert lower - upper == D("4.5339")
    return {
        "top_trace_to_remaining_plated_terminal_y_gap_mm": lower - upper,
        "basis": "no signal via remains, and all top trace ink lies above the through terminal copper",
    }


def variants(raw, name):
    spec = FIXTURES[name]

    def altered(mode):
        doc = ET.fromstring(raw)
        board, s = source_geometry(doc, spec["signal"])
        if mode == "strip_contacts":
            for c in s.findall("contactref"):
                s.remove(c)
        elif mode == "cut":
            # The preregistered source record identifies each edit independently.
            if name == "due":
                w = next(w for w in s.findall("wire") if w.get("y2") == "29.3138")
                s.remove(w)
            else:
                (via,) = s.findall("via")
                s.remove(via)
        elif mode in ("drill_cut", "remote_drill"):
            x = "12.4049" if mode == "drill_cut" else "14.4049"
            ET.SubElement(board.find("plain"), "hole", x=x, y="28.4999", drill="0.60")
        elif mode == "wrong_layer":
            for w in s.findall("wire"):
                w.set("layer", "16")
        elif mode == "airwire":
            w = next(w for w in s.findall("wire") if w.get("y2") == "29.3138")
            w.set("layer", "19")
        elif mode == "pour_only":
            for w in s.findall("wire"):
                s.remove(w)
            poly = ET.SubElement(s, "polygon", width="0.254", layer="1")
            for x, y in ((10, 27), (14, 27), (14, 31), (10, 31)):
                ET.SubElement(poly, "vertex", x=str(x), y=str(y))
        elif mode == "blind_via":
            (via,) = s.findall("via")
            via.set("extent", "1-2")
        return doc

    yield "original", ET.fromstring(raw)
    for mode in (
        ["cut", "wrong_layer", "airwire", "pour_only", "strip_contacts", "drill_cut", "remote_drill"]
        if name == "due"
        else ["cut", "blind_via", "strip_contacts"]
    ):
        yield mode, altered(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from hamon_eagle_netlist import schematic_netlist, compare_board

    args.out.mkdir(exist_ok=True, parents=True)
    report = []
    for name, spec in FIXTURES.items():
        raw = (args.raw / spec["board"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec["hash"]
        pred = schematic_netlist(args.raw / spec["sch"])
        for mode, doc in variants(raw, name):
            dest = args.out / f"{name}-{mode}.brd"
            if mode == "original":
                dest.write_bytes(raw)
            else:
                ET.ElementTree(doc).write(dest, encoding="utf-8", xml_declaration=True)
            flat = compare_board(pred, dest)
            assert flat["agreement"] == (mode != "strip_contacts")
            if flat["agreement"]:
                assert (flat["exact_contacts"], len(flat["exact_nets"])) == spec[
                    "membership"
                ]
            witness = source_ordered_witness(doc, name)
            drill = due_board_drill(doc) if name == "due" else None
            if drill:
                assert mode == "drill_cut" and witness
                witness = None
            assert bool(witness) == (mode in ("original", "strip_contacts", "remote_drill"))
            report.append(
                {
                    "board": name,
                    "control": mode,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                    "flat_agreement": flat["agreement"],
                    "exact_contacts": flat["exact_contacts"],
                    "exact_nets": len(flat["exact_nets"]),
                    "source_ordered_witness": witness,
                    "cut_separation": cut_bound(doc, name) if mode == "cut" else drill,
                }
            )
    (args.out / "independent.json").write_text(
        json.dumps(report, default=str, indent=2) + "\n"
    )
    verify_toolkit(args.raw, args.repo, args.out, args.out / "cli-evidence.json")
    for r in report:
        print(
            r["board"],
            r["control"],
            r["flat_agreement"],
            bool(r["source_ordered_witness"]),
            r["cut_separation"] or "",
        )


def verify_path(path, reference):
    expected = [list(pair) for pair in reference["terminals"]]
    actual = [path["start"][1:], path["end"][1:]]
    assert actual == expected or actual == expected[::-1]
    reverse = actual == expected[::-1]
    emitted = path["steps"]
    assert emitted[0]["start"] == path["start"] and emitted[-1]["end"] == path["end"]
    assert all(a["end"] == b["start"] for a, b in zip(emitted, emitted[1:]))
    assert emitted[0]["source"] == ["terminal", *actual[0]]
    assert emitted[-1]["source"] == ["terminal", *actual[1]]
    steps = [s for s in emitted if s["source"][0] != "terminal"]
    reference_steps = reference["steps"][::-1] if reverse else reference["steps"]
    assert len(steps) == len(reference_steps)

    def point(layer, coords):
        return ["copper", layer, *[int(D(v) * 1000000) for v in coords]]

    for s, e in zip(steps, reference_steps):
        if "wire" in e:
            assert s["source"] == ["wire", e["wire"]]
            ends = [point(e["layer"], e["from_mm"]), point(e["layer"], e["to_mm"])]
        else:
            assert s["source"] == ["via", e["via"]]
            ends = [point(layer, e["center_mm"]) for layer in (e["from"], e["to"])]
        assert [s["start"], s["end"]] == (ends[::-1] if reverse else ends)


def verify_toolkit(raw, repo, cases, output):
    source = {
        "due": ("due-0.sch", "N$3"),
        "feather": ("feather/Adafruit Feather RP2040 Original.sch", "MISO"),
    }
    rows = []
    for c in json.loads((cases / "independent.json").read_text()):
        name, mode = c["board"], c["control"]
        sch, net = source[name]
        argv = [
            sys.executable,
            str(repo / "engineering_toolkit.py"),
            "eagle-netlist",
            str(raw / sch),
            "--board",
            str(cases / f"{name}-{mode}.brd"),
            "--routed-net",
            net,
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, check=True)
        out = json.loads(proc.stdout)
        topology, flat = out["routed_continuity"], out["board_comparison"]
        assert topology["board_sha256"] == c["sha256"] == flat["board_sha256"]
        assert (
            flat["agreement"] == c["flat_agreement"]
            and flat["exact_contacts"] == c["exact_contacts"]
        )
        assert len(flat["exact_nets"]) == c["exact_nets"]
        if c["source_ordered_witness"]:
            assert topology["status"] == "witnessed" and len(topology["witnesses"]) == 1
            verify_path(topology["witnesses"][0], c["source_ordered_witness"])
        else:
            assert topology["status"] == "unresolved" and topology["witnesses"] == []
        if mode in ("blind_via", "airwire", "pour_only"):
            assert topology["unsupported"]
        rows.append(
            {
                "board": name,
                "control": mode,
                "flat": flat["agreement"],
                "physical": topology,
            }
        )
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(
        f"{len(rows)}/12 ordinary CLI controls verified including every witness point, layer, and source primitive."
    )


if __name__ == "__main__":
    main()
