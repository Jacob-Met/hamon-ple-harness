"""Elaborate parameter-free linear SPICE subcircuit occurrences into primitive cards."""
import re
from urllib.parse import quote


def _lines(deck):
    lines = []
    for raw in deck.splitlines():
        line = re.split(r"[;$]", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            if not lines:
                raise ValueError("Orphan SPICE continuation")
            lines[-1].extend(line[1:].split())
        else:
            lines.append(line.split())
    return lines


def _is_card(tokens, number, rest):
    """Recognize the usual titleless primitive extension on the first line."""
    first = tokens[0].upper()
    kind = first[0]
    if first.startswith("."):
        return True
    if kind == "X":
        # The first physical line is conventionally a title. A declaration
        # identifies a titleless call. Retain a dangling call for diagnosis
        # when it names a numbered instance or a terminal used by the deck.
        declared = {line[1].casefold() for line in rest
                    if len(line) > 1 and line[0].casefold() == ".subckt"}
        terminals = {part.casefold() for line in rest
                     if line[0][0] != "." for part in line[1:3]}
        return (tokens[-1].casefold() in declared or re.match(r"^X\d", first) is not None
                or any(port.casefold() in terminals for port in tokens[1:-1]))
    if kind in "RLC":
        return len(tokens) >= 4 and bool(number.fullmatch(tokens[3]))
    if kind in "VI":
        return len(tokens) >= 4 and (tokens[3].upper() in {"AC", "DC"}
                                    or bool(number.fullmatch(tokens[3])))
    if kind in "EGFH":
        return len(tokens) >= (6 if kind in "EG" else 5) and bool(number.fullmatch(tokens[-1]))
    return kind == "K" and len(tokens) >= 4 and tokens[1].upper().startswith("L")


def elaborate_subcircuits(deck, gain):
    number = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[a-zA-Z]*")
    lines = _lines(deck)
    first_physical = deck.splitlines()[0].strip() if deck.splitlines() else ""
    if (lines and first_physical and first_physical[0] not in "*;$"
            and not _is_card(lines[0], number, lines[1:])):
        if re.match(r"^[a-zA-Z]\d", lines[0][0]):
            raise ValueError(f"Malformed or unsupported initial device {lines[0][0]!r}")
        lines = lines[1:]
    definitions = {}
    top = []
    active = None
    for tokens in lines:
        tag = tokens[0].casefold()
        if tag == ".subckt":
            if active is not None:
                raise ValueError("Lexically nested SPICE definitions are unsupported")
            if len(tokens) < 3:
                raise ValueError("Subcircuit requires a name and external ports")
            name = tokens[1].casefold()
            pins = [p.casefold() for p in tokens[2:]]
            if (name in definitions or len(set(pins)) != len(pins) or
                    any(p in {"0", "gnd", "params:"} or "=" in p for p in pins)):
                raise ValueError("Duplicate subcircuit, invalid ports, or unsupported parameters")
            active = name
            definitions[name] = (pins, [])
        elif tag == ".ends":
            if active is None or len(tokens) > 2 or (len(tokens) == 2 and tokens[1].casefold() != active):
                raise ValueError("Mismatched subcircuit end")
            active = None
        elif tag == ".end":
            if active is not None:
                raise ValueError("Unclosed subcircuit before end of deck")
            break
        else:
            (top if active is None else definitions[active][1]).append(tokens)
    if active is not None:
        raise ValueError("Unclosed subcircuit definition")

    # Select a namespace marker absent from the input. Quoted path components
    # also distinguish an instance named X1/X2 from two nested occurrences.
    marker = "@"
    while marker in deck:
        marker += "@"
    flat = ["* Expanded linear subcircuits"]
    def expand(cards, ports, path, stack):
        devices = set()
        local_nodes = dict(ports)

        def scoped(name):
            return marker + "/".join(quote(part, safe="") for part in (*path, name))

        def node(name):
            key = name.casefold()
            if key in {"0", "gnd"}:
                return "0"
            if key not in local_nodes:
                local_nodes[key] = scoped(name) if path else name
            return local_nodes[key]

        def device(name):
            return name[0] + scoped(name) if path else name

        for tokens in cards:
            name, *args = tokens
            kind = name[0].upper()
            if kind == ".":
                directive = name.casefold()
                if not path and directive in {".title", ".op", ".save", ".print", ".plot"}:
                    continue
                if not path and directive == ".ac":
                    if len(args) != 4 or args[0].upper() not in {"LIN", "DEC", "OCT"}:
                        raise ValueError("Unsupported hierarchical AC directive")
                    flat.append(" ".join(tokens))
                    continue
                raise ValueError(f"Unsupported hierarchical directive {name!r}")
            if any("*" in token for token in tokens):
                raise ValueError("Expressions are unsupported in hierarchical primitive cards")
            if name.casefold() in devices:
                raise ValueError(f"Repeated device or occurrence name {name!r}")
            devices.add(name.casefold())
            if kind == "X":
                if len(args) < 2 or args[-1].casefold() not in definitions:
                    raise ValueError(f"Unknown or parameterized subcircuit instance {name!r}")
                target = args[-1].casefold()
                formal, body = definitions[target]
                if len(formal) != len(args)-1 or target in stack:
                    raise ValueError(f"Subcircuit port count or recursive occurrence at {name!r}")
                expand(body, dict(zip(formal, map(node, args[:-1]))), (*path, name), (*stack, target))
                continue

            if kind in "RLC" and len(args) == 3 and number.fullmatch(args[2]):
                args[:2] = map(node, args[:2])
            elif kind in "VI" and len(args) >= 2:
                args[:2] = map(node, args[:2])
            elif kind in "EG" and len(args) == 5 and gain.fullmatch(args[4]):
                args[:4] = map(node, args[:4])
            elif kind in "FH" and len(args) == 4 and gain.fullmatch(args[3]):
                args[:2] = map(node, args[:2])
                args[2] = device(args[2])
            elif kind == "K" and len(args) >= 3 and gain.fullmatch(args[-1]):
                args[:-1] = map(device, args[:-1])
            else:
                raise ValueError(f"Unsupported or malformed hierarchical device {name!r}")
            flat.append(" ".join([device(name), *args]))

    expand(top, {}, (), ())
    return "\n".join(flat)
