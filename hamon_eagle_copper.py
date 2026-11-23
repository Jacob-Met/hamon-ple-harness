"""Sufficient EAGLE routed-copper witnesses for explicit outer-layer artwork."""

from collections import defaultdict, deque
from decimal import Decimal, InvalidOperation
import re

from hamon_eagle_netlist import _named, _physical_pads, _read


FACES = (1, 16)


def nm(value):
    try:
        number = Decimal(value)
    except (TypeError, InvalidOperation) as exc:
        raise ValueError("invalid millimetre coordinate") from exc
    if (
        not number.is_finite()
        or abs(number) > 10**7
        or number.as_tuple().exponent < -30
    ):
        raise ValueError("unsupported coordinate range")
    numerator, denominator = number.as_integer_ratio()
    integer, leftover = divmod(numerator * 1_000_000, denominator)
    if leftover:
        raise ValueError("coordinate finer than the supported nanometre grid")
    return integer


def plating(row):
    drill = nm(row.get("drill"))
    diameter = nm(row.get("diameter", "0"))
    if drill <= 0 or diameter < 0 or (diameter and diameter <= drill):
        raise ValueError("unsupported plated hole or annulus")
    return drill


def pad_nodes(element, pad):
    rot = re.fullmatch(r"S?(M?)R(-?\d+(?:\.\d+)?)", element.get("rot", "R0"))
    if not rot or Decimal(rot[2]) % 90:
        raise ValueError("non-cardinal placement")
    mirror = bool(rot[1])
    turns = int(Decimal(rot[2])) // 90 % 4
    if mirror and turns % 2:
        raise ValueError("unsupported mirrored quarter-turn placement")
    x, y = nm(pad.get("x")), nm(pad.get("y"))
    if mirror:
        x = -x
    for _ in range(turns):
        x, y = -y, x
    x += nm(element.get("x"))
    y += nm(element.get("y"))
    if pad.tag == "smd":
        layer = int(pad.attrib["layer"])
        if layer not in FACES or min(nm(pad.get("dx")), nm(pad.get("dy"))) <= 0:
            raise ValueError("unsupported surface pad")
        layers = (17 - layer if mirror else layer,)
    else:
        plating(pad)
        layers = FACES
    return [("copper", layer, x, y) for layer in layers]


def position(layer, row, suffix=""):
    return ("copper", layer, nm(row.get("x" + suffix)), nm(row.get("y" + suffix)))


def on_segment(point, left, right):
    x, y = point[2:]
    ax, ay = left[2:]
    bx, by = right[2:]
    return (
        min(ax, bx) <= x <= max(ax, bx)
        and min(ay, by) <= y <= max(ay, by)
        and (x - ax) * (by - ay) == (y - ay) * (bx - ax)
    )


def reaches_plating(steps, drills):
    """Each used face must reach past the drilled void on this actual walk."""
    for i, step in enumerate(steps):
        diameter = drills.get(tuple(step["source"]))
        if diameter is None:
            continue
        for center, direction in ((step["start"], -1), (step["end"], 1)):
            if center[0] != "copper":
                continue
            j = i + direction
            outside = False
            while 0 <= j < len(steps):
                edge = steps[j]
                if edge["source"][0] != "wire":
                    break
                far = edge["start"] if direction < 0 else edge["end"]
                dx, dy = far[2] - center[2], far[3] - center[3]
                if 4 * (dx * dx + dy * dy) > diameter * diameter:
                    outside = True
                    break
                j += direction
            if not outside:
                return False
    return True


def routed_continuity(prediction, board_path, signal_name):
    """Find source-addressed walks without using any board contactref as evidence."""
    board, digest = _read(board_path, "board")
    signal = _named(board.findall("signals/signal")).get(signal_name)
    terminals = sorted(
        {
            ("pad", p["part"], p["pad"])
            for p in prediction["physical_contacts"]
            if p["net"] == signal_name
        }
    )
    result = dict(
        status="unresolved",
        signal=signal_name,
        board_sha256=digest,
        schematic_sha256=prediction["schematic_sha256"],
        coordinate_unit="nm",
        terminals=[list(t) for t in terminals],
        witnesses=[],
        unsupported=[],
        scope="Explicit straight outer-layer centerlines, supported pad centers and through plating",
    )
    if signal is None or len(terminals) < 2:
        result["reason"] = "missing_signal_or_fewer_than_two_schematic_terminals"
        return result
    edges = defaultdict(list)
    points = set()
    traces = []
    drills = {}

    def join(a, b, ref):
        edges[a].append((b, ref))
        edges[b].append((a, ref))

    def unsupported(ref, why):
        result["unsupported"].append(dict(source=ref, reason=str(why)))

    elements = _named(board.findall("elements/element"))
    libraries = _named(board.findall("libraries/library"))
    footprints = {}
    for term in terminals:
        _, part, pin = term
        ref = ("terminal", part, pin)
        try:
            e = elements[part]
            key = (e.attrib["library"], e.attrib["package"])
            if key not in footprints:
                packages = _named(libraries[key[0]].findall("packages/package"))
                footprints[key] = _physical_pads(packages[key[1]])
            pad = footprints[key][pin]
            if pad.tag == "pad":
                drills[ref] = plating(pad)
            for node in pad_nodes(e, pad):
                points.add(node)
                join(term, node, ref)
        except (KeyError, ValueError, InvalidOperation) as exc:
            unsupported(ref, exc)
    ordinal = defaultdict(int)
    for row in signal:
        kind = row.tag
        if kind == "contactref":
            continue
        ref = (kind, ordinal[kind])
        ordinal[kind] += 1
        try:
            if kind == "wire":
                layer = int(row.attrib["layer"])
                if (
                    layer not in FACES
                    or nm(row.get("width")) <= 0
                    or Decimal(row.get("curve", "0")) != 0
                ):
                    raise ValueError(
                        "requires a straight positive-width outer-layer trace"
                    )
                a, b = position(layer, row, "1"), position(layer, row, "2")
                if a == b:
                    raise ValueError("zero-length trace")
                traces.append((a, b, ref))
                points.update((a, b))
            elif kind == "via":
                if row.get("extent") != "1-16":
                    raise ValueError("only full-span plated vias are supported")
                drills[ref] = plating(row)
                a, b = [position(layer, row) for layer in FACES]
                points.update((a, b))
                join(a, b, ref)
            else:
                raise ValueError("artwork excluded from this witness")
        except (KeyError, ValueError, InvalidOperation) as exc:
            unsupported(ref, exc)

    by_layer = {layer: [p for p in points if p[1] == layer] for layer in FACES}
    for left, right, ref in traces:
        on_line = sorted(p for p in by_layer[left[1]] if on_segment(p, left, right))
        for a, b in zip(on_line, on_line[1:]):
            join(a, b, ref)
    root = terminals[0]
    previous = {root: None}
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b, ref in edges[a]:
            if b not in previous:
                previous[b] = (a, ref)
                queue.append(b)
    missing = [list(t) for t in terminals if t not in previous]
    if missing:
        result.update(reason="no_supported_copper_path", unreachable=missing)
        return result
    witnesses = []
    for destination in terminals[1:]:
        current = destination
        steps = []
        while current != root:
            parent, ref = previous[current]
            steps.append(dict(source=list(ref), start=list(parent), end=list(current)))
            current = parent
        steps.reverse()
        if not reaches_plating(steps, drills):
            result["reason"] = "trace_does_not_demonstrably_reach_plating"
            return result
        witnesses.append(dict(start=list(root), end=list(destination), steps=steps))
    result.update(status="witnessed", witnesses=witnesses)
    return result
