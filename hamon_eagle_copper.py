"""Sufficient EAGLE routed-copper witnesses for explicit outer-layer artwork."""

from collections import defaultdict, deque
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import partial
import re

from hamon_eagle_netlist import _named, _physical_pads, _read


FACES = (1, 16)


def nm(value, *, subgrid=False):
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
        if subgrid:
            return Fraction(numerator * 1_000_000, denominator)
        raise ValueError("coordinate finer than the supported nanometre grid")
    return integer


def plating(row):
    drill = nm(row.get("drill"))
    diameter = nm(row.get("diameter", "0"))
    if drill <= 0 or diameter < 0 or (diameter and diameter <= drill):
        raise ValueError("unsupported plated hole or annulus")
    return drill


def placed_center(element, pad, coordinate=nm):
    rot = re.fullmatch(r"S?(M?)R(-?\d+(?:\.\d+)?)", element.get("rot", "R0"))
    if not rot or Decimal(rot[2]) % 90:
        raise ValueError("non-cardinal placement")
    mirror = bool(rot[1])
    turns = int(Decimal(rot[2])) // 90 % 4
    if mirror and turns % 2:
        raise ValueError("unsupported mirrored quarter-turn placement")
    x, y = coordinate(pad.get("x")), coordinate(pad.get("y"))
    if mirror:
        x = -x
    for _ in range(turns):
        x, y = -y, x
    x += coordinate(element.get("x"))
    y += coordinate(element.get("y"))
    return x, y, mirror


def pad_nodes(element, pad):
    x, y, mirror = placed_center(element, pad)
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


def inside(point, bore):
    x, y, diameter, _ = bore
    return 4 * ((point[2] - x) ** 2 + (point[3] - y) ** 2) <= diameter ** 2


def hits_segment(a, b, bore):
    """Closed disk against a centerline segment, using exact integer arithmetic."""
    x, y, diameter, _ = bore
    ax, ay = a[2:]
    bx, by = b[2:]
    if (2 * x < 2 * min(ax, bx) - diameter or
            2 * x > 2 * max(ax, bx) + diameter or
            2 * y < 2 * min(ay, by) - diameter or
            2 * y > 2 * max(ay, by) + diameter):
        return False
    dx, dy = bx - ax, by - ay
    ux, uy = x - ax, y - ay
    dot = dx * ux + dy * uy
    length2 = dx * dx + dy * dy
    if dot <= 0:
        return inside(a, bore)
    if dot >= length2:
        return inside(b, bore)
    return 4 * (ux * dy - uy * dx) ** 2 <= diameter ** 2 * length2


def board_bores(board, elements, libraries, signal, terminals):
    """Every known drill excludes centerline copper; only selected plating can join it."""
    bores = []
    coordinate = partial(nm, subgrid=True)

    def add(x, y, row, owner=None, margin=0):
        diameter = coordinate(row.get("drill"))
        if diameter <= 0:
            raise ValueError("nonpositive board drill")
        bores.append((x, y, diameter + 2 * margin, owner))

    for hole in board.findall("plain/hole"):
        add(coordinate(hole.get("x")), coordinate(hole.get("y")), hole)
    for net in board.findall("signals/signal"):
        for i, via in enumerate(net.findall("via")):
            owner = None
            if net is signal and via.get("extent") == "1-16":
                try:
                    plating(via)
                    owner = ("via", i)
                except ValueError:
                    pass
            add(coordinate(via.get("x")), coordinate(via.get("y")), via, owner)
    packages = {
        (libname, name): package
        for libname, lib in libraries.items()
        for name, package in _named(lib.findall("packages/package")).items()
    }
    for name, element in elements.items():
        pkg = packages[element.attrib["library"], element.attrib["package"]]
        for row in pkg:
            if row.tag not in ("pad", "hole"):
                continue
            try:
                x, y, _ = placed_center(element, row, coordinate)
            except ValueError:
                # An unsupported angle still permits a conservative enclosure about
                # the placement origin, including every rotated/reflected center.
                px, py = coordinate(row.get("x")), coordinate(row.get("y"))
                add(coordinate(element.get("x")), coordinate(element.get("y")), row,
                    margin=abs(px) + abs(py))
                continue
            owner = None
            if row.tag == "pad" and ("pad", name, row.get("name")) in terminals:
                try:
                    plating(row)
                    owner = ("terminal", name, row.get("name"))
                except ValueError:
                    pass
            add(x, y, row, owner)
    return bores


def connect_landings(interior, edges):
    """A plated center is reachable only by a wire chain that exits its bore.

    Keep the original XML wire steps in each approach. No drilled-away interior
    vertex becomes a junction for another terminal or the ordinary copper graph.
    """
    for (bore, layer), local in interior.items():
        x, y, _, _ = bore
        center = ("copper", layer, x, y)
        parents = {center: None}
        queue = deque([center])
        while queue:
            a = queue.popleft()
            for b, ref in local[a]:
                if b in parents:
                    continue
                parents[b] = a, ref
                if inside(b, bore):
                    queue.append(b)
                    continue
                reverse = []
                point = b
                while point != center:
                    parent, wire = parents[point]
                    reverse.append((point, parent, wire, bore[3]))
                    point = parent
                edges[b].append((center, reverse))
                edges[center].append((b, [(v, u, w, plating_ref) for u, v, w, plating_ref in reversed(reverse)]))


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
        scope="Straight outer-layer centerlines with drill clearance and supported plated approaches",
    )
    if signal is None or len(terminals) < 2:
        result["reason"] = "missing_signal_or_fewer_than_two_schematic_terminals"
        return result
    edges = defaultdict(list)
    points = set()
    traces = []
    landings = []

    def join(a, b, ref):
        edges[a].append((b, [(a, b, ref, None)]))
        edges[b].append((a, [(b, a, ref, None)]))

    def unsupported(ref, why):
        result["unsupported"].append(dict(source=ref, reason=str(why)))

    elements = _named(board.findall("elements/element"))
    libraries = _named(board.findall("libraries/library"))
    try:
        bores = board_bores(board, elements, libraries, signal, set(terminals))
    except (KeyError, ValueError, InvalidOperation) as exc:
        unsupported(("board", "drills"), exc)
        result["reason"] = "unknown_drill_geometry"
        return result
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
            for node in pad_nodes(e, pad):
                points.add(node)
                landings.append((term, node, ref))
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
                plating(row)
                a, b = [position(layer, row) for layer in FACES]
                points.update((a, b))
                landings.append((a, b, ref))
            else:
                raise ValueError("artwork excluded from this witness")
        except (KeyError, ValueError, InvalidOperation) as exc:
            unsupported(ref, exc)

    blocked_terminal = False
    for a, b, ref in landings:
        if all(bore[3] == ref or not inside(b, bore) for bore in bores):
            join(a, b, ref)
        elif ref[0] == "terminal":
            blocked_terminal = True

    interior = defaultdict(lambda: defaultdict(list))
    by_layer = {layer: [p for p in points if p[1] == layer] for layer in FACES}
    for left, right, ref in traces:
        candidates = [bore for bore in bores if hits_segment(left, right, bore)]
        on_line = sorted(p for p in by_layer[left[1]] if on_segment(p, left, right))
        for a, b in zip(on_line, on_line[1:]):
            touched = [bore for bore in candidates if hits_segment(a, b, bore)]
            if not touched:
                join(a, b, ref)
            elif len(touched) == 1:
                bore = touched[0]
                if bore[3] is not None and (inside(a, bore) or inside(b, bore)):
                    local = interior[bore, a[1]]
                    local[a].append((b, ref))
                    local[b].append((a, ref))
    connect_landings(interior, edges)
    root = terminals[0]
    previous = {root: None}
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b, walk in edges[a]:
            if b not in previous:
                previous[b] = (a, walk)
                queue.append(b)
    missing = [list(t) for t in terminals if t not in previous]
    if missing:
        reason = "trace_does_not_demonstrably_reach_plating" if blocked_terminal else "no_supported_copper_path"
        result.update(reason=reason, unreachable=missing)
        return result
    witnesses = []
    for destination in terminals[1:]:
        current = destination
        steps = []
        while current != root:
            parent, walk = previous[current]
            for a, b, ref, plating_ref in reversed(walk):
                step = dict(source=list(ref), start=list(a), end=list(b))
                if plating_ref is not None:
                    step["plated_approach"] = list(plating_ref)
                steps.append(step)
            current = parent
        steps.reverse()
        witnesses.append(dict(start=list(root), end=list(destination), steps=steps))
    result.update(status="witnessed", witnesses=witnesses)
    return result
