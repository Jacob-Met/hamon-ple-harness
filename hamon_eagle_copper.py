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
    # The virtual center joins all approaches through one continuous plated rim.
    # A neighboring drill can break that rim without touching an approaching wire.
    for i, (x, y, diameter, owner) in enumerate(bores):
        if owner is None:
            continue
        if any(i != j and 4 * ((x - u)**2 + (y - v)**2) <= (diameter + d)**2
               for j, (u, v, d, _) in enumerate(bores)):
            bores[i] = (x, y, diameter, None)
    return bores


def escape_on_pad(center, exterior, diameter, bore):
    """Does this wire reach outside the drill within the pad's inscribed disk?"""
    dx, dy = exterior[2] - center[2], exterior[3] - center[3]
    length2 = dx * dx + dy * dy
    if 4 * length2 <= diameter**2:
        return True
    # Test the point one pad radius along the wire. Its squared clearance has
    # the sign of constant * sqrt(length2) + linear, without taking a root.
    x, y, drill, _ = bore
    ux, uy = center[2] - x, center[3] - y
    constant = 4 * (ux * ux + uy * uy) + diameter**2 - drill**2
    linear = 4 * diameter * (ux * dx + uy * dy)
    if linear > 0:
        return constant >= 0 or linear**2 > constant**2 * length2
    return constant > 0 and constant**2 * length2 > linear**2


def connect_landings(interior, edges, surface):
    """A plated center is reachable only by a wire chain that exits its bore.

    Keep the original XML wire steps in each approach. No drilled-away interior
    vertex becomes a junction for another terminal or the ordinary copper graph.
    """
    for (bore, layer), local in interior.items():
        x, y, _, _ = bore
        center = ("copper", layer, x, y)
        # A clipped SMD can still feed a wire through surviving pad copper.
        # Keep this approach local to the terminal: its center is not a wire
        # junction inside the bore.
        for pad_center in local.keys() & surface.keys():
            for term, ref, diameter in surface[pad_center]:
                for end, wire in local[pad_center]:
                    if inside(end, bore) or not escape_on_pad(pad_center, end, diameter, bore):
                        continue
                    edges[term].append((end, [(term, pad_center, ref, None),
                                              (pad_center, end, wire, None)]))
                    edges[end].append((term, [(end, pad_center, wire, None),
                                              (pad_center, term, ref, None)]))
        parents = {center: None}
        queue = deque([center])
        while queue:
            a = queue.popleft()
            for b, ref in local[a]:
                if b in parents:
                    continue
                parents[b] = a, ref
                if inside(b, bore):
                    if b not in surface:
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


def rooted_tree(edges, root):
    previous = {root: None}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, source in edges[current]:
            if neighbor not in previous:
                previous[neighbor] = current, source
                queue.append(neighbor)
    return previous


def material_fallback(result, board, signal, terminals, elements, footprints, bores):
    """Prove additional routes using a conservative subset of finite copper."""
    try:
        from hamon_eagle_material import material_continuity
        from shapely.errors import ShapelyError
    except ImportError:
        result["material"] = {"reason": "finite_width_geometry_dependency_unavailable"}
        return result
    primitives, approximations = [], []
    rules = {p.get("name"): p.get("value") for p in board.findall("designrules/param")}

    def omitted(ref, why):
        entry = dict(source=ref, reason=str(why))
        if entry not in result["unsupported"]:
            result["unsupported"].append(entry)

    def zero_rule(name):
        value = rules.get(name, "0")
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:mm|mil)?\s*", value)
        return bool(match and Decimal(match[1]) == 0)

    for terminal in terminals:
        _, part, pin = terminal
        ref = ("terminal", part, pin)
        try:
            element = elements[part]
            pad = footprints[element.attrib["library"], element.attrib["package"]][pin]
            nodes = pad_nodes(element, pad)
            primitive = dict(source=ref, terminal=terminal, center=nodes[0][2:],
                             layers=tuple(node[1] for node in nodes))
            if pad.tag == "smd":
                dx, dy = nm(pad.get("dx")), nm(pad.get("dy"))
                roundness = Decimal(pad.get("roundness", "0"))
                rotation = re.fullmatch(r"R(-?\d+(?:\.\d+)?)", pad.get("rot", "R0"))
                if not rotation or not roundness.is_finite() or not 0 <= roundness <= 100:
                    raise ValueError("unsupported surface-pad shape or rotation")
                angle = Decimal(rotation[1])
                square = (roundness == 0 and angle % 90 == 0
                          and all(zero_rule(key) for key in
                                  ("srRoundness", "srMinRoundness", "srMaxRoundness")))
                if square:
                    placement = re.fullmatch(r"S?(M?)R(-?\d+(?:\.\d+)?)", element.get("rot", "R0"))
                    turns = int(Decimal(placement[2]) + angle) // 90
                    primitive.update(shape="rectangle", size=(dy, dx) if turns % 2 else (dx, dy))
                else:
                    # All rounded rectangular pads contain this centered disk.
                    # It also remains a subset under arbitrary local rotation.
                    primitive.update(shape="disk", diameter=min(dx, dy))
                    approximations.append(dict(source=ref, shape="inscribed_surface_pad_disk"))
            else:
                diameter = nm(pad.get("diameter", "0"))
                if (diameter <= 0 or pad.get("shape", "round") != "round"
                        or any(rules.get(k, "-1") != "-1" for k in ("psTop", "psBottom", "psFirst"))):
                    raise ValueError("material proof requires an explicit round plated-pad annulus")
                primitive.update(shape="disk", diameter=diameter, plated=True)
            primitives.append(primitive)
        except (KeyError, ValueError, InvalidOperation) as exc:
            omitted(ref, exc)

    ordinal = defaultdict(int)
    for row in signal:
        kind = row.tag
        if kind == "contactref":
            continue
        ref = (kind, ordinal[kind])
        ordinal[kind] += 1
        try:
            if kind == "wire":
                layer, width = int(row.attrib["layer"]), nm(row.get("width"))
                cap = row.get("cap", "round")
                if (layer not in FACES or width <= 0 or cap not in ("round", "flat")
                        or row.get("style", "continuous") != "continuous"):
                    raise ValueError("requires a continuous positive-width outer-layer trace")
                curve = Decimal(row.get("curve", "0"))
                if not curve.is_finite() or abs(curve) >= 360 or (curve and cap != "round"):
                    raise ValueError("unsupported curved-wire sweep or cap")
                a, b = [tuple(float(nm(row.get(axis + suffix), subgrid=True))
                              for axis in ("x", "y")) for suffix in ("1", "2")]
                if a == b:
                    raise ValueError("zero-length trace")
                primitives.append(dict(source=ref, layers=(layer,), shape="arc" if curve else "segment",
                                       a=a, b=b, width=width, cap=cap, curve=float(curve)))
            elif kind == "via":
                plating(row)
                diameter = nm(row.get("diameter", "0"))
                if row.get("extent") != "1-16" or row.get("shape", "round") != "round" or diameter <= 0:
                    raise ValueError("material proof requires an explicit round full-span via annulus")
                primitives.append(dict(source=ref, layers=FACES, shape="disk",
                                       center=position(1, row)[2:], diameter=diameter, plated=True))
            else:
                raise ValueError("artwork excluded from this witness")
        except (KeyError, ValueError, InvalidOperation) as exc:
            omitted(ref, exc)
    try:
        proof = material_continuity(result["terminals"], primitives, bores)
    except (ShapelyError, ValueError, OverflowError) as exc:
        result["material"] = {"reason": "finite_width_geometry_unresolved", "detail": str(exc)}
        return result
    result["material"] = proof["material"]
    if approximations:
        result["material"]["conservative_shapes"] = approximations
    if proof["witnesses"] and not proof["unreachable"]:
        result.update(status="witnessed", witnesses=proof["witnesses"],
                      scope="Finite outer-layer copper regions minus all known drill voids")
        result.pop("reason", None)
        result.pop("unreachable", None)
    return result


def routed_continuity(prediction, board_path, signal_name, *, pad_pair=None):
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
    targets = terminals
    if pad_pair is not None:
        if len(pad_pair) != 4:
            raise ValueError("a pad pair requires two part and pad names")
        targets = sorted([("pad", *pad_pair[:2]), ("pad", *pad_pair[2:])])
        if targets[0] == targets[1] or any(t not in terminals for t in targets):
            raise ValueError("choose two distinct schematic pads on the selected net")
    result = dict(
        status="unresolved",
        signal=signal_name,
        board_sha256=digest,
        schematic_sha256=prediction["schematic_sha256"],
        coordinate_unit="nm",
        terminals=[list(t) for t in targets],
        witnesses=[],
        unsupported=[],
        scope="Straight outer-layer centerlines with drill clearance and supported plated approaches",
    )
    if pad_pair is not None:
        result.update(query="pad_pair", net_terminal_count=len(terminals))
    if signal is None or len(targets) < 2:
        result["reason"] = "missing_signal_or_fewer_than_two_schematic_terminals"
        return result
    edges = defaultdict(list)
    points = set()
    traces = []
    landings = []
    surface = defaultdict(list)

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
                if pad.tag == "smd":
                    diameter = min(nm(pad.get("dx")), nm(pad.get("dy")))
                    surface[node].append((term, ref, diameter))
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
                if row.get("style", "continuous") != "continuous":
                    raise ValueError("wire style does not establish a continuous centerline")
                layer = int(row.attrib["layer"])
                if (
                    layer not in FACES
                    or nm(row.get("width")) <= 0
                    or Decimal(row.get("curve", "0")) != 0
                    or row.get("cap", "round") != "round"
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
    connect_landings(interior, edges, surface)
    root = targets[0]
    previous = rooted_tree(edges, root)
    missing = [list(t) for t in targets if t not in previous]
    if missing:
        reason = "trace_does_not_demonstrably_reach_plating" if blocked_terminal else "no_supported_copper_path"
        result.update(reason=reason, unreachable=missing)
        return material_fallback(result, board, signal, terminals, elements, footprints, bores)
    witnesses = []
    for destination in targets[1:]:
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
