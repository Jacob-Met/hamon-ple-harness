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


def rule_via_annulus(board, drill):
    """Return the outer diameter and its source rules for an automatic through via."""
    decks = board.findall("designrules")
    if len(decks) != 1:
        raise ValueError("expected a single active board design-rule set")
    values = {}
    for name in ("rvViaOuter", "rlMinViaOuter", "rlMaxViaOuter"):
        entries = decks[0].findall(f"param[@name='{name}']")
        if len(entries) != 1:
            raise ValueError(f"expected one board design rule for {name}")
        values[name] = entries[0].get("value", "").strip()

    def number(text, name):
        if len(text) > 32 or not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", text):
            raise ValueError(f"unsupported nonnegative design rule {name}")
        return Fraction(text)

    def length(name):
        text = values[name]
        match = re.fullmatch(r"(.+?)\s*(mm|mil)", text)
        if not match:
            raise ValueError(f"expected an explicit mm or mil dimension for {name}")
        return number(match[1], name) * {"mm": 1_000_000, "mil": 25_400}[match[2]]

    ratio = number(values["rvViaOuter"], "rvViaOuter")
    lower, upper = length("rlMinViaOuter"), length("rlMaxViaOuter")
    if upper <= 0 or lower > upper:
        raise ValueError("inconsistent outer annulus rule bounds")
    ring = min(upper, max(lower, ratio * drill))
    if ring <= 0:
        raise ValueError("no positive outer annulus established")
    # Round inward if a legal dimension contains a fraction of a nanometre.
    diameter = int(drill + 2 * ring)
    return diameter, dict(rules=values, drill_nm=drill, outer_diameter_lower_bound_nm=diameter)


def wire_material(row, ref):
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
    return dict(source=ref, layers=(layer,), shape="arc" if curve else "segment",
                           a=a, b=b, width=width, cap=cap, curve=float(curve))


def milling_sources(board, layer=46):
    """Selected fabrication-layer artwork in the board and placed packages."""
    def on_milling_layer(row):
        return re.fullmatch(rf"\+?0*{layer}", row.get("layer", "").strip()) is not None

    plain = [row for row in board.findall("plain/*") if on_milling_layer(row)]
    groups = [(None, plain)] if plain else []
    packages = {
        (lib.get("name"), package.get("name")): rows
        for lib in board.findall("libraries/library")
        for package in lib.findall("packages/package")
        if (rows := [row for row in package if on_milling_layer(row)])
    }
    if packages:
        for element in board.findall("elements/element"):
            rows = packages.get((element.get("library"), element.get("package")))
            if rows:
                groups.append((element, rows))
    return groups


def milling_edges(groups):
    """Yield placed straight edges or signed axis-quarter milling edges."""
    coordinate = partial(nm, subgrid=True)
    for element, rows in groups:
        edges = []
        for row in rows:
            curve = Decimal(row.get("curve", "0"))
            if (row.tag != "wire" or coordinate(row.get("width")) != 0
                    or not curve.is_finite() or curve not in (0, -90, 90)
                    or row.get("style", "continuous") != "continuous"):
                raise ValueError("unsupported layer 46 milling primitive")
            ends, mirror = [], False
            for suffix in ("1", "2"):
                vertex = {axis: row.attrib[axis + suffix] for axis in ("x", "y")}
                if element is None:
                    point = coordinate(vertex["x"]), coordinate(vertex["y"])
                else:
                    x, y, mirror = placed_center(element, vertex, coordinate)
                    point = x, y
                ends.append(tuple(Fraction(value) for value in point))
            sweep = -int(curve) if mirror else int(curve)
            edges.append((*ends, sweep) if sweep else tuple(ends))
        yield edges



def dimension_records(board):
    """Parse placed layer-20 wires without treating their pen width as substrate."""
    coordinate = partial(nm, subgrid=True)
    records = []
    libraries = _named(board.findall("libraries/library"))
    for element, rows in milling_sources(board, 20):
        if element is None:
            parent = board.find("plain")
            prefix = ("plain",)
        else:
            parent = _named(libraries[element.get("library")].findall(
                "packages/package"))[element.get("package")]
            prefix = ("element", element.get("name"), "library", element.get("library"),
                      "package", element.get("package"))
        for row in rows:
            if (row.tag != "wire" or coordinate(row.get("width")) < 0
                    or row.get("style", "continuous") != "continuous"):
                raise ValueError("unsupported layer 20 outline primitive")
            curve = Decimal(row.get("curve", "0"))
            if not curve.is_finite() or curve not in (0, -90, 90, -180, 180):
                raise ValueError("unsupported layer 20 outline sweep")
            points, mirror = [], False
            for suffix in ("1", "2"):
                vertex = {axis: row.attrib[axis+suffix] for axis in ("x", "y")}
                if element is None:
                    point = coordinate(vertex["x"]), coordinate(vertex["y"])
                else:
                    x, y, mirror = placed_center(element, vertex, coordinate)
                    point = x, y
                points.append(tuple(Fraction(v) for v in point))
            records.append(dict(a=points[0], b=points[1],
                curve=-int(curve) if mirror else int(curve),
                source=prefix+(row.tag, parent.findall(row.tag).index(row))))
    return records


def resolved_board_domain(board, result):
    """Keep absence of an outline distinct from an unsupported declared outline."""
    from hamon_eagle_material import nominal_board_domain
    result["board_domain"] = {"status": "unresolved", "layer": 20}
    domain, evidence = nominal_board_domain(dimension_records(board))
    result["board_domain"] = evidence
    return domain


def material_fallback(result, board, signal, terminals, elements, footprints, bores, milling=()):
    """Prove additional routes using a conservative subset of finite copper."""
    try:
        from hamon_eagle_material import material_continuity, milling_cutouts
        from shapely.errors import ShapelyError
    except ImportError:
        result["material"] = {"reason": "finite_width_geometry_dependency_unavailable"}
        return result
    primitives, approximations, derived_vias = [], [], []
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
                primitives.append(wire_material(row, ref))
            elif kind == "via":
                drill = plating(row)
                if row.get("extent") != "1-16" or row.get("shape", "round") != "round":
                    raise ValueError("material proof requires a round full-span via")
                diameter = nm(row.get("diameter", "0"))
                if diameter == 0:
                    diameter, evidence = rule_via_annulus(board, drill)
                    derived_vias.append(dict(source=ref, **evidence))
                primitives.append(dict(source=ref, layers=FACES, shape="disk",
                                       center=position(1, row)[2:], diameter=diameter, plated=True))
            else:
                raise ValueError("artwork excluded from this witness")
        except (KeyError, ValueError, InvalidOperation) as exc:
            omitted(ref, exc)
    try:
        domain = resolved_board_domain(board, result)
        milling_evidence = {}
        cutouts = milling_cutouts(milling_edges(milling), milling_evidence)
        proof = material_continuity(result["terminals"], primitives, bores, cutouts, domain)
    except (ShapelyError, KeyError, InvalidOperation, ValueError, OverflowError) as exc:
        result["material"] = {"reason": "finite_width_geometry_unresolved", "detail": str(exc)}
        return result
    result["material"] = proof["material"]
    if milling:
        result["material"]["milling"] = milling_evidence
    if derived_vias:
        result["material"]["rule_derived_vias"] = derived_vias
    if approximations:
        result["material"]["conservative_shapes"] = approximations
    if proof["witnesses"] and not proof["unreachable"]:
        result.update(status="witnessed", witnesses=proof["witnesses"],
                      scope=("Finite outer-layer copper clipped to the nominal board domain, drills and milling"
                             if domain is not None else
                             "Unbounded nominal artwork minus drills and milling; no board outline supplied"))
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
    dimensions = milling_sources(board, 20)
    result = dict(
        status="unresolved",
        signal=signal_name,
        board_sha256=digest,
        schematic_sha256=prediction["schematic_sha256"],
        coordinate_unit="nm",
        terminals=[list(t) for t in targets],
        witnesses=[],
        unsupported=[],
        board_domain=({"status": "unresolved", "layer": 20} if dimensions else
                      {"status": "unbounded_artwork", "reason": "no_layer_20_outline"}),
        scope="Unbounded straight outer-layer artwork with drill clearance and plated approaches",
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
    milling = milling_sources(board)
    if milling or dimensions:
        result["board_domain"] = {"status": "unresolved", "layer": 20}
        # Centerline routes do not model the bounded substrate or its cutouts.
        result["reason"] = "no_supported_copper_path"
        return material_fallback(result, board, signal, terminals, elements, footprints, bores, milling)
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


def inter_net_contact(board_path, net_names):
    """Witness overlap of signal wires, without asserting whole-board clearance."""
    if len(net_names) != 2 or net_names[0] == net_names[1]:
        raise ValueError("Select two distinct board signals")
    board, digest = _read(board_path, "board")
    signals = _named(board.findall("signals/signal"))
    if any(name not in signals for name in net_names):
        raise ValueError("Unknown board signal in contact query")
    dimensions = milling_sources(board, 20)
    result = dict(status="unresolved", signals=list(net_names), board_sha256=digest,
                  board_domain=({"status": "unresolved", "layer": 20} if dimensions else
                                {"status": "unbounded_artwork", "reason": "no_layer_20_outline"}),
                  scope="Same-layer outer wires minus known circular drills and supported milling cutouts",
                  clearance="not_certified", contacts=[], unsupported=[])
    try:
        from hamon_eagle_material import milling_cutouts, wire_contact_witnesses
        from shapely.errors import ShapelyError
    except ImportError:
        result["reason"] = "finite_width_geometry_dependency_unavailable"
        return result
    try:
        bores = board_bores(board, _named(board.findall("elements/element")),
                            _named(board.findall("libraries/library")), None, set())
        groups = []
        for name in net_names:
            wires = []
            ordinal = defaultdict(int)
            for row in signals[name]:
                ref = (row.tag, ordinal[row.tag])
                ordinal[row.tag] += 1
                if row.tag == "contactref":
                    continue
                try:
                    if row.tag != "wire":
                        raise ValueError("only explicit wire copper considered")
                    wires.append(wire_material(row, ref))
                except (KeyError, ValueError, InvalidOperation) as exc:
                    result["unsupported"].append(dict(signal=name, source=ref, reason=str(exc)))
            groups.append(wires)
        domain = resolved_board_domain(board, result)
        milling = milling_sources(board)
        milling_evidence = {}
        cutouts = milling_cutouts(milling_edges(milling), milling_evidence)
        if milling:
            result["milling"] = milling_evidence
        result["contacts"] = wire_contact_witnesses(
            groups, bores, net_names, result["unsupported"], cutouts, domain)
        result["scope"] = ("Same-layer outer wires clipped to the nominal board domain, drills and milling"
                           if domain is not None else
                           "Unbounded nominal wire artwork minus drills and milling; no board outline supplied")
    except (KeyError, ValueError, InvalidOperation, ShapelyError, OverflowError) as exc:
        result["reason"] = "material_geometry_unresolved"
        result["detail"] = str(exc)
        return result
    if result["contacts"]:
        result["status"] = "contact_witnessed"
    else:
        result["reason"] = "no_positive_overlap_witness"
    return result
