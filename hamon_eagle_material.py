"""Sufficient finite-width routing proofs for explicit outer-layer EAGLE copper.

All input lengths and output coordinates are nanometres. Copper polygons lie
inside supported artwork; cut polygons contain the true drilled disks. A path
inside the guarded polygons has at least 1 nm nominal in-plane clearance.
"""

from collections import Counter, defaultdict, deque
from fractions import Fraction
from math import floor, ceil, cos, hypot, pi, radians, sin, sqrt

from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize_full, unary_union
from shapely.strtree import STRtree


# At the parser's maximum coordinate (10**13 nm), floating point ulps are
# below .002 nm. Reserve a second nanometre beyond the reported clearance.
_GUARD = 2.0
_CLEARANCE = 1.0
_QUADS = 32
_MITRE_LIMIT = 5


def _polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _polygons(part)


def _arc_copper(primitive):
    """An inner tube around a chordal approximation of a round-capped EAGLE arc."""
    a, b = primitive["a"], primitive["b"]
    dx, dy = b[0] - a[0], b[1] - a[1]
    chord = hypot(dx, dy)
    half = radians(primitive["curve"]) / 2
    width = float(primitive["width"])
    if (primitive.get("cap", "round") != "round" or
            not (chord > 0 and width > 2 * _GUARD and 0 < abs(half) < pi)):
        return None
    radius = chord / (2 * sin(half))
    if abs(radius) > 1e13:
        return None
    # Bound centerline error by one sixteenth of the nominal trace width.
    # 2 R sin(step/4)^2 is the maximum distance from chord to circular arc.
    steps = max(1, ceil(abs(half) * sqrt(2 * abs(radius) / (width / 16))))
    if steps > 4096:
        return None
    error = 2 * abs(radius) * sin(half / (2 * steps)) ** 2
    tube_radius = width / 2 - error
    pts = [a]
    for i in range(1, steps):
        q = (2 * i / steps - 1) * half
        along = 0.5 + radius * sin(q) / chord
        normal = -2 * radius * sin((half + q) / 2) * sin((half - q) / 2) / chord
        pts.append((a[0] + along * dx - normal * dy, a[1] + along * dy + normal * dx))
    pts.append(b)
    # Rounded joins and caps stay inside the union of disks swept by these chords.
    return LineString(pts).buffer(tube_radius, quad_segs=_QUADS)


def _inner_copper(primitive):
    shape = primitive["shape"]
    if shape == "segment":
        a, b = primitive["a"], primitive["b"]
        cap = primitive.get("cap", "round")
        if cap not in ("round", "flat") or a == b or primitive["width"] <= 0:
            return None
        copper = LineString([a, b]).buffer(float(primitive["width"]) / 2,
                                           quad_segs=_QUADS, cap_style=cap)
    elif shape == "arc":
        copper = _arc_copper(primitive)
        if copper is None:
            return None
    elif shape == "rectangle":
        x, y = map(float, primitive["center"])
        w, h = map(float, primitive["size"])
        if min(w, h) <= 0:
            return None
        copper = box(x - w/2, y - h/2, x + w/2, y + h/2)
    elif shape == "disk":
        if primitive["diameter"] <= 0:
            return None
        copper = Point(primitive["center"]).buffer(float(primitive["diameter"]) / 2,
                                                    quad_segs=_QUADS)
    else:
        return None
    return copper.buffer(-_GUARD, join_style="mitre")



def _outline_cycles(records):
    """Orient degree-two endpoint components without depending on wire order."""
    adjacent = defaultdict(list)
    for index, row in enumerate(records):
        a, b = tuple(row["a"]), tuple(row["b"])
        if a == b:
            raise ValueError("zero-length layer 20 edge")
        adjacent[a].append(index)
        adjacent[b].append(index)
    if any(len(indices) != 2 for indices in adjacent.values()):
        raise ValueError("open or branched layer 20 outline")

    cycles = []
    unused_vertices = set(adjacent)
    used_edges = set()
    while unused_vertices:
        if len(cycles) == 3:
            raise ValueError("more than three layer 20 contours are unresolved")
        start = current = min(unused_vertices)
        ordered = []
        while True:
            unused_vertices.remove(current)
            choices = [i for i in adjacent[current] if i not in used_edges]
            if not choices:
                raise ValueError("incomplete layer 20 endpoint cycle")
            def destination(i):
                row = records[i]
                return tuple(row["b"]) if tuple(row["a"]) == current else tuple(row["a"])
            index = min(choices, key=destination)
            row = dict(records[index])
            if tuple(row["a"]) != current:
                row["a"], row["b"] = row["b"], row["a"]
                row["curve"] = -row["curve"]
            row["a"], row["b"] = tuple(row["a"]), tuple(row["b"])
            ordered.append(row)
            used_edges.add(index)
            current = row["b"]
            if current == start:
                break
        cycles.append(ordered)
    return cycles


def _contour_domain(ordered, *, removed=False):
    """Enclose removed material, or inset retained material, at validated quarters."""
    curved = [row for row in ordered if row["curve"]]
    if removed and len(curved) > 1:
        # The supported multi-arc opening has four equal convex quarters and
        # four straight tangent bridges. The contour checks below establish
        # the joins, winding, and isolation of their local corner squares.
        squared_chords = {(r["b"][0]-r["a"][0])**2 + (r["b"][1]-r["a"][1])**2
                        for r in curved}
        if (len(ordered) != 8 or len(curved) != 4 or len(squared_chords) != 1
                or len({r["curve"] for r in curved}) != 1):
            raise ValueError("multi-quarter inner contour must be a simple rounded rectangle")
        guard2 = (4 * Fraction(_GUARD))**2
        for row in ordered:
            dx, dy = row["b"][0]-row["a"][0], row["b"][1]-row["a"][1]
            if not row["curve"] and dx*dx + dy*dy <= guard2:
                raise ValueError("rounded opening has unresolved corner separation")
    start = ordered[0]["a"]

    def sub(a, b):
        return (a[0]-b[0], a[1]-b[1])

    def dot(a, b):
        return a[0]*b[0]+a[1]*b[1]

    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    def tangent(radial, sign):
        return (-sign*radial[1], sign*radial[0])

    sharp_points = [start]
    corners = {}
    for index, row in enumerate(ordered):
        a, b, sweep = row["a"], row["b"], row["curve"]
        if sweep:
            if abs(sweep) != 90:
                raise ValueError("only tangent quarter outlines are supported")
            sign = 1 if sweep > 0 else -1
            dx, dy = sub(b, a)
            center = ((a[0]+b[0]-sign*dy)/2, (a[1]+b[1]+sign*dx)/2)
            u, v = sub(a, center), sub(b, center)
            if dot(u, v) or dot(u, u) != dot(v, v) or not dot(u, u):
                raise ValueError("layer 20 arc is not a circular quarter")
            previous, following = ordered[index-1], ordered[(index+1) % len(ordered)]
            incoming, outgoing = sub(a, previous["a"]), sub(following["b"], b)
            ta, tb = tangent(u, sign), tangent(v, sign)
            if (previous["curve"] or following["curve"]
                    or cross(incoming, ta) or dot(incoming, ta) <= 0
                    or cross(outgoing, tb) or dot(outgoing, tb) <= 0):
                raise ValueError("layer 20 arc does not join its straight tangents")
            vertex = (a[0]+b[0]-center[0], a[1]+b[1]-center[1])
            # The two radii define the corner's local square at any orientation.
            # Its world-axis bounding box can include unrelated board edges.
            square = Polygon([tuple(map(float, p)) for p in (center, a, vertex, b)])
            corners[index] = (center, u, hypot(*u), vertex, square)
            sharp_points.append(vertex)
        sharp_points.append(b)
    area2 = sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(sharp_points, sharp_points[1:]))
    if area2 == 0:
        raise ValueError("zero-area layer 20 outline")
    winding = 1 if area2 > 0 else -1
    sharp = Polygon([(float(x), float(y)) for x, y in sharp_points])
    if (not sharp.is_valid or not sharp.exterior.is_simple
            or sharp.minimum_clearance <= 4*_GUARD):
        raise ValueError("self-crossing or numerically unresolved layer 20 outline")
    checked = []
    reflex = {}
    for index, (_, _, _, _, square) in corners.items():
        row = ordered[index]
        if row["curve"]*winding < 0:
            if removed and len(curved) > 1:
                raise ValueError("rounded opening has a reflex corner")
            if sharp.intersection(square).area > 0:
                raise ValueError("reflex layer 20 corner box intersects the board interior")
            reflex[index] = square
        elif not sharp.covers(square):
            raise ValueError("convex layer 20 corner box leaves the board")
        if any(square.intersection(other).area > 0 for other in checked):
            raise ValueError("overlapping layer 20 quarter corners")
        for j, other in enumerate(ordered):
            if j == index:
                continue
            points = [other["a"], other["b"]]
            if j in corners:
                points.insert(1, corners[j][3])
            hit = LineString([(float(x), float(y)) for x, y in points]).intersection(square)
            if hit.is_empty:
                continue
            expected = (row["a"] if j == (index-1) % len(ordered)
                        else row["b"] if j == (index+1) % len(ordered) else None)
            if expected is None or not hit.equals(Point(tuple(map(float, expected)))):
                raise ValueError("unrelated boundary enters a layer 20 corner box")
        checked.append(square)

    points = [tuple(map(float, start))]
    largest_error = 0.0
    for index, row in enumerate(ordered):
        if index in corners:
            center, radial, radius, _, _ = corners[index]
            radius = float(radius)
            # R*theta^2/(8*n^2) bounds the circular sagitta without cancellation.
            count = max(1, ceil((pi/2)*sqrt(radius/8)))
            # A convex removed corner needs circumscribed tangents. At a
            # reflex corner the interior side reverses, as it does on an outer rim.
            use_tangents = (index in reflex) != removed
            while True:
                if count > 4096 or len(points)+count+1 > 32768:
                    raise ValueError("layer 20 arc exceeds the bounded approximation")
                half_step = pi/(4*count)
                error = radius*half_step**2/2
                if use_tangents:
                    error /= cos(half_step)
                if error <= 1:
                    break
                count += 1
            largest_error = max(largest_error, error)
            sign = 1 if row["curve"] > 0 else -1
            cx, cy = map(float, center)
            ux, uy = map(float, radial)
            if use_tangents:
                # The adjacent tangent intersection lies on the angular bisector
                # at radius R / cos(half_step). Each polygon edge is tangent.
                scale = 1/cos(half_step)
                for step in range(count):
                    angle = sign*(pi/2)*(step+0.5)/count
                    points.append((cx+scale*(ux*cos(angle)-uy*sin(angle)),
                                   cy+scale*(ux*sin(angle)+uy*cos(angle))))
            else:
                for step in range(1, count):
                    angle = sign*(pi/2)*step/count
                    points.append((cx+ux*cos(angle)-uy*sin(angle),
                                   cy+ux*sin(angle)+uy*cos(angle)))
        points.append(tuple(map(float, row["b"])))
    polygon = Polygon(points)
    envelope = sharp.union(unary_union(list(reflex.values()))) if reflex else sharp
    if not polygon.is_valid:
        raise ValueError("invalid layer 20 polygon")
    if removed and not envelope.covers(polygon):
        # A tangent intersection can round just outside an oblique straight side.
        # The independently validated sharp envelope contains the real cutout,
        # so clipping to it preserves the removal bound before the outward guard.
        polygon = polygon.intersection(envelope)
        if polygon.geom_type != "Polygon" or not polygon.is_valid or polygon.is_empty:
            raise ValueError("unresolved enclosed layer 20 polygon")
    elif not envelope.covers(polygon):
        raise ValueError("unresolved inner layer 20 polygon")
    domain = polygon.buffer(_GUARD if removed else -_GUARD, join_style="mitre",
                            mitre_limit=_MITRE_LIMIT)
    if removed and not domain.covers(polygon):
        raise ValueError("unresolved outward layer 20 guard")
    if domain.is_empty or not domain.is_valid:
        raise ValueError("layer 20 interior is unresolved at the numeric guard")
    return domain, {
        "status": "bounded", "layer": 20,
        "model": ("single_straight_outline_with_validated_tangent_quarters" if reflex else
                  "single_straight_outline_with_validated_convex_quarters"),
        "sources": [list(row["source"]) for row in ordered],
        "wire_count": len(ordered), "quarter_count": len(corners),
        "arc_sagitta_upper_bound_nm": largest_error,
        "boundary_guard_nm": _GUARD,
        ("removed_area_upper_bound_mm2" if removed else "interior_area_lower_bound_mm2"): domain.area/1e12,
    }



def _hole_clearance_features(rows):
    """Exact source boundary pieces, enclosing each validated quarter in its local square."""
    features = []
    for row in rows:
        a, b = row["a"], row["b"]
        if row["curve"]:
            sign = 1 if row["curve"] > 0 else -1
            dx, dy = b[0] - a[0], b[1] - a[1]
            center = (Fraction(a[0] + b[0] - sign * dy) / 2,
                      Fraction(a[1] + b[1] + sign * dx) / 2)
            vertex = (a[0] + b[0] - center[0], a[1] + b[1] - center[1])
            points = (a, center, b, vertex)
        else:
            points = (a, b)
        xs, ys = zip(*points)
        features.append((points, (min(xs), min(ys), max(xs), max(ys))))
    return features


def _convex_supports(rows):
    """Map outward straight-edge normals to one supporting point on a convex loop."""
    if any(row["curve"] for row in rows):
        return {}
    vertices = [row["a"] for row in rows]
    edges = [(row["b"][0] - row["a"][0], row["b"][1] - row["a"][1]) for row in rows]
    turns = [dx * ey - dy * ex for (dx, dy), (ex, ey) in zip(edges, edges[1:] + edges[:1])]
    if min(turns) < 0 < max(turns) or not any(turns):
        return {}
    winding = 1 if max(turns) > 0 else -1
    supports = {}
    for (dx, dy), p in zip(edges, vertices):
        nx, ny = winding * dy, -winding * dx
        scale = abs(nx or ny)
        supports[(Fraction(nx) / scale, Fraction(ny) / scale)] = p
    return supports


def _segments_far(a, b, c, d, distance):
    """Verify disjoint finite segments and all four exact point-to-segment distances."""
    def side(p, q, r):
        return (q[0]-p[0]) * (r[1]-p[1]) - (q[1]-p[1]) * (r[0]-p[0])

    def point_far(p, q, r):
        dx, dy = r[0] - q[0], r[1] - q[1]
        ux, uy = p[0] - q[0], p[1] - q[1]
        projection = ux * dx + uy * dy
        length2 = dx * dx + dy * dy
        if projection >= length2:
            ux, uy = p[0] - r[0], p[1] - r[1]
        elif projection > 0:
            return (ux * dy - uy * dx) ** 2 > distance ** 2 * length2
        return ux * ux + uy * uy > distance ** 2

    # For collinear disjoint segments the bounding boxes or endpoint tests suffice.
    s1, s2, s3, s4 = side(a, b, c), side(a, b, d), side(c, d, a), side(c, d, b)
    if s1 * s2 <= 0 and s3 * s4 <= 0 and any((s1, s2, s3, s4)):
        return False
    return all((point_far(a, c, d), point_far(b, c, d),
                point_far(c, a, b), point_far(d, a, b)))


def _inside_chord_loop(point, rows):
    """Exact parity for a point already separated from all quarter-corner squares."""
    x, y = point
    inside = False
    for row in rows:
        (ax, ay), (bx, by) = row["a"], row["b"]
        if (ay > y) != (by > y):
            cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
            if (cross > 0) == (by > ay):
                inside = not inside
    return inside


def _certified_hole_separation(left, right):
    """Prove the strict post-guard clearance using authored coordinates."""
    guard = Fraction(_GUARD)
    required = 2 * guard
    # Convex mitred offsets remain inside every outward-shifted source half-plane.
    # For opposing faces this is sharp, including at the strict 4 nm threshold.
    left_faces, right_faces = _convex_supports(left), _convex_supports(right)
    for (nx, ny), a in left_faces.items():
        b = right_faces.get((-nx, -ny))
        if b is not None:
            projection = nx * (b[0] - a[0]) + ny * (b[1] - a[1])
            if projection > 0 and projection ** 2 > (required + 2 * guard) ** 2 * (nx ** 2 + ny ** 2):
                return True

    # Every tangent/chord of a validated quarter stays inside its exact local
    # square. A mitre moves a boundary by at most (limit + 1) * guard: uncut
    # tips stop at limit * guard; each clipped end is at most one further
    # guard from the bisector clip point. Compare both boundary envelopes.
    distance = required + 2 * (_MITRE_LIMIT + 1) * guard
    first, second = _hole_clearance_features(left), _hole_clearance_features(right)
    for p, (x0, y0, x1, y1) in first:
        for q, (u0, v0, u1, v1) in second:
            dx, dy = max(0, u0-x1, x0-u1), max(0, v0-y1, y0-v1)
            if dx * dx + dy * dy > distance ** 2:
                continue
            if len(p) == 2 and len(q) == 2 and _segments_far(*p, *q, distance):
                continue
            return False
    # Disjoint boundaries can still nest. Replacing a quarter by its chord
    # preserves parity at points outside its enclosing square.
    return not (_inside_chord_loop(left[0]["a"], right) or _inside_chord_loop(right[0]["a"], left))


def nominal_board_domain(records):
    """Qualify one outer boundary and up to two disjoint, strictly interior removed contours."""
    if not records:
        return None, {"status": "unbounded_artwork", "reason": "no_layer_20_outline"}
    if len(records) > 2048:
        raise ValueError("layer 20 outline exceeds the supported edge count")
    cycles = _outline_cycles(records)
    if len(cycles) == 1:
        return _contour_domain(cycles[0])

    # A qualified outer inset strictly contains the entire inner enclosure.
    # Its area must therefore exceed the inner inset's area. This ordering is
    # independent of traversal direction, wire order and which contour is curved.
    candidates = [(_contour_domain(rows), rows) for rows in cycles]
    candidates.sort(key=lambda candidate: candidate[0][0].area, reverse=True)
    (domain, info), _ = candidates[0]
    cutouts = []
    # Order by physical position so swapping loops of equal area preserves their roles.
    inner_rows = sorted((candidate[1] for candidate in candidates[1:]), key=lambda rows: rows[0]["a"])
    for rows in inner_rows:
        removed, cut_info = _contour_domain(rows, removed=True)
        if not domain.contains_properly(removed):
            raise ValueError("layer 20 hole crosses, touches, or lies outside the guarded board")
        for other, _ in cutouts:
            if removed.distance(other) <= 2 * _GUARD or not _certified_hole_separation(inner_rows[0], rows):
                raise ValueError("layer 20 holes intersect, nest, touch, or have unresolved separation")
        if cut_info["quarter_count"]:
            sources = [list(row["source"]) for row in rows if row["curve"]]
            if len(sources) == 1:
                cut_info["quarter_source"] = sources[0]
            else:
                cut_info["quarter_sources"] = sources
        cutouts.append((removed, cut_info))

    removed = cutouts[0][0] if len(cutouts) == 1 else unary_union([geometry for geometry, _ in cutouts])
    result = domain.difference(removed)
    if (result.geom_type != "Polygon" or not result.is_valid
            or len(result.interiors) != len(cutouts)):
        raise ValueError("unresolved layer 20 board-with-hole topology")
    holes = [cut_info for _, cut_info in cutouts]
    hole_sources = [source for cut_info in holes for source in cut_info["sources"]]
    if len(holes) == 1:
        hole_info = holes[0]
        hole_model = "outer_outline_with_one_straight_dimension_hole"
        if hole_info["quarter_count"]:
            hole_model = "outer_outline_with_one_tangent_quarter_dimension_hole"
            if hole_info["quarter_count"] == 1:
                info = dict(info, hole_quarter_source=hole_info["quarter_source"])
            else:
                hole_model = "outer_outline_with_rounded_rectangle_dimension_hole"
                info = dict(info, hole_quarter_sources=hole_info["quarter_sources"])
            info = dict(info, hole_quarter_count=hole_info["quarter_count"],
                        hole_arc_error_upper_bound_nm=hole_info["arc_sagitta_upper_bound_nm"],
                        removed_area_upper_bound_mm2=hole_info["removed_area_upper_bound_mm2"])
    else:
        hole_model = "outer_outline_with_two_disjoint_dimension_holes"
        info = dict(info, dimension_holes=holes, hole_count=len(holes),
                    guarded_hole_separation_nm=cutouts[0][0].distance(cutouts[1][0]))
    return result, dict(info,
        model=hole_model,
        outer_model=info["model"], hole_sources=hole_sources,
        sources=info["sources"]+hole_sources, wire_count=len(records),
        hole_wire_count=len(hole_sources), interior_area_lower_bound_mm2=result.area/1e12)


def _milling_capsule_bounds(edges):
    """Exact recognition of one simple axis-aligned, six-edge round slot."""
    quarters = []
    for a, b, sweep in edges:
        if not sweep:
            continue
        if abs(sweep) != 90:
            raise ValueError("unsupported curved layer 46 contour")
        sign = 1 if sweep > 0 else -1
        dx, dy = b[0]-a[0], b[1]-a[1]
        center = ((a[0]+b[0]-sign*dy)/2, (a[1]+b[1]+sign*dx)/2)
        u, v = (a[0]-center[0], a[1]-center[1]), (b[0]-center[0], b[1]-center[1])
        if (u[0]*u[1] or v[0]*v[1] or u[0]*v[0]+u[1]*v[1]
                or u[0]*u[0]+u[1]*u[1] != v[0]*v[0]+v[1]*v[1]):
            raise ValueError("layer 46 slot is not axis-tangent")
        quarters.append((center, abs(u[0]+u[1])))
    centers = sorted({c for c, _ in quarters})
    radii = {r for _, r in quarters}
    if len(edges) != 6 or len(quarters) != 4 or len(centers) != 2 or len(radii) != 1:
        raise ValueError("curved layer 46 contour is not a supported capsule")
    radius, = radii
    if radius <= 0:
        raise ValueError("zero-radius layer 46 capsule")
    a, b = centers
    if a[0] == b[0] and a[1] < b[1]:
        x, y0, y1, r = a[0], a[1], b[1], radius
        points = [(x-r,y0),(x,y0-r),(x+r,y0),(x+r,y1),(x,y1+r),(x-r,y1)]
    elif a[1] == b[1] and a[0] < b[0]:
        x0, x1, y, r = a[0], b[0], a[1], radius
        points = [(x0,y+r),(x0-r,y),(x0,y-r),(x1,y-r),(x1+r,y),(x1,y+r)]
    else:
        raise ValueError("layer 46 capsule axis is unsupported")
    def canonical(a, b, sweep):
        return (a, b, sweep) if a < b else (b, a, -sweep)
    expected = Counter(canonical(a,b,s) for a,b,s in
                       zip(points, points[1:]+points[:1], (90,90,0,90,90,0)))
    if Counter(canonical(a,b,s) for a,b,s in edges) != expected:
        raise ValueError("layer 46 quarters do not form the declared simple capsule")
    # Every endpoint is a cardinal extremum; the exact capsule lies in this box.
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (floor(min(xs)), floor(min(ys)), ceil(max(xs)), ceil(max(ys)))


def milling_cutouts(groups, evidence=None):
    """Outward-enclose straight contours and explicitly recognized round slots."""
    cutouts, envelopes = [], []
    straight_count = 0
    for group_index, raw_edges in enumerate(groups):
        edges = [(a,b,rest[0] if rest else 0) for a,b,*rest in raw_edges]
        adjacent = defaultdict(list)
        for index, (a,b,_) in enumerate(edges):
            adjacent[a].append(index)
            adjacent[b].append(index)
        if any(a == b for a,b,_ in edges) or any(len(v) != 2 for v in adjacent.values()):
            raise ValueError("open or branched layer 46 milling contour")
        remaining = set(range(len(edges)))
        while remaining:
            component, queue = set(), [min(remaining)]
            while queue:
                index = queue.pop()
                if index in component:
                    continue
                component.add(index)
                a,b,_ = edges[index]
                queue.extend(adjacent[a]+adjacent[b])
            remaining.difference_update(component)
            selected = [edges[i] for i in sorted(component)]
            if any(s for _,_,s in selected):
                bounds = _milling_capsule_bounds(selected)
                cutouts.append(box(*bounds))
                envelopes.append({"source_group":group_index,
                                  "edge_indices":sorted(component),
                                  "bounds_nm":list(bounds)})
            else:
                polygons, cuts, dangles, invalid = polygonize_full(
                    [LineString([a,b]) for a,b,_ in selected])
                if polygons.is_empty or not all(g.is_empty for g in (cuts,dangles,invalid)):
                    raise ValueError("invalid layer 46 milling contour")
                cutouts.extend(polygons.geoms)
                straight_count += len(polygons.geoms)
    if evidence is not None:
        evidence.update(straight_contours=straight_count,
                        conservative_capsule_boxes=envelopes,
                        boundary_guard_nm=_GUARD)
    # All cutout boxes/interiors are removed, including overlapping/nested ones.
    return unary_union(cutouts).buffer(_GUARD, join_style="mitre") if cutouts else None


class _Drills:
    def __init__(self, bores):
        self.disks = []
        self.owners = {}
        self.centers = []
        for x, y, diameter, owner in bores:
            x, y = float(x), float(y)
            # GEOS approximates a circular buffer by an inscribed polygon.
            # Scale the radius so even its apothem exceeds the physical bore.
            outer_radius = (float(diameter) / 2 + _GUARD) / cos(pi / (4 * _QUADS))
            disk = Point(x, y).buffer(outer_radius, quad_segs=_QUADS)
            if owner is not None:
                self.owners[tuple(owner)] = len(self.disks)
            self.disks.append(disk)
            self.centers.append((x, y, float(diameter) / 2, outer_radius))
        self.tree = STRtree(self.disks)

    def clip(self, copper):
        nearby = [int(i) for i in self.tree.query(copper, predicate="intersects")]
        if not nearby:
            return copper, nearby
        cuts = unary_union([self.disks[i] for i in nearby])
        return copper.difference(cuts), nearby

    def collar_port(self, source, copper, nearby):
        """Choose an accessible full circular collar around this approved bore."""
        own = self.owners.get(source)
        if own is None:
            return None
        x, y, radius, outer = self.centers[own]
        center = Point(x, y)
        if not copper.contains(center):
            return None
        limit = center.distance(copper.boundary)
        for index in nearby:
            if index != own:
                limit = min(limit, center.distance(self.disks[index]))
        if limit <= outer + 2 * _CLEARANCE:
            return None
        return {"point_nm": [x + (outer + limit)/2, y],
                "drill_center_nm": [x, y], "drill_radius_nm": radius,
                "barrel_wall_point_nm": [x + radius, y],
                "clear_collar_outer_radius_nm": limit}


def _overlap_disk(overlap):
    """Return a strictly interior disk in a common positive-area component."""
    best = None
    for patch in _polygons(overlap):
        center = patch.representative_point()
        radius = center.distance(patch.boundary) / 2
        if radius > 0 and (best is None or radius > best[1]):
            best = (tuple(center.coords[0]), radius)
    return best


def _parents(origin, neighbors):
    parents = {origin: None}
    queue = deque([origin])
    while queue:
        node = queue.popleft()
        for other, evidence in neighbors.get(node, ()):
            if other not in parents:
                parents[other] = node, evidence
                queue.append(other)
    return parents


def _walk(parents, end):
    nodes, links = [end], []
    while parents[end] is not None:
        end, link = parents[end]
        links.append(link)
        nodes.append(end)
    return nodes[::-1], links[::-1]


def _in_piece_path(piece, a, b):
    """Return an explicit polyline in a guarded polygon, including around holes."""
    if a == b:
        return [list(a)]
    if piece.covers(LineString([a, b])):
        return [list(a), list(b)]
    # Ordinary Delaunay triangulation can bridge a hole. The constrained
    # tessellation respects the exterior and every drilled boundary.
    triangles = constrained_delaunay_triangles(piece).geoms
    start = end = None
    edges = {}
    neighbors = defaultdict(list)
    for i, triangle in enumerate(triangles):
        if triangle.covers(Point(a)):
            start = i
        if triangle.covers(Point(b)):
            end = i
        corners = list(triangle.exterior.coords)
        for u, v in zip(corners, corners[1:]):
            key = tuple(sorted((u, v)))
            if key in edges:
                j = edges[key]
                midpoint = ((u[0] + v[0])/2, (u[1] + v[1])/2)
                neighbors[i].append((j, midpoint))
                neighbors[j].append((i, midpoint))
            else:
                edges[key] = i
    if start is None or end is None:
        return None
    parents = _parents(start, neighbors)
    if end not in parents:
        return None
    _, portals = _walk(parents, end)
    path = [a, *portals, b]
    if not piece.covers(LineString(path)):
        return None
    # Remove tessellation-only waypoints while checking each simplified chord.
    compact = [a]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not piece.covers(LineString([path[i], path[j]])):
            j -= 1
        compact.append(path[j])
        i = j
    return [list(p) for p in compact]


def _piece_graph(primitives, bores, cutouts, domain=None):
    drills = _Drills(bores)
    pieces, labels = [], []
    by_layer, terminal_nodes, graph = defaultdict(list), defaultdict(list), defaultdict(list)

    for primitive in primitives:
        source = tuple(primitive["source"])
        copper = _inner_copper(primitive)
        if copper is None or copper.is_empty:
            continue
        if domain is not None:
            copper = copper.intersection(domain)
        if cutouts is not None:
            copper = copper.difference(cutouts)
        clipped, nearby = drills.clip(copper)
        parts = sorted(_polygons(clipped), key=lambda p: (p.bounds, p.area))
        plated = primitive.get("plated", False)
        collar = drills.collar_port(source, copper, nearby) if plated else None
        port = collar["point_nm"] if collar else None
        terminal = primitive.get("terminal")
        barrel = {}
        for layer in primitive["layers"]:
            if layer not in (1, 16):
                continue
            for index, part in enumerate(parts):
                node = len(pieces)
                pieces.append(part)
                reaches_port = port is not None and part.contains(Point(port))
                labels.append((source, layer, index, collar if reaches_port else None))
                by_layer[layer].append(node)
                if reaches_port:
                    barrel[layer] = node
                if terminal is not None and (not plated or reaches_port):
                    terminal_nodes[tuple(terminal)].append(node)
        if set(barrel) == {1, 16}:
            link = {"kind": "plated_barrel", "source": list(source), **collar}
            bottom, top = barrel[16], barrel[1]
            graph[bottom].append((top, link))
            graph[top].append((bottom, link))

    for layer_nodes in by_layer.values():
        tree = STRtree([pieces[i] for i in layer_nodes])
        for position, node in enumerate(layer_nodes):
            for j in sorted(tree.query(pieces[node], predicate="intersects")):
                if j <= position:
                    continue
                neighbor = layer_nodes[j]
                if labels[node][0] == labels[neighbor][0]:
                    continue
                disk = _overlap_disk(pieces[node].intersection(pieces[neighbor]))
                if disk is None:
                    continue
                point, radius = disk
                link = {"kind": "overlap", "point_nm": list(point), "radius_nm": radius}
                graph[node].append((neighbor, link))
                graph[neighbor].append((node, link))
    return pieces, labels, terminal_nodes, graph


def _copper_node(layer, xy):
    return ["copper", layer, *xy]


def _witness(pieces, labels, parents, end_node, start, end):
    nodes, links = _walk(parents, end_node)
    steps = []
    for i, node in enumerate(nodes):
        source, layer, component, collar = labels[node]
        # The pad is the terminal, so its overlap/port is an admissible terminal
        # contact location. A clipped-away pad center is never a virtual bridge.
        enter = tuple(links[max(0, i-1)]["point_nm"])
        leave = tuple(links[min(i, len(links)-1)]["point_nm"])
        if i == 0 and collar is not None:
            enter = tuple(collar["point_nm"])
        if i == len(nodes)-1 and collar is not None:
            leave = tuple(collar["point_nm"])
        path = _in_piece_path(pieces[node], enter, leave)
        if path is None:
            return None
        material = {"kind": "component", "layer": layer, "component": component,
                    "path_nm": path, "in_plane_clearance_nm": _CLEARANCE}
        if collar is not None and i in (0, len(nodes)-1):
            material["terminal_collar"] = collar
        if i < len(links) and links[i]["kind"] == "overlap":
            material["exit_overlap"] = links[i]
        steps.append({"source": list(source),
                      "start": list(start) if i == 0 else _copper_node(layer, enter),
                      "end": list(end) if i == len(nodes)-1 else _copper_node(layer, leave),
                      "material": material})
        if i < len(links) and links[i]["kind"] == "plated_barrel":
            next_layer = labels[nodes[i+1]][1]
            wall = links[i]["barrel_wall_point_nm"]
            barrel = dict(links[i], path=[_copper_node(layer, leave),
                                         ["barrel_wall", layer, *wall],
                                         ["barrel_wall", next_layer, *wall],
                                         _copper_node(next_layer, leave)])
            steps.append({"source": list(source), "start": _copper_node(layer, leave),
                          "end": _copper_node(next_layer, leave), "material": barrel})
    return {"start": list(start), "end": list(end), "steps": steps}


def material_continuity(terminals, primitives, bores, cutouts=None, domain=None):
    """Return finite-area witnesses for already parsed, source-addressed artwork.

    A primitive describes a straight segment or round-capped circular arc,
    axis-aligned rectangle (center, size) or disk (center, diameter), its source
    tuple and outer layers. An optional terminal names its physical pad. Plated
    pieces bridge layers only at their own independently approved drill collar.
    Bores are (x, y, diameter, owner) as returned by board_bores.
    """
    terminals = sorted({tuple(t) for t in terminals})
    pieces, labels, contacts, graph = _piece_graph(primitives, bores, cutouts, domain)
    result = {"witnesses": [], "unreachable": [list(t) for t in terminals],
              "material": {"model": "finite_outer_layer_regions",
                           "in_plane_clearance_nm": _CLEARANCE}}
    if len(terminals) < 2:
        return result
    root, *others = terminals
    for root_node in contacts[root]:
        # Seed exactly one connected piece, even when a pad is cut into islands.
        parents = _parents(root_node, graph)
        missing = [list(t) for t in others if not any(n in parents for n in contacts[t])]
        if missing:
            if len(missing) < len(result["unreachable"]):
                result["unreachable"] = missing
            continue
        walks = []
        for term in others:
            for node in contacts[term]:
                if node not in parents:
                    continue
                path = _witness(pieces, labels, parents, node, root, term)
                if path is not None:
                    walks.append(path)
                    break
            else:
                break
        if len(walks) == len(others):
            result.update(witnesses=walks, unreachable=[])
            break
    return result



def wire_contact_witnesses(groups, bores, names, unsupported, cutouts=None, domain=None):
    """Find a strictly interior disk in both drilled wire regions on one layer."""
    drills = _Drills(bores)
    layers = []
    for name, group in zip(names, groups):
        pieces = defaultdict(list)
        for wire in group:
            inner = _inner_copper(wire)
            if inner is None:
                unsupported.append(dict(signal=name, source=wire["source"],
                                        reason="wire material outside geometric proof limits"))
                continue
            if domain is not None:
                inner = inner.intersection(domain)
            if cutouts is not None:
                inner = inner.difference(cutouts)
            clipped, _ = drills.clip(inner)
            if not clipped.is_empty:
                pieces[wire["layers"][0]].append((wire["source"], clipped))
        layers.append(pieces)
    contacts = []
    left, right = layers
    for layer in sorted(left.keys() & right.keys()):
        counterparts = right[layer]
        tree = STRtree([geom for _, geom in counterparts])
        for source, geom in left[layer]:
            for j in sorted(tree.query(geom, predicate="intersects")):
                other, target = counterparts[int(j)]
                overlap = geom.intersection(target)
                disk = _overlap_disk(overlap)
                if disk is None:
                    continue
                center, radius = disk
                contacts.append(dict(
                    layer=layer, center_nm=list(center), disk_radius_nm=radius,
                    sources=[dict(signal=name, primitive=ref)
                             for name, ref in zip(names, (source, other))],
                    overlap_lower_bound_mm2=overlap.area / 1e12))
    return contacts
