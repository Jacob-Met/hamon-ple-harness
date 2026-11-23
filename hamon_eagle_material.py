"""Sufficient finite-width routing proofs for explicit outer-layer EAGLE copper.

All input lengths and output coordinates are nanometres. Copper polygons lie
inside supported artwork; cut polygons contain the true drilled disks. A path
inside the guarded polygons has at least 1 nm nominal in-plane clearance.
"""

from collections import defaultdict, deque
from math import ceil, cos, hypot, pi, radians, sin, sqrt

from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree


# At the parser's maximum coordinate (10**13 nm), floating point ulps are
# below .002 nm. Reserve a second nanometre beyond the reported clearance.
_GUARD = 2.0
_CLEARANCE = 1.0
_QUADS = 32


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


def _overlap_disk(a, b):
    """Return a strictly interior disk in a common positive-area component."""
    best = None
    for patch in _polygons(a.intersection(b)):
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


def _piece_graph(primitives, bores):
    drills = _Drills(bores)
    pieces, labels = [], []
    by_layer, terminal_nodes, graph = defaultdict(list), defaultdict(list), defaultdict(list)

    for primitive in primitives:
        source = tuple(primitive["source"])
        copper = _inner_copper(primitive)
        if copper is None or copper.is_empty:
            continue
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
                disk = _overlap_disk(pieces[node], pieces[neighbor])
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


def material_continuity(terminals, primitives, bores):
    """Return finite-area witnesses for already parsed, source-addressed artwork.

    A primitive describes a straight segment or round-capped circular arc,
    axis-aligned rectangle (center, size) or disk (center, diameter), its source
    tuple and outer layers. An optional terminal names its physical pad. Plated
    pieces bridge layers only at their own independently approved drill collar.
    Bores are (x, y, diameter, owner) as returned by board_bores.
    """
    terminals = sorted({tuple(t) for t in terminals})
    pieces, labels, contacts, graph = _piece_graph(primitives, bores)
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
