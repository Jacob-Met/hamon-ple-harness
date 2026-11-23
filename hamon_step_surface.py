"""Exact embedding predicates for an already triangulated planar STEP shell.

The mass integral of a closed oriented chain need not describe a solid: sheets
can cross, overlap or touch at a nonmanifold vertex.  Work on the original
binary64 vertices, without a distance tolerance or coordinate welding.
"""
from fractions import Fraction


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _coplanar_excess(a, b, normal, shared):
    """Whether the convex intersection extends beyond the shared simplex."""
    drop = max(range(3), key=lambda k: abs(normal[k]))
    axes = [k for k in range(3) if k != drop]
    project = lambda p: tuple(p[k] for k in axes)
    a, b, shared = tuple(map(project, a)), tuple(map(project, b)), tuple(map(project, shared))

    def allowed(p):
        if len(shared) == 1:
            return p == shared[0]
        if len(shared) == 2:
            u, v = shared
            return _orient(u, v, p) == 0 and all(min(u[k], v[k]) <= p[k] <= max(u[k], v[k])
                                               for k in range(2))
        return False

    def inside(p, tri):
        signs = [_orient(tri[k], tri[(k+1) % 3], p) for k in range(3)]
        return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

    # Every extreme point of the intersection of two triangles is either an
    # included input vertex or an edge/edge intersection. Collinear overlaps
    # have an included endpoint and are covered by the first two checks.
    if any(inside(p, b) and not allowed(p) for p in a):
        return True
    if any(inside(p, a) and not allowed(p) for p in b):
        return True
    for i in range(3):
        p, q = a[i], a[(i+1) % 3]
        r = _sub(q, p)
        for j in range(3):
            u, v = b[j], b[(j+1) % 3]
            s, offset = _sub(v, u), _sub(u, p)
            denominator = r[0]*s[1] - r[1]*s[0]
            if denominator == 0:
                continue
            t = (offset[0]*s[1] - offset[1]*s[0]) / denominator
            w = (offset[0]*r[1] - offset[1]*r[0]) / denominator
            if 0 <= t <= 1 and 0 <= w <= 1:
                intersection = tuple(p[k] + t*r[k] for k in range(2))
                if not allowed(intersection):
                    return True
    return False


def _plane_slice(triangle, distances, axis):
    """The exact interval cut from a triangle by a nonparallel plane."""
    values = [p[axis] for p, d in zip(triangle, distances) if d == 0]
    for i in range(3):
        j = (i+1) % 3
        if distances[i]*distances[j] < 0:
            t = distances[i] / (distances[i] - distances[j])
            values.append(triangle[i][axis] + t*(triangle[j][axis] - triangle[i][axis]))
    return min(values), max(values)


def _intersection_exceeds_shared(a, b, na, nb, shared):
    # Triangles in distinct planes that share a whole edge cannot meet
    # anywhere else; their plane intersection is that edge's supporting line.
    direction = _cross(na, nb)
    if len(shared) == 2 and any(direction):
        return False
    db = [_dot(na, _sub(p, a[0])) for p in b]
    if all(d > 0 for d in db) or all(d < 0 for d in db):
        return False
    da = [_dot(nb, _sub(p, b[0])) for p in a]
    if all(d > 0 for d in da) or all(d < 0 for d in da):
        return False
    if not any(direction):
        # The same-side exclusions have already removed distinct parallel
        # planes. These triangles lie in exactly the same plane.
        return _coplanar_excess(a, b, na, shared)
    axis = max(range(3), key=lambda k: abs(direction[k]))
    alo, ahi = _plane_slice(a, da, axis)
    blo, bhi = _plane_slice(b, db, axis)
    lo, hi = max(alo, blo), min(ahi, bhi)
    if lo > hi:
        return False
    if len(shared) == 1:
        return not lo == hi == shared[0][axis]
    return True


def verify_surface_embedding(vertices, triangles, source_faces=None):
    """Qualify a closed edge-paired mesh; vertices are already unique.

    A single cyclic vertex link excludes pinched sheets. Exact intersections
    exclude crossing/overlap/touching outside a common mesh vertex or edge.
    Faces from one qualified planar polygon partition have disjoint projected
    interiors, so their constituent triangles need no pairwise retest.
    """
    triangles = [tuple(map(int, tri)) for tri in triangles]
    links = [{} for _ in vertices]
    for tri in triangles:
        for i, vertex in enumerate(tri):
            a, b = tri[(i+1) % 3], tri[(i+2) % 3]
            links[vertex].setdefault(a, []).append(b)
            links[vertex].setdefault(b, []).append(a)
    for vertex, link in enumerate(links):
        if not link:
            continue
        seen, pending = set(), [next(iter(link))]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(link[current])
        if len(seen) != len(link) or any(len(neighbors) != 2 for neighbors in link.values()):
            return {"embedded_surface_verified": False, "watertight_manifold": False,
                    "nonmanifold_vertex_m": [float(x) for x in vertices[vertex]],
                    "reason": "Declared STEP shell has a pinched or nonmanifold vertex link"}

    exact = [tuple(Fraction(float(x)) for x in point) for point in vertices]
    points = [tuple(exact[v] for v in tri) for tri in triangles]
    normals = [_cross(_sub(b, a), _sub(c, a)) for a, b, c in points]
    if any(not any(normal) for normal in normals):
        return {"embedded_surface_verified": False,
                "reason": "Declared STEP shell contains a degenerate triangle"}
    bounds = [(tuple(min(vertices[v][k] for v in tri) for k in range(3)),
               tuple(max(vertices[v][k] for v in tri) for k in range(3))) for tri in triangles]
    axis = max(range(3), key=lambda k: max(p[1][k] for p in bounds) - min(p[0][k] for p in bounds))
    active = []
    for i in sorted(range(len(triangles)), key=lambda t: bounds[t][0][axis]):
        low, high = bounds[i]
        active = [j for j in active if bounds[j][1][axis] >= low[axis]]
        for j in active:
            if source_faces is not None and source_faces[i] == source_faces[j]:
                continue
            other_low, other_high = bounds[j]
            if any(high[k] < other_low[k] or other_high[k] < low[k] for k in range(3)):
                continue
            shared = tuple(exact[v] for v in sorted(set(triangles[i]) & set(triangles[j])))
            if _intersection_exceeds_shared(points[i], points[j], normals[i], normals[j], shared):
                result = {"embedded_surface_verified": False,
                          "reason": "Declared STEP shell surfaces intersect beyond a shared edge or vertex"}
                if source_faces is not None:
                    result["intersecting_face_entity_ids"] = [int(source_faces[i]), int(source_faces[j])]
                return result
        active.append(i)
    return {"embedded_surface_verified": True}
