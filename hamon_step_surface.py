"""Exact embedding predicates for an already triangulated planar STEP shell.

The mass integral of a closed oriented chain need not describe a solid: sheets
can cross, overlap or touch at a nonmanifold vertex.  Work on the original
binary64 vertices, without a distance tolerance or coordinate welding.
"""
from fractions import Fraction
import math


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


def integrate_closed_surface(vertices, triangles, density):
    """Integrate exact polyhedral moments before rounding public quantities.

    An embedded surface may enclose a very thin region between nearly equal
    pyramids. Floating sums of their first and second moments can lose the
    result even when volume remains accurate. All signed tetrahedral moments
    and the parallel-axis subtraction therefore use the parsed coordinates
    as exact rational numbers.
    """
    if not math.isfinite(density) or density < 0:
        return {"reason": "Solid density must be finite and nonnegative"}
    raw = [tuple(Fraction(float(x)) for x in point) for point in vertices]
    anchor = raw[0]
    local = [_sub(point, anchor) for point in raw]
    total = Fraction(0)
    first = [Fraction(0) for _ in range(3)]
    second = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for indices in triangles:
        tri = [local[int(index)] for index in indices]
        det = _dot(tri[0], _cross(tri[1], tri[2]))
        total += det
        summed = [sum(p[k] for p in tri) for k in range(3)]
        for i in range(3):
            first[i] += det*summed[i]
            for j in range(i, 3):
                second[i][j] += det*(summed[i]*summed[j] + sum(p[i]*p[j] for p in tri))
    if total == 0:
        return {"reason": "Declared STEP shell encloses zero volume"}
    if total < 0:
        total = -total
        first = [-x for x in first]
        second = [[-x for x in row] for row in second]
    volume = total/6
    center = [x/(4*total) for x in first]
    covariance = [[second[min(i, j)][max(i, j)]/120 - volume*center[i]*center[j]
                   for j in range(3)] for i in range(3)]
    trace = sum(covariance[i][i] for i in range(3))
    rho = Fraction(density)
    inertia = [[rho*((trace if i == j else 0) - covariance[i][j]) for j in range(3)] for i in range(3)]

    def wire(value, positive=False):
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("out of binary64 range") from exc
        if not math.isfinite(result) or (positive and result <= 0):
            raise ValueError("out of binary64 range")
        return result

    try:
        result = {
            "volume_m3": wire(volume, True),
            "mass_kg": wire(rho*volume, density > 0),
            "density_kg_m3": density,
            "centroid_m": [wire(x+y) for x, y in zip(anchor, center)],
            "aabb_extents_m": [wire(max(p[k] for p in raw) - min(p[k] for p in raw)) for k in range(3)],
            "inertia_tensor_centroid_kg_m2": [[wire(x) for x in row] for row in inertia],
            "mass_moment_method": "exact_polyhedral_integrals",
        }
        if density == 0:
            result["principal_moments_kg_m2"] = [0., 0., 0.]
            return result
        determinant = _dot(inertia[0], _cross(inertia[1], inertia[2]))
        if (inertia[0][0] <= 0 or inertia[0][0]*inertia[1][1] - inertia[0][1]**2 <= 0
                or determinant <= 0):
            return {"reason": "Exact STEP inertia is not positive definite"}
        # For positive eigenvalues, lambda_min >= det(I)/trace(I)^2 and
        # lambda_max <= trace(I). Their ratio bounds the precision needed to
        # recover the smallest principal moment without cancellation.
        condition_bound = sum(inertia[k][k] for k in range(3))**3 / determinant
        bits = max(0, condition_bound.numerator.bit_length() - condition_bound.denominator.bit_length())
        digits = 60 + math.ceil(bits*math.log10(2))
        import mpmath as mp
        with mp.workdps(digits):
            matrix = mp.matrix([[mp.mpf(x.numerator)/x.denominator for x in row] for row in inertia])
            eigenvalues = mp.eigsy(matrix, eigvals_only=True)
            result["principal_moments_kg_m2"] = [wire(x, True) for x in eigenvalues]
        return result
    except ValueError:
        return {"reason": "STEP mass properties cannot be represented as finite binary64 quantities"}


def _point_inside_closed_surface(point, triangles):
    """Exact ray parity for a point disjoint from an embedded closed surface.

    Integer moment-curve directions avoid edge/vertex hits without geometric
    tolerances. Each plane/edge degeneracy excludes finitely many seeds, so
    the finite retry bound covers every triangle's possible bad directions.
    """
    prepared = []
    for a, b, c in triangles:
        normal = _cross(_sub(b, a), _sub(c, a))
        drop = max(range(3), key=lambda k: abs(normal[k]))
        axes = tuple(k for k in range(3) if k != drop)
        projected = [tuple(p[k] for k in axes) for p in (a,b,c)]
        prepared.append((a, normal, axes, projected))
    for seed in range(1, 8*len(triangles)+2):
        direction = (1, seed, seed*seed)
        crossings, ambiguous = 0, False
        for anchor, normal, axes, projected in prepared:
            denominator = _dot(normal, direction)
            if denominator == 0:
                ambiguous = True
                break
            distance = _dot(normal, _sub(anchor, point)) / denominator
            if distance <= 0:
                continue
            hit = tuple(point[k] + distance*direction[k] for k in axes)
            sides = [_orient(projected[i], projected[(i+1) % 3], hit) for i in range(3)]
            if all(s >= 0 for s in sides) or all(s <= 0 for s in sides):
                if any(s == 0 for s in sides):
                    ambiguous = True
                    break
                crossings += 1
        if not ambiguous:
            return bool(crossings % 2)
    raise ValueError("Could not construct a transverse STEP containment ray")


def verify_void_shells(vertices, triangles, shell_ids):
    """Check the orientation and containment of pairwise disjoint shell surfaces.

    Embedding and one connected component per declared shell are prerequisites.
    Consequently a single boundary point locates each entire shell relative to
    every other shell. Cavities must be inside the outer boundary and may not
    contain one another.
    """
    raw = [tuple(Fraction(float(x)) for x in point) for point in vertices]
    groups = {}
    for tri, shell in zip(triangles, shell_ids):
        groups.setdefault(int(shell), []).append(tuple(raw[int(v)] for v in tri))
    result = {"void_shell_count": len(groups)-1, "void_containment_verified": False}
    if 0 not in groups or len(groups) < 2:
        return {**result, "reason": "STEP void topology lacks an outer or inner shell"}
    anchor = groups[0][0][0]
    signed = {}
    for shell, faces in groups.items():
        signed[shell] = sum(_dot(_sub(a, anchor), _cross(_sub(b, anchor), _sub(c, anchor)))
                            for a,b,c in faces)
    if any(signed[0]*signed[shell] >= 0 for shell in groups if shell != 0):
        return {**result, "reason": "STEP void shell winding must oppose its outer shell"}
    for shell, faces in groups.items():
        if shell != 0 and not _point_inside_closed_surface(faces[0][0], groups[0]):
            return {**result, "reason": "STEP void shell is outside the declared outer solid"}
    for first, faces in groups.items():
        if first == 0:
            continue
        for second, other in groups.items():
            if second != 0 and second != first and _point_inside_closed_surface(faces[0][0], other):
                return {**result, "reason": "STEP void shells may not be nested"}
    return {**result, "void_containment_verified": True}
