"""Bounded exact shell relations for finite polygonal source coordinates.

All predicates operate on one integer lattice.  No gap is welded and no
perturbed point is substituted for the supplied boundary.  Accepted shells are
embedded, connected, oriented two-manifolds; cavity boundaries are strictly
inside the exterior and disjoint from one another without nesting.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


@dataclass
class _Triangle:
    points: tuple

    def __post_init__(self):
        a, b, c = self.points
        self.normal = _cross(_sub(b, a), _sub(c, a))
        if not any(self.normal):
            raise ValueError("degenerate shell triangle")
        self.offset = _dot(self.normal, a)
        self.low = tuple(min(p[i] for p in self.points) for i in range(3))
        self.high = tuple(max(p[i] for p in self.points) for i in range(3))

    def contains(self, p):
        return all(_dot(self.normal, _cross(_sub(b, a), _sub(p, a))) >= 0
                   for a, b in zip(self.points, self.points[1:] + self.points[:1]))


class _Budget:
    def __init__(self, limit):
        self.remaining = limit

    def spend(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("shell relations exceed the exact predicate budget")


def _intersection_points(a, b):
    """Vertices of the convex intersection of two closed triangles, exactly."""
    if any(x < y or u < v for x, y, u, v in zip(a.high, b.low, b.high, a.low)):
        return
    da = [_dot(b.normal, p) - b.offset for p in a.points]
    db = [_dot(a.normal, p) - a.offset for p in b.points]
    if any(all(d > 0 for d in ds) or all(d < 0 for d in ds) for ds in (da, db)):
        return
    for source, target, distances in ((a, b, da), (b, a, db)):
        for p, d in zip(source.points, distances):
            if d == 0 and target.contains(p):
                yield p
        for i in range(3):
            j = (i + 1) % 3
            if distances[i] * distances[j] < 0:
                t = Fraction(distances[i], distances[i] - distances[j])
                point = tuple(x + t * (y - x) for x, y in zip(source.points[i], source.points[j]))
                if target.contains(point):
                    yield point
    if any(da) or any(db):
        return
    # Coplanar edges can cross even when neither triangle contains a vertex.
    omit = max(range(3), key=lambda i: abs(a.normal[i]))
    axes = [i for i in range(3) if i != omit]

    def cross2(u, v):
        return u[axes[0]] * v[axes[1]] - u[axes[1]] * v[axes[0]]

    for p, q in zip(a.points, a.points[1:] + a.points[:1]):
        u = _sub(q, p)
        for r, s in zip(b.points, b.points[1:] + b.points[:1]):
            v = _sub(s, r)
            determinant = cross2(u, v)
            if determinant:
                t = Fraction(cross2(_sub(r, p), v), determinant)
                w = Fraction(cross2(_sub(r, p), u), determinant)
                if 0 <= t <= 1 and 0 <= w <= 1:
                    yield tuple(x + t * d for x, d in zip(p, u))
            # Collinear overlap endpoints were already included by contains().


def _on_common_simplex(point, common):
    if len(common) == 1:
        return point == common[0]
    if len(common) == 2:
        a, b = common
        return (not any(_cross(_sub(point, a), _sub(b, a)))
                and all(min(x, y) <= p <= max(x, y) for p, x, y in zip(point, a, b)))
    return False


def _connected(graph):
    pending = [next(iter(graph))]
    seen = set()
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(graph[current])
    return len(seen) == len(graph)


def _shell(points, faces, inward, budget):
    triangles = []
    edges = defaultdict(list)
    links = defaultdict(lambda: defaultdict(list))
    neighbors = {i: [] for i in range(len(faces))}
    for i, face in enumerate(faces):
        triangle = _Triangle(tuple(points[int(index)] for index in face))
        triangles.append(triangle)
        a, b, c = triangle.points
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            edges[tuple(sorted((u, v)))].append((i, u < v))
            links[u][v].append(w)
            links[u][w].append(v)
    for incident in edges.values():
        if len(incident) != 2 or incident[0][1] == incident[1][1]:
            raise ValueError("shell is not closed and consistently oriented")
        a, b = (entry[0] for entry in incident)
        neighbors[a].append(b)
        neighbors[b].append(a)
    if not triangles or not _connected(neighbors):
        raise ValueError("each shell must be a single connected component")
    if any(any(len(adjacent) != 2 for adjacent in link.values()) or not _connected(link)
           for link in links.values()):
        raise ValueError("shell has a non-manifold vertex link")
    origin = triangles[0].points[0]
    volume6 = sum(_dot(_sub(a, origin), _cross(_sub(b, origin), _sub(c, origin)))
                  for triangle in triangles for a, b, c in [triangle.points])
    if not volume6 or (volume6 < 0) != inward:
        raise ValueError("shell lacks the required outward or inward winding")
    # Sweep on the longest axis. Two disjoint projections cannot intersect.
    # The budget limits all overlapping intervals, including later AABB rejects.
    axis = max(range(3), key=lambda k: max(t.high[k] for t in triangles) -
                                       min(t.low[k] for t in triangles))
    active = []
    for a in sorted(triangles, key=lambda t: t.low[axis]):
        active = [b for b in active if b.high[axis] >= a.low[axis]]
        for b in active:
            budget.spend()
            if any(x < y or u < v for x, y, u, v in zip(a.high, b.low, b.high, a.low)):
                continue
            common = tuple(set(a.points).intersection(b.points))
            if len(common) == 2 and any(_cross(a.normal, b.normal)):
                continue
            if any(not _on_common_simplex(p, common) for p in _intersection_points(a, b)):
                raise ValueError("shell has intersecting or overlapping faces")
        active.append(a)
    return triangles, volume6


def _positive_integer_roots(normal):
    """Roots of nx + ny*k + nz*k*k, without approximate square roots."""
    c, b, a = normal
    if a == 0:
        if b and (-c) % b == 0 and -c // b > 0:
            yield -c // b
        return
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        return
    for numerator in {-b - root, -b + root}:
        if numerator % (2 * a) == 0 and numerator // (2 * a) > 0:
            yield numerator // (2 * a)


def _transverse_direction(point, triangles):
    # A face-parallel ray, or one passing through an edge, has direction in
    # one of finitely many planes through the point. Each nonzero plane normal
    # excludes at most two integers from d=(1,k,k²). Select a remaining integer
    # exactly; subdividing a face cannot exhaust a fixed list of trial rays.
    excluded = set()
    for triangle in triangles:
        excluded.update(_positive_integer_roots(triangle.normal))
        for a, b in zip(triangle.points, triangle.points[1:] + triangle.points[:1]):
            normal = _cross(_sub(a, point), _sub(b, point))
            if not any(normal):
                edge = _sub(b, a)
                normal = _cross(edge, (1, 0, 0))
                if not any(normal):
                    normal = _cross(edge, (0, 1, 0))
            excluded.update(_positive_integer_roots(normal))
    k = 1
    while k in excluded:
        k += 1
    return (1, k, k * k)


def _inside(point, triangles, budget):
    direction = _transverse_direction(point, triangles)
    hits = 0
    for triangle in triangles:
        budget.spend()
        denominator = _dot(triangle.normal, direction)
        numerator = triangle.offset - _dot(triangle.normal, point)
        if denominator == 0:
            raise ValueError("containment direction is not transverse")
        if numerator == 0 and triangle.contains(point):
            raise ValueError("containment point touches a shell boundary")
        if denominator < 0:
            denominator, numerator = -denominator, -numerator
        if numerator <= 0:
            continue
        q = tuple(p * denominator + numerator * d for p, d in zip(point, direction))
        signs = [_dot(triangle.normal, _cross(_sub(b, a),
                 tuple(x - y * denominator for x, y in zip(q, a))))
                 for a, b in zip(triangle.points, triangle.points[1:] + triangle.points[:1])]
        if min(signs) < 0:
            continue
        if min(signs) == 0:
            raise ValueError("containment direction meets a boundary edge")
        hits += 1
    return bool(hits % 2)


def certify_void_shells(shells, *, max_checks=2_000_000):
    """Accept only an exterior plus disjoint, strictly enclosed cavity shells."""
    if len(shells) < 2 or any(not len(vertices) or not len(faces) for vertices, faces in shells):
        raise ValueError("expected nonempty exterior and cavity shells")
    count = sum(len(faces) for _, faces in shells)
    if count * (count - 1) // 2 > max_checks:
        raise ValueError("shell relations exceed the exact predicate budget")
    coordinates = [float(value) for vertices, _ in shells for row in vertices for value in row]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("shell coordinates must be finite")
    denominator = max(value.as_integer_ratio()[1] for value in coordinates)

    def integer(value):
        n, d = float(value).as_integer_ratio()
        return n * (denominator // d)

    origin = tuple(integer(value) for value in shells[0][0][0])
    budget = _Budget(max_checks)
    certified = []
    for index, (vertices, faces) in enumerate(shells):
        points = [tuple(integer(value) - offset for value, offset in zip(row, origin)) for row in vertices]
        triangles, volume6 = _shell(points, faces, index > 0, budget)
        for other, _ in certified:
            for a in triangles:
                for b in other:
                    budget.spend()
                    if next(_intersection_points(a, b), None) is not None:
                        raise ValueError("shell boundaries touch or intersect")
        if certified:
            if not _inside(triangles[0].points[0], certified[0][0], budget):
                raise ValueError("void shell is not strictly inside the outer shell")
            for other, _ in certified[1:]:
                if (_inside(triangles[0].points[0], other, budget)
                        or _inside(other[0].points[0], triangles, budget)):
                    raise ValueError("void shells must not enclose one another")
        certified.append((triangles, volume6))
    if sum(volume for _, volume in certified) <= 0:
        raise ValueError("shells do not enclose positive material")


def certify_exterior_shell(vertices, faces, *, max_checks=2_000_000):
    """Require an embedded, closed, outward-wound connected triangle boundary."""
    if not len(faces) or len(faces) > 32768:
        raise ValueError("advanced BRep exceeds bounded embedded-shell qualification")
    coordinates = [tuple(map(float, row)) for row in vertices]
    scale = max(value.as_integer_ratio()[1] for row in coordinates for value in row)
    def integer(value):
        numerator, denominator = value.as_integer_ratio()
        return numerator * (scale // denominator)
    points = [tuple(integer(value) for value in row) for row in coordinates]
    _shell(points, faces, False, _Budget(max_checks))
