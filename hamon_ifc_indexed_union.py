"""Volume-only validation for indexed output of a successful material UNION.

Only the Boolean kernel's output may enter this path. Input mesh admission is
unchanged. Distinct indexed vertices can occupy the same point at a material
contact; position welding is not a valid way to judge that output's topology.
This module certifies no replacement surface, area, centroid or clearance.
"""
from collections import defaultdict
from fractions import Fraction
import math
import sys
import numpy as np

def _closed_indexed_boundary(points, faces):
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not 1 <= len(faces) <= 100000
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError("Invalid indexed Boolean output")
    edges = defaultdict(list)
    links = defaultdict(lambda: defaultdict(list))
    for row in faces:
        a, b, c = map(int, row)
        if len({a, b, c}) != 3:
            raise ValueError("Repeated triangle vertex")
        for u, v in ((a, b), (b, c), (c, a)):
            edges[min(u, v), max(u, v)].append(1 if u < v else -1)
        for center, u, v in ((a, b, c), (b, c, a), (c, a, b)):
            links[center][u].append(v)
            links[center][v].append(u)
    if any(len(uses) != 2 or sum(uses) != 0 for uses in edges.values()):
        raise ValueError("Indexed Boolean output is open or inconsistently oriented")
    for link in links.values():
        if any(len(neighbors) != 2 for neighbors in link.values()):
            raise ValueError("Boolean vertex link is not a cycle")
        pending, reached = [next(iter(link))], set()
        while pending:
            vertex = pending.pop()
            if vertex not in reached:
                reached.add(vertex)
                pending.extend(set(link[vertex]) - reached)
        if len(reached) != len(link):
            raise ValueError("Boolean vertex joins separate surface fans")

def _exact_axis_volumes(points, faces):
    # Fractions are exact values of the binary64 coordinates, not decimal
    # approximations. Each Cartesian divergence field independently integrates
    # the enclosed volume. Translation is removed before multiplying values.
    reference = tuple(Fraction.from_float(float(v)) for v in points[0])
    vertices = [tuple(Fraction.from_float(float(v)) - reference[j]
                      for j, v in enumerate(point)) for point in points]
    totals = [Fraction(0), Fraction(0), Fraction(0)]
    for ia, ib, ic in faces:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        u = tuple(b[j] - a[j] for j in range(3))
        v = tuple(c[j] - a[j] for j in range(3))
        cross = (u[1]*v[2] - u[2]*v[1],
                 u[2]*v[0] - u[0]*v[2],
                 u[0]*v[1] - u[1]*v[0])
        if not any(cross):
            raise ValueError("Boolean triangle collapses in the reconstructed coordinates")
        for j in range(3):
            totals[j] += cross[j] * (a[j] + b[j] + c[j])
    values = [value / 6 for value in totals]
    if values[0] <= 0 or values[0] != values[1] or values[0] != values[2]:
        raise ValueError("Cartesian enclosed-volume integrals disagree")
    try:
        result = float(values[0])
    except OverflowError as exc:
        raise ValueError("Indexed volume is not representable") from exc
    if not math.isfinite(result) or result < sys.float_info.min:
        raise ValueError("No representable normal positive indexed volume")
    return result

def certified_indexed_union_volume(vertices, faces, *, kernel_volume,
                                   operand_volumes, world_vertices, scale):
    """Return only volume in metres cubed; reject every unqualified output."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    world_vertices = np.asarray(world_vertices)
    _closed_indexed_boundary(vertices, faces)
    if world_vertices.shape != vertices.shape or not np.isfinite(world_vertices).all():
        raise ValueError("Invalid reconstruction of the Boolean output")
    values = [float(kernel_volume), float(scale), *map(float, operand_volumes)]
    if len(operand_volumes) < 2 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Invalid material-union measures")
    local = _exact_axis_volumes(vertices, faces)
    if not math.isfinite(local) or not math.isclose(
            local, kernel_volume, rel_tol=2e-10, abs_tol=64*math.ulp(kernel_volume)):
        raise ValueError("Independent indexed volume disagrees with the Boolean kernel")
    maximum, total = max(operand_volumes), math.fsum(operand_volumes)
    margin = max(2e-10 * total, 64 * math.ulp(total))
    if not maximum - margin <= local <= total + margin:
        raise ValueError("Material union violates operand volume bounds")
    reconstructed = _exact_axis_volumes(world_vertices, faces)
    try:
        expected = local * scale**3
    except OverflowError as exc:
        raise ValueError("World volume overflow") from exc
    if (not math.isfinite(expected) or expected < sys.float_info.min
            or not math.isfinite(reconstructed) or reconstructed < sys.float_info.min
            or not math.isclose(reconstructed, expected, rel_tol=2e-9,
                                abs_tol=128*math.ulp(expected))):
        raise ValueError("World reconstruction changes the certified volume")
    return reconstructed
