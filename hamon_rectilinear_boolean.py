"""Regularized Boolean meshes on exact source-coordinate rectilinear cells.

Only axis-aligned triangle surfaces enter this path. All predicates use integer
representations of the supplied binary64 coordinates. No tolerance, normalized
coordinate, or rounded cell midpoint can merge distinct source planes.
"""
from __future__ import annotations

import math
import numpy as np

MAX_CELLS = 250_000
MAX_RASTER_TESTS = 3_000_000
MAX_OUTPUT_TRIANGLES = 100_000


def _integer_coordinates(values):
    ratios = [float(v).as_integer_ratio() for v in values]
    exponent = max(d.bit_length() - 1 for _, d in ratios)
    return [n << (exponent - (d.bit_length() - 1)) for n, d in ratios]


def _triangle_rows(meshes):
    """Return None for genuinely oblique faces, before any grid allocation."""
    result = []
    for vertices, faces in meshes:
        rows = []
        for face in faces:
            triangle = vertices[face]
            axes = [a for a in range(3) if triangle[0, a] == triangle[1, a] == triangle[2, a]]
            if not axes:
                return None
            if len(axes) != 1:
                raise ValueError("rectilinear operand contains a degenerate triangle")
            rows.append((axes[0], triangle))
        result.append(rows)
    return result


def _raster_operand(rows, coordinates, indices, integer_axes, shape, budget):
    # Cyclic projected axes give the correct outward normal component.
    crossings = [np.zeros((shape[a] + 1, shape[(a+1) % 3], shape[(a+2) % 3]), dtype=np.int32)
                 for a in range(3)]
    for axis, triangle in rows:
        u, v = (axis + 1) % 3, (axis + 2) % 3
        plane = indices[axis][float(triangle[0, axis])]
        uv = [(indices[u][float(p[u])], indices[v][float(p[v])]) for p in triangle]
        lo_u, hi_u = min(p[0] for p in uv), max(p[0] for p in uv)
        lo_v, hi_v = min(p[1] for p in uv), max(p[1] for p in uv)
        budget[0] += (hi_u - lo_u) * (hi_v - lo_v)
        if budget[0] > MAX_RASTER_TESTS:
            raise ValueError("rectilinear Boolean exceeds the exact face raster budget")
        points = [(2*integer_axes[u][i], 2*integer_axes[v][j]) for i, j in uv]
        a, b, c = points
        signed = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if signed == 0:
            raise ValueError("rectilinear operand contains a degenerate triangle")
        sign = 1 if signed > 0 else -1
        if sign < 0:
            points.reverse()
        edges = []
        for a, b in zip(points, points[1:] + points[:1]):
            du, dv = b[0]-a[0], b[1]-a[1]
            # One of the two triangles sharing an edge owns its exact midpoint.
            inclusive = dv > 0 or (dv == 0 and du < 0)
            edges.append((-dv, du, dv*a[0]-du*a[1], inclusive))
        for i in range(lo_u, hi_u):
            pu = integer_axes[u][i] + integer_axes[u][i+1]
            for j in range(lo_v, hi_v):
                pv = integer_axes[v][j] + integer_axes[v][j+1]
                inside = True
                for aa, bb, cc, inclusive in edges:
                    value = aa*pu + bb*pv + cc
                    if value < 0 or (value == 0 and not inclusive):
                        inside = False
                        break
                if inside:
                    crossings[axis][plane, i, j] -= sign

    volumes = []
    orientations = set()
    for axis, events in enumerate(crossings):
        winding = np.cumsum(events, axis=0, dtype=np.int32)
        if np.any(winding[-1]) or np.any(np.abs(winding) > 1):
            raise ValueError("rectilinear operand has an unbounded or overlapping oriented boundary")
        if np.any(winding > 0):
            orientations.add(1)
        if np.any(winding < 0):
            orientations.add(-1)
        order = [axis, (axis+1) % 3, (axis+2) % 3]
        volumes.append(np.transpose(winding[:-1] != 0, np.argsort(order)))
    if len(orientations) != 1 or not volumes[0].any():
        raise ValueError("rectilinear operand has no consistently oriented material")
    if any(not np.array_equal(volumes[0], other) for other in volumes[1:]):
        raise ValueError("rectilinear boundary disagrees across independent axis sweeps")
    return volumes[0]


def _boundary(occupied, coordinates):
    vertices, faces, indexed = [], [], {}

    def vertex(index):
        if index not in indexed:
            indexed[index] = len(vertices)
            vertices.append(tuple(coordinates[a][index[a]] for a in range(3)))
        return indexed[index]

    for axis in range(3):
        order = [axis, (axis+1) % 3, (axis+2) % 3]
        cells = np.transpose(occupied, order)
        padded = np.pad(cells, ((1, 1), (0, 0), (0, 0)))
        boundary = padded[:-1].astype(np.int8) - padded[1:].astype(np.int8)
        for plane, i, j in np.argwhere(boundary):
            local = [(plane, i, j), (plane, i+1, j), (plane, i+1, j+1), (plane, i, j+1)]
            if boundary[plane, i, j] < 0:
                local.reverse()
            quad = []
            for point in local:
                key = [0, 0, 0]
                for a, value in zip(order, point):
                    key[a] = int(value)
                quad.append(vertex(tuple(key)))
            faces.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
            if len(faces) > MAX_OUTPUT_TRIANGLES:
                raise ValueError("rectilinear Boolean exceeds the boundary triangle budget")
    # Closed edges alone do not qualify a vertex at which separate shells touch.
    # Its link must be one cycle, including after exact source-position welding.
    links = {}
    for face in faces:
        for i, vertex_id in enumerate(face):
            a, b = face[(i+1) % 3], face[(i+2) % 3]
            graph = links.setdefault(vertex_id, {})
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    for graph in links.values():
        if any(len(neighbors) != 2 for neighbors in graph.values()):
            raise ValueError("rectilinear Boolean has a nonmanifold material contact")
        pending = [next(iter(graph))]
        seen = set()
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(graph[node] - seen)
        if len(seen) != len(graph):
            raise ValueError("rectilinear Boolean has a nonmanifold material contact")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def rectilinear_boolean(meshes, operator):
    """Return an exact-cell mesh, or None when a source face is not axis aligned.

    The caller qualifies finite coordinates, indices and closed topology. An
    unsupported orientation may use a general polyhedral kernel; an invalid or
    over-budget rectilinear operand must stay unresolved rather than be snapped.
    """
    rows = _triangle_rows(meshes)
    if rows is None:
        return None
    coordinates = [sorted({float(p[a]) for vertices, _ in meshes for p in vertices}) for a in range(3)]
    shape = tuple(len(c)-1 for c in coordinates)
    if min(shape) < 1 or math.prod(shape) > MAX_CELLS:
        raise ValueError("rectilinear Boolean exceeds the exact cell budget")
    indices = [{value: i for i, value in enumerate(c)} for c in coordinates]
    integer_axes = [_integer_coordinates(c) for c in coordinates]
    budget = [0]
    material = None
    for operand in rows:
        cells = _raster_operand(operand, coordinates, indices, integer_axes, shape, budget)
        if material is None:
            material = cells
        elif operator == "UNION":
            material |= cells
        elif operator == "INTERSECTION":
            material &= cells
        elif operator == "DIFFERENCE":
            material &= ~cells
        else:
            raise ValueError("invalid rectilinear Boolean operation")
    if material is None or not material.any():
        raise ValueError("Boolean produced no qualified bounded solid")
    return _boundary(material, coordinates)
