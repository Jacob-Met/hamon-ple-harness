"""A contained tetrahedron must survive a box boundary's coplanar subdivision."""

import unittest

from hamon_shell_relations import certify_void_shells


def exterior_box(y_levels):
    # The face orders below point outward for x-, x+, z-, z+, y-, y+.
    vertices = []
    indices = {}
    triangles = []

    def vertex(point):
        if point not in indices:
            indices[point] = len(vertices)
            vertices.append(point)
        return indices[point]

    def quad(a, b, c, d):
        a, b, c, d = map(vertex, (a, b, c, d))
        triangles.extend(((a, b, c), (a, c, d)))

    xmin, xmax = -1, 1
    zmin, zmax = -1, 1025
    for lo, hi in zip(y_levels, y_levels[1:]):
        quad((xmin, lo, zmin), (xmin, lo, zmax), (xmin, hi, zmax), (xmin, hi, zmin))
        quad((xmax, lo, zmin), (xmax, hi, zmin), (xmax, hi, zmax), (xmax, lo, zmax))
        quad((xmin, lo, zmin), (xmin, hi, zmin), (xmax, hi, zmin), (xmax, lo, zmin))
        quad((xmin, lo, zmax), (xmax, lo, zmax), (xmax, hi, zmax), (xmin, hi, zmax))

    ymin, ymax = y_levels[0], y_levels[-1]
    quad((xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax))
    quad((xmin, ymax, zmin), (xmin, ymax, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin))
    return vertices, triangles


class StrictGapRetriangulationTests(unittest.TestCase):
    def test_folded_closed_boundary_cannot_certify_material(self):
        vertices, triangles = exterior_box([-1, 33])
        vertices[vertices.index((-1, -1, -1))] = (2, 16, 512)
        cavity = (
            [(0, 0, 0), (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25)],
            [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
        )
        with self.assertRaisesRegex(ValueError, "intersecting or overlapping faces"):
            certify_void_shells([(vertices, triangles), cavity])

    def test_box_boundary_subdivision_preserves_positive_clearance(self):
        cavity = (
            [(0, 0, 0), (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25)],
            [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
        )
        # Each ray (1, k, k*k) for 1 <= k <= 32 hits x=1 on the
        # subdivision edge y=k. The tetrahedron has 0.75 minimum clearance.
        for levels, expected_count in [([-1, 33], 12), ([-1, *range(1, 33), 33], 268)]:
            with self.subTest(exterior_triangles=expected_count):
                outer = exterior_box(levels)
                self.assertEqual(len(outer[1]), expected_count)
                certify_void_shells([outer, cavity])


if __name__ == "__main__":
    unittest.main()
