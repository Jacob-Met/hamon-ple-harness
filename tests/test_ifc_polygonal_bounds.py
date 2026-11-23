from collections import Counter
import sys
import unittest
from unittest.mock import patch

from hamon_ifc_spatial_clash import SolidMesh3D, UnresolvedSolidGeometry, _bounded_face_triangles, _polygonal_solid_mesh


class PolygonalTube(unittest.TestCase):
    def tube(self, *, floor=True):
        # A square tube 4 by 4 by 3 with a one-by-one through-hole.
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        points = [[x, y, z] for z in (0, 3) for x, y in outer + hole]
        entities = {1: ("IFCCARTESIANPOINTLIST3D", [points])}
        refs = []

        def face(outside, inside=None):
            idx = len(entities) + 1
            kind = "IFCINDEXEDPOLYGONALFACE" if inside is None else "IFCINDEXEDPOLYGONALFACEWITHVOIDS"
            entities[idx] = (kind, [outside] if inside is None else [outside, [inside]])
            refs.append(("ref", idx))

        face([9, 10, 11, 12], [16, 15, 14, 13])
        if floor:
            face([4, 3, 2, 1], [5, 6, 7, 8])
        for loop in ([1, 2, 3, 4], [8, 7, 6, 5]):
            for first, second in zip(loop, loop[1:] + loop[:1]):
                face([first, second, second + 8, first + 8])
        return _polygonal_solid_mesh(entities, [("ref", 1), True, refs, None], 1.0)

    def test_material_volume_and_boundary_area(self):
        (vertices, faces), = self.tube()
        tube = SolidMesh3D("id", "IFCBUILDINGELEMENTPROXY", "tube", "other", vertices, faces)
        self.assertAlmostEqual(tube.volume_m3(), 45.0)
        self.assertAlmostEqual(tube.surface_area_m2(), 90.0)

    def test_closed_flag_alone_does_not_establish_a_floor(self):
        with self.assertRaisesRegex(UnresolvedSolidGeometry, "shell is open"):
            self.tube(floor=False)

    def test_no_backend_open_shell_and_holed_solid(self):
        with patch.dict(sys.modules, {"shapely": None}):
            with self.assertRaisesRegex(UnresolvedSolidGeometry, "shell is open"):
                self.tube(floor=False)
            self.test_material_volume_and_boundary_area()

    def test_concave_outline_and_two_holes_without_backend(self):
        # The upper central notch is exterior. Two closed voids stay inside.
        outer = [(0, 0), (12, 0), (12, 9), (8, 9), (8, 5), (5, 5), (5, 9), (0, 9)]
        left = [(1, 1), (1, 3), (3, 3), (3, 1)]
        right = [(9, 2), (9, 5), (11, 5), (11, 2)]
        rings = [outer, left, right]
        with patch.dict(sys.modules, {"shapely": None}):
            triangles = _bounded_face_triangles(rings)
        self.assertEqual(len(triangles), 18)
        def signed_area(points):
            return sum(x*v-y*u for (x,y),(u,v) in zip(points, points[1:]+points[:1]))/2
        self.assertAlmostEqual(sum(map(signed_area, triangles)), 86.)
        directed = Counter((a,b) for tri in triangles for a,b in zip(tri,tri[1:]+tri[:1]))
        border = Counter((a,b) for ring in rings for a,b in zip(ring,ring[1:]+ring[:1]))
        self.assertEqual(directed - Counter({(b,a): v for (a,b),v in directed.items()}), border)

    def test_crossing_or_nested_bounds_are_rejected_without_backend(self):
        border = [(0,0), (10,0), (10,10), (0,10)]
        hole = [(2,2), (2,8), (8,8), (8,2)]
        nested = [(4,4), (4,6), (6,6), (6,4)]
        crossing = [(1,1), (1,9), (5,9), (5,1)]
        with patch.dict(sys.modules, {"shapely": None}):
            for rings in ([border,hole,nested], [border,hole,crossing]):
                with self.subTest(rings=rings), self.assertRaises(UnresolvedSolidGeometry):
                    _bounded_face_triangles(rings)
