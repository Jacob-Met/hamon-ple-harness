import unittest

from hamon_ifc_spatial_clash import SolidMesh3D, UnresolvedSolidGeometry, _polygonal_solid_mesh


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
