"""Held-out exact triangle surface-gap and threshold contracts."""
import math
import unittest
import hamon_geometry as h
from test_ifc_narrowphase import tetra


class DistanceContracts(unittest.TestCase):
    def test_false_positive_envelope_has_large_true_separation(self):
        a=tetra()
        b=tetra((1.6,1.6,1.6))
        self.assertTrue(h.overlaps(a.bbox,b.bbox))
        observed=h.surface_distance(a,b)
        expected=2.8/math.sqrt(3.0)
        self.assertAlmostEqual(observed["surface_distance_m"],expected,places=8)
        self.assertGreater(observed["distance_triangle_tests"],0)
        self.assertAlmostEqual(h._d2(observed["closest_a_world_m"],
                                     observed["closest_b_world_m"]),
                               expected*expected,places=8)

    def test_close_but_aabb_disjoint_requires_distance(self):
        a=tetra();b=tetra((2.1,0,0))
        self.assertFalse(h.overlaps(a.bbox,b.bbox))
        observed=h.surface_distance(a,b)
        self.assertAlmostEqual(observed["surface_distance_m"],0.1,places=8)

    def test_crossing_surfaces_zero_gap(self):
        a=tetra();b=tetra((0.3,0.3,0.3))
        d=h.surface_distance(a,b)
        self.assertEqual(d["surface_distance_m"],0.0)
        self.assertIsNone(d["closest_a_world_m"])

    def test_surface_containment_has_positive_surface_gap_but_material_overlap(self):
        a=tetra();b=tetra((0.15,0.15,0.15),0.1)
        self.assertEqual(h.mesh_relation(a,b)["relation"],"solid_containment")
        self.assertGreater(h.surface_distance(a,b)["surface_distance_m"],0.0)

    def test_coplanar_parallel_planes(self):
        a=h.make_mesh([0,0,0,1,0,0,0,1,0],[0,1,2])
        b=h.make_mesh([0,0,.35,1,0,.35,0,1,.35],[0,1,2])
        self.assertAlmostEqual(h.surface_distance(a,b)["surface_distance_m"],0.35,places=8)

    def test_distance_comparison_budget_is_fail_closed(self):
        a=tetra();b=tetra((1.6,1.6,1.6))
        with self.assertRaisesRegex(ValueError,"budget exhausted"):
            h.surface_distance(a,b,triangle_budget=1)


if __name__=="__main__":unittest.main()
