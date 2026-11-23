"""Closed STEP shells must also be embedded, without crossing surface sheets."""
import math
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry


def surface_step(points, polygons):
    rows = []
    def add(kind, args):
        rows.append(f"#{len(rows)+1}={kind}({args});")
        return len(rows)
    def refs(values):
        return "(" + ",".join(f"#{v}" for v in values) + ")"
    ids = [add("CARTESIAN_POINT", "'',(" + ",".join(repr(float(x)) for x in p) + ")") for p in points]
    faces = []
    for polygon in polygons:
        loop = add("POLY_LOOP", "''," + refs([ids[i] for i in polygon]))
        bound = add("FACE_OUTER_BOUND", f"'',#{loop},.T.")
        faces.append(add("FACE", f"'',(#{bound})"))
    shell = add("CLOSED_SHELL", "''," + refs(faces))
    add("FACETED_BREP", f"'bounded surface',#{shell}")
    return "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n" + "\n".join(rows) + "\nENDSEC;\nEND-ISO-10303-21;"


def fans(x, z=1.):
    points = [(-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.), (0., 0., 2.), (x, 0., z)]
    triangles = []
    for i in range(4):
        j = (i+1) % 4
        triangles += [(i, j, 4), (j, i, 5)]
    return np.asarray(points), triangles


def dimples(gap):
    ring = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
    points = [(x, y, z) for z in (-1., 1.) for x, y in ring] + [(0., 0., gap), (0., 0., -gap)]
    triangles = []
    for i in range(4):
        j = (i+1) % 4
        triangles += [(i, j, j+4), (i, j+4, i+4), (i+4, j+4, 8), (j, i, 9)]
    return points, triangles


class StepSurfaceEmbeddingTests(unittest.TestCase):
    def verify(self, points, polygons):
        return verify_step_cad_geometry({"step_text": surface_step(points, polygons), "density_kg_m3": 1.})

    def assert_withheld(self, points, triangles):
        got = self.verify(points, triangles)
        self.assertFalse(got["conservation_verified"], got)
        for key in ("volume_m3", "mass_kg", "centroid_m", "inertia_tensor_centroid_kg_m2"):
            self.assertNotIn(key, got)

    def test_crossing_and_coplanar_folded_fans_are_refused(self):
        # (.75,.5,.5) is strictly inside two faces at x=.55, with barycentrics
        # (1/8,5/8,1/4) and (1/80,39/80,1/2), although they share only one vertex.
        for x in (.55, .5, math.nextafter(.5, math.inf)):
            with self.subTest(apex_x=x):
                self.assert_withheld(*fans(x))

    def test_disjoint_nested_fans_keep_their_positive_volume(self):
        for x, z in ((0., 1.), (.45, 1.), (math.nextafter(.5, 0.), 1.), (.55, -1.)):
            with self.subTest(apex=(x, z)):
                got = self.verify(*fans(x, z))
                self.assertTrue(got["conservation_verified"], got)
                self.assertAlmostEqual(got["volume_m3"], 4*(2-z)/3, places=13)

    def test_pinched_vertex_and_crossed_dimples_are_refused(self):
        # Outer walls connect both fans. At zero gap, the common apex has two
        # local surface sheets even though all edges are paired consistently.
        for gap in (0., -.1):
            with self.subTest(gap=gap):
                self.assert_withheld(*dimples(gap))
        got = self.verify(*dimples(.1))
        self.assertTrue(got["conservation_verified"], got)
        self.assertAlmostEqual(got["volume_m3"], 5.6, places=13)

    def test_parallel_cap_gap_cannot_be_welded_into_a_solid(self):
        ring = ((0., 0.), (2., 0.), (2., 2.), (0., 2.))
        points = [(x, y, z) for z in (0., 2.) for x, y in ring]
        points += [(x, y, -2e-10) for x, y in ring]
        faces = [(8, 11, 10, 9), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self.assert_withheld(points, faces)


if __name__ == "__main__":
    unittest.main()
