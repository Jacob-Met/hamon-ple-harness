"""Closed STEP shells must also be embedded, without crossing surface sheets."""
import math
from fractions import Fraction as F
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
                self.assertTrue(got["embedded_surface_verified"], got)
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

    def test_signed_axis_permutations_and_translation_preserve_classification(self):
        # Reflection reverses all face senses together without changing whether
        # the surface is embedded; well-separated controls survive wire rounding.
        for axes, signs, scale in (((1, 2, 0), (1, -1, 1), .125),
                                   ((2, 0, 1), (-1, -1, 1), 16.)):
            for x, valid in ((.45, True), (.5, False), (.55, False)):
                with self.subTest(axes=axes, scale=scale, x=x):
                    points, faces = fans(x)
                    points = points[:, axes] * np.asarray(signs) * scale + (64., -128., 32.)
                    got = self.verify(points, faces)
                    self.assertEqual(got["conservation_verified"], valid, got)
                    if valid:
                        self.assertAlmostEqual(got["volume_m3"] / scale**3, 4/3, places=12)
                    else:
                        self.assertNotIn("volume_m3", got)

    def test_many_sided_prism_retains_area_and_volume(self):
        count, radius, height = 128, 2., 3.
        ring = [(radius*math.cos(2*math.pi*i/count), radius*math.sin(2*math.pi*i/count))
                for i in range(count)]
        points = [(x, y, z) for z in (0., height) for x, y in ring]
        faces = [tuple(reversed(range(count))), tuple(range(count, 2*count))]
        faces.extend((i, (i+1) % count, (i+1) % count+count, i+count) for i in range(count))
        got = self.verify(points, faces)
        self.assertTrue(got["conservation_verified"], got)
        area = count*radius**2*math.sin(2*math.pi/count)/2
        perimeter = 2*count*radius*math.sin(math.pi/count)
        self.assertAlmostEqual(got["volume_m3"], area*height, places=11)
        self.assertAlmostEqual(got["surface_area_m2"], 2*area+perimeter*height, places=11)

    def test_newly_resolved_thin_solids_have_exact_centered_moments(self):
        # Integrate horizontal square cross-sections analytically, subtracting
        # the smaller pyramid in rational arithmetic. A dyadic shear checks
        # products of inertia as well as the diagonal terms.
        for height in (math.nextafter(2., 0.), 2.-2.**-25):
            h = F(height)
            volume = 4*(2-h)/3
            center = (2+h)/4
            variance = [F(1, 5), F(1, 5), (4+2*h+h*h)/10-center*center]
            for shear in (False, True):
                matrix = [[F(1), F(0), F(1, 2) if shear else F(0)],
                          [F(0), F(1), -F(1, 4) if shear else F(0)],
                          [F(0), F(0), F(1)]]
                covariance = [[sum(matrix[i][k]*variance[k]*matrix[j][k] for k in range(3))
                               for j in range(3)] for i in range(3)]
                trace = sum(covariance[k][k] for k in range(3))
                for scale in (2.**-100, 1., 2.**100):
                    with self.subTest(height=height, shear=shear, scale=scale):
                        points, faces = fans(0., height)
                        points = (points @ np.array(matrix, dtype=float).T)*scale
                        got = verify_step_cad_geometry({"step_text": surface_step(points, faces),
                                                        "density_kg_m3": 7.5})
                        self.assertTrue(got["conservation_verified"], got)
                        self.assertEqual(got["volume_m3"], float(volume*F(scale)**3))
                        self.assertEqual(got["centroid_m"], [float(row[2]*center*F(scale)) for row in matrix])
                        expected = [[float(F(7.5)*volume*F(scale)**5*((trace if i == j else 0)-covariance[i][j]))
                                     for j in range(3)] for i in range(3)]
                        self.assertEqual(got["inertia_tensor_centroid_kg_m2"], expected)
                        if not shear:
                            self.assertEqual(got["principal_moments_kg_m2"], sorted(expected[k][k] for k in range(3)))
                        else:
                            normalized = np.asarray(expected)/max(abs(x) for row in expected for x in row)
                            principal = np.linalg.eigvalsh(normalized)
                            actual = np.asarray(got["principal_moments_kg_m2"])/max(abs(x) for row in expected for x in row)
                            np.testing.assert_allclose(actual, principal, rtol=2e-15, atol=0)

    def test_nonfinite_or_unrepresentable_mass_withholds_all_mass_properties(self):
        for height, density in ((1., -1.), (1., math.inf),
                                (math.nextafter(2., 0.), math.ulp(0.)), (-1., 1e308)):
            with self.subTest(height=height, density=density):
                points, faces = fans(0., height)
                got = verify_step_cad_geometry({"step_text": surface_step(points, faces),
                                                "density_kg_m3": density})
                self.assertFalse(got["conservation_verified"], got)
                for key in ("volume_m3", "mass_kg", "centroid_m", "inertia_tensor_centroid_kg_m2"):
                    self.assertNotIn(key, got)

    def test_scaled_cubes_keep_representable_surface_area_and_moments(self):
        corners = np.array([(0,0,0), (1,0,0), (1,1,0), (0,1,0),
                            (0,0,1), (1,0,1), (1,1,1), (0,1,1)], dtype=float)
        faces = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
        for exponent, multiplier in ((-290, 1.), (290, 1.), (-268, 1.5)):
            with self.subTest(exponent=exponent, multiplier=multiplier):
                side, density = multiplier*2.**exponent, 2.**(-900 if exponent > 0 else 900)
                got = verify_step_cad_geometry({"step_text": surface_step(side*corners, faces),
                                                "density_kg_m3": density})
                self.assertTrue(got["conservation_verified"], got)
                self.assertEqual(got["surface_area_m2"], float(6*F(side)**2))
                self.assertEqual(got["volume_m3"], float(F(side)**3))
                self.assertEqual(got["mass_kg"], float(F(density)*F(side)**3))
                self.assertEqual(got["centroid_m"], [side/2]*3)
                moment = float(F(density)*F(side)**5/6)
                self.assertEqual(got["principal_moments_kg_m2"], [moment]*3)


if __name__ == "__main__":
    unittest.main()
