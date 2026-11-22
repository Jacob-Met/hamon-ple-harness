import math
import unittest
from fractions import Fraction

from hamon_wave6_physics_verifiers import solve_mohr_vessel_buckling


def pipe(**parameters):
    return solve_mohr_vessel_buckling({"subkind": "pressure_vessel", **parameters})


class PipeWallStressTests(unittest.TestCase):
    def test_material_boundaries_and_out_of_wall_refusals(self):
        base = dict(r_i=1., r_o=2., p_i=3., p_o=1.)
        for r, normal_traction in [(1., -3.), (2., -1.)]:
            with self.subTest(r=r):
                result = pipe(**base, r_eval=r)
                self.assertTrue(result["verified"])
                self.assertAlmostEqual(result["sigma_radial_lame"], normal_traction)
        for changes in [
            {"r_eval": math.nextafter(1., 0.)}, {"r_eval": math.nextafter(2., math.inf)},
            {"r_eval": 0}, {"r_o": 1.}, {"r_i": 0.}, {"r_i": 3.}, {"r_eval": math.nan},
        ]:
            with self.subTest(changes=changes):
                result = pipe(**{**base, **changes})
                self.assertIs(result["verified"], False)
                self.assertFalse(any(key.startswith("sigma_") for key in result))

    def test_cut_resultants_and_unloaded_open_ends(self):
        ri, ro, r = Fraction(10), Fraction(11), Fraction(21, 2)
        for pi, po in [(3, 0), (0, 3), (3, 3), (4, 3)]:
            for closed in [False, True]:
                with self.subTest(pi=pi, po=po, closed=closed):
                    result = pipe(r_i=float(ri), t=float(ro-ri), r_eval=float(r),
                                  p_i=pi, p_o=po, closed_ends=closed)
                    self.assertTrue(result["verified"])
                    # The full cut force includes both inner and outer pressure.
                    mean_hoop = (pi*ri - po*ro) / (ro-ri)
                    cap_force_over_pi = pi*ri**2 - po*ro**2
                    membrane_axial = -po + (pi-po)*ri / (2*(ro-ri)) if closed else 0
                    self.assertAlmostEqual(result["sigma_hoop_thin"], float(mean_hoop))
                    self.assertAlmostEqual(result["sigma_axial_thin"], float(membrane_axial))
                    self.assertAlmostEqual(result["sigma_axial_lame"] * float(ro**2-ri**2),
                                           float(cap_force_over_pi) if closed else 0.)
                    inner = pipe(r_i=float(ri), r_o=float(ro), p_i=pi, p_o=po)
                    outer = pipe(r_i=float(ri), r_o=float(ro), r_eval=float(ro), p_i=pi, p_o=po)
                    self.assertAlmostEqual(inner["sigma_radial_lame"], -pi)
                    self.assertAlmostEqual(outer["sigma_radial_lame"], -po)

    def test_hydrostatic_closed_tube_and_default_internal_only(self):
        hydrostatic = pipe(r_i=.07, r_o=.071, r_eval=.0705, p_i=210000., p_o=210000.)
        self.assertTrue(hydrostatic["verified"])
        for field in ["sigma_hoop_lame", "sigma_radial_lame", "sigma_axial_lame",
                      "sigma_hoop_thin", "sigma_axial_thin"]:
            self.assertEqual(hydrostatic[field], -210000.)
        self.assertEqual(hydrostatic["sigma_von_mises"], 0.)
        internal = pipe(r_i=1., t=.125, p_i=2.)
        self.assertTrue(internal["verified"])
        self.assertEqual(internal["sigma_hoop_thin"], 16.)
        self.assertEqual(internal["sigma_axial_thin"], 8.)
