"""Mass-quality and component-direction checks for the Rankine callback."""
import math
import unittest
from fractions import Fraction

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier

BASE = dict(subkind="rankine", P1=10000, P2=8000000, v1=.00101, h1=191800,
            h3=3348400, h4s=2100000, eta_p=.85, eta_t=.88)
WATER = dict(h_f=191810, h_fg=2392090, s_f=649.2, s_fg=7499.6)


def solve(**updates):
    return dispatch_wave7_verifier("thermo_cycles", {**BASE, **updates})


class RankinePhaseTests(unittest.TestCase):
    def test_existing_direct_enthalpy_fixture_preserves_energy_without_fabricated_quality(self):
        result = solve()
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["eta_thermal"], .346063335648688)
        self.assertEqual(result["isentropic_outlet_phase"], "unknown")
        self.assertIsNone(result["x4s_quality"])
        self.assertIsNone(result["x4_quality"])
        self.assertFalse(result["carnot_checked"])

    def test_published_water_corner_and_independent_actual_state(self):
        result = solve(**WATER)
        self.assertTrue(result["verified"])
        self.assertEqual(result["isentropic_outlet_phase"], "liquid_vapor_mixture")
        self.assertEqual(result["x4s_quality"], float(Fraction(190819, 239209)))
        self.assertEqual(result["x4_quality"], float(Fraction(1028999, 1196045)))
        self.assertEqual(result["first_turbine_outlet_enthalpy_J_kg"], 2249808)
        self.assertNotEqual(result["x4_quality"], result["x4s_quality"])

    def test_first_turbine_actual_outlet_crosses_declared_saturation_boundary(self):
        result = solve(h_f=200000, h_fg=2400000, h3=3500000, h4s=2552000, eta_t=.8)
        self.assertTrue(result["verified"])
        self.assertEqual(result["x4s_quality"], .98)
        self.assertEqual(result["first_turbine_outlet_enthalpy_J_kg"], 2741600)
        self.assertEqual(result["first_turbine_outlet_phase"], "above_saturated_vapor")
        self.assertIsNone(result["x4_quality"])

    def test_supplied_quality_only_is_kept_as_an_unchecked_declaration(self):
        result = solve(x4s=.7)
        self.assertTrue(result["verified"])
        self.assertIsNone(result["x4s_quality"])
        self.assertEqual(result["declared_x4s_quality"], .7)
        self.assertFalse(result["declared_x4s_quality_checked"])
        checked = solve(**WATER, x4s=float(Fraction(190819, 239209)))
        self.assertTrue(checked["declared_x4s_quality_checked"])
        self.assertFalse(solve(**WATER, x4s=.7)["verified"])

    def test_unavailable_mixture_interpolation_and_valid_single_phase_direct_input(self):
        for entropy in [600., 8250.]:
            with self.subTest(entropy=entropy):
                spec = {**BASE, **WATER, "s3": entropy}
                del spec["h4s"]
                result = dispatch_wave7_verifier("thermo_cycles", spec)
                self.assertFalse(result["verified"])
                self.assertEqual(result["error"], "isentropic_entropy_outside_saturation_bracket")
        result = solve(**WATER, h4s=2700000, s3=8250)
        self.assertTrue(result["verified"])
        self.assertEqual(result["isentropic_outlet_phase"], "above_saturated_vapor")
        self.assertIsNone(result["x4s_quality"])

    def test_redundant_state_observations_and_identity_efficiency(self):
        entropy_only = solve(s_f=500, s_fg=7000, s3=4000, eta_t=1)
        self.assertEqual(entropy_only["x4s_quality"], .5)
        self.assertEqual(entropy_only["x4_quality"], .5)
        self.assertIsNone(solve(s_f=500, s_fg=7000, s3=4000)["x4_quality"])
        self.assertFalse(solve(**WATER, s3=700)["verified"])
        self.assertFalse(solve(**WATER, h4s=2700000, s3=700)["verified"])

    def test_structured_refusals(self):
        for spec in [dict(x4s=1.01), dict(h_fg=1), dict(h_f=200000, h_fg=0),
                     dict(s_f=100, s_fg=0, s3=100), dict(h5=3600000),
                     dict(h5=3550000, h6s=2500000, eta_t2=1.2),
                     dict(h5=3550000, h6s=3700000), dict(h5=2000000, h6s=1500000),
                     dict(h4s=80000, eta_t=1), dict(T_cond=300), dict(h4s=math.nan)]:
            with self.subTest(spec=spec):
                refused = solve(**spec)
                self.assertFalse(refused["verified"])
                self.assertIn("error", refused)
