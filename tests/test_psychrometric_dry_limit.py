"""Positive liquid-wick wet-bulbs at the zero-humidity limit."""
import unittest

from hamon_pde_control_psychro_fem import verify_psychrometric_process


class DryWetBulbTests(unittest.TestCase):
    def test_exact_dry_limit_and_nearby_state(self):
        # Liquid-water energy balance with IAPWS saturation pressure, in degrees C.
        for dry, pressure, reference in (
            (20, 101325, 5.83594), (25, 101325, 8.27080),
            (30, 101325, 10.52945), (40, 101325, 14.58554),
            (30, 80000, 8.46607), (30, 95000, 9.97351),
        ):
            state = {"t_db_c": dry, "p_atm_pa": pressure}
            near = verify_psychrometric_process({**state, "rh": 1e-12})
            self.assertTrue(near["passed"])
            for humidity in ({"rh": 0.0}, {"w_kg_per_kgda": 0.0}):
                with self.subTest(dry=dry, pressure=pressure, humidity=humidity):
                    result = verify_psychrometric_process({**state, **humidity})
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["w_kg_per_kgda"], 0.0)
                    self.assertAlmostEqual(result["t_wb_c"], reference, delta=.002)
                    self.assertAlmostEqual(result["t_wb_c"], near["t_wb_c"], delta=1e-8)

    def test_false_dry_wet_bulb_claim_is_rejected(self):
        result = verify_psychrometric_process({
            "t_db_c": 30, "rh": 0, "claimed_field": "t_wb_c",
            "claimed_value": -60, "atol": .01, "rtol": 0,
        })
        self.assertFalse(result["passed"])
        self.assertFalse(result["claim_check"]["within_tolerance"])

    def test_ashrae_fundamentals_chapter_one_example_one(self):
        # The published 40/20 C dry/wet chart example reads approximately 6.5 g/kg.
        result = verify_psychrometric_process({"t_db_c": 40, "t_wb_c": 20, "p_atm_pa": 101325})
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(1000 * result["w_kg_per_kgda"], 6.5, delta=.15)
        self.assertAlmostEqual(result["t_wb_c"], 20, places=8)
