import unittest

from hamon_dual_verifier import solve_domain_problem


def coil(**changes):
    spec = dict(m_dry_air_kg_s=1., T_in_C=30., T_out_C=15.,
                w_in_kg_kg=.006, w_out_kg_kg=.010,
                duct_width_m=.5, duct_height_m=.4, duct_length_m=2.)
    return solve_domain_problem("hvac_coil_and_duct", dict(spec, **changes))


class CoilInventoryTests(unittest.TestCase):
    def test_air_acquires_water_without_any_water_input(self):
        result = coil()
        self.assertFalse(result["dual_verified"])
        self.assertFalse(result["coil_water_balance_ok"])
        self.assertAlmostEqual(result["coil_water_residual_kg_s"], .004)
        self.assertAlmostEqual(result["Q_total_cooling_W"], 5141.8)
