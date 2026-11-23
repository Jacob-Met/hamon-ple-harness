"""Cooling and dehumidifying coils, 2024 ASHRAE Handbook ch. 23, SI example 4."""
import unittest

from hamon_pde_control_psychro_fem import verify_psychrometric_process


def coil(**updates):
    spec = {"mode": "cooling_coil", "load_split": "ashrae_dew_point", "p_atm_pa": 101325,
            "inlet": {"t_db_c": 32, "t_wb_c": 24}, "outlet": {"t_db_c": 16, "t_wb_c": 14.5},
            "m_da_kg_s": 1, "t_condensate_c": 12}
    return verify_psychrometric_process({**spec, **updates})


class CoolingCoilSplit(unittest.TestCase):
    def test_handbook_example_4(self):
        # Table states are rounded; this solver evaluates its existing moist-air correlations.
        for args in ({}, {"inlet": {"t_db_c": 32, "w_kg_per_kgda": .01557},
                          "outlet": {"t_db_c": 16, "w_kg_per_kgda": .00972}}):
            with self.subTest(args=args):
                r = coil(**args)
                self.assertTrue(r["passed"])
                water = .01557 - .00972
                latent = water * 2451.35
                total = 72.04 - 40.69 - water * 50.23
                self.assertAlmostEqual(r["condensation_t_dp_c"], 20.85, delta=.08)
                self.assertAlmostEqual(r["q_latent_kw"], latent, delta=.10)
                self.assertAlmostEqual(r["q_sensible_kw"], total - latent, delta=.10)
                self.assertAlmostEqual(r["q_total_kw"], total, delta=.10)
                enthalpy_balance = (r["inlet_h_kj_per_kgda"] - r["outlet_h_kj_per_kgda"]
                                    - r["condensate_kg_s"] * 4.186 * 12)
                self.assertAlmostEqual(r["q_total_kw"], enthalpy_balance, places=12)

    def test_drain_temperature_moves_only_sensible_heat(self):
        warm, cold = coil(), coil(t_condensate_c=2)
        self.assertEqual(cold["q_latent_kw"], warm["q_latent_kw"])
        self.assertEqual(cold["condensate_kg_s"], warm["condensate_kg_s"])
        for field in ("q_total_kw", "q_sensible_kw", "condensate_sensible_kw"):
            self.assertAlmostEqual(cold[field] - warm[field], 10 * 4.186 * warm["condensate_kg_s"], places=13)

    def test_flow_and_claims(self):
        base = coil()
        for flow in (1e-14, .3, 1e6):
            result = coil(m_da_kg_s=flow)
            self.assertEqual(result["sensible_heat_ratio"], base["sensible_heat_ratio"])
            for field in ("q_sensible_kw", "q_latent_kw", "q_total_kw", "condensate_kg_s"):
                self.assertAlmostEqual(result[field] / flow, base[field], places=12)
        self.assertTrue(coil(claimed_field="q_latent_kw", claimed_value=14.3394, atol=.1, rtol=0)["passed"])
        self.assertFalse(coil(claimed_field="q_latent_kw", claimed_value=20)["passed"])

    def test_dry_phase_and_idle(self):
        dry_in = {"t_db_c": 14, "w_kg_per_kgda": 0}
        r = coil(inlet=dry_in, outlet={"t_db_c": 4, "w_kg_per_kgda": 0})
        self.assertTrue(r["passed"])
        self.assertIsNone(r["condensation_t_dp_c"])
        self.assertEqual((r["q_latent_kw"], r["condensate_sensible_kw"]), (0, 0))
        self.assertAlmostEqual(r["q_sensible_kw"], 10.06)
        no_load = coil(inlet=dry_in, outlet=dry_in)
        self.assertEqual((no_load["q_total_kw"], no_load["q_sensible_kw"], no_load["q_latent_kw"]), (0, 0, 0))

    def test_refused_phase_flow_and_selector(self):
        for args in ({"t_condensate_c": -2}, {"m_da_kg_s": 0}, {"m_da_kg_s": float("inf")},
                     {"load_split": "unrecognized"}, {"inlet": {"t_db_c": 5, "t_dp_c": -3},
                                                       "outlet": {"t_db_c": 1, "t_dp_c": -7}}):
            with self.subTest(args=args):
                self.assertFalse(coil(**args)["passed"])

    def test_default_path_remains_the_midpoint_split(self):
        spec = {"mode": "cooling_coil", "inlet": {"t_db_c": 32, "t_wb_c": 24},
                "outlet": {"t_db_c": 16, "t_wb_c": 14.5}, "t_condensate_c": 12, "m_da_kg_s": 1}
        default = verify_psychrometric_process(spec)
        self.assertEqual(default, coil(load_split="moist_air_midpoint"))
        self.assertAlmostEqual(default["q_sensible_kw"], 16.47060179563946, places=10)
        self.assertAlmostEqual(default["q_latent_kw"], 14.504200983836483, places=10)


if __name__ == "__main__":
    unittest.main()
