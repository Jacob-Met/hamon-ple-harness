from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hamon_pde_control_psychro_fem import _ice_enthalpy_kj_kg, verify_psychrometric_process


def mixing(cold, warm, hot_fraction=0.5, model="equilibrium_ice", rh=1.0):
    spec = {"mode": "adiabatic_mixing", "p_atm_pa": 101325.,
            "stream1": {"t_db_c": cold, "rh": rh, "m_da_kg_s": 1-hot_fraction},
            "stream2": {"t_db_c": warm, "rh": rh, "m_da_kg_s": hot_fraction}}
    if model is not None:
        spec["condensate_model"] = model
    return verify_psychrometric_process(spec)


class EquilibriumIceTests(unittest.TestCase):
    def test_primary_ice_properties_and_original_decline(self):
        # ASHRAE Fundamentals (2025), Ch. 1, Table 3 (saturated Ice Ih, kJ/kg).
        for temp, reference in [(-40., -411.47), (-25., -383.59), (-5., -343.86)]:
            self.assertAlmostEqual(_ice_enthalpy_kj_kg(temp), reference, delta=.12)
        self.assertFalse(mixing(-25, -5, model=None)["passed"])
        out = mixing(-25, -5)
        self.assertTrue(out["passed"])
        self.assertAlmostEqual(out["t_db_mix_c"], -14.067, delta=.005)
        self.assertEqual(out["w_liquid_kg_per_kgda"], 0)
        self.assertGreater(out["w_ice_kg_per_kgda"], 0)

    def test_melting_interval_conserves_water_and_enthalpy(self):
        # -25 C and +15 C saturated inlets, fixed total dry-air flow.
        # Hot stream mass fractions at the two phase endpoints.
        low, high = .4992412819222413, .5079862614184052
        points = [low-1e-6, low, (low+high)/2, high, high+1e-6]
        states = [mixing(-25, 15, x) for x in points]
        for out in states:
            self.assertTrue(out["passed"])
            temp = out["t_db_mix_c"]
            vapor, liquid, ice = (out[k] for k in ("w_vapor_kg_per_kgda",
                                  "w_liquid_kg_per_kgda", "w_ice_kg_per_kgda"))
            self.assertAlmostEqual(vapor+liquid+ice, out["w_mix_kg_per_kgda"], places=14)
            h = 1.006*temp + vapor*(2501 + 1.86*temp)
            self.assertAlmostEqual(h+ice*_ice_enthalpy_kj_kg(temp)+liquid*4.186*temp,
                                   out["h_mix_kj_per_kgda"], places=10)
            self.assertTrue(ice >= 0 and liquid >= 0)
        temps = [s["t_db_mix_c"] for s in states]
        self.assertTrue(-1e-4 < temps[0] < 0 < temps[-1] < 1e-4)
        self.assertTrue(all(abs(t)<1e-12 for t in temps[1:-1]))
        self.assertGreater(states[2]["w_liquid_kg_per_kgda"], 0)
        self.assertGreater(states[2]["w_ice_kg_per_kgda"], 0)
        self.assertLess(states[1]["w_liquid_kg_per_kgda"], 1e-12)
        self.assertLess(states[-2]["w_ice_kg_per_kgda"], 1e-12)

    def test_warm_and_clear_outlets(self):
        for cold, warm, rh in [(4, 44, 1.), (-25, -5, .1), (0, 0, 1.)]:
            out = mixing(cold, warm, rh=rh)
            self.assertTrue(out["passed"])
            self.assertEqual(out["w_ice_kg_per_kgda"], 0.)
        self.assertEqual(mixing(4, 44, model=None)["t_db_mix_c"], mixing(4,44)["t_db_mix_c"])
        self.assertFalse(mixing(-25, -5, model="other")["passed"])


if __name__ == "__main__":
    unittest.main()
