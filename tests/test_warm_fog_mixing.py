"""Public thermodynamic regression cases for the registered Wave-5 family."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hamon_pde_control_psychro_fem import verify_psychrometric_process


def mix(temperature1, temperature2, rh=1.0, pressure=75000, flow1=3.0):
    return verify_psychrometric_process({
        "mode": "adiabatic_mixing", "p_atm_pa": pressure,
        "stream1": {"t_db_c": temperature1, "rh": rh, "m_da_kg_s": flow1},
        "stream2": {"t_db_c": temperature2, "rh": rh, "m_da_kg_s": 1.0},
    })


class TestWarmFog(unittest.TestCase):
    def test_positive_temperature_saturated_mixture(self):
        # Independent PsychroLib 2.5.0 saturation properties and liquid-water balance.
        out = mix(4, 44)
        self.assertTrue(out["passed"])
        self.assertTrue(out["fog_saturated"])
        self.assertEqual(out["rh_mix"], 1.0)
        temp = out["t_db_mix_c"]
        vapor = out["w_vapor_kg_per_kgda"]
        liquid = out["w_condensate_kg_per_kgda"]
        self.assertAlmostEqual(temp, 22.467973560871904, delta=2e-5)
        self.assertAlmostEqual(vapor, 0.023415365341022724, delta=2e-9)
        self.assertAlmostEqual(liquid, 0.003198966514030675, delta=2e-9)
        self.assertAlmostEqual(vapor + liquid, out["w_mix_kg_per_kgda"], delta=2e-10)
        enthalpy = 1.006*temp + vapor*(2501.0 + 1.86*temp) + liquid*4.186*temp
        self.assertAlmostEqual(enthalpy, out["h_mix_kj_per_kgda"], delta=2e-5)

    def test_clear_outlet(self):
        out = mix(12, 28, rh=0.4)
        self.assertTrue(out["passed"])
        self.assertFalse(out["fog_saturated"])
        self.assertEqual(out["w_condensate_kg_per_kgda"], 0.0)
        self.assertEqual(out["w_vapor_kg_per_kgda"], out["w_mix_kg_per_kgda"])

    def test_unbracketed_above_saturation_inlets_decline(self):
        # PsychroLib 2.5.0 humidity ratio at 25°C, 101325 Pa and RH 1.00005.
        # This inlet is inside the existing moisture-input roundoff allowance.
        inlet = {"t_db_c": 25.0, "w_kg_per_kgda": 0.02008215922474159, "m_da_kg_s": 1.0}
        out = verify_psychrometric_process({
            "mode": "adiabatic_mixing", "p_atm_pa": 101325.0,
            "stream1": inlet, "stream2": inlet,
        })
        self.assertFalse(out["passed"])
        self.assertIn("bracket", out["reason"])

    def test_subfreezing_fog_declines(self):
        out = mix(-25, -5, pressure=101325, flow1=1.0)
        self.assertFalse(out["passed"])
        self.assertIn("ice", out["reason"])


if __name__ == "__main__":
    unittest.main()
