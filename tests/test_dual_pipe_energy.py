"""Public dual pipe verifier against Hagen–Poiseuille and the Colebrook relation."""
import math
import unittest
from unittest.mock import patch

import hamon_dual_verifier as verifier


def pipe(reynolds, **changes):
    spec = dict(Q_m3_s=reynolds * math.pi / 40000000, D_m=0.1, L_m=100.0,
                rho_kg_m3=1000.0, mu_Pa_s=0.001, roughness_m=0.0001,
                minor_loss_K=2.5, pump_efficiency=0.8)
    spec.update(changes)
    return verifier.run_dual_verify_spec({"kind": "pipe_hydraulics_colebrook", "spec": spec})


class DualPipeEnergyTests(unittest.TestCase):
    def test_laminar_energy_and_reversal(self):
        for reynolds in (1e-16, 1, 1000, 2100):
            for sign in (-1, 1):
                with self.subTest(reynolds=reynolds, sign=sign):
                    result = pipe(sign * reynolds)
                    flow = reynolds * math.pi / 40000000
                    speed = 4 * flow / (math.pi * 0.1**2)
                    pressure = 128 * 0.001 * 100 * flow / (math.pi * 0.1**4) + 2.5 * 1000 * speed**2 / 2
                    self.assertTrue(result["dual_verified"])
                    for field, expected in (
                        ("velocity_m_s", sign * speed), ("reynolds", reynolds),
                        ("friction_factor_f", 64 / reynolds), ("pressure_drop_Pa", pressure),
                        ("head_loss_m", pressure / 9806.65), ("shaft_power_W", flow * pressure / 0.8),
                    ):
                        self.assertTrue(math.isclose(result[field], expected, rel_tol=1e-12), field)

    def test_rough_and_smooth_turbulence_through_both_solver_paths(self):
        for minimal in (False, True):
            with patch.object(verifier, "HAS_SCIPY", verifier.HAS_SCIPY and not minimal), \
                    patch.object(verifier, "HAS_SYMPY", verifier.HAS_SYMPY and not minimal):
                for reynolds, roughness in ((4000, 0.0001), (100000, 0.002), (1e8, 0.0)):
                    with self.subTest(reynolds=reynolds, minimal=minimal):
                        forward = pipe(reynolds, roughness_m=roughness)
                        reverse = pipe(-reynolds, roughness_m=roughness)
                        self.assertEqual(reverse, {**forward, "velocity_m_s": -forward["velocity_m_s"]})
                        f = forward["friction_factor_f"]
                        residual = 1 / math.sqrt(f) + 2 * math.log10(roughness / 0.37 + 2.51 / (reynolds * math.sqrt(f)))
                        self.assertLess(abs(residual), 1e-10)
                        self.assertGreater(forward["shaft_power_W"], 0)

    def test_zero_flow_and_invalid_properties(self):
        stopped = pipe(0)
        self.assertIsNone(stopped["friction_factor_f"])
        for field in ("pressure_drop_Pa", "head_loss_m", "shaft_power_W", "reynolds"):
            self.assertEqual(stopped[field], 0)
        for field, invalid in (("L_m", -1), ("D_m", 0), ("rho_kg_m3", -1), ("mu_Pa_s", 0),
                               ("pump_efficiency", 1.1), ("roughness_m", -0.1),
                               ("minor_loss_K", -1), ("Q_m3_s", math.nan)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipe(0, **{field: invalid})
