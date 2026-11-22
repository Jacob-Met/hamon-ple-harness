"""Signed flow and endpoint energy in the scalar Darcy–Weisbach calculation."""
import math

import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as solve


def pipe(flow, **extra):
    return dict(pipe_D_m=0.1, pipe_L_m=100.0, rho_kg_m3=1000.0, mu_Pa_s=0.001,
                Q_m3_s=flow, pump_efficiency=0.8, **extra)


class ScalarPipeEnergyTests(unittest.TestCase):
    def test_laminar_power_and_loss_reverse_together_at_endpoints(self):
        flow = math.pi / 40000  # Re=1000 for the test geometry and fluid.
        delta_p_loss = 128 * 0.001 * 100 * flow / (math.pi * 0.1**4)
        for sign in (1, -1):
            spec = pipe(sign * flow, delta_P_Pa=sign * 5000, delta_z_m=sign * 3)
            result = solve(spec)
            assert result == solve({"mode": "thermal_fluid", "spec": spec})
            assert result["conservation_verified"]
            self.assertTrue(math.isclose(result["darcy_f"], 0.064, rel_tol=1e-10))
            self.assertTrue(math.isclose(result["head_loss_m"], delta_p_loss / (1000 * 9.80665), rel_tol=1e-10))
            self.assertTrue(math.isclose(result["required_shaft_power_W"],
                                         flow * (5000 + 1000 * 9.80665 * 3 + delta_p_loss) / 0.8, rel_tol=1e-10))


    def test_reverse_turbulent_friction_and_minor_losses_are_dissipative(self):
        flow = 0.01
        pair = [solve(pipe(sign * flow, K_minor=2, roughness_m=0.0001)) for sign in (1, -1)]
        for result in pair:
            f = result["darcy_f"]
            re = 4 * 1000 * flow / (math.pi * 0.1 * 0.001)
            residual = 1 / math.sqrt(f) + 2 * math.log10(0.001 / 3.7 + 2.51 / (re * math.sqrt(f)))
            self.assertLess(abs(residual), 1e-10)
            loss = (f * 1000 + 2) * (4 * flow / (math.pi * 0.1**2))**2 / (2 * 9.80665)
            self.assertTrue(math.isclose(result["head_loss_m"], loss, rel_tol=1e-10))
            self.assertTrue(math.isclose(result["required_shaft_power_W"], 1000 * 9.80665 * flow * loss / 0.8, rel_tol=1e-10))
        assert pair[0] == pair[1]


    def test_stopped_pipe_and_pressure_driven_surplus(self):
        stopped = solve(pipe(0, delta_P_Pa=5000))
        assert stopped["conservation_verified"]
        assert stopped["darcy_f"] is None and stopped["colebrook_dual_rel_error"] is None
        assert stopped["required_shaft_power_W"] == stopped["head_loss_m"] == 0
        assisted = solve(pipe(-0.0001, delta_P_Pa=5000))
        assert assisted["required_hydraulic_power_W"] < 0
        assert assisted["required_shaft_power_W"] == 0
        assert assisted["excess_hydraulic_power_W"] == -assisted["required_hydraulic_power_W"]


    def test_invalid_scalar_pipe_cannot_certify_a_combined_balance(self):
        cases = [
        ("pump_efficiency", 0), ("pump_efficiency", 1.01), ("roughness_m", -0.1),
        ("pipe_D_m", 0), ("pipe_L_m", -1), ("K_minor", -1),
        ("mu_Pa_s", 0), ("rho_kg_m3", -1), ("Q_m3_s", math.nan),
    ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                spec = pipe(0.001, inlets=[{"mdot_kg_s": 1}], outlets=[{"mdot_kg_s": 1}])
                spec[key] = value
                result = solve(spec)
                assert result["mass_balance_ok"]
                assert not result["pipe_balance_ok"] and not result["conservation_verified"]
