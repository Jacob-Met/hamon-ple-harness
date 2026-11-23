"""Changed-parameter, analytic-coordinate and interface-coupling controls."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_layered_consolidation import verify_layered_consolidation


def specification():
    return dict(calculation="layered_consolidation_1d", consolidation_layers=[
        dict(thickness_m=1., cv_m2_s=1., mv_per_Pa=1e-7),
        dict(thickness_m=1., cv_m2_s=.01, mv_per_Pa=1e-6)],
        time_s=10., drainage="single",
        load_history=[dict(time_s=0., surcharge_Pa=10000.)])


def solve(spec):
    return run_router_conservation_cad_verifier({"mode": "auto", "spec": spec})


def matched_impedance_oracle(spec):
    """Exact Fourier series after x=integral dz/sqrt(cv), with mv*sqrt(cv) constant.

    Every material and geometric parameter can change while the analytic
    solution remains known. Neither production algorithm is used here.
    """
    rows = spec["consolidation_layers"]
    widths = [r["thickness_m"] / math.sqrt(r["cv_m2_s"]) for r in rows]
    length = sum(widths)
    impedance = [r["mv_per_Pa"] * math.sqrt(r["cv_m2_s"]) for r in rows]
    assert max(impedance) - min(impedance) < max(impedance) * 1e-12
    edges = np.concatenate(([0.], np.cumsum(widths)))
    if spec["drainage"] == "single":
        k = (np.arange(2048) + .5) * math.pi / length
        amplitude = 2 / (length * k)
    else:
        k = (2 * np.arange(2048) + 1) * math.pi / length
        amplitude = 4 / (length * k)
    coefficients = [amplitude * (np.cos(k*a) - np.cos(k*b)) / (k*(b-a))
                    for a, b in zip(edges[:-1], edges[1:])]
    answer = np.zeros(len(rows))
    t, previous_time, previous_load = spec["time_s"], 0., 0.
    for row in spec["load_history"]:
        end, final = row["time_s"], row["surcharge_Pa"]
        if end > t:
            final = previous_load + (final-previous_load)*(t-previous_time)/(end-previous_time)
            end = t
        span, young = end-previous_time, t-end
        if span == 0 and young == 0:
            response = np.ones(len(rows))
        else:
            kernel = np.exp(-k*k*young)
            if span:
                kernel *= -np.expm1(-k*k*span)/(k*k*span)
            response = np.array([np.sum(c*kernel) for c in coefficients])
        answer += (final-previous_load)*response
        previous_time, previous_load = end, final
        if row["time_s"] > t:
            break
    return answer


class LayeredConsolidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = specification()
        cls.result = solve(cls.base)

    def assert_verified(self, result):
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["conservation_verified"], result)
        for key in ("dual_solver_rel_residual", "component_rel_residual", "inversion_rel_residual", "load_balance_rel_residual"):
            self.assertLess(result[key], 1e-6, result)

    def test_coupled_layers_match_analytic_coordinate_solution(self):
        self.assert_verified(self.result)
        expected = matched_impedance_oracle(self.base)
        np.testing.assert_allclose(self.result["layer_mean_excess_pore_pressure_Pa"], expected,
                                   rtol=2e-10, atol=2e-6)
        expected_settlement = sum(r["thickness_m"] * r["mv_per_Pa"] * (10000-p)
                                  for r, p in zip(self.base["consolidation_layers"], expected))
        self.assertAlmostEqual(self.result["settlement_at_time_m"], expected_settlement, places=12)
        self.assertGreater(abs(self.result["mean_excess_pore_pressure_Pa"]
                               - self.result["storage_weighted_excess_pore_pressure_Pa"]), 2000)

    def test_changed_layers_double_drainage_and_construction_ramp(self):
        spec = specification()
        spec.update(consolidation_layers=[
            dict(thickness_m=.7, cv_m2_s=.36, mv_per_Pa=2e-7),
            dict(thickness_m=1.3, cv_m2_s=.04, mv_per_Pa=6e-7),
            dict(thickness_m=.2, cv_m2_s=1.44, mv_per_Pa=1e-7)], drainage="double", time_s=8.,
            load_history=[dict(time_s=0., surcharge_Pa=0.), dict(time_s=5., surcharge_Pa=17000.)])
        result = solve(spec)
        self.assert_verified(result)
        np.testing.assert_allclose(result["layer_mean_excess_pore_pressure_Pa"],
                                   matched_impedance_oracle(spec), rtol=2e-9, atol=2e-5)

    def test_internal_drain_or_averaged_cv_cannot_replace_coupling(self):
        # Treating each stratum as independently drained invents an internal drain.
        separate = 0.
        for layer in self.base["consolidation_layers"]:
            tv = layer["cv_m2_s"] * self.base["time_s"] / layer["thickness_m"]**2
            remaining = sum(8/(math.pi*k)**2 * math.exp(-(math.pi*k)**2*tv/4)
                            for k in range(1, 501, 2))
            separate += layer["thickness_m"]*layer["mv_per_Pa"]*10000*(1-remaining)
        self.assertGreater(abs(separate-self.result["settlement_at_time_m"])
                           / self.result["settlement_at_time_m"], .1)
        # Matched hydraulic impedance has an order-invariant total settlement.
        # Break that symmetry before testing an averaged-cv shortcut.
        forward_spec = copy.deepcopy(self.base)
        forward_spec["consolidation_layers"][1]["mv_per_Pa"] = 1e-7
        forward_result = solve(forward_spec)
        self.assert_verified(forward_result)
        reversed_spec = copy.deepcopy(forward_spec)
        reversed_spec["consolidation_layers"].reverse()
        reversed_result = solve(reversed_spec)
        self.assert_verified(reversed_result)
        self.assertGreater(abs(reversed_result["settlement_at_time_m"]-forward_result["settlement_at_time_m"]), .0001)
        # An averaged-cv model is invariant under this reversal.

    def test_homogeneous_subdivision_and_length_time_scaling(self):
        spec = specification()
        spec["consolidation_layers"] = [dict(thickness_m=2., cv_m2_s=.1, mv_per_Pa=1e-7)]
        spec["time_s"] = 3.
        one = solve(spec)
        split = copy.deepcopy(spec)
        split["consolidation_layers"] = [dict(spec["consolidation_layers"][0], thickness_m=1.)] * 2
        two = solve(split)
        self.assert_verified(one)
        self.assert_verified(two)
        np.testing.assert_allclose(two["mean_excess_pore_pressure_Pa"], one["mean_excess_pore_pressure_Pa"], rtol=1e-12)
        scaled = copy.deepcopy(self.base)
        for row in scaled["consolidation_layers"]:
            row["thickness_m"] *= 7
        scaled["time_s"] *= 49
        result = solve(scaled)
        self.assert_verified(result)
        self.assertAlmostEqual(result["settlement_at_time_m"], self.result["settlement_at_time_m"]*7, places=12)

    def test_unloading_preserves_settlement_at_jump_and_rebounds(self):
        pulse = copy.deepcopy(self.base)
        pulse["load_history"] += [dict(time_s=10., surcharge_Pa=10000.), dict(time_s=10., surcharge_Pa=0.)]
        at = solve(pulse)
        self.assert_verified(at)
        self.assertAlmostEqual(at["settlement_at_time_m"], self.result["settlement_at_time_m"], places=12)
        np.testing.assert_allclose(at["layer_mean_excess_pore_pressure_Pa"],
                                   np.array(self.result["layer_mean_excess_pore_pressure_Pa"])-10000, atol=1e-7)
        pulse["time_s"] = 20.
        later = solve(pulse)
        self.assert_verified(later)
        self.assertGreater(later["settlement_at_time_m"], 0)
        self.assertLess(later["settlement_at_time_m"], at["settlement_at_time_m"])
        np.testing.assert_allclose(later["layer_mean_excess_pore_pressure_Pa"], matched_impedance_oracle(pulse), atol=1e-6)

    def test_instantaneous_and_not_yet_loaded_states(self):
        at = solve(dict(self.base, time_s=0.))
        self.assert_verified(at)
        self.assertEqual(at["settlement_at_time_m"], 0.)
        self.assertEqual(at["layer_mean_excess_pore_pressure_Pa"], [10000., 10000.])
        later = solve(dict(self.base, load_history=[dict(time_s=0., surcharge_Pa=0.),
                                                  dict(time_s=20., surcharge_Pa=0.),
                                                  dict(time_s=20., surcharge_Pa=10000.)]))
        self.assert_verified(later)
        self.assertEqual(later["settlement_at_time_m"], 0.)

    def test_invalid_or_unsupported_physics_and_wrong_claims_are_refused(self):
        for changes in (dict(Cc=.2), dict(rectangular_loads=[]), dict(drainage="internal"),
                        dict(time_s=True), dict(time_s=-1), dict(consolidation_layers=[]),
                        dict(consolidation_layers=[dict(thickness_m=1., cv_m2_s=0., mv_per_Pa=1e-7)]),
                        dict(consolidation_layers=[dict(thickness_m=1., cv_m2_s=1., mv_per_Pa=.1)]),
                        dict(consolidation_layers=[dict(thickness_m=1., cv_m2_s=1., mv_per_Pa=1e-7, Cc=.2)]),
                        dict(load_history=[dict(time_s=0., surcharge_Pa=-1.)])):
            with self.subTest(changes=changes):
                self.assertFalse(solve({**self.base, **changes})["conservation_verified"])
        self.assertFalse(solve(dict(self.base, time_s=0., proposed_settlement_m=1e-15))["conservation_verified"])
        self.assertFalse(solve(dict(self.base, proposed_mean_excess_pore_pressure_Pa=0.))["conservation_verified"])

    def test_independent_disagreement_abstains(self):
        with patch("hamon_layered_consolidation._finite_volume", return_value=np.zeros((2, 2))):
            self.assertFalse(verify_layered_consolidation(self.base)["conservation_verified"])

    def test_unresolved_short_pulse_cannot_erase_dimensional_settlement(self):
        spec = specification()
        spec.update(consolidation_layers=[dict(thickness_m=1e80, cv_m2_s=1e160, mv_per_Pa=1e-6)],
                    time_s=1., load_history=[dict(time_s=0., surcharge_Pa=1e5),
                                            dict(time_s=1e-80, surcharge_Pa=1e5),
                                            dict(time_s=1e-80, surcharge_Pa=0.)])
        # q*duration*U'(1), then h*mv: a finite 0.01696099454 m response.
        reference = .2 * sum(math.exp(-(math.pi*(n+.5))**2) for n in range(20))
        self.assertGreater(reference, .016)
        result = solve(spec)
        self.assertFalse(result["conservation_verified"], result)
        self.assertIn("unresolved", result["reason"])

    def test_dimensional_reconstruction_does_not_round_compliance_first(self):
        spec = specification()
        spec.update(consolidation_layers=[dict(thickness_m=1e-100, cv_m2_s=1e-200, mv_per_Pa=1e-220)],
                    time_s=4., load_history=[dict(time_s=0., surcharge_Pa=1e218)])
        remaining = sum(8/(math.pi*k)**2 * math.exp(-(math.pi*k)**2)
                        for k in range(1, 101, 2))
        expected = 1e-100 * (1e-220 * 1e218) * (1-remaining)
        result = solve(dict(spec, proposed_settlement_m=expected))
        self.assert_verified(result)
        self.assertTrue(math.isclose(result["settlement_at_time_m"], expected, rel_tol=1e-12))

    def test_tiny_component_disagreement_cannot_hide_below_peak_load_scale(self):
        spec = dict(self.base, time_s=20., load_history=[dict(time_s=0., surcharge_Pa=10000.),
                    dict(time_s=10., surcharge_Pa=10000.), dict(time_s=10., surcharge_Pa=0.)])
        tiny = np.array([[1e-80, 1e-80], [-1e-80, -1e-80]])
        with patch("hamon_layered_consolidation._finite_volume", return_value=tiny), \
                patch("hamon_layered_consolidation._laplace_history", return_value=np.zeros((2, 2))):
            result = verify_layered_consolidation(spec)
        self.assertFalse(result["conservation_verified"], result)
        self.assertIn("Independent layer solutions", result["reason"])

    def test_unrepresentable_dimensional_settlement_abstains(self):
        spec = specification()
        spec.update(consolidation_layers=[dict(thickness_m=1., cv_m2_s=1., mv_per_Pa=1e-300)],
                    time_s=1., load_history=[dict(time_s=0., surcharge_Pa=1e-50)])
        result = solve(spec)
        self.assertFalse(result["conservation_verified"], result)
        self.assertIn("output range", result["reason"])

    def test_inline_route_requires_all_residuals_and_no_extra_work(self):
        import hamon_pareto
        empty = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
                     by_extension={}, files=[], warnings=[])
        instruction = json.dumps(dict(mode="auto", spec=self.base))
        actual = hamon_pareto.route_from_intake_prompt(json.dumps(empty), instruction=instruction)
        self.assertTrue(actual["presolve_verified"], actual)
        self.assertAlmostEqual(json.loads(actual["presolve_result_json"])["settlement_at_time_m"],
                               self.result["settlement_at_time_m"], places=12)
        with patch.object(hamon_pareto, "dispatch_ple_tool", return_value=self.result):
            route = hamon_pareto.route_from_intake_prompt(json.dumps(empty), instruction=instruction)
            self.assertTrue(route["presolve_verified"])
            self.assertEqual(route["thinking_tier"], "low")
            for extra in (dict(deliverable_count=1), dict(mcp_server_count=1)):
                self.assertFalse(hamon_pareto.route_from_intake_prompt(
                    json.dumps(empty), instruction=instruction, **extra)["presolve_verified"])
        for field in ("dual_solver_rel_residual", "component_rel_residual", "inversion_rel_residual", "load_balance_rel_residual"):
            for value in (1e-6, float("nan"), True):
                with patch.object(hamon_pareto, "dispatch_ple_tool", return_value={**self.result, field: value}):
                    self.assertFalse(hamon_pareto.route_from_intake_prompt(
                        json.dumps(empty), instruction=instruction)["presolve_verified"])


if __name__ == "__main__":
    unittest.main()
