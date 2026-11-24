import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


class PumpDutyTests(unittest.TestCase):
    def check(self, **changes):
        spec = {
            "calculation": "pump_duty",
            "curve": [{"flow_m3_s": 0.0, "head_m": 60.0},
                      {"flow_m3_s": 0.02, "head_m": 40.0},
                      {"flow_m3_s": 0.03, "head_m": 15.0}],
            "total_flow_m3_s": 0.05,
            "identical_pumps": 2,
            "required_head_m": 27.0,
            **changes,
        }
        return run_router_conservation_cad_verifier({"spec": spec})

    def test_parallel_units_share_flow_and_each_sees_the_system_head(self):
        result = self.check(per_pump_input_W=30000, available_panel_W=60000)
        self.assertTrue(result["duty_verified"])
        self.assertAlmostEqual(result["available_head_m"], 27.5)
        self.assertAlmostEqual(result["head_margin_m"], 0.5)
        self.assertEqual(result["total_input_W"], 60000)
        self.assertEqual(result["panel_margin_W"], 0)
        self.assertFalse(self.check(required_head_m=28)["duty_verified"])
        self.assertFalse(self.check(per_pump_input_W=30000, available_panel_W=59999)["duty_verified"])

    def test_explicit_zero_connected_power_cannot_deliver_positive_hydraulic_work(self):
        for panel, required in [(0, 27), (60000, 27), (60000, 0)]:
            with self.subTest(panel=panel, required=required):
                result = self.check(per_pump_input_W=0, available_panel_W=panel,
                                    required_head_m=required)
                self.assertTrue(result["ok"])
                self.assertTrue(result["checks"]["head_adequate"])
                self.assertTrue(result["checks"]["panel_adequate"])
                self.assertFalse(result["checks"]["connected_power_consistent"])
                self.assertFalse(result["duty_verified"])

    def test_zero_power_guard_uses_positive_flow_and_actual_curve_head(self):
        self.assertTrue(self.check()["duty_verified"])
        electrical = {"per_pump_input_W": 0, "available_panel_W": 0}
        self.assertTrue(self.check(total_flow_m3_s=0, required_head_m=60,
                                   **electrical)["duty_verified"])
        zero_curve = [{"flow_m3_s": 0, "head_m": 0}, {"flow_m3_s": .03, "head_m": 0}]
        self.assertTrue(self.check(curve=zero_curve, required_head_m=0,
                                   **electrical)["duty_verified"])
        tiny = 1e-200
        tiny_curve = [{"flow_m3_s": 0, "head_m": tiny}, {"flow_m3_s": 1, "head_m": tiny}]
        self.assertFalse(self.check(total_flow_m3_s=tiny, curve=tiny_curve,
                                    required_head_m=0, **electrical)["duty_verified"])

    def test_curve_ends_and_knot(self):
        for flow, head in [(0, 60), (0.04, 40), (0.06, 15)]:
            with self.subTest(flow=flow):
                self.assertTrue(self.check(total_flow_m3_s=flow, required_head_m=head)["duty_verified"])
        self.assertFalse(self.check(total_flow_m3_s=math.nextafter(0.06, math.inf))["ok"])

    def test_unsupported_or_incomplete_inputs_are_rejected(self):
        for change in [
            {"identical_pumps": 1.5}, {"identical_pumps": True},
            {"required_head_m": math.nan}, {"per_pump_input_W": 30000},
            {"available_panel_W": -1}, {"curve": []}, {"speed_ratio": 1.5},
            {"curve": [{"flow_m3_s": 0, "head_m": 20}, {"flow_m3_s": 0, "head_m": 12}]},
        ]:
            with self.subTest(change=change):
                self.assertFalse(self.check(**change)["ok"])


class EquippedBranchTests(unittest.TestCase):
    def setUp(self):
        self.branch = {
            "common_head_m": 2, "hose_length_m": 40, "hose_diameter_m": .03,
            "hose_roughness_m": .00002, "kinematic_viscosity_m2_s": .0001,
            "strainer_reference_flow_m3_s": .0001, "strainer_reference_head_m": 3,
        }
        self.spec = {
            "calculation": "pump_duty", "curve": [{"flow_m3_s": 0, "head_m": 15},
                                                   {"flow_m3_s": .001, "head_m": 15}],
            "total_flow_m3_s": .0002, "identical_pumps": 2, "equipped_branch": self.branch,
            "per_pump_input_W": 50, "available_panel_W": 100,
        }

    def run_spec(self, **changes):
        return run_router_conservation_cad_verifier({"mode": "thermal_fluid",
                                                      "spec": {**self.spec, **changes}})

    def test_parallel_hose_and_strainer_with_independent_laminar_solution(self):
        q = self.spec["total_flow_m3_s"] / 2
        hose = 128 * .0001 * 40 * q / (math.pi * 9.80665 * .03**4)
        result = self.run_spec()
        self.assertTrue(result["duty_verified"])
        self.assertAlmostEqual(result["required_head_m"], 2 + hose + 3)
        self.assertAlmostEqual(result["hose_head_loss_m"], hose)
        self.assertAlmostEqual(result["strainer_head_loss_m"], 3)
        single = self.run_spec(identical_pumps=1)
        self.assertAlmostEqual(single["required_head_m"], 2 + 2 * hose + 12)
        self.assertFalse(single["checks"]["head_adequate"])
        self.assertTrue(single["checks"]["panel_adequate"])
        self.assertFalse(self.run_spec(available_panel_W=99)["checks"]["panel_adequate"])

    def test_positive_branch_losses_survive_rounding_at_the_curve_head(self):
        # A positive laminar loss is a strict physical shortfall even when both
        # reported heads round to the same binary64 value.
        branch = {**self.branch, "common_head_m": 16, "hose_length_m": 1,
                  "hose_diameter_m": 1, "kinematic_viscosity_m2_s": 1e-6,
                  "strainer_reference_flow_m3_s": 1e-4, "strainer_reference_head_m": 1e-6}
        def solve(flow, head):
            curve = [{"flow_m3_s": q, "head_m": head} for q in (0, 1)]
            return self.run_spec(total_flow_m3_s=flow, identical_pumps=1,
                                 curve=curve, equipped_branch=branch)
        q = 1e-10
        loss = 128 * 1e-6 * q / (math.pi * 9.80665) + 1e-6 * (q / 1e-4)**2
        self.assertTrue(0 < loss < math.ulp(16) / 2)
        result = solve(q, 16)
        self.assertTrue(result["ok"])
        self.assertEqual(result["required_head_m"], result["available_head_m"])
        self.assertFalse(result["checks"]["head_adequate"])
        self.assertFalse(result["duty_verified"])
        self.assertTrue(math.isclose(result["head_margin_m"], -loss, rel_tol=1e-12))
        self.assertTrue(solve(0, 16)["duty_verified"])
        self.assertTrue(solve(q, math.nextafter(16, math.inf))["duty_verified"])

    def test_strainer_ratio_square_can_exceed_float_range_with_finite_head(self):
        # 2^-1000 * (1 / 2^-600)^2 = 2^200 metres.
        branch = {**self.branch, "common_head_m": 0, "hose_length_m": 1,
                  "hose_diameter_m": 1, "kinematic_viscosity_m2_s": 1,
                  "strainer_reference_flow_m3_s": 2.0**-600, "strainer_reference_head_m": 2.0**-1000}
        curve = [{"flow_m3_s": q, "head_m": 2.0**200} for q in (0, 1)]
        result = self.run_spec(total_flow_m3_s=1, identical_pumps=1, curve=curve,
                               equipped_branch=branch)
        self.assertTrue(result["ok"], result.get("reason"))
        self.assertEqual(result["strainer_head_loss_m"], 2.0**200)
        self.assertFalse(result["duty_verified"])
        self.assertLess(result["head_margin_m"], -4)

    def test_invalid_mixed_or_missing_branch_inputs_and_curve_limit(self):
        self.assertFalse(self.run_spec(required_head_m=0)["ok"])
        for name, value in [("hose_diameter_m", 0), ("hose_roughness_m", .1),
                            ("kinematic_viscosity_m2_s", math.nan), ("gravity_m_s2", 0),
                            ("strainer_reference_flow_m3_s", 0), ("extra", 4)]:
            with self.subTest(name=name):
                self.assertFalse(self.run_spec(equipped_branch={**self.branch, name: value})["ok"])
        partial = {k: v for k, v in self.branch.items() if k != "hose_length_m"}
        self.assertFalse(self.run_spec(equipped_branch=partial)["ok"])
        terminal = self.run_spec(total_flow_m3_s=.003)
        self.assertFalse(terminal["ok"])
        self.assertEqual(terminal["total_input_W"], 100)
        self.assertIn("outside", terminal["reason"])
        zero = self.run_spec(total_flow_m3_s=0)
        self.assertEqual(zero["required_head_m"], self.branch["common_head_m"])
