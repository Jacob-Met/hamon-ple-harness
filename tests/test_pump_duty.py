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
