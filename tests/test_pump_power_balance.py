import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec


def design(**overrides):
    return {
        "calculation": "pump_duty",
        "curve": [{"flow_m3_s": 0, "head_m": 60}, {"flow_m3_s": 0.02, "head_m": 40}],
        "total_flow_m3_s": 0.01,
        "required_head_m": 49,
        **overrides,
    }


class PumpPowerBalance(unittest.TestCase):
    def solve(self, spec):
        normal = run_router_conservation_cad_verifier({"spec": spec})
        dual = run_dual_verify_spec({"mode": "domain", "kind": "pump_duty", "spec": spec})
        self.assertEqual(normal["duty_verified"], dual["dual_verified"])
        for name in (
            "available_head_m",
            "total_hydraulic_power_W",
            "per_pump_power_margin_W",
            "max_fluid_density_at_rating_kg_m3",
            "hydraulic_rating_verified",
            "checks",
        ):
            self.assertEqual(normal.get(name), dual.get(name))
        return normal

    def test_actual_head_and_water_power(self):
        power = {"per_pump_input_W": 1, "available_panel_W": 1}
        partial = self.solve(design(**power))
        self.assertTrue(partial["duty_verified"])
        self.assertIsNone(partial["hydraulic_rating_verified"])
        self.assertAlmostEqual(
            partial["max_fluid_density_at_rating_kg_m3"], 1 / (0.01 * 50 * 9.80665)
        )
        bad = self.solve(design(**power, fluid_density_kg_m3=1000))
        self.assertTrue(bad["checks"]["head_adequate"])
        self.assertTrue(bad["checks"]["panel_adequate"])
        self.assertFalse(bad["duty_verified"])
        self.assertFalse(bad["hydraulic_rating_verified"])
        self.assertAlmostEqual(bad["total_hydraulic_power_W"], 4903.325)
        self.assertAlmostEqual(bad["per_pump_power_margin_W"], -4902.325)
        self.assertFalse(
            self.solve(design(**power, fluid_density_kg_m3=1000, required_head_m=0))[
                "duty_verified"
            ]
        )

    def test_bank_power_and_separate_panel_constraint(self):
        spec = design(
            per_pump_input_W=6200,
            available_panel_W=12400,
            identical_pumps=2,
            total_flow_m3_s=0.02,
            fluid_density_kg_m3=1000,
            wire_to_fluid_efficiency=0.8,
        )
        r = self.solve(spec)
        self.assertTrue(r["hydraulic_rating_verified"] and r["duty_verified"])
        self.assertAlmostEqual(r["required_input_per_pump_W"], 6129.15625)
        self.assertAlmostEqual(r["required_total_input_W"], 12258.3125)
        self.assertAlmostEqual(r["total_hydraulic_power_W"], 9806.65)
        # Panel sizing uses both connected ratings even when the demand is lower.
        low = self.solve({**spec, "available_panel_W": 12300})
        self.assertTrue(low["hydraulic_rating_verified"])
        self.assertFalse(low["duty_verified"])

    def test_inverse_limit_and_exact_equality(self):
        spec = design(
            curve=[{"flow_m3_s": 0, "head_m": 4}, {"flow_m3_s": 1, "head_m": 4}],
            total_flow_m3_s=0.125,
            required_head_m=4,
            gravity_m_s2=8,
            wire_to_fluid_efficiency=0.5,
            per_pump_input_W=16,
            available_panel_W=16,
        )
        limit = self.solve(spec)["max_fluid_density_at_rating_kg_m3"]
        self.assertEqual(limit, 2)
        for rho, expected in (
            (math.nextafter(limit, 0), True),
            (limit, True),
            (math.nextafter(limit, math.inf), False),
        ):
            with self.subTest(rho=rho):
                r = self.solve({**spec, "fluid_density_kg_m3": rho})
                self.assertIs(r["hydraulic_rating_verified"], expected)
                self.assertIs(r["duty_verified"], expected)

    def test_head_only_efficiency_unknown_and_shutoff(self):
        self.assertTrue(self.solve(design())["duty_verified"])
        required = self.solve(design(fluid_density_kg_m3=1000))
        self.assertEqual(required["power_basis"], "ideal_lower_bound")
        self.assertIsNone(required["hydraulic_rating_verified"])
        adequate = self.solve(
            design(fluid_density_kg_m3=1000, per_pump_input_W=6000, available_panel_W=6000)
        )
        self.assertTrue(adequate["duty_verified"])
        self.assertIsNone(adequate["hydraulic_rating_verified"])
        stopped = self.solve(
            design(
                total_flow_m3_s=0,
                fluid_density_kg_m3=1000,
                wire_to_fluid_efficiency=0.4,
                per_pump_input_W=0,
                available_panel_W=0,
            )
        )
        self.assertEqual(stopped["total_hydraulic_power_W"], 0)
        self.assertTrue(stopped["duty_verified"])
        self.assertNotIn("max_fluid_density_at_rating_kg_m3", stopped)

    def test_exact_branch_when_display_value_rounds_to_the_limit(self):
        spec = design(
            curve=[{"flow_m3_s": 0, "head_m": 10}, {"flow_m3_s": 1, "head_m": 9}],
            total_flow_m3_s=2**-55,
            required_head_m=10,
        )
        r = self.solve(spec)
        self.assertEqual(r["available_head_m"], 10)
        self.assertEqual(r["head_margin_m"], -(2**-55))
        self.assertFalse(r["duty_verified"])
        r = self.solve(
            design(
                identical_pumps=5, total_flow_m3_s=0, per_pump_input_W=0.1, available_panel_W=0.5
            )
        )
        self.assertEqual(r["total_input_W"], 0.5)
        self.assertLess(r["panel_margin_W"], 0)
        self.assertFalse(r["duty_verified"])

    def test_invalid_units_and_partial_electrical_inputs(self):
        for fields in (
            {"fluid_density_kg_m3": 0},
            {"fluid_density_kg_m3": math.nan},
            {"wire_to_fluid_efficiency": 1.01, "fluid_density_kg_m3": 1000},
            {"wire_to_fluid_efficiency": 0},
            {"wire_to_fluid_efficiency": 0.8},
            {"per_pump_input_W": 1},
            {"available_panel_W": 1},
            {"gravity_m_s2": 0},
            {"fluid_density_lbm_ft3": 62.4},
        ):
            with self.subTest(fields=fields):
                self.assertFalse(self.solve(design(**fields))["ok"])

    def test_equipped_branch_uses_one_gravity(self):
        branch = {
            "common_head_m": 0,
            "hose_length_m": 1,
            "hose_diameter_m": 0.1,
            "hose_roughness_m": 0,
            "kinematic_viscosity_m2_s": 1e-6,
            "strainer_reference_flow_m3_s": 1,
            "strainer_reference_head_m": 0,
        }
        spec = design(equipped_branch=branch, total_flow_m3_s=1e-6, fluid_density_kg_m3=800)
        del spec["required_head_m"]
        normal = self.solve(spec)
        local = self.solve({**spec, "gravity_m_s2": 3})
        self.assertAlmostEqual(
            local["total_hydraulic_power_W"] / normal["total_hydraulic_power_W"], 3 / 9.80665
        )
        self.assertAlmostEqual(local["required_head_m"] / normal["required_head_m"], 9.80665 / 3)
        self.assertFalse(
            self.solve(
                {**spec, "gravity_m_s2": 3, "equipped_branch": {**branch, "gravity_m_s2": 9}}
            )["ok"]
        )

    def test_pressure_work_oracle(self):
        # Construct the flow from independently specified pressure and electrical
        # power. Every input is dyadic, so the equality is representable.
        for count in (1, 2, 4):
            for pressure_pa in (16, 64, 1024):
                for density in (1, 4, 32):
                    for efficiency in (0.25, 0.5, 1):
                        gravity = 8
                        hydraulic_watts = 2.0
                        flow = hydraulic_watts / pressure_pa
                        head = pressure_pa / (density * gravity)
                        electrical_watts = hydraulic_watts / efficiency
                        spec = design(
                            identical_pumps=count,
                            gravity_m_s2=gravity,
                            curve=[
                                {"flow_m3_s": 0, "head_m": head},
                                {"flow_m3_s": 1, "head_m": head},
                            ],
                            required_head_m=head,
                            total_flow_m3_s=count * flow,
                            fluid_density_kg_m3=density,
                            wire_to_fluid_efficiency=efficiency,
                        )
                        for rating in (
                            math.nextafter(electrical_watts, 0),
                            electrical_watts,
                            math.nextafter(electrical_watts, math.inf),
                        ):
                            with self.subTest(
                                count=count,
                                pressure=pressure_pa,
                                density=density,
                                eta=efficiency,
                                rating=rating,
                            ):
                                result = self.solve(
                                    {
                                        **spec,
                                        "per_pump_input_W": rating,
                                        "available_panel_W": count * rating,
                                    }
                                )
                                self.assertEqual(
                                    result["duty_verified"], rating >= electrical_watts
                                )
                                self.assertEqual(
                                    result["total_hydraulic_power_W"], count * hydraulic_watts
                                )
                                self.assertEqual(
                                    result["required_total_input_W"], count * electrical_watts
                                )
                                self.assertEqual(
                                    result["max_fluid_density_at_rating_kg_m3"],
                                    density * (rating / electrical_watts),
                                )
