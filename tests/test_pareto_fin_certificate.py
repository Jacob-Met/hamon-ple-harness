"""A tier downgrade must cover the entire passive-fin task on the observed inputs."""
import asyncio
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from unittest import TestCase, mock

from fin_provider_replay import EMPTY, INSTRUCTION, SPEC, independent_flux, replay
import hamon_pareto as router


def fence(spec):
    return "```wave5_spec\n" + json.dumps(spec) + "\n```"


def route(instruction=INSTRUCTION, inventory=EMPTY, **kwargs):
    return router.route_from_intake_prompt(json.dumps(inventory), instruction=instruction, **kwargs)


class PassiveFinCertificateTests(TestCase):
    def test_source_bound_passive_fin(self):
        for base in (125, 25, -25):
            spec = {**SPEC, "t_base_c": base}
            source = fence(spec)
            plan = route(source)
            assert plan["thinking_tier"] == "low"
            assert plan["presolve_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
            result = json.loads(plan["presolve_result_json"])
            assert result["passed"]
            expected = independent_flux(spec)
            assert math.isclose(result["q_fin_w"], expected, rel_tol=1e-14)
            assert math.isclose(result["q_fvm_w"], expected, rel_tol=1e-3)

    def test_verified_flux_uses_the_source_temperature_difference(self):
        # k*A = h*P = 1e16, length=1, so for an insulated tip mL=1
        # and q = 1e16 * (Tbase - Tambient) * tanh(1).
        spec = dict(mode="convective_fin_1d", tip_bc="adiabatic", k_w_mk=1e16,
                    h_w_m2k=1e16, perimeter_m=1, area_m2=1, length_m=1,
                    t_base_c="BASE", t_inf_c="AMBIENT")
        template = fence(spec)
        for base, ambient, scale in (
            ("20.0000000000000001", "20", 1),
            ("20", "20.0000000000000001", -1),
            ("1.0000000000000001", "1", 1),
            ("0.0000000000000001", "0", 1),
            ("20.000000000000004", "20", 40),
            ("20", "20", 0),
        ):
            with self.subTest(base=base, ambient=ambient):
                source = template.replace('"BASE"', base).replace('"AMBIENT"', ambient)
                plan = route(source)
                self.assertTrue(plan["presolve_verified"], plan)
                result = json.loads(plan["presolve_result_json"])
                expected = scale*math.tanh(1)
                self.assertTrue(math.isclose(result["q_fin_w"], expected, rel_tol=2e-15))
                self.assertTrue(math.isclose(result["q_fvm_w"], expected, rel_tol=1e-3))
                exact_excess = Fraction(base) - Fraction(ambient)
                tip = float(Fraction(ambient) + Fraction(float(exact_excess)/math.cosh(1)))
                for field in ("t_eval_c", "t_tip_c"):
                    self.assertTrue(math.isclose(result[field], tip, rel_tol=2e-15))
                self.assertEqual(plan["presolve_source_sha256"], hashlib.sha256(source.encode()).hexdigest())
        extreme = template.replace('"BASE"', '1e-999999999').replace('"AMBIENT"', '0')
        self.assertFalse(route(extreme)["presolve_verified"])

    def test_convective_tip_and_source_spelling(self):
        spec = {**SPEC, "tip_bc": "convective", "h_tip_w_m2k": 22., "x_eval_m": .04}
        source = fence(spec)
        first, second = route(source), route(source + "\n")
        assert first["thinking_tier"] == second["thinking_tier"] == "low"
        assert first["presolve_source_sha256"] != second["presolve_source_sha256"]
        assert first["presolve_result_json"] == second["presolve_result_json"]
        assert json.loads(first["presolve_result_json"]) == router.verify_wave5_engineering(spec)

    def test_unsupported_request_retains_normal_budget(self):
        for source in (
            fence({**SPEC, "length_m": -1}), fence({**SPEC, "area_m2": True}),
            fence({**SPEC, "length_m": float("inf")}), fence({**SPEC, "x_eval_m": .121}),
            fence({**SPEC, "tip_bc": "unspecified"}), fence({**SPEC, "h_tip_w_m2k": 0}),
            fence({**SPEC, "also_report": "stress"}), fence({k:v for k,v in SPEC.items() if k != "t_inf_c"}),
            INSTRUCTION.replace('"length_m":', '"length_m": 2, "length_m":'),
            INSTRUCTION + "\nCalculate a second part.", INSTRUCTION.replace("```wave5_spec", "```wave5_spec_extra"),
        ):
            plan = route(source)
            assert plan["thinking_tier"] == "medium"
            assert plan["presolve_abstention"]
            assert "presolve_result_json" not in plan

    def test_unconfirmed_physical_witness_and_external_work(self):
        for result in ({"passed": False}, {"passed": True, "q_fin_w": 0., "q_fvm_w": 1e-16},
                       {"passed": True, "q_fin_w": float("nan"), "q_fvm_w": 1}):
            with mock.patch.object(router, "verify_wave5_engineering", return_value=result):
                assert route()["presolve_abstention"]
        for changes, options in (({"warnings": ["incomplete"]}, {}), ({"scanned_files": 1}, {}),
                                 ({}, {"deliverable_count": 1}), ({}, {"mcp_server_count": 1})):
            plan = route(inventory={**EMPTY, **changes}, **options)
            assert not plan["presolve_verified"]
            assert plan["presolve_abstention"] == "additional task inputs or delivery obligations"

    def test_locked_pi_provider_wire(self):
        root = Path(__file__).resolve().parents[1]
        for expected in ("verified", "refused"):
            with self.subTest(expected=expected):
                asyncio.run(replay(root, expected))

    def test_archived_task_prompts_still_require_the_participant(self):
        root = Path(__file__).resolve().parents[1] / "artifacts"
        tasks = json.loads((root / "manifest.json").read_text())["tasks"]
        self.assertEqual(len(tasks), 29)
        for name in tasks:
            history = json.loads((root / name / "trajectory.json").read_text())
            instruction = next(step["message"] for step in history["steps"] if step["source"] == "user")
            with self.subTest(task=name):
                # Even optimistic empty inventory cannot qualify these complete requests.
                plan = route(instruction)
                self.assertFalse(plan["presolve_verified"])
                self.assertNotIn("presolve_result_json", plan)
