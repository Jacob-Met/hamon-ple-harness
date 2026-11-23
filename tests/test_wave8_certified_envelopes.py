"""Coverage of the complete instruction determines admission to the low-effort tier."""
import json
import math
from unittest import mock

import pytest
import hamon_pareto as router

from test_wave8_certified_budget import EMPTY

SPEC = {
    "calculation": "consolidation_1d", "H_clay_m": 5, "e0": 0.8, "Cc": 0.25,
    "sigma_v0_eff_Pa": 100000, "delta_sigma_v_Pa": 50000,
    "cv_m2_s": 1e-7, "time_s": 31536000, "drainage": "double",
}
CASE = {"mode": "civil_geotech", "spec": SPEC}
SECOND = {"mode": "civil_geotech", "spec": {**SPEC, "delta_sigma_v_Pa": 60000}}
SINGLE = "Run the calculation below and return its JSON result."
BATCH = "Run the calculations below and return their JSON results in the same order."


def fenced(data):
    return "```json\n" + json.dumps(data, indent=2) + "\n```"


def route(instruction, inventory=EMPTY, **options):
    return router.route_from_intake_prompt(json.dumps(inventory), instruction=instruction, **options)


def settlement(delta):
    # Terzaghi's independent series solution for a double-drained layer.
    t = SPEC["cv_m2_s"] * SPEC["time_s"] / (SPEC["H_clay_m"] / 2) ** 2
    degree = 1 - sum(8 / (math.pi * (2*n+1))**2 * math.exp(
        -(math.pi * (2*n+1))**2 * t / 4) for n in range(40))
    return degree * SPEC["H_clay_m"] * SPEC["Cc"] / (1 + SPEC["e0"]) * math.log10(
        (SPEC["sigma_v0_eff_Pa"] + delta) / SPEC["sigma_v0_eff_Pa"])


def test_equivalent_briefs_and_ordered_load_cases():
    expected = json.loads(route(json.dumps(CASE))["presolve_result_json"])
    for text in (fenced(CASE), SINGLE + "\n\n" + fenced(CASE),
                 SINGLE + "\n" + json.dumps(CASE)):
        plan = route(text)
        assert plan["thinking_tier"] == "low"
        assert json.loads(plan["presolve_result_json"]) == expected
    for ordered in ([CASE, SECOND], [SECOND, CASE], [CASE, CASE], [CASE]):
        for text in (json.dumps(ordered), BATCH + "\n" + fenced(ordered)):
            plan = route(text)
            assert plan["thinking_tier"] == "low"
            output = json.loads(plan["presolve_result_json"])
            assert len(output) == len(ordered)
            for case, result in zip(ordered, output):
                assert result == json.loads(route(json.dumps(case))["presolve_result_json"])
                assert result["settlement_at_time_m"] == pytest.approx(
                    settlement(case["spec"]["delta_sigma_v_Pa"]), abs=1e-10)


@pytest.mark.parametrize("text", [
    "Convert the settlement to inches.\n" + fenced(CASE),
    fenced(CASE) + "\nThen compare against the permissible settlement.",
    SINGLE + "\n" + fenced([CASE, SECOND]),
    BATCH + "\n" + fenced(CASE),
    fenced([]), fenced([CASE, SECOND, CASE]), fenced([CASE, {"invalid": True}]),
    fenced([CASE, {**SECOND, "spec": {**SPEC, "time_s": -1}}]),
    fenced(CASE) + "\n" + fenced(SECOND),
    "```json\n" + json.dumps(CASE),
    "```python\n" + json.dumps(CASE) + "\n```",
    '{"mode":"auto","mode":"auto","spec":{}}',
])
def test_uncovered_or_failed_request_preserves_original_budget(text):
    plan = route(text)
    assert not plan["presolve_verified"]
    assert plan["thinking_tier"] == "medium"
    assert "presolve_result_json" not in plan


def test_ambient_evidence_and_combined_size_limit():
    brief = BATCH + "\n" + fenced([CASE, SECOND])
    assert not route(brief, {**EMPTY, "files": [{"path": "borehole.pdf"}]})["presolve_verified"]
    assert not route(brief, mcp_server_count=1)["presolve_verified"]
    assert not route(brief, deliverable_count=1)["presolve_verified"]
    huge = {"calculation": "consolidation_1d", "ok": True, "conservation_verified": True,
            "dual_solver_abs_residual": 0., "diagnostics": "x" * 6400}
    with mock.patch.object(router, "dispatch_ple_tool", return_value=huge):
        assert route(json.dumps(CASE))["presolve_verified"]
        assert not route(brief)["presolve_verified"]
