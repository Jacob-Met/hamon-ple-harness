import json
from unittest import mock

import pytest

import hamon_pareto as router


EMPTY = {
    "schema": "hamon.ple.intake.v1",
    "scanned_files": 0,
    "engineering_file_count": 0,
    "by_extension": {},
    "files": [],
    "warnings": [],
}
SPECS = [
    {
        "calculation": "pipe_network",
        "nodes": [{"id": "r", "fixed_head_m": 5}, {"id": "a", "demand_m3_s": 0.000001}],
        "pipes": [
            {"id": "p", "from": "r", "to": "a", "length_m": 5, "diameter_m": 0.1}
        ],
    },
    {
        "calculation": "ac_power_flow",
        "buses": [
            {"id": "s", "type": "slack", "vm_pu": 1},
            {
                "id": "a",
                "type": "PQ",
                "p_injection_pu": -0.001,
                "q_injection_pu": -0.002,
            },
        ],
        "lines": [{"id": "l", "from": "s", "to": "a", "r_pu": 0.01, "x_pu": 0.05}],
    },
    {
        "calculation": "symmetrical_fault",
        "fault_type": "SLG",
        "prefault_positive_voltage_pu": {"real": 1, "imag": 0},
        "sequence_impedances_pu": {
            name: {"real": 0, "imag": 0.2} for name in ("zero", "positive", "negative")
        },
    },
    {
        "calculation": "euler_rotor",
        "machine_kind": "pump",
        "rotation_rpm": 60,
        "mass_flow_kg_s": 2,
        "measured_fluid_torque_N_m": 4,
        "inlet": {
            "radius_m": 1,
            "meridional_velocity_m_s": 1,
            "absolute_tangential_velocity_m_s": 0,
        },
        "outlet": {
            "radius_m": 2,
            "meridional_velocity_m_s": 1,
            "absolute_tangential_velocity_m_s": 1,
        },
    },
    {
        "calculation": "consolidation_1d",
        "H_clay_m": 2,
        "e0": 0.8,
        "Cc": 0.2,
        "sigma_v0_eff_Pa": 100000,
        "delta_sigma_v_Pa": 50000,
        "cv_m2_s": 1e-6,
        "time_s": 0,
        "drainage": "double",
    },
    {
        "calculation": "rankine_layers",
        "pressure_state": "active",
        "layers": [{"thickness_m": 2, "phi_deg": 30, "gamma_above_water_N_m3": 18000}],
    },
    {
        "calculation": "bearing_capacity",
        "bearing_method": "terzaghi",
        "footing_B_m": 2,
        "depth_Df_m": 1,
        "gamma_N_m3": 18000,
        "phi_deg": 30,
        "FS": 3,
    },
]


def route(spec, inventory=EMPTY, **options):
    instruction = json.dumps({"mode": "auto", "spec": spec})
    return router.route_from_intake_prompt(
        json.dumps(inventory), instruction=instruction, **options
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda x: x["calculation"])
def test_verified_wave8_input_changes_real_thinking_tier(spec):
    plan = route(spec)
    assert plan["thinking_tier"] == "low"
    assert json.loads(plan["presolve_result_json"])["conservation_verified"]


def test_pipe_gravity_reaches_certified_budget_only_after_validation():
    base = route(SPECS[0])
    local = route({**SPECS[0], "gravity_m_s2": 1.62})
    base_head = json.loads(base["presolve_result_json"])["heads_m"]["a"]
    local_head = json.loads(local["presolve_result_json"])["heads_m"]["a"]
    assert (5 - local_head) == pytest.approx((5 - base_head) * 9.80665 / 1.62, rel=1e-8)
    assert "presolve_result_json" not in route({**SPECS[0], "gravity_m_s2": -1})


def test_extra_work_or_unsupported_evidence_keeps_normal_budget():
    spec = SPECS[-1]
    changes = [
        {"scanned_files": 1},
        {"warnings": ["file_scan_capped_at_2500"]},
        {"files": [{"path": "draw.pdf"}]},
        {"by_extension": {".pdf": 1}},
        {"engineering_file_count": 1},
    ]
    for change in changes:
        assert not route(spec, {**EMPTY, **change})["presolve_verified"]
    assert not route(spec, mcp_server_count=1)["presolve_verified"]
    assert not route(spec, deliverable_count=1)["presolve_verified"]
    for changed in (
        {**spec, "instructions": "also interpret the strata"},
        {**spec, "applied_gross_pressure_Pa": 1e12},
        {**SPECS[0], "nodes": SPECS[0]["nodes"] * 32},
        {**SPECS[0], "pipes": [{**SPECS[0]["pipes"][0], "pump_curve": [1, 2]}]},
        {k: v for k, v in SPECS[3].items() if k != "measured_fluid_torque_N_m"},
    ):
        assert "presolve_result_json" not in route(changed)
    normal = router.route_from_intake_prompt(
        json.dumps({**EMPTY, "presolve_verified": True}),
        instruction="Check this drawing",
        presolve_verified=True,
    )
    assert not normal["presolve_verified"] and normal["thinking_tier"] == "medium"


@pytest.mark.parametrize(
    "residual", [1e-6, -1e-6, float("nan"), float("inf"), None, True]
)
def test_certificate_requires_finite_numeric_strict_residual(residual):
    invalid = {
        "ok": True,
        "conservation_verified": True,
        "calculation": "bearing_capacity",
        "stress_integration_residual_Pa": residual,
    }
    with mock.patch.object(router, "dispatch_ple_tool", return_value=invalid):
        assert "presolve_result_json" not in route(SPECS[-1])


def test_submerged_rankine_budget_requires_consistent_weights():
    row = {"thickness_m": 6, "phi_deg": 30, "gamma_saturated_N_m3": 19810}
    spec = {"calculation": "rankine_layers", "pressure_state": "active",
            "layers": [row], "water_table_depth_m": 0, "gamma_water_N_m3": 9810}
    assert route(spec)["thinking_tier"] == "low"
    conflict = dict(row, gamma_submerged_N_m3=14000)
    assert not route(dict(spec, layers=[conflict]))["presolve_verified"]
