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


def test_staged_consolidation_keeps_unloading_and_physical_claims_in_the_route():
    history = [{"time_s": 0, "surcharge_Pa": 10000},
               {"time_s": 1, "surcharge_Pa": 10000},
               {"time_s": 1, "surcharge_Pa": 0}]
    spec = {"calculation": "consolidation_history_1d", "H_clay_m": 1,
            "cv_m2_s": 1, "mv_per_Pa": 1e-7, "time_s": 1, "drainage": "single",
            "load_history": history}
    plan = route(spec)
    assert plan["thinking_tier"] == "low"
    result = json.loads(plan["presolve_result_json"])
    pressure = result["mean_excess_pore_pressure_Pa"]
    assert pressure < 0 and result["settlement_at_time_m"] > 0
    assert result["mean_effective_increment_Pa"] == pytest.approx(-pressure)
    assert result["dual_solver_rel_residual"] == result["dual_solver_abs_residual_Pa"] / 10000
    assert not route(dict(spec, proposed_mean_excess_pore_pressure_Pa=-pressure))["presolve_verified"]
    assert not route(dict(spec, load_history=[dict(history[0], surcharge_kPa=10)]))["presolve_verified"]
    assert not route(dict(spec, load_history=history * 11))["presolve_verified"]

    unloaded = dict(spec, load_history=[{"time_s": 0, "surcharge_Pa": 0},
                                        {"time_s": 2, "surcharge_Pa": 0},
                                        {"time_s": 2, "surcharge_Pa": 10000}])
    zero = json.loads(route(unloaded)["presolve_result_json"])
    assert zero["dual_solver_abs_residual_Pa"] == zero["dual_solver_rel_residual"] == 0


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


def test_opposing_front_water_reaches_the_verified_inline_route():
    spec = {"calculation": "rankine_layers", "pressure_state": "active",
            "water_table_depth_m": 0, "front_water_table_depth_m": 0,
            "gamma_water_N_m3": 10000,
            "layers": [{"thickness_m": 6, "phi_deg": 30, "gamma_saturated_N_m3": 20000}]}
    plan = route(spec)
    assert plan["presolve_verified"]
    result = json.loads(plan["presolve_result_json"])
    assert result["thrust_N_m"] == pytest.approx(60000)
    assert result["moment_Nm_m"] == pytest.approx(120000)
    assert not route(dict(spec, front_water_table_depth_m=-1))["presolve_verified"]


def test_opposed_earth_pure_couple_reaches_inline_certificate():
    rear = {"thickness_m": 6, "phi_deg": 30, "gamma_above_water_N_m3": 18000}
    front = dict(rear, thickness_m=2)
    spec = {"calculation": "rankine_layers", "pressure_state": "active", "layers": [rear],
            "front_layers": [front], "front_surface_depth_m": 4, "front_surcharge_Pa": 0}
    plan = route(spec)
    assert plan["presolve_verified"]
    result = json.loads(plan["presolve_result_json"])
    assert result["thrust_N_m"] == 0
    assert result["moment_Nm_m"] == pytest.approx(144000)
    assert result["resultant_above_base_m"] is None
    assert not route(dict(spec, front_layers=[dict(front, unidentified_input=1)]))["presolve_verified"]
    assert not route(dict(spec, front_surface_depth_m=0))["presolve_verified"]


def test_pump_datum_correction_controls_certified_budget():
    gauge = {"density_kg_m3": 1000, "suction_absolute_pressure_Pa": 100000,
             "vapor_pressure_Pa": 2500, "suction_velocity_m_s": 2, "required_npsh_m": 8,
             "gauge_elevation_above_npsh_datum_m": 1, "loss_to_npsh_datum_m": 2}
    pump = dict(SPECS[3], suction_flange=gauge)
    qualified = route(pump)
    assert qualified["thinking_tier"] == "low"
    assert json.loads(qualified["presolve_result_json"])["meets_required_npsh"]
    gauge["loss_to_npsh_datum_m"] = 4
    assert not route(pump)["presolve_verified"]



def _notre_dame_jacket(coolant):
    # NDCBE controls, Exothermic-CSTR.ipynb, Git blob 36251f18f6214fa5f39ecbb732b9658d5c0feaa6.
    # Convert the published litre/minute model to SI and retain its Ea/R ratio.
    return {"reactor_type": "cstr", "order": 1, "C_A0": 1000, "v0": 0.1 / 60,
            "volume": 0.1, "T0": 350, "A_preexp": 7.2e10 / 60,
            "E_a": 72750 * (8.31446261815324 / 8.314),
            "delta_T_adiabatic": 50000 / 239, "steady_state_branch": "all",
            "heat_transfer": {"UA_W_K": 50000 / 60, "rho_kg_m3": 1000,
                              "Cp_J_kg_K": 239, "coolant_T_K": coolant}}


def _dimensional_route(spec):
    return router.route_from_intake_prompt(json.dumps(EMPTY), instruction=json.dumps(
        {"mode": "reactor_kinetics", "spec": spec}))


@pytest.mark.parametrize("coolant,expected_branches", [(295, 1), (300, 3), (305, 1)])
def test_dimensional_jacket_uses_both_original_dynamic_balances(coolant, expected_branches):
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    # Independently evaluate the published two-state differential equations
    # in minutes and mol/L. Never substitute the steady heat manifold into the ODE.
    def k(temp):
        return 7.2e10 * np.exp(-72750 / (8.314 * temp))

    def ode(time, state):
        concentration, temp = state
        reacted = k(temp) * concentration
        return np.array([1 - concentration - reacted,
                         350 - temp + 50000/239 * reacted + 50000/23900 * (coolant - temp)])

    def energy(temp):
        return ode(0, (1/(1+k(temp)), temp))[1]

    grid = np.linspace(280, 440, 641)
    roots = [brentq(energy, a, b) for a, b in zip(grid, grid[1:]) if energy(a)*energy(b) < 0]
    plan = _dimensional_route(_notre_dame_jacket(coolant))
    assert plan["presolve_verified"] and plan["thinking_tier"] == "low"
    states = json.loads(plan["presolve_result_json"])["steady_states"]
    assert len(states) == len(roots) == expected_branches
    for row, temp in zip(states, roots):
        state = np.array([1/(1+k(temp)), temp])
        assert row["exit_T_K"] == pytest.approx(temp, abs=1e-7)
        assert row["C_A_mol_m3"] == pytest.approx(1000*state[0], abs=1e-6)
        steps = np.diag([1e-5, 1e-4])
        def derivative(function):
            return np.column_stack([(function(state+step) - function(state-step))/(2*step[i])
                                    for i, step in enumerate(steps)])
        jac = derivative(lambda x: ode(0, x))
        trace, det = np.trace(jac), np.linalg.det(jac)
        expected = "unstable_saddle" if det < 0 else ("unstable" if trace > 0 else "asymptotically_stable")
        assert row["linear_stability"] == expected
        assert row["trace_s_inv"] == pytest.approx(trace/60, rel=1e-6)
        assert row["determinant_s_inv2"] == pytest.approx(det/3600, rel=1e-6)
        # Check finite-time response to small physical perturbations in both directions.
        def propagate(x):
            result = solve_ivp(ode, (0, .1), x, method="DOP853", rtol=1e-11, atol=1e-12)
            assert result.success
            return result.y[:, -1]
        radius = max(abs(np.linalg.eigvals(derivative(propagate))))
        assert bool(radius < 1) is (expected == "asymptotically_stable")


def test_dimensional_jacket_rejects_extra_inputs_and_marginal_classification():
    spec = _notre_dame_jacket(300)
    assert not _dimensional_route(dict(spec, coolant_volume_m3=4))["presolve_verified"]
    assert not _dimensional_route(dict(spec, gas_phase=True))["presolve_verified"]
    assert not _dimensional_route(dict(spec, steady_state_branch=None))["presolve_verified"]
    assert not _dimensional_route(dict(spec, heat_transfer=dict(spec["heat_transfer"], UA_W_K=True)))["presolve_verified"]
    # The certificate must carry a strictly decisive interval even when the
    # root solver finds a balanced equilibrium.
    row = {"linear_stability": "asymptotically_stable",
           "trace_bounds_s_inv": [-1.1, -.9], "determinant_bounds_s_inv2": [-.1, .1]}
    assert not router._bounded_jacket_stability(row)
    row.update(linear_stability="nonhyperbolic", determinant_bounds_s_inv2=[0, 0])
    assert not router._bounded_jacket_stability(row)


def test_notre_dame_printed_equilibrium_and_jacobian():
    import numpy as np
    middle = json.loads(_dimensional_route(_notre_dame_jacket(300))["presolve_result_json"])["steady_states"][1]
    # Public notebook cell 30 rounds its middle branch to two decimal places.
    assert middle["C_A_mol_m3"] / 1000 == pytest.approx(.50, abs=.005)
    assert middle["exit_T_K"] == pytest.approx(350.08, abs=.005)
    low = json.loads(_dimensional_route(_notre_dame_jacket(290))["presolve_result_json"])["steady_states"][0]
    # Cell 36 prints the forward-difference A matrix at the first 290 K coolant.
    # Convert the SI Jacobian back to the notebook's mol/L, K and minutes.
    a = np.asarray(low["jacobian"]) * 60
    a[0, 1] /= 1000
    a[1, 0] *= 1000
    np.testing.assert_allclose(a, [[-1.05041791, -.00429658087], [10.5476802, -2.19318466]],
                               rtol=3e-6, atol=1e-8)
