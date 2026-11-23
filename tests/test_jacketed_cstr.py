import json
import math
from unittest.mock import patch

import pytest

from hamon_pareto import route_from_intake_prompt
from hamon_wave6_physics_verifiers import solve_reactor_kinetics as solve


BASE = dict(reactor_type="cstr", order=1., C_A0=1., v0=1., T0=300.,
            delta_T_adiabatic=200., k_ref=1., T_ref=350., E_a=80000., volume=1.,
            jacket_to_feed_heat_capacity_ratio=2., coolant_T_K=300.)
EMPTY = dict(schema="hamon.ple.intake.v1", scanned_files=0, engineering_file_count=0,
             files=[], warnings=[], by_extension={})


def route(spec, inventory=EMPTY, **kwargs):
    request = json.dumps({"mode": "reactor_kinetics", "spec": spec})
    return route_from_intake_prompt(json.dumps(inventory), instruction=request, **kwargs)


@pytest.mark.parametrize(("ratio", "roots"), [
    (0., [.01342170533580371, .14297312550405095, .99973756033601]),
    (.5, [.0119874598828557, .293883454709249, .994785123895742]),
    (1., [.011429318110917, .5, .960760627440046]),
    (2., [.0109436129208311]),
])
def test_cooled_and_adiabatic_branches_close_original_balances(ratio, roots):
    out = solve(dict(BASE, jacket_to_feed_heat_capacity_ratio=ratio, steady_state_branch="all"))
    assert out["verified"] and out["steady_state_count"] == len(roots)
    for row, expected in zip(out["steady_states"], roots):
        x, temp = row["conversion_X"], row["exit_T_K"]
        assert x == pytest.approx(expected, abs=3e-14, rel=0)
        k = math.exp(-80000 / 8.31446261815324 * (1 / temp - 1 / 350))
        assert (1-x) * k == pytest.approx(x, rel=5e-12)
        assert (300-temp) + 200*x == pytest.approx(ratio*(temp-300), abs=1e-12)
        assert row["mass_balance_residual"] < 1e-12 and row["energy_balance_residual"] < 1e-12


def test_exact_fold_is_retained_between_neighboring_volumes():
    # Effective T=256+512*X, including the exact reference-temperature fold at 1/2.
    base = dict(BASE, T0=128., coolant_T_K=384., jacket_to_feed_heat_capacity_ratio=1.,
                T_ref=512., E_a=2048*8.31446261815324, delta_T_adiabatic=1024., steady_state_branch="all")
    outcomes = [solve(dict(base, volume=v)) for v in (math.nextafter(1., 0.), 1., math.nextafter(1., math.inf))]
    assert [out["steady_state_count"] for out in outcomes] == [1, 2, 3]
    fold = outcomes[1]["steady_states"][-1]
    assert fold["conversion_X"] == .5 and fold["exit_T_K"] == 512.
    assert fold["energy_balance_residual_K"] == 0


def test_hot_jacket_signed_heat_and_prescribed_conversion():
    p = dict(BASE, coolant_T_K=360., delta_T_adiabatic=0., volume=1.)
    rated = solve(p)
    assert rated["exit_T_K"] == 340. and rated["jacket_heat_removed_per_feed_heat_capacity_K"] == -40.
    design = solve({k:v for k,v in {**p, "conversion":rated["conversion_X"]}.items() if k != "volume"})
    assert design["volume_m3"] == pytest.approx(1., rel=1e-12)
    assert design["verified"] and design["mass_balance_residual"] < 1e-12


@pytest.mark.parametrize("update", [dict(coolant_T_K=0), dict(jacket_to_feed_heat_capacity_ratio=-1),
    dict(jacket_to_feed_heat_capacity_ratio=True), dict(T=310.), dict(gas_phase=True),
    dict(epsilon_A=.1), dict(F_A0=2.), dict(k_ref=-1., conversion=.5),
    dict(reactor_type="pfr"), dict(reactor_type="cstr_series")])
def test_incomplete_or_incompatible_jacket_models_fail(update):
    with pytest.raises(ValueError):
        solve(dict(BASE, **update))


def test_certified_inline_request_and_abstentions():
    plan = route(BASE)
    cert = json.loads(plan["presolve_result_json"])
    assert plan["thinking_tier"] == "low" and cert["verified"] and cert["steady_state_count"] == 1
    for spec in (dict(BASE, claimed_conversion=.99), dict(BASE, gas_phase=True),
                 dict(BASE, order=True), dict(BASE, A_preexp=2.), dict(BASE, steady_state_branch=None),
                 dict(BASE, jacket_to_feed_heat_capacity_ratio=0)):
        assert "presolve_result_json" not in route(spec)
    for kwargs in (dict(deliverable_count=1), dict(mcp_server_count=1), dict(inventory={**EMPTY, "warnings":["partial scan"]})):
        assert "presolve_result_json" not in route(BASE, **kwargs)


def test_every_returned_branch_must_have_two_finite_residuals():
    spec = dict(BASE, steady_state_branch="all", jacket_to_feed_heat_capacity_ratio=0)
    for key in ("energy_balance_residual", "mass_balance_residual"):
        fake = solve(spec)
        fake["steady_states"][1][key] = math.nan
        with patch("hamon_pareto.dispatch_ple_tool", return_value=fake):
            assert "presolve_result_json" not in route(spec)
