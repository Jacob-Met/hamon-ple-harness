"""Tests for PLE Wave-7 deterministic engineering verifiers & fast-path pruner."""
from __future__ import annotations

import math
try:
    import pytest
except ModuleNotFoundError:
    class _Approx:
        def __init__(self, expected, rel=1e-6, abs_tol=1e-12):
            self.expected = expected
            self.rel = rel
            self.abs_tol = abs_tol
        def __eq__(self, actual):
            import numpy as _np
            return bool(_np.allclose(actual, self.expected, rtol=self.rel if self.rel is not None else 0.0, atol=self.abs_tol if self.abs_tol is not None else 0.0))
    class _PytestShim:
        @staticmethod
        def approx(expected, rel=1e-6, abs=1e-12):
            return _Approx(expected, rel=rel, abs_tol=abs)
    pytest = _PytestShim()  # type: ignore[assignment]

from hamon_wave6_physics_verifiers import prune_deterministic_fast_path
from hamon_wave7_engineering_verifiers import (
    WAVE7_DOMAINS,
    dispatch_wave7_verifier,
    infer_wave7_domain,
    solve_electrochem_or_separation,
    solve_exchanger_or_radiation,
    solve_lagrangian_orbital,
    solve_thermodynamic_cycle,
)


def test_thermo_cycles_brayton_rankine_otto_diesel_vcr():
    assert set(WAVE7_DOMAINS) == {"thermo_cycles", "exchangers_radiation", "lagrangian_orbital", "electrochem_separations"}
    br = solve_thermodynamic_cycle({"subkind": "brayton", "T1": 300.0, "T3": 1400.0, "r_p": 10.0, "eta_c": 0.86, "eta_t": 0.90, "regenerator_eff": 0.80})
    assert br["verified"] is True and br["s_gen_J_kgK"] > 0.0 and br["eta_thermal"] < br["eta_carnot"]
    bad_reg = solve_thermodynamic_cycle({"subkind": "brayton", "T1": 300.0, "T3": 900.0, "r_p": 30.0, "regenerator_eff": 0.7})
    assert bad_reg["verified"] is False

    otto = solve_thermodynamic_cycle({"subkind": "otto", "r": 9.5, "T1": 300.0, "T3": 2100.0})
    assert otto["verified"] is True and math.isclose(otto["eta_thermal"], 1.0 - 9.5 ** (-0.4), rel_tol=1e-11)

    rk = solve_thermodynamic_cycle({"subkind": "rankine", "P1": 10e3, "P2": 8e6, "v1": 0.00101, "h1": 191.8e3, "h3": 3348.4e3, "h4s": 2100.0e3, "eta_p": 0.85, "eta_t": 0.88})
    assert rk["verified"] is True and rk["eta_thermal"] > 0.30

    vcr = solve_thermodynamic_cycle({"subkind": "vapor_compression", "h1": 250e3, "h2s": 280e3, "h3": 100e3, "eta_c": 0.82, "T_L": 265.0, "T_H": 310.0})
    assert vcr["verified"] is True and math.isclose(vcr["COP_HP"] - vcr["COP_R"], 1.0, abs_tol=1e-12)


def test_exchangers_ntu_lmtd_duality_and_radiation_enclosure():
    hx = solve_exchanger_or_radiation({"subkind": "heat_exchanger", "flow_arrangement": "counterflow", "C_h": 2200.0, "C_c": 2800.0, "T_h_in": 430.0, "T_c_in": 300.0, "UA": 3500.0})
    assert hx["verified"] is True and hx["residual"] < 1e-11
    inv = solve_exchanger_or_radiation({"subkind": "heat_exchanger", "flow_arrangement": "counterflow", "C_h": 2200.0, "C_c": 2800.0, "T_h_in": 430.0, "T_c_in": 300.0, "effectiveness": hx["effectiveness"]})
    assert inv["verified"] is True and math.isclose(inv["UA_W_K"], 3500.0, rel_tol=1e-9)

    rad = solve_exchanger_or_radiation({"subkind": "radiation_network", "F": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], "areas": [2.0, 2.0, 2.0], "emissivities": [0.9, 0.7, 0.5], "T_K": [950.0, 450.0, None]})
    assert rad["verified"] is True and abs(rad["Q_net_W"][2]) < 1e-7 and abs(sum(rad["Q_net_W"])) < 1e-6


def test_lagrangian_modes_euler_rigid_body_and_orbital_transfers():
    eu = solve_lagrangian_orbital({"subkind": "euler_rigid_body", "I": [2.0, 3.5, 5.0], "omega0": [1.0, 0.5, 2.2], "t_final": 1.5, "n_steps": 2000})
    assert eu["verified"] is True and eu["residual"] < 1e-9

    md = solve_lagrangian_orbital({"subkind": "lagrangian_modes", "M": [[2.0, 0.0], [0.0, 1.0]], "K": [[6.0, -2.0], [-2.0, 4.0]]})
    assert md["verified"] is True and len(md["natural_frequencies_rad_s"]) == 2

    orb = solve_lagrangian_orbital({"subkind": "orbital_transfer", "mu": 3.986004418e14, "r1": 6678e3, "r2": 42164e3, "delta_inc_deg": 28.5, "isp_s": 315.0})
    assert orb["verified"] is True and orb["dv_total_m_s"] > 4000.0 and 0.0 < orb["propellant_mass_fraction"] < 1.0


def test_electrochem_and_mccabe_thiele_distillation():
    ec = dispatch_wave7_verifier("nernst electrochemistry", {"subkind": "electrochemistry", "E0_V": 1.10, "n_electrons": 2.0, "T_K": 298.15, "Q_reaction": 0.05})
    assert ec["verified"] is True and ec["E_nernst_V"] > 1.10

    sep = solve_electrochem_or_separation({"subkind": "distillation", "z_F": 0.50, "x_D": 0.95, "x_B": 0.05, "alpha": 2.5, "q": 0.6, "R_mult": 1.30})
    assert sep["verified"] is True and sep["N_theoretical"] > sep["N_min_fenske"]
    pinch = solve_electrochem_or_separation({"subkind": "distillation", "z_F": 0.50, "x_D": 0.95, "x_B": 0.05, "alpha": 2.5, "q": 1.0, "R_mult": 0.95})
    assert pinch["verified"] is False


def test_wave1_to_7_fast_path_pruning_guards():
    assert infer_wave7_domain("McCabe-Thiele binary distillation") == "electrochem_separations"
    p_w2 = prune_deterministic_fast_path("medium", 5.3532, presolve_verified=True, domain="dxf_takeoff", residual=1e-12)
    assert p_w2["fast_path_pruned"] is True and p_w2["tier"] == "light" and p_w2["est_cost_usd"] == pytest.approx(1.9331)
    p_w5 = prune_deterministic_fast_path("low", 3.2715, presolve_verified=True, domain="convective_fin_1d", residual=1e-12)
    assert p_w5["fast_path_pruned"] is True and p_w5["tier"] == "light"
    p_bad = prune_deterministic_fast_path("low", 3.2715, presolve_verified=True, domain="convective_fin_1d", residual=1e-3)
    assert p_bad["fast_path_pruned"] is False and p_bad["tier"] == "low"


import unittest
import _scipy_compat as _sc
import hamon_wave7_engineering_verifiers as _w7_mod


def test_wave7_pure_python_scipy_fallback_parity():
    orig_i, orig_l, orig_o = _w7_mod.integrate, _w7_mod.linalg, _w7_mod.optimize
    try:
        from types import SimpleNamespace
        _w7_mod.integrate = SimpleNamespace(quad=_sc.quad_fallback, solve_ivp=_sc.solve_ivp_fallback)
        _w7_mod.optimize = SimpleNamespace(brentq=_sc.brentq_fallback)
        _w7_mod.linalg = SimpleNamespace(
            lu_factor=_sc.lu_factor_fallback,
            lu_solve=_sc.lu_solve_fallback,
            eigh=_sc.eigh_fallback,
        )
        test_thermo_cycles_brayton_rankine_otto_diesel_vcr()
        test_exchangers_ntu_lmtd_duality_and_radiation_enclosure()
        rb = solve_lagrangian_orbital({"subkind": "euler_rigid_body", "I": [2.0, 3.0, 5.0], "omega0": [1.0, 0.4, -0.7], "t_final": 1.5})
        assert rb["verified"] is True and rb["residual"] < 1e-9
        eig = solve_lagrangian_orbital({"subkind": "lagrangian_modes", "M": [[2.0, 0.2], [0.2, 1.5]], "K": [[12.0, -3.0], [-3.0, 8.0]]})
        assert eig["verified"] is True and eig["residual"] < 1e-11
    finally:
        _w7_mod.integrate, _w7_mod.linalg, _w7_mod.optimize = orig_i, orig_l, orig_o


class Wave7EngineeringVerifiersUnittest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(Wave7EngineeringVerifiersUnittest, _name, staticmethod(_fn))


def test_inverse_counterflow_near_pinch_and_balanced_capacity_rates():
    from decimal import Decimal, localcontext

    # Independent heat-duty/LMTD sizing, including NTU far above 250.
    for ratio in ("1", ".99999999999", ".99", ".9"):
        for eff in (".2", ".997", ".9999"):
            with localcontext() as context:
                context.prec = 60
                cr, target = Decimal(ratio), Decimal(eff)
                cmin = Decimal(1200)
                duty = cmin * 120 * target
                for ch, cc in ((cmin, cmin / cr), (cmin / cr, cmin)):
                    hot = 400 - duty / ch
                    cold = 280 + duty / cc
                    dh, dc = 400 - cold, hot - 280
                    lmtd = dh if dh == dc else (dh - dc) / (dh / dc).ln()
                    base = dict(subkind="heat_exchanger", flow_arrangement="counterflow",
                                T_h_in=400, T_c_in=280, C_h=float(ch), C_c=float(cc))
                    for field, value in (("effectiveness", target), ("Q", duty),
                                         ("T_h_out", hot), ("T_c_out", cold)):
                        result = solve_exchanger_or_radiation({**base, field: float(value)})
                        assert result["verified"], (ratio, eff, field, result)
                        assert math.isclose(result["UA_W_K"], float(duty / lmtd), rel_tol=1e-8)
    for invalid in (-.01, 1, 1.01, float("nan")):
        result = solve_exchanger_or_radiation({**base, "effectiveness": invalid})
        assert result["verified"] is False


def test_exchanger_checks_every_rating_and_sizing_observation():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier

    base = {"subkind": "heat_exchanger", "flow_arrangement": "counterflow",
            "T_h_in": 400., "T_c_in": 300., "C_h": 1000., "C_c": 1000.}
    # For matched rates and NTU=1, epsilon=1/2 and both streams move by 50 K.
    observed = {"UA": 1000., "effectiveness": .5, "Q": 50000., "T_h_out": 350., "T_c_out": 350.}

    def solve(changes):
        return run_router_conservation_cad_verifier({"mode": "exchangers_radiation", "spec": {**base, **changes}})

    for first in range(len(observed)):
        fields = list(observed)[first:]
        values = {k: observed[k] for k in fields}
        answer = solve(values)
        assert answer["verified"], answer
        for field in fields[1:]:
            incorrect = solve({**values, field: observed[field] * 1.1})
            assert not incorrect["verified"], (first, field, incorrect)
            assert (incorrect.get("invalid_observation") == field
                    or incorrect["observation_residuals"][field] > 1e-9)

    for field in observed:
        for invalid in (float("nan"), None, True):
            assert not solve({**observed, field: invalid})["verified"]
    assert not solve({"UA": 0., "flow_arrangement": "unknown"})["verified"]


def test_phase_change_exchanger_uses_sensible_outlet_to_size_and_checks_both():
    for arrangement, active, fixed in (("condenser", "T_c_out", "T_h_out"), ("boiler", "T_h_out", "T_c_out")):
        base = {"subkind": "heat_exchanger", "flow_arrangement": arrangement,
                "T_h_in": 400., "T_c_in": 300., "C_h": 1000., "C_c": 1000.}
        inlet = 400. if fixed == "T_h_out" else 300.
        both = solve_exchanger_or_radiation({**base, active: 350., fixed: inlet})
        assert both["verified"] and math.isclose(both["UA_W_K"], 1000 * math.log(2), rel_tol=1e-10)
        assert not solve_exchanger_or_radiation({**base, active: 350., fixed: inlet + 1.})["verified"]
        assert not solve_exchanger_or_radiation({**base, fixed: inlet})["verified"]
        capacity = "C_c" if arrangement == "condenser" else "C_h"
        for invalid in (0., -1000., float("inf")):
            assert not solve_exchanger_or_radiation({**base, "UA": 1000., capacity: invalid})["verified"]


def test_phase_change_sizing_at_the_pinch_and_finite_rate_with_rounded_outlet():
    for arr, sensible in (("condenser", "T_c_out"), ("boiler", "T_h_out")):
        base = {"flow_arrangement": arr, "T_h_in": 400., "T_c_in": 300., "C_h": 1000., "C_c": 1000.}
        ntu = 38 * math.log(2)
        rated = solve_exchanger_or_radiation({**base, "UA": 1000 * ntu})
        assert rated["verified"] and rated["effectiveness"] == 1.0 - 2.0**-38
        for key, value in (("effectiveness", rated["effectiveness"]), ("Q", rated["Q_W"]),
                           (sensible, rated[sensible + "_K"])):
            sized = solve_exchanger_or_radiation({**base, key: value})
            assert sized["verified"] and math.isclose(sized["NTU"], ntu, rel_tol=1e-6)
        # Exactly saturated outlet temperatures cannot identify a finite inverse
        # rating. A known finite conductance can nevertheless round to that outlet.
        for extreme in (1e-16, 50., 1000.):
            result = solve_exchanger_or_radiation({**base, "UA": 1000 * extreme})
            assert result["verified"], result
            lo, hi = result["LMTD_bounds_K"]
            assert lo - 1e-13 <= result["LMTD_K"] <= hi + 1e-13
        assert not solve_exchanger_or_radiation({**base, "effectiveness": 1.0})["verified"]

def test_lower_priority_phase_observations_cannot_cross_physical_endpoints():
    for arr, sensible, crossover in (("condenser", "T_c_out", 400.00000005),
                                     ("boiler", "T_h_out", 299.99999995)):
        base = {"flow_arrangement": arr, "T_h_in": 400., "T_c_in": 300., "C_h": 1000., "C_c": 1000.,
                "UA": 38000. * math.log(2)}
        assert solve_exchanger_or_radiation(base)["verified"]
        for key, value in (("effectiveness", 1.0000000005), ("Q", 100000.00005), (sensible, crossover)):
            failed = solve_exchanger_or_radiation({**base, key: value})
            assert not failed["verified"] and failed["invalid_observation"] == key
        inactive_capacity = "C_h" if arr == "condenser" else "C_c"
        assert not solve_exchanger_or_radiation({**base, inactive_capacity: -1000.})["verified"]


def test_parallel_flow_rejects_measured_crossovers_inside_global_duty_bounds():
    base = {"flow_arrangement": "parallel", "T_h_in": 400., "T_c_in": 300., "C_h": 1000., "C_c": 1000.,
            "UA": 16000. * math.log(2)}
    assert solve_exchanger_or_radiation(base)["verified"]
    # Equal-rate streams cannot cross their 350 K equilibrium in parallel flow.
    for key, value in (("T_h_out", 349.99999999), ("T_c_out", 350.00000001),
                       ("effectiveness", .50000000008), ("Q", 50000.000008)):
        result = solve_exchanger_or_radiation({**base, key: value})
        assert not result["verified"] and result["invalid_observation"] == key


def test_finite_arrangement_pinch_inverse_and_mixed_small_ratio_ratings():
    for arr, ch, cc, ntu in (("parallel", 1000., 1000., 19 * math.log(2)),
                             ("shell_tube_1_2", 1000., 2000., 26.),
                             ("crossflow_cmax_mixed", 1000., 2000., 26.),
                             ("crossflow_cmin_mixed", 1000., 2000., 52.)):
        base = {"flow_arrangement": arr, "T_h_in": 400., "T_c_in": 300., "C_h": ch, "C_c": cc}
        rated = solve_exchanger_or_radiation({**base, "UA": ntu * min(ch, cc)})
        assert rated["verified"], rated
        for key, output in (("effectiveness", "effectiveness"), ("Q", "Q_W"),
                            ("T_h_out", "T_h_out_K"), ("T_c_out", "T_c_out_K")):
            sized = solve_exchanger_or_radiation({**base, key: rated[output]})
            assert sized["verified"] and math.isclose(sized["NTU"], ntu, rel_tol=.001), (key, sized)

    base = {"flow_arrangement": "crossflow_cmax_mixed", "T_h_in": 400., "T_c_in": 300.,
            "C_h": 2.**40, "C_c": 1., "UA": 40.}
    rated = solve_exchanger_or_radiation(base)
    assert rated["verified"] and rated["effectiveness"] == -math.expm1(-2.**-40) / 2.**-40
    assert solve_exchanger_or_radiation({**base, "effectiveness": rated["effectiveness"]})["verified"]


def test_phase_inverse_preserves_deficit_in_supplied_units():
    from decimal import Decimal, localcontext

    base = {"flow_arrangement": "condenser", "T_h_in": 480., "T_c_in": 300., "C_h": 1000., "C_c": 1000.}
    duty = math.nextafter(180000., 0.)
    outlet = math.nextafter(480., 0.)
    with localcontext() as ctx:
        ctx.prec = 70
        gaps = (Decimal(1) - Decimal(duty) / 180000, (480 - Decimal(outlet)) / 180)
        for key, measurement, gap in zip(("Q", "T_c_out"), (duty, outlet), gaps):
            sized = solve_exchanger_or_radiation({**base, key: measurement})
            assert sized["verified"] and math.isclose(sized["NTU"], float(-gap.ln()), rel_tol=1e-14)


def test_counterflow_mean_temperature_check_handles_a_rounded_terminal_pinch():
    for hot_capacity, cold_capacity in ((1000., 2000.), (2000., 1000.), (1000., 1000.)):
        base = {"flow_arrangement": "counterflow", "T_h_in": 400., "T_c_in": 300.,
                "C_h": hot_capacity, "C_c": cold_capacity}
        for ntu in (40., 80., 1e17):
            result = solve_exchanger_or_radiation({**base, "UA": ntu * min(hot_capacity, cold_capacity)})
            assert result["verified"], result
            low, high = result["LMTD_bounds_K"]
            assert low <= result["LMTD_K"] <= high


def test_zero_duty_exchanger_inverse_binds_all_observations():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as solve

    for arrangement in ("counterflow", "parallel", "shell_tube_1_2", "crossflow_unmixed",
                        "crossflow_cmin_mixed", "crossflow_cmax_mixed", "condenser", "boiler"):
        base = {"subkind": "heat_exchanger", "flow_arrangement": arrangement, "T_h_in": 400.,
                "T_c_in": 300., "C_h": 1000., "C_c": 1000.}

        def check(**observations):
            return solve({"mode": "exchangers_radiation", "spec": {**base, **observations}})

        fixed = {"condenser": "T_h_out", "boiler": "T_c_out"}.get(arrangement)
        controls = {"effectiveness": 0., "Q": 0., "T_h_out": 400., "T_c_out": 300.}
        for key, value in controls.items():
            result = check(**{key: value})
            if key == fixed:
                assert not result["verified"]
            else:
                assert result["verified"] and result["UA_W_K"] == result["NTU"] == result["Q_W"] == 0.
        recovered = check(**controls)
        assert recovered["verified"] and all(value == 0. for value in recovered["observation_residuals"].values())
        assert not check(Q=0., effectiveness=.25)["verified"]
        assert not check(effectiveness=0., Q=1.)["verified"]
        assert not check(Q=0., effectiveness=0., T_h_out=390.)["verified"]
        assert not check(Q=0., T_h_in=300.)["verified"]
        assert not check(Q=0., **{"C_c" if arrangement == "condenser" else "C_h": 0.})["verified"]

        if arrangement == "counterflow":
            # Precise nonzero duty can coexist with an outlet rounded to its inlet.
            tiny = check(Q=1e-13, T_h_out=400., T_c_out=300.)
            assert tiny["verified"] and tiny["UA_W_K"] > 0.


def test_electrochem_species_and_written_reaction_normalization():
    from fractions import Fraction

    spec = dict(E0_V=0.7996, n_electrons=1, Q_reaction=0.25, current_A=0.5,
                time_s=7200, molar_mass_g_mol=107.8682, faradaic_efficiency=0.75)
    base = solve_electrochem_or_separation(spec)
    charge = Fraction(2700)
    silver = charge / Fraction("96485.33212")
    for scale in range(1, 101):
        row = dispatch_wave7_verifier("electrochem_separations", {
            **spec, "n_electrons": scale, "product_stoich": scale,
            "Q_reaction": 0.25 ** scale,
        })
        assert row["verified"]
        assert row["E_nernst_V"] == pytest.approx(base["E_nernst_V"], rel=1e-14)
        assert row["delta_G_J_mol"] == pytest.approx(scale * base["delta_G_J_mol"], rel=1e-14)
        assert row["n_electrons"] == row["product_stoich"] == scale
        assert row["charge_passed_C"] == 3600 and row["reaction_charge_C"] == charge
        assert row["reaction_extent_mol"] == row["moles_reacted"] == float(silver / scale)
        assert row["product_moles"] == float(silver)
        assert row["mass_deposited_g"] == float(silver * Fraction("107.8682"))


def test_electrochem_mass_information_and_time_for_target():
    spec = dict(E0_V=0.8, n_electrons=2, product_stoich=3, current_A=5, time_s=10)
    solve = lambda **fields: solve_electrochem_or_separation({**spec, **fields})
    assert solve()["verified"] and solve()["mass_deposited_g"] is None
    assert solve(time_s=0)["mass_deposited_g"] == 0
    zero = solve(faradaic_efficiency=0)
    assert zero["verified"] and zero["charge_passed_C"] == 50
    assert zero["reaction_charge_C"] == zero["product_moles"] == zero["mass_deposited_g"] == 0
    target = solve(molar_mass_g_mol=10, faradaic_efficiency=0.5, target_mass_g=0.4)
    expected = 0.4 * 2 * 96485.33212 / (5 * 0.5 * 3 * 10)
    assert target["verified"] and target["target_time_s"] == pytest.approx(expected)
    assert target["target_mass_status"] == "unique"
    replay = solve(time_s=target["target_time_s"], molar_mass_g_mol=10, faradaic_efficiency=0.5)
    assert replay["mass_deposited_g"] == pytest.approx(0.4, rel=1e-14)
    assert solve(target_mass_g=1)["target_mass_status"] == "molar_mass_required"
    for stop in ({"current_A": 0}, {"faradaic_efficiency": 0}):
        failed = solve(target_mass_g=1, **stop)
        assert not failed["verified"] and failed["target_time_s"] is None
        assert failed["target_mass_status"] == "unreachable_at_zero_rate"
        flat = solve(target_mass_g=0, **stop)
        assert flat["verified"] and flat["target_mass_status"] == "all_nonnegative_times"
        assert flat["target_time_s"] is None
    assert solve(target_mass_g=0)["target_time_s"] == 0


def test_electrochem_invalid_input_returns_structured_refusal():
    spec = {"E0_V": 0.8, "n_electrons": 1}
    for patch in ({"time_s": -1}, {"molar_mass_g_mol": -1}, {"molar_mass_g_mol": 0},
                  {"product_stoich": 0}, {"product_stoich": -1}, {"mode": "typo"},
                  {"faradaic_efficiency": -0.1}, {"faradaic_efficiency": 1.1},
                  {"target_mass_g": -1}, {"current_A": True}, {"T_K": float("inf")},
                  {"E0_V": float("nan")}, {"time_s": "bad"}, {"n_electrons": None}):
        row = solve_electrochem_or_separation({**spec, **patch})
        assert row["verified"] is False and row["error"]
        assert "mass_deposited_g" not in row


def test_registered_distillation_uses_tabulated_equilibrium_over_scalar_volatility():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    spec = {"subkind": "distillation", "z_F": .5, "x_D": .9, "x_B": .1,
            "alpha": 3, "R": 1.5, "equilibrium_points":
            [[0,0], [.1,.2], [.3,.4], [.5,.75], [.7,.78], [.9,.95], [1,1]]}
    result = run_router_conservation_cad_verifier({"mode": "electrochem_separations", "spec": spec})
    assert result["verified"] is False
    assert result["minimum_operating_gap"] == 0
    assert result["limiting_pinch_x"] == .7
    spec["R"] = 2
    result = run_router_conservation_cad_verifier({"mode": "electrochem_separations", "spec": spec})
    assert result["verified"] is True
    assert (result["N_stages_integer"], result["feed_stage"]) == (16, 7)
