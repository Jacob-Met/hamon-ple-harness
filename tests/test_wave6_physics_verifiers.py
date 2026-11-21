"""Cold deterministic unit + integration tests for PLE Wave-6 physics/engineering verifiers."""
from __future__ import annotations

import math

from hamon_wave6_physics_verifiers import (
    WAVE6_DOMAINS,
    dispatch_wave6_verifier,
    infer_wave6_domain,
    prune_deterministic_fast_path,
    solve_compressible_gas_acoustics,
    solve_mohr_vessel_buckling,
    solve_reactor_kinetics,
    solve_transmission_line_rlc,
)


def test_compressible_isentropic_area_mach_inversion_sub_and_supersonic():
    sup = solve_compressible_gas_acoustics({"subkind": "isentropic", "area_ratio": 1.6875, "regime": "supersonic"})
    assert sup["verified"] is True
    assert math.isclose(sup["mach"], 2.0, rel_tol=1e-10)
    assert math.isclose(sup["T0_over_T"], 1.8, rel_tol=1e-12)
    assert math.isclose(sup["P0_over_P"], 1.8 ** 3.5, rel_tol=1e-11)

    sub = solve_compressible_gas_acoustics({"subkind": "isentropic", "area_ratio": 1.6875, "regime": "subsonic"})
    assert sub["verified"] is True
    assert 0.3 < sub["mach"] < 0.4
    assert math.isclose(sub["A_over_Astar"], 1.6875, rel_tol=1e-11)


def test_normal_shock_rankine_hugoniot_exact():
    res = solve_compressible_gas_acoustics({"subkind": "normal_shock", "M1": 2.0, "gamma": 1.4})
    assert res["verified"] is True
    assert math.isclose(res["P2_over_P1"], 4.5, rel_tol=1e-12)
    assert math.isclose(res["rho2_over_rho1"], 8.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(res["M2"], math.sqrt(1.8 / 5.4), rel_tol=1e-12)
    assert res["P02_over_P01"] < 1.0


def test_fanno_and_rayleigh_duct_flows():
    fanno = solve_compressible_gas_acoustics({"subkind": "fanno", "M1": 0.3, "f": 0.005, "L": 12.0, "D_h": 0.05})
    assert fanno["verified"] is True
    assert 0.3 < fanno["M2"] < 1.0
    assert fanno["choked"] is False

    ray = solve_compressible_gas_acoustics({"subkind": "rayleigh", "M1": 0.25, "T01": 300.0, "q_J_kg": 1.5e5})
    assert ray["verified"] is True
    assert 0.25 < ray["M2"] <= 1.0


def test_acoustics_impedance_and_helmholtz():
    ac = solve_compressible_gas_acoustics(
        {"subkind": "acoustics", "rho1": 1.21, "c1": 343.0, "rho2": 1000.0, "c2": 1480.0,
         "A_neck": 1.0e-3, "V_cavity": 2.0e-3, "L_eff": 0.05}
    )
    assert ac["verified"] is True
    assert ac["transmission_loss_dB"] > 25.0
    assert ac["helmholtz_f0_Hz"] > 50.0


def test_transmission_line_quarter_wave_and_smith_stub():
    tl = solve_transmission_line_rlc(
        {"subkind": "transmission_line", "Z0": 50.0, "R_L": 100.0, "X_L": 0.0, "length_wavelengths": 0.25}
    )
    assert tl["verified"] is True
    assert math.isclose(tl["Z_in_real"], 25.0, rel_tol=1e-11)
    assert abs(tl["Z_in_imag"]) < 1e-11
    assert math.isclose(tl["VSWR"], 2.0, rel_tol=1e-12)

    stub = solve_transmission_line_rlc(
        {"subkind": "smith_stub", "Z0": 50.0, "R_L": 60.0, "X_L": -80.0, "stub_type": "short"}
    )
    assert stub["verified"] is True
    assert stub["residual"] < 1e-11
    assert 0.0 <= stub["d_over_lambda"] < 0.5
    assert 0.0 <= stub["l_stub_over_lambda"] < 0.5


def test_rlc_transient_all_three_damping_regimes():
    for r_val, expected_regime in [(10.0, "underdamped"), (200.0, "critically_damped"), (500.0, "overdamped")]:
        res = solve_transmission_line_rlc(
            {"subkind": "rlc_transient", "R": r_val, "L": 0.1, "C": 1.0e-5, "V_s": 12.0, "t": 0.002}
        )
        assert res["regime"] == expected_regime
        assert res["verified"] is True
        assert res["residual"] < 1e-9


def test_mohr_stress_pressure_vessel_and_column_buckling():
    mohr = solve_mohr_vessel_buckling(
        {"subkind": "mohr_stress", "sigma_x": 80e6, "sigma_y": -40e6, "tau_xy": 80e6, "sigma_yield": 250e6}
    )
    assert mohr["verified"] is True
    assert math.isclose(mohr["tau_max_inplane"], 100e6, rel_tol=1e-12)
    assert math.isclose(mohr["sigma_1"], 120e6, rel_tol=1e-12)
    assert math.isclose(mohr["sigma_3"], -80e6, rel_tol=1e-12)

    vessel = solve_mohr_vessel_buckling({"subkind": "pressure_vessel", "r_i": 0.10, "r_o": 0.15, "p_i": 50e6})
    assert vessel["verified"] is True
    assert math.isclose(vessel["sigma_hoop_lame"], 50e6 * (0.15**2 + 0.10**2) / (0.15**2 - 0.10**2), rel_tol=1e-12)
    assert math.isclose(vessel["sigma_radial_lame"], -50e6, rel_tol=1e-12)

    euler = solve_mohr_vessel_buckling(
        {"subkind": "column_buckling", "E": 200e9, "sigma_y": 250e6, "L": 4.0, "A": 2.0e-3, "I": 1.5e-6, "K": 1.0}
    )
    assert euler["verified"] is True
    assert euler["regime"] == "euler_elastic"

    johnson = solve_mohr_vessel_buckling(
        {"subkind": "column_buckling", "E": 200e9, "sigma_y": 250e6, "L": 1.0, "A": 2.0e-3, "I": 1.5e-6, "K": 0.5}
    )
    assert johnson["verified"] is True
    assert johnson["regime"] == "johnson_inelastic"


def test_reactor_kinetics_pfr_cstr_series_and_adiabatic():
    pfr = solve_reactor_kinetics(
        {"reactor_type": "pfr", "order": 2.0, "k": 0.25, "C_A0": 2.0, "v0": 0.05, "conversion": 0.8}
    )
    assert pfr["verified"] is True
    assert math.isclose(pfr["volume_m3"], 0.4, rel_tol=1e-11)

    pfr_inv = solve_reactor_kinetics(
        {"reactor_type": "pfr", "order": 2.0, "k": 0.25, "C_A0": 2.0, "v0": 0.05, "volume": 0.4}
    )
    assert pfr_inv["verified"] is True
    assert math.isclose(pfr_inv["conversion_X"], 0.8, rel_tol=1e-10)

    cstr_ad = solve_reactor_kinetics(
        {"reactor_type": "cstr", "order": 1.0, "A_preexp": 1.0e8, "E_a": 55000.0,
         "T0": 320.0, "delta_T_adiabatic": 60.0, "C_A0": 1.5, "v0": 0.02, "conversion": 0.85}
    )
    assert cstr_ad["verified"] is True
    assert math.isclose(cstr_ad["exit_T_K"], 320.0 + 60.0 * 0.85, rel_tol=1e-12)

    series = solve_reactor_kinetics(
        {"reactor_type": "cstr_series", "n_stages": 3, "order": 1.0, "k": 0.4, "C_A0": 1.0, "v0": 0.1, "conversion": 0.875}
    )
    assert series["verified"] is True
    assert math.isclose(series["volume_m3"], 0.75, rel_tol=1e-10)


def test_dispatcher_and_fast_path_cost_pruner():
    assert len(WAVE6_DOMAINS) == 4
    assert infer_wave6_domain("Compute downstream Mach across a normal shock at M1=2.4") == "compressible_acoustics"
    out = dispatch_wave6_verifier({"domain": "compressible_acoustics", "params": {"subkind": "normal_shock", "M1": 2.5}})
    assert out["verified"] is True

    pruned = prune_deterministic_fast_path(
        "medium", 5.3532, presolve_verified=True, domain="compressible_acoustics", residual=1e-12
    )
    assert pruned["fast_path_pruned"] is True
    assert pruned["tier"] == "light"
    assert pruned["cost_saved_usd"] > 3.0

    unpruned = prune_deterministic_fast_path(
        "medium", 5.3532, presolve_verified=False, domain="compressible_acoustics", residual=1e-2
    )
    assert unpruned["fast_path_pruned"] is False
    assert unpruned["tier"] == "medium"
