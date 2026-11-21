"""Tests for PLE Wave-7 deterministic engineering verifiers & fast-path pruner."""
from __future__ import annotations

import math
try:
    import pytest
except ModuleNotFoundError:
    pytest = None  # type: ignore[assignment]

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
