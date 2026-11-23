"""Exercise the real Pi registration -> TypeScript -> Python solver boundary.

These synthetic inputs qualify runtime reachability, not benchmark accuracy.
"""
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import tempfile
import unittest

try:
    import pytest
except ModuleNotFoundError:
    class _PytestShim:
        @staticmethod
        def fixture(*_a, **_kw):
            def _dec(fn):
                return fn
            return _dec

        class mark:
            @staticmethod
            def parametrize(*_a, **_kw):
                def _dec(fn):
                    return fn
                return _dec

    pytest = _PytestShim()

from hamon_pde_control_psychro_fem import verify_wave5_engineering
from hamon_wave6_physics_verifiers import dispatch_wave6_verifier
from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("pde", {"mode": "convective_fin_1d", "k_w_mk": 190.0,
             "area_m2": 1.6e-4, "perimeter_m": 0.052, "h_w_m2k": 42.0,
             "length_m": 0.11, "t_base_c": 120.0, "t_inf_c": 23.0}),
    ("control", {"mode": "state_space", "A": [[-1.7]], "B": [[2.0]],
                 "C": [[3.0]], "D": [[0.0]]}),
    ("psychrometrics", {"mode": "state", "t_db_c": 26.0, "rh": 0.55}),
    ("fem", {"mode": "euler_bernoulli_beam_1d", "x_nodes_m": [0.0, 1.0, 2.0],
             "EI_Nm2": 2.2e7, "q_uniform_N_m": -11000.0,
             "supports": [{"node": 0, "fix_w": True, "fix_theta": True},
                          {"node": 2, "fix_w": True, "fix_theta": False}]}),
    ("compressible_acoustics", {"subkind": "normal_shock", "M1": 2.4, "gamma": 1.4}),
    ("rf_transmission_rlc", {"subkind": "transmission_line", "Z0": 75.0,
                             "R_L": 150.0, "X_L": 0.0, "length_wavelengths": 0.25}),
    ("solid_mohr_vessel_buckling", {"subkind": "mohr_stress", "sigma_x": 90e6,
                                   "sigma_y": -30e6, "tau_xy": 45e6,
                                   "sigma_yield": 250e6}),
    ("reactor_kinetics", {"reactor_type": "pfr", "order": 1.0, "k": 0.21,
                          "C_A0": 1.5, "v0": 0.012, "conversion": 0.63}),
    ("thermo_cycles", {"subkind": "otto", "r": 9.1, "T1": 295.0, "T3": 2050.0}),
    ("exchangers_radiation", {"subkind": "heat_exchanger", "flow_arrangement": "counterflow",
                              "C_h": 2000.0, "C_c": 2700.0,
                              "T_h_in": 410.0, "T_c_in": 290.0, "UA": 3200.0}),
    ("lagrangian_orbital", {"subkind": "lagrangian_modes", "M": [[2.0, 0.0], [0.0, 1.0]],
                            "K": [[6.0, -2.0], [-2.0, 4.0]]}),
    ("electrochem_separations", {"subkind": "electrochemistry", "E0_V": 1.05,
                                 "n_electrons": 2.0, "T_K": 298.15, "Q_reaction": 0.12}),
]
def beam_profile(q0, q1, count, rigidity=1000.0):
    nodes = [2.0 * k / count for k in range(count + 1)]
    loads = [q0 + (q1 - q0) * x / 2.0 for x in nodes]
    return {
        "mode": "euler_bernoulli_beam_1d", "x_nodes_m": nodes, "EI_Nm2": rigidity,
        "supports": [{"node": 0, "fix_w": True, "fix_theta": True}],
        "elements": [{"q_start_N_m": a, "q_end_N_m": b} for a, b in zip(loads, loads[1:])],
    }


BEAM_PROFILES = {
    "beam-ramp-coarse": beam_profile(0, -600, 1),
    "beam-ramp-fine": beam_profile(0, -600, 4),
    "beam-reverse-small": beam_profile(-600, 0, 3, rigidity=1e6),
    "beam-trapezoid": beam_profile(-100, -600, 2),
    "beam-uniform-ends": beam_profile(-350, -350, 1),
    "beam-uniform": {**beam_profile(-350, -350, 1), "elements": [{"q_N_m": -350}]},
    "beam-antisymmetric": {**beam_profile(-600, 600, 1),
        "supports": [{"node": i, "fix_w": True, "fix_theta": True} for i in (0, 1)]},
    "beam-extra-element": {**beam_profile(0, -600, 1),
                           "elements": [{"q_N_m": 0}, {"q_start_N_m": 0, "q_end_N_m": -600}]},
    "beam-incomplete": {**beam_profile(0, -600, 1), "elements": [{"q_end_N_m": -600}]},
    "beam-ambiguous": {**beam_profile(0, -600, 1), "elements": [
        {"q_N_m": -600, "q_start_N_m": 0, "q_end_N_m": -600}]},
}


NEGATIVE_PHYSICS = [
    ("pde", {**CASES[0][1], "length_m": -0.11}),
    ("thermo_cycles", {"subkind": "otto", "r": 0.9, "T1": 295.0, "T3": 2050.0}),
]


def direct(family, spec):
    if family in {"pde", "control", "psychrometrics", "fem"}:
        result = verify_wave5_engineering({**spec, "domain": family})
    elif family in {"compressible_acoustics", "rf_transmission_rlc",
                    "solid_mohr_vessel_buckling", "reactor_kinetics"}:
        result = dispatch_wave6_verifier(family, spec)
    else:
        result = dispatch_wave7_verifier(family, spec)
    return json.loads(json.dumps(result, allow_nan=False))


# All three expectations are algebraic ranks of the supplied binary64 data.
CONTROL_RANK_CASES = [
    ("control-rank-scaled", {"A": [[-1, 0], [0, -2]],
                             "B": [[1], [2.0**-42]], "C": [[1, 2.0**42]]}, 2),
    ("control-rank-rounded-product", {"A": [[-1, 0], [2.0**-55, -1]],
                                      "B": [[1], [1]], "C": [[1, 1]]}, 2),
    ("control-rank-invariant-ports", {"A": [[-1, 0], [2.0**-55, -1]],
                                      "B": [[0], [1]], "C": [[1, 0]]}, 1),
]


# These four registered examples use two states. An absent requested P denotes
# the inconsistent identity-forcing equation in the last example.
IDENTITY_2 = [[1, 0], [0, 1]]
STABILITY_CASES = [
    ("slow-stable", {"A": [[-1, 0], [0, -(2.0**-55)]]},
     True, [[0.5, 0], [0, 2**54]]),
    ("slow-unstable", {"A": [[-1, 0], [0, 2.0**-55]]},
     False, [[0.5, 0], [0, -(2**54)]]),
    ("coupled-semidefinite", {"A": [[-1, 1], [0, -1]], "Q": [[1, 0], [0, 0]]},
     True, [[0.5, 0.25], [0.25, 0.25]]),
    ("singular-forcing", {"A": [[-1, 0], [0, 0]]}, False, None),
]


# Use binary64 inputs on the Pi JSON wire, including the finite 3T/2 entries.
MIXED_FAST = 2.0**512
SPIRAL_FAST = 2.0**520
RANGE_SCALE = 2.0**1023
DIAGNOSTIC_CASES = [
    ("ordinary-poles", [[-1, 0], [0, -2]], "finite", [(-2, 0), (-1, 0)]),
    ("mixed-fast-poles", [[-MIXED_FAST, 0], [0, -1]], "finite",
     [(-MIXED_FAST, 0), (-1, 0)]),
    ("spiral-poles", [[-SPIRAL_FAST, -SPIRAL_FAST], [SPIRAL_FAST, -SPIRAL_FAST]],
     "outside_float_range", [(-SPIRAL_FAST, -SPIRAL_FAST), (-SPIRAL_FAST, SPIRAL_FAST)]),
    ("outside-float-pole", [[-1.5 * RANGE_SCALE, RANGE_SCALE], [RANGE_SCALE, -1.5 * RANGE_SCALE]],
     "outside_float_range", None),
]


TF_DELTA = 2.0**-55
TF_UNIT = Fraction(1)
# Root count triples are ordered right half-plane, imaginary axis, left half-plane.
# Closed coefficients are specified independently at exact wire precision.
TRANSFER_CASES = [
    ("tf-slow", [1, TF_DELTA], [1], (0, 0, 1), [1, TF_UNIT + Fraction(TF_DELTA)], (0, 0, 1)),
    ("tf-feedback-boundary", [1, 1, 1, 1], [-TF_DELTA], (0, 2, 1),
     [1, 1, 1, TF_UNIT - Fraction(TF_DELTA)], (0, 0, 3)),
    ("tf-double-axis-pair", [1, 0, 2, 0, 1], [0], (0, 4, 0), [1, 0, 2, 0, 1], (0, 4, 0)),
    ("tf-undefined-feedback", [1, 1], [-1, -1], (0, 0, 1), [], None),
]


# Decimal literals correspond to the Bode solver's existing decimal-string
# interpretation. Use them directly so precomputing 3B or B/2 cannot round them.
BODE_RANGE_CASES = [
    ("high-gain", [5e200], [4e-140, 3e200], 126.86989764584402),
    ("tiny-gain", [5e-140], [4e200, 3e-140], 126.86989764584402),
    ("high-phase", [1e-140, -1e200, 0], [1e-140, 1e200], 90.0),
    ("high-pole", [5e307], [1e-320, 0, 1e308], 0.0),
    ("negative-unity-hole", [-1e-320, 0, -1e308], [1e-320, 0, 1e308], None),
    ("signed-zero-wrap", [1e-140, -1e200, 0, 0], [1e-140, 1e200], 180.0),
]


# Transfer-function DC ratios use the accepted binary64 values. In particular
# the subnormal tie below has exact value 3u/2 and rounds to the even value 2u.
DC_UNIT = 2.0**-1074
DC_SMALL = Fraction(1e-300) / Fraction(1e100)
DC_RANGE_CASES = [
    ("dc-positive-underflow", [1e-300], [1e100, 1e100], DC_SMALL, None, "outside_float_range"),
    ("dc-common-origin-underflow", [-1e-300, 0], [1e100, 1e100, 0], -DC_SMALL, None, "outside_float_range"),
    ("dc-subnormal-tie", [3 * DC_UNIT], [2, 2], 3 * Fraction(DC_UNIT) / 2, 2 * DC_UNIT, "finite"),
    ("dc-origin-pole", [-1, 0], [1, 1, 0, 0], None, None, "pole_at_zero"),
]


# Last two columns are the expected scalar (None for an invalid claim) and
# the overall pass flag. A valid claim can describe an already failed result.
CLAIM_UNIT_GAIN = {"mode": "transfer_function", "num": [1, 1], "den": [1, 1]}
CLAIM_LARGE_GAIN = {"mode": "transfer_function", "num": [1e308, 1e308], "den": [1, 1]}
CLAIM_EXACT_ZERO_TOL = {"atol": 0, "rtol": 0}
CLAIM_TOLERANCE_CASES = [
    ("claim-rounded-subtraction", CLAIM_UNIT_GAIN,
     {"claimed_field": "dc_gain", "claimed_value": -(2.0**-54), "atol": 1, "rtol": 0}, 1, False),
    ("claim-overflow-outside", CLAIM_LARGE_GAIN,
     {"claimed_field": "dc_gain", "claimed_value": -1e308, "atol": 0, "rtol": 1.9}, 1e308, False),
    ("claim-overflow-boundary", CLAIM_LARGE_GAIN,
     {"claimed_field": "dc_gain", "claimed_value": -1e308, "atol": 0, "rtol": 2}, 1e308, True),
    ("claim-boolean", {"mode": "state_space", "A": [[-1]]},
     {"claimed_field": "hurwitz_stable", "claimed_value": True, **CLAIM_EXACT_ZERO_TOL}, True, True),
    ("claim-rendered-third", {"mode": "transfer_function", "num": [1], "den": [3, 3]},
     {"claimed_field": "dc_gain", "claimed_value": "0.3333333333333333", **CLAIM_EXACT_ZERO_TOL}, 1/3, True),
    ("claim-already-failed", {"mode": "transfer_function", "num": [-1, -1], "den": [1, 1]},
     {"claimed_field": "hurwitz_stable", "claimed_value": True, **CLAIM_EXACT_ZERO_TOL}, True, False),
    ("claim-nan-text", CLAIM_UNIT_GAIN,
     {"claimed_field": "dc_gain", "claimed_value": "NaN", **CLAIM_EXACT_ZERO_TOL}, None, False),
    ("claim-negative-tolerance", CLAIM_UNIT_GAIN,
     {"claimed_field": "dc_gain", "claimed_value": 1, "atol": -1, "rtol": 0}, None, False),
    ("claim-unknown-field", CLAIM_UNIT_GAIN,
     {"claimed_field": "missing_scalar", "claimed_value": 0}, None, False),
    ("claim-evidence-field", CLAIM_UNIT_GAIN,
     {"claimed_field": "dc_gain_exact", "claimed_value": 1}, None, False),
    ("claim-unavailable-scalar", {"mode": "transfer_function", "num": [1], "den": [1, 0]},
     {"claimed_field": "dc_gain", "claimed_value": 0}, None, False),
    ("claim-list-selector", CLAIM_UNIT_GAIN,
     {"claimed_field": ["dc_gain"], "claimed_value": 0}, None, False),
    ("claim-missing-value", CLAIM_UNIT_GAIN, {"claimed_field": "dc_gain"}, None, False),
    ("claim-missing-selector", CLAIM_UNIT_GAIN, {"claimed_value": 1}, None, False),
]


FIN_TINY_DRIVE = 2.0**-45
FIN_MATCHED_ROBIN = {
    "mode": "convective_fin_1d", "tip_bc": "convective", "k_w_mk": 1,
    "area_m2": 1, "perimeter_m": 4, "h_w_m2k": 1, "h_tip_w_m2k": 2,
    "length_m": 1, "t_inf_c": 0,
}
# Matched tip: q=2*drive and q_max_ref=6*drive. Infinite fin: q=drive/2,
# which rounds to zero at the smallest positive binary64 drive.
FIN_NORMALIZATION_CASES = [
    ("fin-positive-tiny", {**FIN_MATCHED_ROBIN, "t_base_c": FIN_TINY_DRIVE},
     2*FIN_TINY_DRIVE, 6*FIN_TINY_DRIVE, 0.5, 2),
    ("fin-negative-tiny", {**FIN_MATCHED_ROBIN, "t_base_c": -FIN_TINY_DRIVE},
     -2*FIN_TINY_DRIVE, -6*FIN_TINY_DRIVE, 0.5, 2),
    ("fin-zero-drive", {**FIN_MATCHED_ROBIN, "t_base_c": 0}, 0, 0, None, None),
    ("fin-heat-rounds-zero", {
        "mode": "convective_fin_1d", "tip_bc": "infinite", "k_w_mk": 0.25,
        "area_m2": 1, "perimeter_m": 1, "h_w_m2k": 1, "length_m": 1,
        "t_base_c": 2.0**-1074, "t_inf_c": 0}, 0, 2.0**-1074, 0.5, 0.5),
]


WET_BULB_LOW_LIQUID = {"t_db_c": 40, "t_wb_c": 5}
WET_BULB_CASES = [
    ("wet-liquid-infeasible", {"mode": "state", **WET_BULB_LOW_LIQUID}, False),
    ("wet-ice-infeasible", {"mode": "state", "t_db_c": 0, "t_wb_c": -20}, False),
    ("wet-clipped-zero-claim", {"mode": "state", **WET_BULB_LOW_LIQUID,
                                "claimed_field": "w_kg_per_kgda", "claimed_value": 0,
                                "atol": 0, "rtol": 0}, False),
    ("wet-mixing-inlet", {"mode": "adiabatic_mixing",
                          "stream1": {**WET_BULB_LOW_LIQUID, "m_da_kg_s": 1},
                          "stream2": {"t_db_c": 20, "w_kg_per_kgda": 0, "m_da_kg_s": 1}}, False),
    ("wet-cooling-inlet", {"mode": "cooling_coil", "inlet": WET_BULB_LOW_LIQUID,
                           "outlet": {"t_db_c": 20, "w_kg_per_kgda": 0}, "m_da_kg_s": 1}, False),
    ("wet-chart-state", {"mode": "state", "t_db_c": 40, "t_wb_c": 20}, True),
    ("wet-dry-control", {"mode": "state", "t_db_c": 40, "w_kg_per_kgda": 0}, True),
    ("wet-saturated-control", {"mode": "state", "t_db_c": 20, "rh": 1}, True),
]


# Each system uses a numeric input. None in the claim column means an ordinary
# request; none of these calls submits null as a dew-point temperature.
DEW_POINT_CASES = [
    ("dew-frost", {"t_dp_c": -90}, None, None, True),
    ("dew-frost-true", {"t_dp_c": -90}, (-90, 1e-8), None, True),
    ("dew-frost-old-floor", {"t_dp_c": -90}, (-80, 0), None, False),
    ("dew-dry", {"rh": 0}, None, "dry_no_finite_point", True),
    ("dew-dry-old-sentinel", {"rh": 0}, (-100, 0), "dry_no_finite_point", False),
    ("dew-ordinary", {"rh": 0.5}, None, None, True),
    ("dew-below-range", {"w_kg_per_kgda": 1e-16}, None, "below_correlation_range", True),
    ("dew-below-old-floor", {"w_kg_per_kgda": 1e-16}, (-80, 0), "below_correlation_range", False),
]


TRUSS_LOAD_SCALE = 2.0**-12
TRUSS_WEAK_MODULUS = 2.0**-60
TRUSS_WEAK_ENERGY = 125 * 2.0**-85
TRUSS_CASES = [
    ("truss-strong", "strong", None, True),
    ("truss-weak", "weak", None, True),
    ("truss-weak-energy", "weak", TRUSS_WEAK_ENERGY, True),
    ("truss-weak-zero", "weak", 0, False),
    ("truss-one-bar-mechanism", "mechanism", None, False),
]


HEAT_LARGE_MODE = 2**53 - 2  # Retains its integer value through the registered JSON route.
HEAT_PHASE_CASES = [
    ("heat-phase-large", HEAT_LARGE_MODE, 0.75, 0, None, True),
    ("heat-phase-true", HEAT_LARGE_MODE, 0.75, 0, ("u_exact", 1), True),
    ("heat-phase-false", HEAT_LARGE_MODE, 0.75, 0, ("u_crank_nicolson", 0), False),
    ("heat-phase-right", HEAT_LARGE_MODE, 1, 0, None, True),
    ("heat-phase-ordinary", 1, 0.5, 0.125, None, True),
]


HEAT_TIME_HARMONIC = 2**40
HEAT_TIME_INPUTS = {
    "underflow": {"alpha_m2_s": 2.0**-542, "length_m": 2.0**-500, "t_eval_s": 2.0**-542,
                  "x_eval_m": 2.0**-541, "modes": [{"n": HEAT_TIME_HARMONIC, "amplitude": 1}]},
    "overflow": {"alpha_m2_s": 2.0**512, "length_m": 2.0**511, "t_eval_s": 2.0**512,
                 "x_eval_m": 2.0**510, "modes": [{"n": 1, "amplitude": 1}]},
    "ordinary": {"alpha_m2_s": 1, "length_m": 1, "t_eval_s": 4,
                 "x_eval_m": 0.5, "modes": [{"n": 1, "amplitude": 1}]},
}
HEAT_TIME_UNDERFLOW_TARGET = math.exp(-math.pi**2 / 16)
HEAT_TIME_OVERFLOW_TARGET = math.exp(-4 * math.pi**2)
HEAT_TIME_CASES = [
    ("heat-time-underflow", "underflow", None, True),
    ("heat-time-underflow-true", "underflow",
     (HEAT_TIME_UNDERFLOW_TARGET, 2e-13 * HEAT_TIME_UNDERFLOW_TARGET), True),
    ("heat-time-underflow-false", "underflow", (1, 0), False),
    ("heat-time-overflow", "overflow", None, True),
    ("heat-time-overflow-true", "overflow",
     (HEAT_TIME_OVERFLOW_TARGET, 2e-13 * HEAT_TIME_OVERFLOW_TARGET), True),
    ("heat-time-overflow-false", "overflow", (0, 0), False),
    ("heat-time-ordinary-twin", "ordinary", None, True),
]


HEAT_AMPLITUDE_SCALE = 2.0**-100
HEAT_AMPLITUDE_PEAK = 2.0**1022
HEAT_AMPLITUDE_CONTINUUM = math.pi * HEAT_AMPLITUDE_SCALE
HEAT_AMPLITUDE_GRID = 120 * math.sin(math.pi / 120) * HEAT_AMPLITUDE_SCALE
HEAT_AMPLITUDE_INPUTS = {
    "large": (2.0**100, 2.0**-1022, HEAT_AMPLITUDE_PEAK),
    "ordinary": (1, HEAT_AMPLITUDE_SCALE, 1),
    "boundary": (2.0**100, 0, HEAT_AMPLITUDE_PEAK),
}
HEAT_AMPLITUDE_CASES = [
    ("heat-amplitude-large", "large", None, True),
    ("heat-amplitude-true", "large",
     ("u_exact", HEAT_AMPLITUDE_CONTINUUM, 2e-13 * HEAT_AMPLITUDE_CONTINUUM), True),
    ("heat-amplitude-false", "large", ("u_crank_nicolson", 0, 0), False),
    ("heat-amplitude-ordinary", "ordinary", None, True),
    ("heat-amplitude-boundary", "boundary", None, True),
]


@pytest.fixture(scope="module")
def routed(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("registered-engineering")
    calls = [{"id": family, "mode": family, "spec": spec} for family, spec in CASES]
    for family, spec in NEGATIVE_PHYSICS:
        calls.append({"id": "unverified-" + family, "mode": family, "spec": spec})
    for family, spec in CASES:
        selector = "mode" if "mode" in spec else (
            "reactor_type" if family == "reactor_kinetics" else "subkind"
        )
        calls.extend([
            {"id": "empty-" + family, "mode": family, "spec": {}},
            {"id": "selector-only-" + family, "mode": family,
             "spec": {selector: spec[selector]}},
            {"id": "no-selector-" + family, "mode": family,
             "spec": {key: value for key, value in spec.items() if key != selector}},
            {"id": "unknown-input-" + family, "mode": family,
             "spec": {selector: spec[selector], "typo_parameter": 1}},
        ])
    calls.append({"id": "rotated-octahedral-plane", "mode": "solid_mohr_vessel_buckling",
                  "spec": {"subkind": "mohr_stress", "stress_tensor": [[72, 24, 0], [24, 58, 0], [0, 0, -30]],
                           "plane_normal": [7, -1, 5]}})
    calls.append({"id": "enriched-product-two-tray", "mode": "electrochem_separations",
                  "spec": {"subkind": "distillation", "z_F": .8, "x_D": .9, "x_B": .7,
                           "alpha": 2., "q": 1., "R": 1000., "F_mol_s": 100.}})
    refrigerator = {"subkind": "vapor_compression", "h1": 250000., "h2s": 280000.,
                    "h3": 100000., "eta_c": .75}
    for name, data in (("specific", {}), ("measured", {"Q_evap_W": 75000.}),
                       ("contradictory", {"Q_evap_W": 75000., "W_comp_W": 99999.})):
        calls.append({"id": "refrigerator-" + name, "mode": "thermo_cycles",
                      "spec": {**refrigerator, **data}})
    cstr = {"reactor_type":"cstr","order":1,"C_A0":1,"v0":1,"T0":300,"delta_T_adiabatic":200,"k_ref":1,"T_ref":350,"E_a":80000,"gas_phase":False,"volume":1}
    calls.extend([
        {"id": "cstr-all-branches", "mode": "reactor_kinetics", "spec": {**cstr, "steady_state_branch": "all"}},
        {"id": "cstr-cold-branch", "mode": "reactor_kinetics", "spec": {**cstr, "steady_state_branch": 0}},
        {"id": "cstr-implicit-branch", "mode": "reactor_kinetics", "spec": cstr},
    ])
    equal_lengths = {"lhs": {"value": 2, "unit": "m"},
                     "rhs": {"value": 200, "unit": "cm"}}
    calls.extend([
        {"id": "fin-hot-tip-at-ambient-base", "mode": "pde", "spec": {
            "mode": "convective_fin_1d", "tip_bc": "prescribed_temp",
            "k_w_mk": 2.0, "area_m2": 0.02, "perimeter_m": 0.4, "h_w_m2k": 0.1,
            "length_m": 1.0, "t_base_c": 20.0, "t_inf_c": 20.0, "t_tip_c": 70.0}},
        {"id": "legacy-positive", "mode": "dimensional", "spec": equal_lengths},
        {"id": "legacy-auto", "mode": "auto", "spec": equal_lengths},
        {"id": "legacy-unverified", "mode": "dimensional", "spec": {
            **equal_lengths, "rhs": {"value": 3, "unit": "m"}}},
        {"id": "conflicting-domain", "mode": "control",
         "spec": {**CASES[1][1], "domain": "pde"}},
        {"id": "matching-domain", "mode": "control",
         "spec": {**CASES[1][1], "domain": "control"}},
        {"id": "unknown-family", "mode": "unknown_engineering_family", "spec": {}},
        {"id": "non-object-spec", "mode": "control", "spec": [1, 2]},
        {"id": "acoustics-unknown-input", "mode": "compressible_acoustics",
         "spec": {"subkind": "acoustics", "typo_parameter": 1}},
        {"id": "acoustics-incomplete-resonator", "mode": "compressible_acoustics",
         "spec": {"subkind": "acoustics", "A_neck": 0.01}},
        {"id": "explicit-zero-stress", "mode": "solid_mohr_vessel_buckling",
         "spec": {"subkind": "mohr_stress", "sigma_x": 0.0}},
        {"id": "explicit-zero-length", "mode": "rf_transmission_rlc",
         "spec": {"subkind": "transmission_line", "length_wavelengths": 0.0}},
    ])
    for case, matrices, _ in CONTROL_RANK_CASES:
        for field in ("controllable", "observable"):
            for claim in (False, True):
                calls.append({"id": f"{case}-{field}-{claim}", "mode": "control", "spec": {
                    "mode": "state_space", **matrices, "claimed_field": field, "claimed_value": claim,
                }})
    for case, matrices, _, _ in STABILITY_CASES:
        spec = {"mode": "state_space", **matrices}
        calls.append({"id": case, "mode": "control", "spec": spec})
        for field in ("hurwitz_stable", "lyapunov_pd"):
            for claim in (False, True):
                calls.append({"id": f"{case}-{field}-{claim}", "mode": "control",
                              "spec": {**spec, "claimed_field": field, "claimed_value": claim}})
    for case, matrix, _, _ in DIAGNOSTIC_CASES:
        spec = {"mode": "state_space", "A": matrix}
        calls.append({"id": case, "mode": "control", "spec": spec})
        for field in ("hurwitz_stable", "lyapunov_pd"):
            for claim in (False, True):
                calls.append({"id": f"{case}-{field}-{claim}", "mode": "control",
                              "spec": {**spec, "claimed_field": field, "claimed_value": claim}})
    for case, den, num, _, _, _ in TRANSFER_CASES:
        spec = {"mode": "transfer_function", "den": den, "num": num}
        calls.append({"id": case, "mode": "control", "spec": spec})
        for field in ("hurwitz_stable", "closed_loop_hurwitz_stable"):
            for claim in (False, True):
                calls.append({"id": f"{case}-{field}-{claim}", "mode": "control",
                              "spec": {**spec, "claimed_field": field, "claimed_value": claim}})
    for case, num, den, pm in BODE_RANGE_CASES:
        spec = {"mode": "transfer_function", "num": num, "den": den}
        calls.append({"id": "bode-" + case, "mode": "control", "spec": spec})
        claims = [("w_gc_rad_s", 0), ("w_pc_rad_s", 0)]
        if pm is not None:
            claims.append(("phase_margin_deg", pm))
        if case == "high-phase":
            claims.append(("gain_margin_db", -6800))
        for field, value in claims:
            atol = 0 if field in ("w_gc_rad_s", "w_pc_rad_s") else 1e-10
            calls.append({"id": f"bode-{case}-{field}", "mode": "control",
                          "spec": {**spec, "claimed_field": field, "claimed_value": value,
                                   "atol": atol, "rtol": 0}})
    for case, num, den, _, rendered, _ in DC_RANGE_CASES:
        spec = {"mode": "transfer_function", "num": num, "den": den}
        calls.append({"id": case, "mode": "control", "spec": spec})
        claims = [("zero", "dc_gain", 0), ("evidence", "dc_gain_exact", 0)]
        if rendered is not None:
            claims.append(("rendered", "dc_gain", rendered))
        for label, field, value in claims:
            calls.append({"id": f"{case}-{label}", "mode": "control",
                          "spec": {**spec, "claimed_field": field, "claimed_value": value, "rtol": 0, "atol": 0}})
    for case, system, claim, _, _ in CLAIM_TOLERANCE_CASES:
        calls.append({"id": case, "mode": "control", "spec": {**system, **claim}})
    for case, system, _, _, eta, epsilon in FIN_NORMALIZATION_CASES:
        calls.append({"id": case, "mode": "pde", "spec": system})
        claims = [("fin_efficiency", eta), ("fin_effectiveness", epsilon)]
        if case == "fin-positive-tiny":
            claims.append(("fin_normalization", 0))
        for field, value in claims:
            calls.append({"id": f"{case}-{field}", "mode": "pde", "spec": {
                **system, "claimed_field": field, "claimed_value": 0 if value is None else value,
                "atol": 0, "rtol": 0}})
    for case, system, _ in WET_BULB_CASES:
        calls.append({"id": case, "mode": "psychrometrics", "spec": {**system, "p_atm_pa": 101325}})
    for case, inputs, claim, _, _ in DEW_POINT_CASES:
        spec = {"mode": "state", "t_db_c": 20, "p_atm_pa": 101325, **inputs}
        if claim is not None:
            value, atol = claim
            spec.update(claimed_field="t_dp_c", claimed_value=value, atol=atol, rtol=0)
        calls.append({"id": case, "mode": "psychrometrics", "spec": spec})
    for case, direction, claim, _ in TRUSS_CASES:
        scale = TRUSS_LOAD_SCALE
        weak = TRUSS_WEAK_MODULUS
        load = (3*scale, 4*scale) if direction == "strong" else (4*scale*weak, -3*scale*weak)
        members = [{"i": 0, "j": 2, "E_pa": 1, "A_m2": 1}]
        if direction != "mechanism":
            members.append({"i": 1, "j": 2, "E_pa": weak, "A_m2": 1})
        spec = {"mode": "truss_2d", "nodes": [[-3, -4], [-4, 3], [0, 0]], "elements": members,
                "supports": [{"node": node, "fix_x": True, "fix_y": True} for node in (0, 1)],
                "loads": [{"node": 2, "fx_N": load[0], "fy_N": load[1]}]}
        if claim is not None:
            spec.update(claimed_field="strain_energy_J", claimed_value=claim, atol=0, rtol=0)
        calls.append({"id": case, "mode": "fem", "spec": spec})
    for case, harmonic, position, time, claim, _ in HEAT_PHASE_CASES:
        spec = {"mode": "heat_1d_transient", "alpha_m2_s": 1, "length_m": 1,
                "t_eval_s": time, "x_eval_m": position, "modes": [{"n": harmonic, "amplitude": 1}]}
        if claim is not None:
            field, value = claim
            spec.update(claimed_field=field, claimed_value=value, atol=0, rtol=0)
        calls.append({"id": case, "mode": "pde", "spec": spec})
    for case, system, claim, _ in HEAT_TIME_CASES:
        spec = {"mode": "heat_1d_transient", **HEAT_TIME_INPUTS[system]}
        if claim is not None:
            value, atol = claim
            spec.update(claimed_field="u_exact", claimed_value=value, atol=atol, rtol=0)
        calls.append({"id": case, "mode": "pde", "spec": spec})
    for case, system, claim, _ in HEAT_AMPLITUDE_CASES:
        length, position, amplitude = HEAT_AMPLITUDE_INPUTS[system]
        spec = {"mode": "heat_1d_transient", "alpha_m2_s": 1, "length_m": length,
                "t_eval_s": 0, "x_eval_m": position, "modes": [{"n": 1, "amplitude": amplitude}]}
        if claim is not None:
            field, value, atol = claim
            spec.update(claimed_field=field, claimed_value=value, atol=atol, rtol=0)
        calls.append({"id": case, "mode": "pde", "spec": spec})
    calls.extend({"id": name, "mode": "fem", "spec": spec}
                 for name, spec in BEAM_PROFILES.items())
    source = r"""
import registerTaskTools, { PLE_DOMAIN_TOOL_NAMES } from './task-tools.ts';
const tools = new Map();
registerTaskTools({ registerTool(t) { tools.set(t.name, t); }, on() {} });
const tool = tools.get('hamon_conservation_cad_verifier');
const results = {};
for (const c of JSON.parse(process.env.ENGINEERING_TOOL_CASES)) {
  try {
    const result = await tool.execute(c.id, {mode: c.mode, spec: c.spec}, null, () => {}, {});
    results[c.id] = {completed: true, details: result.details};
  } catch (error) {
    results[c.id] = {completed: false, error: String(error)};
  }
}
console.log(JSON.stringify({names: PLE_DOMAIN_TOOL_NAMES, results}));
"""
    env = {**os.environ, "TASK_RELAY_URL": "http://127.0.0.1:1",
           "TASK_WORKDIR": str(workdir), "TASK_HARNESS_SRC": str(ROOT),
           "TASK_HARNESS_PYTHON": sys.executable,
           "ENGINEERING_TOOL_CASES": json.dumps(calls)}
    proc = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", source],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    observed = json.loads(proc.stdout.strip().splitlines()[-1])
    assert observed["names"] == ["hamon_ifc_spatial_clash", "hamon_drawing_reader",
                                  "hamon_conservation_cad_verifier", "check_deliverable_preflight"]
    return observed["results"]


@pytest.mark.parametrize("family,spec", CASES, ids=[family for family, _ in CASES])
def test_registered_tool_reaches_each_family_without_changing_calculation(routed, family, spec):
    expected = direct(family, spec)
    assert expected.get("passed", expected.get("verified")) is True
    result = routed[family]
    assert result["completed"], result
    assert result["details"] == expected


@pytest.mark.parametrize("family,spec", NEGATIVE_PHYSICS)
def test_completed_transport_preserves_failed_physics(routed, family, spec):
    expected = direct(family, spec)
    assert expected.get("passed", expected.get("verified")) is False
    result = routed["unverified-" + family]
    assert result["completed"], result
    assert result["details"] == expected


@pytest.mark.parametrize("family", [family for family, _ in CASES])
@pytest.mark.parametrize("prefix", ["empty-", "selector-only-", "no-selector-"])
def test_registered_tool_refuses_unspecified_calculations(routed, family, prefix):
    assert routed[prefix + family]["completed"] is False
    assert "requires" in routed[prefix + family]["error"]


def test_registered_fin_preserves_signed_flux_at_an_ambient_base(routed):
    hot_tip = routed["fin-hot-tip-at-ambient-base"]
    assert hot_tip["completed"], hot_tip
    details = hot_tip["details"]
    assert details["passed"], details
    assert details["q_fin_w"] == pytest.approx(-1.7018362564786431)
    assert details["fin_efficiency"] is None


def test_legacy_checks_and_explicit_family_contract(routed):
    for name in ["legacy-positive", "legacy-auto"]:
        assert routed[name]["completed"]
        assert routed[name]["details"]["conservation_verified"] is True
    assert routed["legacy-unverified"]["completed"]
    assert routed["legacy-unverified"]["details"]["conservation_verified"] is False
    assert routed["matching-domain"]["details"] == routed["control"]["details"]
    for name in ["conflicting-domain", "unknown-family", "non-object-spec"]:
        assert routed[name]["completed"] is False


@pytest.mark.parametrize("family", [family for family, _ in CASES])
def test_unknown_keys_cannot_verify_an_unsupplied_physical_problem(routed, family):
    result = routed["unknown-input-" + family]
    assert not result["completed"] or result["details"].get(
        "passed", result["details"].get("verified")) is False


def test_default_capable_solvers_distinguish_unknown_keys_from_explicit_zero(routed):
    assert routed["acoustics-unknown-input"]["completed"] is False
    assert routed["acoustics-incomplete-resonator"]["completed"] is False
    for name in ["explicit-zero-stress", "explicit-zero-length"]:
        assert routed[name]["completed"], routed[name]
        assert routed[name]["details"]["verified"] is True


@pytest.mark.parametrize("case,matrices,rank", CONTROL_RANK_CASES,
                         ids=[case for case, _, _ in CONTROL_RANK_CASES])
def test_registered_algebraic_control_ranks(routed, case, matrices, rank):
    full_rank = rank == len(matrices["A"])
    for field in ("controllable", "observable"):
        for claim in (False, True):
            result = routed[f"{case}-{field}-{claim}"]
            assert result["completed"], result
            details = result["details"]
            assert details["domain"] == "control" and details["mode"] == "state_space"
            assert (details["controllability_rank"], details["observability_rank"]) == (rank, rank)
            assert details["controllable"] is full_rank and details["observable"] is full_rank
            accepted = claim == full_rank
            assert details["passed"] is accepted
            check = details["claim_check"]
            assert check["field"] == field and check["claimed"] == float(claim)
            assert check["actual"] == float(full_rank) and check["within_tolerance"] is accepted


def checked_pi_lyapunov_matrix(encoded, a, q):
    assert isinstance(encoded, list) and len(encoded) == 2
    assert all(len(row) == 2 and all(isinstance(x, str) for x in row) for row in encoded)
    p = [[Fraction(x) for x in row] for row in encoded]
    assert p[0][1] == p[1][0]
    for i in range(2):
        for j in range(2):
            lhs = sum(Fraction(a[k][i]) * p[k][j] + p[i][k] * Fraction(a[k][j])
                      for k in range(2))
            assert lhs == -Fraction(q[i][j])
    return p


@pytest.mark.parametrize("case,matrices,stable,expected_p", STABILITY_CASES,
                         ids=[row[0] for row in STABILITY_CASES])
def test_registered_state_space_stability(routed, case, matrices, stable, expected_p):
    result = routed[case]
    assert result["completed"], result
    details = result["details"]
    assert details["domain"] == "control" and details["mode"] == "state_space"
    assert details["passed"] is True and details["hurwitz_stable"] is stable, details
    cert = details["stability_certificate"]
    if expected_p is None:
        expected_pd = None
        assert cert["solution_status"] == details["lyapunov_solution_status"] == "inconsistent"
        assert cert["P_exact"] is None and cert["residual_exact"] is None
        assert cert["positive_definite"] is None
        assert details["lyapunov_P"] is None and details["lyapunov_P_exact"] is None
        assert details["lyapunov_residual_exact"] is None
        assert details["lyapunov_float_status"] == "not_unique_solution"
    else:
        assert cert["solution_status"] == details["lyapunov_solution_status"] == "unique"
        assert cert["residual_exact"] is True and details["lyapunov_residual_exact"] is True
        p = checked_pi_lyapunov_matrix(details["lyapunov_P_exact"], matrices["A"],
                                       matrices.get("Q", IDENTITY_2))
        assert p == expected_p and details["lyapunov_P"] == expected_p
        assert details["lyapunov_float_status"] == "finite"
        expected_pd = p[0][0] > 0 and p[0][0] * p[1][1] > p[0][1]**2
        identity_p = checked_pi_lyapunov_matrix(cert["P_exact"], matrices["A"], IDENTITY_2)
        identity_pd = identity_p[0][0] > 0 and identity_p[0][0] * identity_p[1][1] > identity_p[0][1]**2
        assert cert["positive_definite"] is stable and identity_pd is stable
    assert details["lyapunov_pd"] is expected_pd
    for field, truth in (("hurwitz_stable", stable), ("lyapunov_pd", expected_pd)):
        for claim in (False, True):
            reply = routed[f"{case}-{field}-{claim}"]
            assert reply["completed"], reply
            claimed = reply["details"]
            assert claimed[field] is truth
            assert claimed["passed"] is (claim is truth)
            if truth is None:
                assert isinstance(claimed.get("reason"), str) and claimed["reason"].strip()
            else:
                check = claimed["claim_check"]
                assert check["field"] == field and check["claimed"] == float(claim)
                assert check["actual"] == float(truth) and check["within_tolerance"] is (claim is truth)


@pytest.mark.parametrize("case,matrix,poly_status,poles", DIAGNOSTIC_CASES,
                         ids=[row[0] for row in DIAGNOSTIC_CASES])
def test_registered_state_space_diagnostic_range(routed, case, matrix, poly_status, poles):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "control" and result["mode"] == "state_space"
    assert result["passed"] is True and result["hurwitz_stable"] is True
    assert result["lyapunov_pd"] is True and result["lyapunov_solution_status"] == "unique"
    exact_p = checked_pi_lyapunov_matrix(result["lyapunov_P_exact"], matrix, IDENTITY_2)
    assert exact_p[0][0] > 0 and exact_p[0][0] * exact_p[1][1] > exact_p[0][1]**2
    assert result["lyapunov_float_status"] == "finite"
    assert result["lyapunov_P"] == [[float(value) for value in row] for row in exact_p]

    (a, b), (c, d) = [[Fraction(v) for v in row] for row in matrix]
    coefficients = [Fraction(1), -(a + d), a*d - b*c]
    encoded = result["char_poly_exact"]
    assert isinstance(encoded, list) and all(isinstance(s, str) for s in encoded)
    assert [Fraction(s) for s in encoded] == coefficients
    assert result["char_poly_float_status"] == poly_status
    if poly_status == "finite":
        assert result["char_poly"] == [float(v) for v in coefficients]
    else:
        assert result["char_poly"] is None
    assert result["cayley_hamilton_residual_exact"] == "0"
    assert result["cayley_hamilton_residual"] == 0.0
    assert result["cayley_hamilton_residual_basis"] == "char_poly_exact"
    assert type(result["zero_eigenvalue_multiplicity"]) is int
    assert result["zero_eigenvalue_multiplicity"] == 0

    if poles is None:
        assert result["eigenvalues_real_imag"] is None
        assert result["eigenvalues_status"] == "outside_float_range"
    else:
        assert result["eigenvalues_status"] == "finite_approximation"
        actual = result["eigenvalues_real_imag"]
        assert len(actual) == 2 and all(len(z) == 2 for z in actual)
        # Accept either pole order. Each nonzero component gets its own scale,
        # so the large real pole cannot hide the loss of its slow companion.
        def matches(got, wanted):
            return all(math.isclose(x, y, rel_tol=1e-12, abs_tol=0.0) for x, y in zip(got, wanted))
        assert any(all(matches(actual[i], order[i]) for i in range(2))
                   for order in (poles, poles[::-1])), actual

    for field in ("hurwitz_stable", "lyapunov_pd"):
        for claim in (False, True):
            claimed = routed[f"{case}-{field}-{claim}"]
            assert claimed["completed"], claimed
            details = claimed["details"]
            assert details[field] is True and details["passed"] is claim
            check = details["claim_check"]
            assert check["field"] == field and check["claimed"] == float(claim)
            assert check["actual"] == 1.0 and check["within_tolerance"] is claim


def check_pi_transfer_certificate(cert, coefficients, counts):
    assert cert["method"] == "infinitesimal_shift_hurwitz"
    encoded = cert["normalized_coefficients"]
    assert isinstance(encoded, list) and all(type(value) is str for value in encoded)
    assert [Fraction(value) for value in encoded] == [Fraction(c) for c in coefficients]
    keys = ("rhp_poles", "jw_axis_poles", "lhp_poles")
    if counts is None:
        assert cert["status"] == "identically_zero" and cert["minor_leading_terms"] == []
        assert isinstance(cert.get("reason"), str) and cert["reason"].strip()
        assert all(cert[key] is None for key in (*keys, "degree", "hurwitz_stable",
                                                "first_column_signs_plus", "first_column_signs_minus"))
        return
    degree = len(coefficients) - 1
    assert cert["status"] == "classified" and type(cert["degree"]) is int and cert["degree"] == degree
    assert all(type(cert[key]) is int for key in keys)
    assert tuple(cert[key] for key in keys) == counts
    assert cert["hurwitz_stable"] is (counts[:2] == (0, 0))
    # Reconstruct the first column from the recorded least nonzero determinant
    # coefficients. Each first-column sign is the sign of consecutive minors' ratio.
    terms = cert["minor_leading_terms"]
    assert len(terms) == degree
    for direction, key, variations in ((1, "first_column_signs_plus", counts[0]),
                                       (-1, "first_column_signs_minus", counts[0] + counts[1])):
        signs, previous = [1], 1
        for term in terms:
            power = term["power"]
            assert type(power) is int and power >= 0 and type(term["coefficient"]) is str
            value = Fraction(term["coefficient"])
            assert value != 0
            current = (1 if value > 0 else -1) * direction**power
            signs.append(previous * current)
            previous = current
        assert cert[key] == signs
        assert sum(x != y for x, y in zip(signs, signs[1:])) == variations


@pytest.mark.parametrize("case,den,num,open_counts,closed_coeffs,closed_counts", TRANSFER_CASES,
                         ids=[row[0] for row in TRANSFER_CASES])
def test_registered_transfer_hurwitz(routed, case, den, num, open_counts, closed_coeffs, closed_counts):
    result = routed[case]
    assert result["completed"], result
    details = result["details"]
    assert details["domain"] == "control" and details["mode"] == "transfer_function"
    assert details["passed"] is (closed_counts is not None)
    check_pi_transfer_certificate(details["open_loop_root_certificate"], den, open_counts)
    check_pi_transfer_certificate(details["closed_loop_root_certificate"], closed_coeffs, closed_counts)
    assert (details["rhp_poles_routh"], details["jw_axis_poles"]) == open_counts[:2]
    assert details["rhp_poles_eigen_is_approximate"] is True
    for status, poles in (("poles_status", "poles_real_imag"),
                          ("closed_loop_poles_status", "closed_loop_poles_real_imag")):
        if status == "closed_loop_poles_status" and closed_counts is None:
            continue
        if details[status] == "finite_approximation":
            assert isinstance(details[poles], list) and all(
                len(z) == 2 and all(math.isfinite(x) for x in z) for z in details[poles])
        else:
            assert details[status] == "unavailable_numerical_diagnostic" and details[poles] is None
    assert details["hurwitz_stable"] is (open_counts[:2] == (0, 0))
    assert details["closed_loop_hurwitz_stable"] is (
        None if closed_counts is None else closed_counts[:2] == (0, 0)
    )
    if closed_counts is None:
        assert isinstance(details.get("reason"), str) and details["reason"].strip()
        assert details["closed_loop_poles_real_imag"] is None
        assert details["closed_loop_poles_status"] == "undefined_characteristic"

    for field in ("hurwitz_stable", "closed_loop_hurwitz_stable"):
        for claim in (False, True):
            reply = routed[f"{case}-{field}-{claim}"]
            assert reply["completed"], reply
            claimed = reply["details"]
            assert claimed[field] is details[field]
            assert claimed["passed"] is (closed_counts is not None and claim is details[field])
            if closed_counts is None:
                assert isinstance(claimed.get("reason"), str) and claimed["reason"].strip()
            else:
                check = claimed["claim_check"]
                assert check["field"] == field and check["claimed"] == float(claim)
                assert check["actual"] == float(details[field]) and check["within_tolerance"] is (claim is details[field])


def check_pi_frequency(metadata, status, expected_x):
    assert metadata["status"] == status
    if expected_x is None:
        assert status == "unbounded" and metadata["squared_interval"] is None
        return
    bounds = metadata["squared_interval"]
    assert isinstance(bounds, list) and len(bounds) == 2 and all(type(x) is str for x in bounds)
    lo, hi = (Fraction(x) for x in bounds)
    if status == "zero":
        assert lo == hi == expected_x == 0
    else:
        assert 0 < lo <= expected_x <= hi and (hi - lo) * 2**56 <= hi


@pytest.mark.parametrize("case,num,den,pm", BODE_RANGE_CASES, ids=[row[0] for row in BODE_RANGE_CASES])
def test_registered_bode_frequency_range(routed, case, num, den, pm):
    reply = routed["bode-" + case]
    assert reply["completed"], reply
    details = reply["details"]
    assert details["mode"] == "transfer_function" and details["domain"] == "control"
    assert details["passed"] is (case != "negative-unity-hole")
    evidence = details["bode_frequency_evidence"]
    assert evidence["coordinate"] == "omega_squared_rad2_per_s2"
    assert evidence["rendering_basis"] == "isolating_interval_midpoint"
    gain_index = {"high-gain": 0, "tiny-gain": 0, "high-phase": 0, "high-pole": 1,
                  "negative-unity-hole": None, "signed-zero-wrap": 0}[case]
    phase_index = 0 if case == "high-phase" else None
    assert evidence["selected_gain_index"] == gain_index and evidence["selected_phase_index"] == phase_index
    for group in ("gain_crossovers", "phase_crossovers", "unity_gain_intervals_rad_s",
                  "negative_real_intervals_rad_s", "imaginary_axis_poles_rad_s", "removable_holes_rad_s"):
        assert len(details[group]) == len(evidence[group])

    # Squared frequencies here are derived from the decimal wire coefficients.
    pole_x = 10**628
    gain_xs = {"high-gain": [10**680], "tiny-gain": [Fraction(1, 10**680)], "high-phase": [1],
               "high-pole": [Fraction(pole_x, 2), Fraction(3 * pole_x, 2)],
               "negative-unity-hole": [], "signed-zero-wrap": [1]}[case]
    status = {"high-gain": "above_float_range", "tiny-gain": "below_float_range",
              "high-phase": "finite", "high-pole": "above_float_range", "signed-zero-wrap": "finite"}.get(case)
    assert len(evidence["gain_crossovers"]) == len(gain_xs)
    for i, x in enumerate(gain_xs):
        check_pi_frequency(evidence["gain_crossovers"][i], status, x)
        assert details["gain_crossovers"][i]["w_rad_s"] == (1.0 if case in ("high-phase", "signed-zero-wrap") else None)
    assert details["w_gc_rad_s"] == (1.0 if case in ("high-phase", "signed-zero-wrap") else None)
    assert details["w_pc_rad_s"] is None
    assert len(evidence["phase_crossovers"]) == int(case == "high-phase")
    if pm is not None:
        assert math.isclose(details["phase_margin_deg"], pm, rel_tol=1e-12, abs_tol=0)
        assert details["phase_margin_deg"] == details["gain_crossovers"][gain_index]["phase_margin_deg"]
    else:
        assert details["phase_margin_deg"] is None
    if case == "high-phase":
        check_pi_frequency(evidence["phase_crossovers"][0], "above_float_range", 10**680)
        assert details["phase_crossovers"][0]["w_rad_s"] is None
        assert math.isclose(details["gain_margin_db"], -6800, rel_tol=1e-12)
    else:
        assert details["gain_margin_db"] is None

    if case == "signed-zero-wrap":
        # At w=1 the exact imaginary part of (B-iA)/(B+iA) is negative.
        # Its angle rounds to -0.0, so the wrapped margin must round to +180.
        assert details["dc_gain"] == 0.0
        assert details["phase_margin_deg"] == 180.0
        assert details["phase_crossovers"] == [] and details["negative_real_intervals_rad_s"] == []
    if case == "high-pole":
        assert [v["phase_margin_deg"] for v in details["gain_crossovers"]] == [-180, 0]
        assert details["imaginary_axis_poles_rad_s"] == [None] and details["removable_holes_rad_s"] == []
        check_pi_frequency(evidence["imaginary_axis_poles_rad_s"][0], "above_float_range", pole_x)
        assert details["negative_real_intervals_rad_s"] == [[None, None]]
        left, right = evidence["negative_real_intervals_rad_s"][0]
        check_pi_frequency(left, "above_float_range", pole_x)
        check_pi_frequency(right, "unbounded", None)
    if case == "negative-unity-hole":
        assert details["closed_loop_hurwitz_stable"] is None
        assert details["removable_holes_rad_s"] == [None] and details["imaginary_axis_poles_rad_s"] == []
        check_pi_frequency(evidence["removable_holes_rad_s"][0], "above_float_range", pole_x)
        for key in ("unity_gain_intervals_rad_s", "negative_real_intervals_rad_s"):
            assert details[key] == [[0.0, None], [None, None]]
            first, second = evidence[key]
            check_pi_frequency(first[0], "zero", 0)
            check_pi_frequency(first[1], "above_float_range", pole_x)
            check_pi_frequency(second[0], "above_float_range", pole_x)
            check_pi_frequency(second[1], "unbounded", None)

    for field in ("w_gc_rad_s", "w_pc_rad_s"):
        claimed = routed[f"bode-{case}-{field}"]
        assert claimed["completed"] and claimed["details"]["passed"] is False, claimed
        assert claimed["details"]["reason"]
    for field, truth in (("phase_margin_deg", pm), ("gain_margin_db", -6800 if case == "high-phase" else None)):
        if truth is not None:
            claimed = routed[f"bode-{case}-{field}"]
            assert claimed["completed"] and claimed["details"]["passed"] is True, claimed
            check = claimed["details"]["claim_check"]
            assert check["field"] == field and check["claimed"] == truth and check["within_tolerance"] is True


@pytest.mark.parametrize("case,num,den,exact,rendered,status", DC_RANGE_CASES,
                         ids=[row[0] for row in DC_RANGE_CASES])
def test_registered_dc_gain_range(routed, case, num, den, exact, rendered, status):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "control" and result["mode"] == "transfer_function"
    assert result["passed"] is True and result["dc_gain_status"] == status
    assert result["dc_gain"] == rendered
    if exact is None:
        assert result["dc_gain_exact"] is None
    else:
        assert isinstance(result["dc_gain_exact"], str) and Fraction(result["dc_gain_exact"]) == exact
    if rendered is not None:
        assert 0 < result["dc_gain"] == 2 * DC_UNIT and Fraction(result["dc_gain"]) != exact

    opened = result["open_loop_root_certificate"]
    closed = result["closed_loop_root_certificate"]
    assert opened["status"] == closed["status"] == "classified"
    assert result["hurwitz_stable"] is opened["hurwitz_stable"]
    assert result["closed_loop_hurwitz_stable"] is closed["hurwitz_stable"]
    evidence = result["bode_frequency_evidence"]
    assert evidence["coordinate"] == "omega_squared_rad2_per_s2"
    for key in ("gain_crossovers", "phase_crossovers", "imaginary_axis_poles_rad_s",
                "removable_holes_rad_s", "unity_gain_intervals_rad_s", "negative_real_intervals_rad_s"):
        assert isinstance(result[key], list) and len(result[key]) == len(evidence[key])

    for label in ("zero", "evidence"):
        checked = routed[f"{case}-{label}"]
        assert checked["completed"], checked
        details = checked["details"]
        assert details["passed"] is False and details["dc_gain_exact"] == result["dc_gain_exact"]
        assert isinstance(details.get("reason"), str) and details["reason"].strip()
        if label == "evidence":
            assert "dc_gain_exact" in details["reason"] and "evidence" in details["reason"].lower()
        elif rendered is not None:
            check = details["claim_check"]
            assert check["field"] == "dc_gain" and check["claimed"] == 0
            assert check["actual"] == rendered and check["within_tolerance"] is False
    if rendered is not None:
        checked = routed[f"{case}-rendered"]
        assert checked["completed"] and checked["details"]["passed"] is True, checked
        check = checked["details"]["claim_check"]
        assert check["field"] == "dc_gain" and check["actual"] == check["claimed"] == rendered
        assert check["within_tolerance"] is True


def check_pi_error_quantity(check, name, expected):
    evidence = check[name + "_exact"]
    assert type(evidence) is str and evidence == str(expected)
    try:
        value = float(expected)
    except OverflowError:
        value, status = None, "above_float_range"
    else:
        if expected and value == 0:
            value, status = None, "below_float_range"
        else:
            status = "finite"
    assert check[name + "_float_status"] == status
    assert check[name] == value


@pytest.mark.parametrize("case,system,claim,actual,passed", CLAIM_TOLERANCE_CASES,
                         ids=[row[0] for row in CLAIM_TOLERANCE_CASES])
def test_registered_scalar_claim_tolerance(routed, case, system, claim, actual, passed):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "control" and result["mode"] == system["mode"]
    assert result["passed"] is passed, result
    if passed is False:
        assert isinstance(result.get("reason"), str) and result["reason"].strip()
    if actual is None:
        assert "claim_check" not in result
        return
    field = claim["claimed_field"]
    check = result["claim_check"]
    assert result[field] == actual and check["actual"] == float(actual)
    if type(actual) is bool:
        assert result[field] is actual
    assert check["field"] == field and check["claimed"] == float(claim["claimed_value"])
    # Compare the two accepted binary64 inputs without rounding their difference.
    a = Fraction(float(actual))
    c = Fraction(float(claim["claimed_value"]))
    e = abs(a - c)
    atol = Fraction(float(claim["atol"]))
    rtol = Fraction(float(claim["rtol"]))
    limit = max(atol, rtol * max(abs(a), Fraction(1e-12)))
    check_pi_error_quantity(check, "abs_error", e)
    check_pi_error_quantity(check, "tolerance_limit", limit)
    assert check["within_tolerance"] is (e <= limit)
    if case == "claim-already-failed":
        assert check["within_tolerance"] is True and result["closed_loop_hurwitz_stable"] is None


@pytest.mark.parametrize("case,system,q,q_max,eta,epsilon", FIN_NORMALIZATION_CASES,
                         ids=[row[0] for row in FIN_NORMALIZATION_CASES])
def test_registered_passive_fin_normalization(routed, case, system, q, q_max, eta, epsilon):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["passed"] is True and result["domain"] == "pde"
    assert result["mode"] == system["mode"]
    assert result["q_fin_w"] == q and result["q_max_ref_w"] == q_max
    assert (result["fin_efficiency"], result["fin_effectiveness"]) == (eta, epsilon)
    status = "undefined_zero_base_drive" if eta is None else "finite"
    assert result["fin_normalization"] == {
        "method": "passive_unit_drive", "efficiency_reference": "lateral_h_P_L",
        "effectiveness_reference": "bare_base_h_A", "length_reference": "provided_length_m",
        "zero_drive_policy": "undefined_heat_rate_ratio",
        "efficiency_float_status": status, "effectiveness_float_status": status,
    }
    fields = ["fin_efficiency", "fin_effectiveness"]
    if case == "fin-positive-tiny":
        fields.append("fin_normalization")
    for field in fields:
        claimed = routed[f"{case}-{field}"]
        assert claimed["completed"], claimed
        checked = claimed["details"]
        if field == "fin_normalization" or eta is None:
            assert checked["passed"] is False and "claim_check" not in checked
            assert isinstance(checked.get("reason"), str) and checked["reason"].strip()
        else:
            assert checked["passed"] is True
            comparison = checked["claim_check"]
            assert comparison["field"] == field
            assert comparison["actual"] == comparison["claimed"] == result[field]
            assert comparison["within_tolerance"] is True


@pytest.mark.parametrize("case,system,passed", WET_BULB_CASES, ids=[row[0] for row in WET_BULB_CASES])
def test_registered_wet_bulb_feasibility(routed, case, system, passed):
    reply = routed[case]
    assert reply["completed"], reply
    state = reply["details"]
    assert state["domain"] == "psychrometrics" and state["mode"] == system["mode"]
    assert state["passed"] is passed, state
    assert "wet_bulb_dry_bracket_c" not in state
    if not passed:
        reason = state.get("reason")
        assert isinstance(reason, str) and reason.strip()
        assert "claim_check" not in state
        assert {"w_kg_per_kgda", "w_mix_kg_per_kgda", "q_total_kw"}.isdisjoint(state)
        if case == "wet-clipped-zero-claim":
            assert reason == routed["wet-liquid-infeasible"]["details"]["reason"]
        return

    assert state["t_db_c"] == system["t_db_c"]
    if case == "wet-chart-state":
        # ASHRAE Fundamentals (2025 SI), ch. 1, Example 1 reads rounded values
        # from chart 1: W=6.5 g/kg dry air and RH=14% at 101.325 kPa.
        assert state["w_kg_per_kgda"] == pytest.approx(0.0065, rel=0, abs=0.0002)
        assert state["rh"] == pytest.approx(0.14, rel=0, abs=0.01)
        assert state["t_wb_c"] == pytest.approx(20, rel=0, abs=1e-8)
    elif case == "wet-dry-control":
        assert state["w_kg_per_kgda"] == state["pw_pa"] == state["rh"] == 0
        assert 0 < state["t_wb_c"] < state["t_db_c"]
    else:
        assert state["rh"] == 1 and state["w_kg_per_kgda"] > 0
        assert state["pw_pa"] == state["pws_pa"]
        assert state["t_wb_c"] == pytest.approx(state["t_db_c"], rel=0, abs=1e-8)


@pytest.mark.parametrize("case,inputs,claim,status,passed", DEW_POINT_CASES,
                         ids=[row[0] for row in DEW_POINT_CASES])
def test_registered_dew_point_domain(routed, case, inputs, claim, status, passed):
    reply = routed[case]
    assert reply["completed"], reply
    state = reply["details"]
    assert state["domain"] == "psychrometrics" and state["mode"] == "state"
    assert state["passed"] is passed, state
    assert state["t_db_c"] == 20
    if status is None:
        assert "t_dp_status" not in state
        assert state["pw_pa"] > 0 and state["w_kg_per_kgda"] > 0
        if "t_dp_c" in inputs:
            assert state["t_dp_c"] == pytest.approx(-90, rel=0, abs=1e-8)
        else:
            assert state["rh"] == 0.5 and 9 < state["t_dp_c"] < 10
    else:
        assert state["t_dp_c"] is None and state["t_dp_status"] == status
        if status == "dry_no_finite_point":
            assert state["pw_pa"] == state["w_kg_per_kgda"] == state["rh"] == 0
        else:
            assert state["w_kg_per_kgda"] == 1e-16 and 0 < state["pw_pa"] < 2e-11
    if not passed:
        assert isinstance(state.get("reason"), str) and state["reason"].strip()
    if claim is None or status is not None:
        assert "claim_check" not in state
    else:
        comparison = state["claim_check"]
        assert comparison["field"] == "t_dp_c" and comparison["claimed"] == claim[0]
        assert comparison["actual"] == state["t_dp_c"] and comparison["within_tolerance"] is passed


@pytest.mark.parametrize("case,direction,claim,passed", TRUSS_CASES, ids=[row[0] for row in TRUSS_CASES])
def test_registered_truss_factor_loss(routed, case, direction, claim, passed):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "fem" and result["mode"] == "truss_2d"
    assert result["passed"] is passed, result
    if not passed:
        assert isinstance(result.get("reason"), str) and result["reason"].strip()
    if direction == "mechanism":
        reason = result["reason"].lower()
        assert "mechanism" in reason or "singular" in reason
        assert "displacements_xy_m" not in result and "claim_check" not in result
        return

    # With orthogonal length-five members the radial stiffnesses are E_i/5.
    # A pure load extends only the loaded member. Energy equals half F dot u.
    scale = TRUSS_LOAD_SCALE
    if direction == "strong":
        displacement = [15*scale, 20*scale]
        forces = [5*scale, 0]
        energy = 125 * scale**2 / 2
    else:
        displacement = [20*scale, -15*scale]
        forces = [0, 5*scale*TRUSS_WEAK_MODULUS]
        energy = TRUSS_WEAK_ENERGY
    motion = result["displacements_xy_m"]
    assert len(motion) == 3 and all(len(node) == 2 for node in motion)
    assert len(result["axial_forces_N"]) == 2
    quantities = list(zip(motion[0] + motion[1] + motion[2], [0, 0, 0, 0, *displacement]))
    quantities.extend(zip(result["axial_forces_N"], forces))
    quantities.extend([(result["strain_energy_J"], energy), (result["external_work_J"], energy),
                       (result["equilibrium_residual_N"], 0)])
    for actual, expected in quantities:
        assert type(actual) in (int, float) and math.isfinite(actual)
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0)
    if claim is None:
        assert "claim_check" not in result
    else:
        comparison = result["claim_check"]
        assert comparison["field"] == "strain_energy_J"
        assert comparison["claimed"] == claim
        assert comparison["actual"] == result["strain_energy_J"] == TRUSS_WEAK_ENERGY
        assert comparison["within_tolerance"] is passed


@pytest.mark.parametrize("case,harmonic,position,time,claim,passed", HEAT_PHASE_CASES,
                         ids=[row[0] for row in HEAT_PHASE_CASES])
def test_registered_heat_modal_phase(routed, case, harmonic, position, time, claim, passed):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "pde" and result["mode"] == "heat_1d_transient"
    assert result["passed"] is passed, result
    assert result["error_measure"] == "whole-domain upper bound"
    for key in ("u_exact", "u_crank_nicolson", "query_abs_error", "linf_error", "spatial_cells", "time_steps"):
        assert type(result[key]) in (int, float) and math.isfinite(result[key])
    assert 0 <= result["linf_error"] <= 0.003
    exact = result["u_exact"]
    numeric = result["u_crank_nicolson"]
    if harmonic == HEAT_LARGE_MODE:
        # N = 6 (mod 8), hence 3N/4 = 1/2 (mod 2). The refined grids have
        # a node at 3/4. Both decays equal one at time zero. At x=1 the temperature is fixed at zero.
        expected = 1 if position == 0.75 else 0
        assert exact == numeric == expected
        assert result["time_steps"] == result["query_abs_error"] == 0
    else:
        # For the unit rod, the fundamental mode at its midpoint decays by exp(-pi^2*t).
        reference = math.exp(-math.pi**2 * time)
        assert math.isclose(exact, reference, rel_tol=2e-14, abs_tol=0)
        assert 0 < numeric < 1 and result["time_steps"] > 0
        assert abs(numeric - reference) <= result["linf_error"] + 2e-14 * abs(reference)
        assert result["query_abs_error"] == abs(numeric - exact)
    if claim is None:
        assert "claim_check" not in result
    else:
        field, value = claim
        comparison = result["claim_check"]
        assert comparison["field"] == field and comparison["claimed"] == value
        assert comparison["actual"] == result[field] == 1
        assert comparison["within_tolerance"] is passed
        if not passed:
            assert isinstance(result.get("reason"), str) and result["reason"].strip()


@pytest.mark.parametrize("case,system,claim,passed", HEAT_TIME_CASES, ids=[row[0] for row in HEAT_TIME_CASES])
def test_registered_heat_modal_time(routed, case, system, claim, passed):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "pde" and result["mode"] == "heat_1d_transient"
    assert result["passed"] is passed, result
    assert result["error_measure"] == "whole-domain upper bound", result
    for field in ("u_exact", "u_crank_nicolson", "query_abs_error", "linf_error"):
        assert type(result[field]) in (int, float) and math.isfinite(result[field]), result

    # All three positions are unit-peak nodes. The tabulated scales S and
    # mesh/step counts come from the frozen dimensionless calculation.
    if system == "underflow":
        scale, cells_per_mode, steps = 1/16, 32, 1920
        cells = 32 * HEAT_TIME_HARMONIC
        continuum = HEAT_TIME_UNDERFLOW_TARGET
    else:
        scale, cells_per_mode, steps = 4, 120, 120
        cells = 120
        continuum = HEAT_TIME_OVERFLOW_TARGET
    assert result["spatial_cells"] == cells and result["time_steps"] == steps, result
    rate = 4 * scale * cells_per_mode**2 * math.sin(math.pi / (2 * cells_per_mode))**2
    half_step = rate / (2 * steps)
    discrete = ((1 - half_step) / (1 + half_step))**steps
    actual, numeric = result["u_exact"], result["u_crank_nicolson"]
    assert 0 < actual < 1 and math.isclose(actual, continuum, rel_tol=2e-13, abs_tol=0), result
    assert 0 < numeric < 1 and math.isclose(numeric, discrete, rel_tol=2e-12, abs_tol=0), result
    assert result["query_abs_error"] == abs(actual - numeric), result
    assert 0 <= result["query_abs_error"] <= result["linf_error"] <= 0.003, result
    if case == "heat-time-overflow":
        twin = routed["heat-time-ordinary-twin"]
        assert twin["completed"], twin
        other = twin["details"]
        assert other["passed"] is True, other
        assert math.isclose(actual, other["u_exact"], rel_tol=2e-13, abs_tol=0), other
        assert math.isclose(numeric, other["u_crank_nicolson"], rel_tol=2e-12, abs_tol=0), other
        assert (other["spatial_cells"], other["time_steps"]) == (cells, steps), other
    if claim is None:
        assert "claim_check" not in result
    else:
        comparison = result["claim_check"]
        assert comparison["field"] == "u_exact" and comparison["claimed"] == claim[0]
        assert comparison["actual"] == actual and comparison["within_tolerance"] is passed
        if not passed:
            assert isinstance(result.get("reason"), str) and result["reason"].strip()


@pytest.mark.parametrize("case,system,claim,passed", HEAT_AMPLITUDE_CASES,
                         ids=[row[0] for row in HEAT_AMPLITUDE_CASES])
def test_registered_heat_modal_amplitude(routed, case, system, claim, passed):
    reply = routed[case]
    assert reply["completed"], reply
    result = reply["details"]
    assert result["domain"] == "pde" and result["mode"] == "heat_1d_transient", result
    assert result["passed"] is passed, result
    assert result["error_measure"] == "whole-domain upper bound", result
    for field in ("u_exact", "u_crank_nicolson", "query_abs_error", "linf_error"):
        assert type(result[field]) in (int, float) and math.isfinite(result[field]), result
    assert type(result["spatial_cells"]) is int and result["spatial_cells"] == 120, result
    assert type(result["time_steps"]) is int and result["time_steps"] == 0, result

    exact, grid = result["u_exact"], result["u_crank_nicolson"]
    query_error = result["query_abs_error"]
    if system == "boundary":
        assert exact == grid == query_error == 0, result
    else:
        # Both interior inputs have A*x/L = 2^-100. The cubic sine
        # correction is below roundoff; the first-cell interpolant uses sin(pi/120).
        assert exact > 0 and math.isclose(exact, HEAT_AMPLITUDE_CONTINUUM, rel_tol=2e-13, abs_tol=0), result
        assert grid > 0 and math.isclose(grid, HEAT_AMPLITUDE_GRID, rel_tol=2e-13, abs_tol=0), result
        assert query_error == abs(exact - grid) and query_error > 0, result
    amplitude = HEAT_AMPLITUDE_INPUTS[system][2]
    assert 0 <= query_error <= result["linf_error"] <= 0.003 * amplitude, result
    if claim is None:
        assert "claim_check" not in result
    else:
        field, value, atol = claim
        comparison = result["claim_check"]
        assert comparison["field"] == field and comparison["claimed"] == value
        assert comparison["actual"] == result[field] and comparison["within_tolerance"] is passed
        assert comparison["tolerance_limit"] == atol and comparison["abs_error"] == abs(result[field] - value)
        if not passed:
            assert isinstance(result.get("reason"), str) and result["reason"].strip()



def test_registered_cstr_preserves_all_branches_and_refuses_implicit_choice(routed):
    complete = routed["cstr-all-branches"]
    assert complete["completed"], complete
    states = complete["details"]["steady_states"]
    assert len(states) == 3 and complete["details"]["verified"]
    assert abs(states[0]["conversion_X"] - .01342170533580371) < 2e-14
    assert abs(states[1]["conversion_X"] - .14297312550405095) < 2e-14
    assert abs(states[2]["conversion_X"] - .99973756033601) < 2e-14
    selected = routed["cstr-cold-branch"]
    assert selected["completed"] and selected["details"]["conversion_X"] == states[0]["conversion_X"]
    refused = routed["cstr-implicit-branch"]
    assert not refused["completed"]
    assert "multiple physical CSTR steady states" in refused["error"]


class RegisteredEngineeringVerifiersUnittest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "node_modules" / "@earendil-works" / "pi-coding-agent").exists():
            raise unittest.SkipTest("node_modules/@earendil-works/pi-coding-agent not installed")
        cls._tmpdir = tempfile.TemporaryDirectory()
        class _Factory:
            @staticmethod
            def mktemp(name):
                p = Path(cls._tmpdir.name) / name
                p.mkdir(parents=True, exist_ok=True)
                return p
        cls.routed_data = routed.__wrapped__(_Factory()) if hasattr(routed, "__wrapped__") else routed(_Factory())

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_cstr_branch_boundary(self):
        test_registered_cstr_preserves_all_branches_and_refuses_implicit_choice(self.routed_data)

    def test_all_registered_solver_boundaries(self):
        r = self.routed_data
        for family, spec in CASES:
            test_registered_tool_reaches_each_family_without_changing_calculation(r, family, spec)
            for prefix in ("empty-", "selector-only-", "no-selector-"):
                test_registered_tool_refuses_unspecified_calculations(r, family, prefix)
            test_unknown_keys_cannot_verify_an_unsupplied_physical_problem(r, family)
        for family, spec in NEGATIVE_PHYSICS:
            test_completed_transport_preserves_failed_physics(r, family, spec)
        test_legacy_checks_and_explicit_family_contract(r)
        test_default_capable_solvers_distinguish_unknown_keys_from_explicit_zero(r)

    def test_registered_algebraic_control_ranks(self):
        for case, matrices, rank in CONTROL_RANK_CASES:
            with self.subTest(case=case):
                test_registered_algebraic_control_ranks(self.routed_data, case, matrices, rank)

    def test_registered_state_space_stability(self):
        for case, matrices, stable, expected_p in STABILITY_CASES:
            with self.subTest(case=case):
                test_registered_state_space_stability(self.routed_data, case, matrices, stable, expected_p)

    def test_registered_state_space_diagnostic_range(self):
        for case, matrix, poly_status, poles in DIAGNOSTIC_CASES:
            with self.subTest(case=case):
                test_registered_state_space_diagnostic_range(self.routed_data, case, matrix, poly_status, poles)

    def test_registered_transfer_hurwitz(self):
        for args in TRANSFER_CASES:
            with self.subTest(case=args[0]):
                test_registered_transfer_hurwitz(self.routed_data, *args)

    def test_registered_bode_frequency_range(self):
        for args in BODE_RANGE_CASES:
            with self.subTest(case=args[0]):
                test_registered_bode_frequency_range(self.routed_data, *args)

    def test_registered_refrigeration_capacity(self):
        specific, measured, contradictory = (
            self.routed_data["refrigerator-" + name] for name in
            ("specific", "measured", "contradictory")
        )
        self.assertTrue(all(call["completed"] for call in (specific, measured, contradictory)))
        specific, measured, contradictory = (call["details"] for call in
                                             (specific, measured, contradictory))
        self.assertTrue(specific["verified"])
        self.assertEqual(specific["COP_R"], 3.75)
        self.assertIsNone(specific["Q_evap_W"])
        self.assertTrue(measured["verified"])
        self.assertEqual((measured["m_dot_kg_s"], measured["W_comp_W"], measured["Q_cond_W"]),
                         (.5, 20000., 95000.))
        self.assertFalse(contradictory["verified"])

    def test_registered_dc_gain_range(self):
        for args in DC_RANGE_CASES:
            with self.subTest(case=args[0]):
                test_registered_dc_gain_range(self.routed_data, *args)

    def test_registered_scalar_claim_tolerance(self):
        for args in CLAIM_TOLERANCE_CASES:
            with self.subTest(case=args[0]):
                test_registered_scalar_claim_tolerance(self.routed_data, *args)

    def test_registered_passive_fin_normalization(self):
        for args in FIN_NORMALIZATION_CASES:
            with self.subTest(case=args[0]):
                test_registered_passive_fin_normalization(self.routed_data, *args)

    def test_registered_wet_bulb_feasibility(self):
        for args in WET_BULB_CASES:
            with self.subTest(case=args[0]):
                test_registered_wet_bulb_feasibility(self.routed_data, *args)

    def test_registered_dew_point_domain(self):
        for args in DEW_POINT_CASES:
            with self.subTest(case=args[0]):
                test_registered_dew_point_domain(self.routed_data, *args)

    def test_registered_truss_factor_loss(self):
        for args in TRUSS_CASES:
            with self.subTest(case=args[0]):
                test_registered_truss_factor_loss(self.routed_data, *args)

    def test_registered_heat_modal_phase(self):
        for args in HEAT_PHASE_CASES:
            with self.subTest(case=args[0]):
                test_registered_heat_modal_phase(self.routed_data, *args)

    def test_registered_heat_modal_time(self):
        for args in HEAT_TIME_CASES:
            with self.subTest(case=args[0]):
                test_registered_heat_modal_time(self.routed_data, *args)

    def test_registered_heat_modal_amplitude(self):
        for args in HEAT_AMPLITUDE_CASES:
            with self.subTest(case=args[0]):
                test_registered_heat_modal_amplitude(self.routed_data, *args)

@pytest.mark.parametrize("name,q0,q1,rigidity", [
    ("beam-ramp-coarse", 0, -600, 1000), ("beam-ramp-fine", 0, -600, 1000),
    ("beam-reverse-small", -600, 0, 1e6), ("beam-trapezoid", -100, -600, 1000),
    ("beam-uniform-ends", -350, -350, 1000),
])
def test_beam_linear_load_against_section_equilibrium(routed, name, q0, q1, rigidity):
    outcome = routed[name]
    assert outcome["completed"], outcome
    got = outcome["details"]
    assert got["passed"], got
    length = 2.0
    slope = (q1 - q0) / length
    # Independent section equilibrium: M(x) = integral_x^L q(s) (s-x) ds.
    moment = [q0 * length ** 2 / 2 + slope * length ** 3 / 3,
              -q0 * length - slope * length ** 2 / 2, q0 / 2, slope / 6]
    def deflection(x):
        return sum(c * x ** (i + 2) / ((i + 1) * (i + 2))
                   for i, c in enumerate(moment)) / rigidity
    energy = sum(a * b * length ** (i + j + 1) / (i + j + 1)
                 for i, a in enumerate(moment) for j, b in enumerate(moment)) / (2 * rigidity)
    assert got["reactions_V_M"][0] == pytest.approx([moment[1], -moment[0]])
    assert got["tip_deflection_m"] == pytest.approx(deflection(length))
    assert got["strain_energy_J"] == pytest.approx(energy)
    assert got["external_work_J"] == pytest.approx(energy)
    nodes = BEAM_PROFILES[name]["x_nodes_m"]
    assert got["element_midspan_deflections_m"] == pytest.approx(
        [deflection((a + b) / 2) for a, b in zip(nodes, nodes[1:])])


def test_beam_antisymmetric_load_energy_and_profile_contract(routed):
    same = routed["beam-uniform-ends"]["details"]
    assert same == routed["beam-uniform"]["details"]
    got = routed["beam-antisymmetric"]["details"]
    assert got["passed"] and got["element_midspan_deflections_m"] == [0.0]
    # For q=600(2t-1), the fixed-end bubble is h*(2t-1)t²(1-t)².
    h = 600 * 2**4 / (120 * 1000)
    energy = 0.5 * 1000 / 2**3 * (4 / 7) * h**2
    assert got["strain_energy_J"] == pytest.approx(energy)
    assert got["external_work_J"] == pytest.approx(energy)
    for case in ("beam-incomplete", "beam-ambiguous", "beam-extra-element"):
        assert not routed[case]["details"]["passed"]


def test_registered_enriched_product_uses_whole_equilibrium_trays(routed):
    receipt = routed["enriched-product-two-tray"]
    assert receipt["completed"], receipt
    result = receipt["details"]
    top, bottom, alpha = Fraction(9, 10), Fraction(7, 10), Fraction(2)
    first = top / (alpha - (alpha - 1)*top)
    vapor = (1000*first + top)/1001
    second = vapor / (alpha - (alpha - 1)*vapor)
    assert first > Fraction(4, 5) and second < bottom < first
    graphical = 1 + (first-bottom)/(first-second)
    assert result["N_theoretical"] == float(graphical)
    assert result["N_theoretical"] < result["N_min_fenske"] < 2
    assert result["N_stages_integer"] == result["feed_stage"] == 2
    assert result["residual"] < 1e-10
    assert result["verified"], result


def test_registered_tensor_and_plane_traction(routed):
    row = routed["rotated-octahedral-plane"]
    assert row["completed"], row
    result = row["details"]
    assert result["verified"], result
    # Cauchy traction, evaluated directly from the full Cartesian tensor.
    v = [7, -1, 5]
    force = [480, 110, -150]
    normal = sum(x*y for x,y in zip(v,force)) / 75
    tangent = [x-normal*y for x,y in zip(force,v)]
    plane = result["plane_traction"]
    assert math.isclose(plane["normal_stress"], normal)
    assert math.isclose(plane["shear_magnitude"], math.sqrt(sum(x*x for x in tangent)/75))
    assert math.isclose(result["sigma_oct"], 100/3)
    assert math.isclose(plane["shear_magnitude"], result["tau_oct"])
    trace = sum((e-100/3)**2 for e in [72,58,-30])+2*24**2
    assert math.isclose(result["tau_oct"], math.sqrt(trace/3))
