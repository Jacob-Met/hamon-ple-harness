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
