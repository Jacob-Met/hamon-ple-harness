"""Exercise the real Pi registration -> TypeScript -> Python solver boundary.

These synthetic inputs qualify runtime reachability, not benchmark accuracy.
"""
import json
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
