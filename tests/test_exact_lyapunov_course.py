"""Analytic one- and two-state Lyapunov controls through the registered Python router."""
from dataclasses import dataclass
from fractions import Fraction as F
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def dyadic(exponent):
    return F(2) ** exponent


def diagonal(*values):
    return [[value if i == j else 0 for j in range(len(values))]
            for i, value in enumerate(values)]


@dataclass
class Case:
    name: str
    a: list
    q: object
    stable: bool
    status: str
    p: object


def cases():
    d = dyadic(-55)
    for a, p in ((-d, dyadic(54)), (0, None), (d, -dyadic(54)), (-dyadic(55), dyadic(-56))):
        yield Case(f"scalar-default/{a}", [[a]], None, a < 0,
                   "inconsistent" if p is None else "unique", None if p is None else [[p]])
    for eigenvalue in (-d, d, 0):
        p = diagonal(F(1, 2), -F(1, 2) / eigenvalue) if eigenvalue else None
        yield Case(f"mixed-default/{eigenvalue}", diagonal(-1, eigenvalue), None,
                   eigenvalue < 0, "unique" if eigenvalue else "inconsistent", p)
    for i, (a, q, p) in enumerate(((-d, dyadic(-100), dyadic(-46)), (-1, -1, -F(1, 2)),
                                   (1, -1, F(1, 2)), (-1, 0, 0))):
        yield Case(f"scalar-requested/{i}", [[a]], [[q]], a < 0, "unique", [[p]])
    for k in (-80, 80):
        q, p = dyadic(k), dyadic(k - 1)
        yield Case(f"forcing-scale/{k}", diagonal(-1, -1), diagonal(q, q), True,
                   "unique", diagonal(p, p))
    yield Case("semidefinite/uncoupled", diagonal(-1, -2), diagonal(1, 0), True,
               "unique", diagonal(F(1, 2), 0))
    yield Case("semidefinite/coupled", [[-1, 1], [0, -1]], diagonal(1, 0), True,
               "unique", [[F(1, 2), F(1, 4)], [F(1, 4), F(1, 4)]])
    yield Case("indefinite", diagonal(-1, -2), diagonal(1, -1), True,
               "unique", diagonal(F(1, 2), -F(1, 4)))
    yield Case("nonsymmetric", diagonal(-1, -1), [[1, 1], [0, 1]], True, "invalid", None)
    yield Case("zero-compatible", [[0]], [[0]], False, "nonunique", None)
    rotation = [[0, -1], [1, 0]]
    yield Case("rotation/identity", rotation, diagonal(1, 1), False, "inconsistent", None)
    yield Case("rotation/zero", rotation, diagonal(0, 0), False, "nonunique", None)
    yield Case("identity-nonunique", diagonal(-1, 1), diagonal(1, 1), False, "nonunique", None)
    for exponent in (-20, 0, 20):
        c = dyadic(exponent)
        for label, swapped, stretch in (("identity", False, 0), ("swap", True, 0),
                                         ("scaled-swap", True, 20)):
            # x' = Sx. The swapped map is [[0,2^-stretch],[2^stretch,0]].
            if swapped:
                q1, q2 = dyadic(2 * stretch), dyadic(-2 * stretch)
                a, q = diagonal(-c * d, -c), diagonal(q1, q2)
                p = diagonal(q1 * dyadic(54) / c, q2 / (2 * c))
            else:
                a, q = diagonal(-c, -c * d), diagonal(1, 1)
                p = diagonal(1 / (2 * c), dyadic(54) / c)
            yield Case(f"coordinate-time/{exponent}/{label}", a, q, True, "unique", p)
    yield Case("overflow-P", [[-dyadic(-1074)]], [[1]], True, "unique", [[dyadic(1073)]])
    yield Case("underflow-P", [[-dyadic(1023)]], [[dyadic(-1074)]], True, "unique",
               [[dyadic(-2098)]])


def strict_pd(p):
    if p is None:
        return None
    if len(p) == 1:
        return p[0][0] > 0
    return p[0][1] == p[1][0] and p[0][0] > 0 and p[0][0]*p[1][1] > p[0][1]*p[1][0]


def identity_solution(case):
    if case.name == "semidefinite/coupled":
        return "unique", [[F(1, 2), F(1, 4)], [F(1, 4), F(3, 4)]]
    if case.name == "identity-nonunique":
        return "nonunique", None
    if case.name.startswith("rotation/") or any(row[i] == 0 for i, row in enumerate(case.a)):
        return "inconsistent", None
    return "unique", diagonal(*[-F(1, 2) / row[i] for i, row in enumerate(case.a)])


def matrix_floats(matrix):
    return [[float(v) for v in row] for row in matrix]


def has_residual_zero(a, p, q):
    # Evaluate the equation on the wire values, never on computed eigenvalues.
    n = len(a)
    return all(sum(F(a[k][i]) * p[k][j] + p[i][k] * F(a[k][j]) for k in range(n))
               == -F(q[i][j]) for i in range(n) for j in range(n))


def certificate_matches(status, expected, result, fields, a, q):
    status_key, matrix_key, pd_key, residual_key = fields
    if (any(k not in result for k in fields) or result[status_key] != status
            or result[pd_key] is not strict_pd(expected)):
        return False
    raw = result[matrix_key]
    if expected is None:
        return raw is None and result[residual_key] is None
    n = len(expected)
    if (not isinstance(raw, list) or len(raw) != n
            or any(not isinstance(row, list) or len(row) != n for row in raw)
            or any(type(v) is not str for row in raw for v in row)):
        return False
    try:
        parsed = [[F(v) for v in row] for row in raw]
    except (ValueError, ZeroDivisionError):
        return False
    return (parsed == expected and strict_pd(parsed) is result[pd_key]
            and result[residual_key] is True and has_residual_zero(a, parsed, q))


def rendered_matches(case, result):
    if "lyapunov_P" not in result:
        return False
    rendered, state = result["lyapunov_P"], result.get("lyapunov_float_status")
    if case.p is None:
        return rendered is None and state == "not_unique_solution"
    if case.name in ("overflow-P", "underflow-P"):
        return rendered is None and state == "outside_float_range"
    n = len(case.p)
    if (state != "finite" or not isinstance(rendered, list) or len(rendered) != n
            or any(not isinstance(row, list) or len(row) != n for row in rendered)):
        return False
    for i, row in enumerate(rendered):
        for j, v in enumerate(row):
            reference = case.p[i][j]
            if type(v) is not float or not math.isfinite(v):
                return False
            if v == 0 and reference != 0:
                return False
            if abs(F(v) - reference) > abs(reference) * F(1, 10**12):
                return False
    return True


def call(spec, errors, label):
    try:
        result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:100]])
        return {}
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:100]])
        return {}
    return result


def check_case(case):
    spec = {"mode": "state_space", "A": matrix_floats(case.a)}
    n = len(case.a)
    identity = diagonal(*([1] * n))
    q_wire = identity
    if case.q is not None:
        spec["Q"] = q_wire = matrix_floats(case.q)
    errors = []
    result = call(spec, errors, "unclaimed")
    status = case.status
    invalid = status == "invalid"
    pd = strict_pd(case.p)
    detail = {"case": case.name, "passed": result.get("passed"), "hurwitz": result.get("hurwitz_stable"),
              "pd": result.get("lyapunov_pd"), "solution": result.get("lyapunov_solution_status"),
              "float": result.get("lyapunov_float_status"), "expected_status": status}
    if invalid:
        checks = {"explicit_refusal": (result.get("passed") is False and bool(result.get("reason"))
                                       and result.get("lyapunov_P") is None
                                       and result.get("lyapunov_P_exact") is None)}
    else:
        identity_status, identity_p = identity_solution(case)
        certificate = result.get("stability_certificate")
        checks = {
            "unclaimed_pass": result.get("passed") is True,
            "classification": result.get("hurwitz_stable") is case.stable,
            "requested_pd": "lyapunov_pd" in result and result["lyapunov_pd"] is pd,
            "requested_exact": certificate_matches(
                status, case.p, result,
                ("lyapunov_solution_status", "lyapunov_P_exact", "lyapunov_pd", "lyapunov_residual_exact"),
                spec["A"], q_wire),
            "identity_exact": isinstance(certificate, dict) and certificate_matches(
                identity_status, identity_p, certificate,
                ("solution_status", "P_exact", "positive_definite", "residual_exact"), spec["A"], identity),
            "rendered": rendered_matches(case, result),
        }
    detail["failed_checks"] = [key for key, ok in checks.items() if not ok]
    decisions, truthful_approvals, false_approvals, false_refusals, correct = [], 0, 0, 0, 0
    for field, truth in (("hurwitz_stable", case.stable), ("lyapunov_pd", pd)):
        for supplied in (False, True):
            expected_accept = not invalid and truth is supplied
            checked = call({**spec, "claimed_field": field, "claimed_value": supplied}, errors,
                           f"{field}/{supplied}")
            decision = checked.get("passed")
            decisions.append("A" if decision is True else "R" if decision is False else "E")
            false_approvals += int(decision is True and not expected_accept)
            false_refusals += int(decision is False and expected_accept)
            truthful_approvals += int(decision is True and expected_accept)
            if invalid or truth is None:
                evidence_ok = bool(checked.get("reason"))
            else:
                claim = checked.get("claim_check") or {}
                evidence_ok = (checked.get(field) is truth and claim.get("field") == field
                               and claim.get("actual") == float(truth)
                               and claim.get("claimed") == float(supplied)
                               and claim.get("within_tolerance") is expected_accept)
            correct += int(decision is expected_accept and evidence_ok)
    detail.update(decisions="".join(decisions), correct_claims=correct,
                  truthful_approvals=truthful_approvals, false_approvals=false_approvals,
                  false_refusals=false_refusals, errors=errors)
    return detail


class ExactLyapunovCourseTests(unittest.TestCase):
    def test_frozen_analytic_stability_course(self):
        rows = [check_case(case) for case in cases()]
        for row in rows:
            print("LYAPUNOV_CASE " + json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
        summary = {"systems": len(rows), "claims": sum(len(row["decisions"]) for row in rows),
                   "correct_unclaimed": sum(not row["failed_checks"] for row in rows),
                   "correct_claims": sum(row["correct_claims"] for row in rows),
                   "false_approvals": sum(row["false_approvals"] for row in rows),
                   "false_refusals": sum(row["false_refusals"] for row in rows),
                   "solver_errors": sum(err[1] == "solver" for row in rows for err in row["errors"]),
                   "serialization_errors": sum(err[1] == "json" for row in rows for err in row["errors"])}
        print("LYAPUNOV_TOTAL " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
        self.assertEqual(summary, {"systems": 32, "claims": 128, "correct_unclaimed": 32,
                                   "correct_claims": 128, "false_approvals": 0, "false_refusals": 0,
                                   "solver_errors": 0, "serialization_errors": 0})
