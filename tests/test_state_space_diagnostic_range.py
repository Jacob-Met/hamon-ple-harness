"""Lossless characteristic diagnostics and finite rendering on analytic state-space models."""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import permutations
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def diag(*values):
    return [[v if i == j else 0 for j in range(len(values))] for i, v in enumerate(values)]


@dataclass
class Case:
    name: str
    a: list
    polynomial: list
    roots: list
    p: object
    stable: bool


def diagonal_case(name, rates, polynomial):
    p = diag(*[-F(1, 2) / rate for rate in rates]) if all(rates) else None
    return Case(name, diag(*rates), polynomial, [(v, 0) for v in rates], p, all(v < 0 for v in rates))


def fixtures():
    m, h, tiny, t = (F(2) ** k for k in (512, 520, -550, 1023))
    yield diagonal_case("ordinary-stable", (-1, -2), [1, 3, 2])
    yield diagonal_case("ordinary-unstable", (-1, 2), [1, -1, -2])
    yield diagonal_case("mixed-fast-slow", (-m, -1), [1, m+1, m])
    yield diagonal_case("mixed-swap", (-1, -m), [1, m+1, m])
    yield diagonal_case("mixed-fast-unstable", (-m, 1), [1, m-1, -m])
    yield diagonal_case("repeated-fast", (-h, -h), [1, 2*h, h*h])
    yield diagonal_case("both-fast-unstable", (-h, 2*h), [1, -h, -2*h*h])
    yield Case("damped-rotation", [[-h, -h], [h, -h]], [1, 2*h, 2*h*h],
               [(-h, -h), (-h, h)], diag(1/(2*h), 1/(2*h)), True)
    yield Case("rotation", [[0, -h], [h, 0]], [1, 0, h*h], [(0, -h), (0, h)], None, False)
    yield diagonal_case("tiny-polynomial", (-tiny, -2*tiny), [1, 3*tiny, 2*tiny*tiny])
    yield Case("root-outside-float", [[-3*t/2, t], [t, -3*t/2]],
               [1, 3*t, 5*t*t/4], [(-t/2, 0), (-5*t/2, 0)],
               [[3/(5*t), 2/(5*t)], [2/(5*t), 3/(5*t)]], True)
    u = F(2) ** -52
    yield Case("determinant-cancellation", [[1, 1+u], [-(1-u), -1]], [1, 0, -u*u],
               [(-u, 0), (u, 0)], None, False)
    for k in (0, 520, -520):
        s = F(2) ** k
        yield diagonal_case(f"cubic-stable/{k}", (-s, -2*s, -4*s), [1, 7*s, 14*s*s, 8*s**3])
    yield diagonal_case("cubic-unstable", (-h, 2*h, 4*h), [1, -5*h, 2*h*h, 8*h**3])
    yield diagonal_case("cubic-zero", (-2, 0, 1), [1, 1, -2, 0])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def call(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:100]])
        return {}
    try:
        canonical(result)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:100]])
        return {}
    return result


def rational_strings(raw, length):
    if not isinstance(raw, list) or len(raw) != length or any(type(v) is not str for v in raw):
        return None
    try:
        return [F(v) for v in raw]
    except (ValueError, ZeroDivisionError):
        return None


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def float_coefficients(coefficients):
    try:
        converted = [float(v) for v in coefficients]
    except OverflowError:
        return None
    if any(not math.isfinite(value) or (exact != 0 and value == 0)
           for value, exact in zip(converted, coefficients)):
        return None
    return converted


def ch_zero(a, coefficients):
    # Evaluate the returned polynomial on the accepted binary64 matrix using
    # exact matrix Horner arithmetic, independently of its construction.
    n = len(a)
    p = diag(*([0] * n))
    for c in coefficients:
        p = [[sum(p[i][k] * a[k][j] for k in range(n)) + (c if i == j else 0)
              for j in range(n)] for i in range(n)]
    return all(v == 0 for row in p for v in row)


def roots_match(case, returned):
    n = len(case.roots)
    if case.name == "root-outside-float":
        return ("eigenvalues_real_imag" in returned and returned["eigenvalues_real_imag"] is None
                and returned.get("eigenvalues_status") == "outside_float_range")
    observed = returned.get("eigenvalues_real_imag")
    if (returned.get("eigenvalues_status") != "finite_approximation"
            or not isinstance(observed, list) or len(observed) != n
            or any(not isinstance(row, list) or len(row) != 2 or not all(map(numeric, row))
                   for row in observed)):
        return False

    def close(pole, pair):
        for component, measured in zip(pole, pair):
            if component:
                if abs(F(measured) - component) > F(abs(component), 10**12):
                    return False
            elif n <= 2 or pole == (0, 0):
                if measured != 0:
                    return False
            elif abs(F(measured)) > F(max(abs(pole[0]), abs(pole[1])), 10**12):
                # The three-state fixtures have real poles; their exact
                # magnitude is therefore max(abs(real), abs(imag)).
                return False
        return True

    # One-to-one matching checks repeated roots and is independent of output ordering.
    return any(all(close(pole, pair) for pole, pair in zip(order, observed))
               for order in permutations(case.roots))


def determinant3(p):
    return (p[0][0]*(p[1][1]*p[2][2]-p[1][2]*p[2][1])
            - p[0][1]*(p[1][0]*p[2][2]-p[1][2]*p[2][0])
            + p[0][2]*(p[1][0]*p[2][1]-p[1][1]*p[2][0]))


def positive_definite(p):
    if p is None:
        return None
    return (p[0][0] > 0 and p[0][0]*p[1][1] > p[0][1]*p[1][0]
            and (len(p) == 2 or determinant3(p) > 0))


def lyapunov_matches(case, a, returned, keys):
    status, exact, pd, residual = keys
    if any(k not in returned for k in keys):
        return False
    if case.p is None:
        return (returned[status] == "inconsistent" and returned[exact] is None
                and returned[pd] is None and returned[residual] is None)
    n = len(a)
    raw = returned[exact]
    if not isinstance(raw, list) or len(raw) != n:
        return False
    p = [rational_strings(row, n) for row in raw]
    if p != case.p or returned[status] != "unique" or returned[residual] is not True:
        return False
    return (returned[pd] is positive_definite(p) and all(
        sum(a[k][i]*p[k][j] + p[i][k]*a[k][j] for k in range(n)) == -int(i == j)
        for i in range(n) for j in range(n)))


def lyapunov_rendered(case, returned):
    if "lyapunov_P" not in returned:
        return False
    p, state = returned["lyapunov_P"], returned.get("lyapunov_float_status")
    if case.p is None:
        return p is None and state == "not_unique_solution"
    if state != "finite" or not isinstance(p, list) or len(p) != len(case.p):
        return False
    for row, expected in zip(p, case.p):
        if not isinstance(row, list) or len(row) != len(expected):
            return False
        if any(not numeric(x) or abs(F(x) - y) > F(abs(y), 10**12) for x, y in zip(row, expected)):
            return False
    return True


def result_checks(case, a, result):
    exact = rational_strings(result.get("char_poly_exact"), len(case.polynomial))
    rounded = float_coefficients(case.polynomial)
    reported = result.get("char_poly")
    coefficients_rendered = ("char_poly" in result and result.get("char_poly_float_status") ==
                             ("outside_float_range" if rounded is None else "finite"))
    if rounded is None:
        coefficients_rendered = coefficients_rendered and reported is None
    else:
        coefficients_rendered = (coefficients_rendered and isinstance(reported, list)
                                 and all(map(numeric, reported)) and reported == rounded)
    multiplicity = sum(pole == (0, 0) for pole in case.roots)
    certificate = result.get("stability_certificate")
    return {
        "passed": result.get("passed") is True,
        "stable": result.get("hurwitz_stable") is case.stable,
        "polynomial": exact == case.polynomial,
        "polynomial_float": coefficients_rendered,
        "ch": (exact is not None and ch_zero(a, exact)
               and result.get("cayley_hamilton_residual_basis") == "char_poly_exact"
               and result.get("cayley_hamilton_residual_exact") == "0"
               and numeric(result.get("cayley_hamilton_residual"))
               and result["cayley_hamilton_residual"] == 0.0),
        "zero_multiplicity": (type(result.get("zero_eigenvalue_multiplicity")) is int
                              and result["zero_eigenvalue_multiplicity"] == multiplicity),
        "poles": roots_match(case, result),
        "P": lyapunov_matches(case, a, result,
                             ("lyapunov_solution_status", "lyapunov_P_exact", "lyapunov_pd", "lyapunov_residual_exact")),
        "P_float": lyapunov_rendered(case, result),
        "identity": (isinstance(certificate, dict) and lyapunov_matches(
            case, a, certificate, ("solution_status", "P_exact", "positive_definite", "residual_exact"))),
    }


def exercise(case):
    spec = {"mode": "state_space", "A": [[float(v) for v in row] for row in case.a]}
    a = [[F(v) for v in row] for row in spec["A"]]
    errors = []
    result = call(spec, "unclaimed", errors)
    failures = [k for k, ok in result_checks(case, a, result).items() if not ok]
    decisions = []
    correct, false_approvals, false_refusals = 0, 0, 0
    for field, truth in (("hurwitz_stable", case.stable), ("lyapunov_pd", positive_definite(case.p))):
        for supplied in (False, True):
            expected = supplied is truth
            checked = call({**spec, "claimed_field": field, "claimed_value": supplied},
                           f"{field}/{supplied}", errors)
            actual = checked.get("passed")
            decisions.append("A" if actual is True else "R" if actual is False else "E")
            false_approvals += int(actual is True and not expected)
            false_refusals += int(actual is False and expected)
            if truth is None:
                evidence_matches = bool(checked.get("reason")) and checked.get(field) is None
            else:
                claim = checked.get("claim_check") or {}
                evidence_matches = (checked.get(field) is truth and claim.get("field") == field
                                    and claim.get("claimed") == float(supplied)
                                    and claim.get("actual") == float(truth)
                                    and claim.get("within_tolerance") is expected)
            correct += int(actual is expected and evidence_matches)
    return {"case": case.name, "failed_checks": failures, "decisions": "".join(decisions),
            "correct_claims": correct, "false_approvals": false_approvals, "false_refusals": false_refusals,
            "char_float": result.get("char_poly_float_status"), "roots": result.get("eigenvalues_status"),
            "stability": result.get("hurwitz_stable"), "requested_P": result.get("lyapunov_solution_status"),
            "errors": errors}


class StateSpaceDiagnosticRangeTests(unittest.TestCase):
    def test_characteristic_diagnostics_preserve_exact_stability(self):
        rows = [exercise(case) for case in fixtures()]
        for row in rows:
            print("CONTROL_DIAGNOSTIC_CASE " + canonical(row))
        summary = {"systems": len(rows), "correct_unclaimed": sum(not row["failed_checks"] for row in rows),
                   "claims": sum(len(row["decisions"]) for row in rows),
                   "correct_claims": sum(row["correct_claims"] for row in rows),
                   "approvals": sum(row["decisions"].count("A") for row in rows),
                   "refusals": sum(row["decisions"].count("R") for row in rows),
                   "false_approvals": sum(row["false_approvals"] for row in rows),
                   "false_refusals": sum(row["false_refusals"] for row in rows),
                   "solver_errors": sum(e[1] == "solver" for row in rows for e in row["errors"]),
                   "json_errors": sum(e[1] == "json" for row in rows for e in row["errors"])}
        print("CONTROL_DIAGNOSTIC_TOTAL " + canonical(summary))
        self.assertEqual(summary, {"systems": 17, "correct_unclaimed": 17, "claims": 68,
                                   "correct_claims": 68, "approvals": 31, "refusals": 37,
                                   "false_approvals": 0, "false_refusals": 0,
                                   "solver_errors": 0, "json_errors": 0})
