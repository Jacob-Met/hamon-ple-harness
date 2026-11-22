"""Count roots of known factorizations and independently expand the shifted Hurwitz minors."""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


BODE_DC = ("w_gc_rad_s", "phase_margin_deg", "w_pc_rad_s", "gain_margin_db",
           "gain_crossovers", "phase_crossovers", "unity_gain_intervals_rad_s",
           "negative_real_intervals_rad_s", "imaginary_axis_poles_rad_s", "removable_holes_rad_s",
           "dc_gain", "dc_gain_status")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass
class Case:
    name: str
    den: list
    num: list
    opened: object
    closed: object


def cases():
    delta = F(1, 2**55)
    stable1, unstable1 = (0, 0, 1), (1, 0, 0)
    for sign, open_counts in ((1, stable1), (-1, unstable1)):
        for num in (0, 1):
            yield Case(f"slow/{sign}/{num}", [1, sign*delta], [num], open_counts,
                       open_counts if num == 0 else stable1)
    # s³(s+2), (s²+1)²(s+2), (s²-1)(s²+4), (s²-1)², s²(s²+1)².
    for name, den, counts in (
        ("origin-cube", [1, 2, 0, 0, 0], (0, 3, 1)),
        ("repeated-axis", [1, 2, 2, 4, 1, 2], (0, 4, 1)),
        ("symmetric-real-and-axis", [1, 0, 3, 0, -4], (1, 2, 1)),
        ("repeated-symmetric-real", [1, 0, -2, 0, 1], (2, 0, 2)),
        ("double-origin-repeated-axis", [1, 0, 2, 0, 1, 0, 0], (0, 6, 0)),
        ("zero-first-pivot", [1, 2, 3, 6, 2], (2, 0, 2)),
    ):
        yield Case(name, den, [0], counts, counts)
    for exponent, sign in ((0, 1), (50, -1), (-50, 1)):
        g = sign * F(2)**exponent
        yield Case(f"common-gain/{sign}/{exponent}", [g, 3*g, 2*g], [g], (0, 0, 2), (0, 0, 2))
    yield Case("authored-common-rhp", [1, 1, -2], [1, -1], (1, 0, 1), (1, 0, 1))
    for name, n, closed in (("below", -delta, (0, 0, 3)), ("above", delta, (2, 0, 1)),
                            ("on", 0, (0, 2, 1))):
        yield Case(f"feedback-boundary/{name}", [1, 1, 1, 1], [n], (0, 2, 1), closed)
    yield Case("constant-feedback", [1, 1], [-1, 0], stable1, (0, 0, 0))
    yield Case("identically-zero-feedback", [1, 1], [-1, -1], stable1, None)
    yield Case("invalid-input-constant", [1], [0], None, None)


def monic(coefficients):
    first = next((i for i, v in enumerate(coefficients) if v), len(coefficients))
    return tuple(v / coefficients[first] for v in coefficients[first:])


def add_descending(left, right):
    width = max(len(left), len(right))
    return [a+b for a, b in zip([0]*(width-len(left)) + left, [0]*(width-len(right)) + right)]


def multiply(left, right):
    # Coefficients in increasing powers of epsilon.
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a:
            for j, b in enumerate(right):
                result[i+j] += a*b
    return result


@lru_cache(maxsize=None)
def leading_minor_terms(p):
    """Leibniz expansion over at most 6! permutations; no Routh row recursion or Bareiss."""
    n = len(p) - 1
    # Coefficient at s**(n-i) after substituting s+epsilon in the monic p.
    shifted = [[p[i-j] * math.comb(n-i+j, j) for j in range(i+1)] for i in range(n+1)]
    terms = []
    for k in range(1, n+1):
        determinant = [0] * (k*(k+1)//2 + 1)
        for perm in itertools.permutations(range(k)):
            indices = [2*j-i+1 for i, j in enumerate(perm)]
            if any(index < 0 or index > n for index in indices):
                continue
            inversions = sum(perm[i] > perm[j] for i in range(k) for j in range(i+1, k))
            term = [(-1)**inversions]
            for index in indices:
                term = multiply(term, shifted[index])
            for power, coeff in enumerate(term):
                determinant[power] += coeff
        power = next(i for i, coeff in enumerate(determinant) if coeff)
        terms.append((power, determinant[power]))
    return terms


def signs(terms, direction):
    previous, result = 1, [1]
    for power, coeff in terms:
        current = 1 if coeff * direction**power > 0 else -1
        result.append(current * previous)
        previous = current
    return result


def rational_strings(values):
    if not isinstance(values, list) or any(type(v) is not str for v in values):
        return None
    try:
        return tuple(F(v) for v in values)
    except (ValueError, ZeroDivisionError):
        return None


def check_certificate(cert, p, counts):
    if not isinstance(cert, dict) or cert.get("method") != "infinitesimal_shift_hurwitz":
        return False
    if rational_strings(cert.get("normalized_coefficients")) != p:
        return False
    keys = ("rhp_poles", "jw_axis_poles", "lhp_poles")
    if counts is None:
        return (cert.get("status") == "identically_zero" and cert.get("minor_leading_terms") == []
                and all(key in cert and cert[key] is None for key in (
                    *keys, "degree", "hurwitz_stable", "first_column_signs_plus", "first_column_signs_minus")))
    degree = len(p)-1
    if (cert.get("status") != ("classified" if degree else "constant_nonzero")
            or type(cert.get("degree")) is not int or cert["degree"] != degree
            or any(type(cert.get(key)) is not int for key in keys)
            or tuple(cert[key] for key in keys) != counts
            or cert.get("hurwitz_stable") is not (counts[0] == 0 and counts[1] == 0)):
        return False
    actual = cert.get("minor_leading_terms")
    expected = leading_minor_terms(p)
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for entry, (power, coeff) in zip(actual, expected):
        if (not isinstance(entry, dict) or type(entry.get("power")) is not int or entry["power"] != power
                or rational_strings([entry.get("coefficient")]) != (coeff,)):
            return False
    return (cert.get("first_column_signs_plus") == signs(expected, 1)
            and cert.get("first_column_signs_minus") == signs(expected, -1))


def query(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(result)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
        return {}
    return result


def pole_rendering(result, field, status_field, degree):
    if field not in result:
        return False
    pairs, status = result[field], result.get(status_field)
    if degree is None:
        return pairs is None and status == "undefined_characteristic"
    if status in ("outside_float_range", "unavailable_numerical_diagnostic"):
        return degree > 0 and pairs is None
    return (status == "finite_approximation" and isinstance(pairs, list) and len(pairs) == degree
            and all(isinstance(pair, list) and len(pair) == 2
                    and all(type(v) in (int, float) and math.isfinite(v) for v in pair)
                    for pair in pairs))


def exercise(case):
    spec = {"mode": "transfer_function", "num": list(map(float, case.num)), "den": list(map(float, case.den))}
    d, numerator = [F(v) for v in spec["den"]], [F(v) for v in spec["num"]]
    op, cl = monic(d), monic(add_descending(d, numerator))
    errors = []
    result = query(spec, "unclaimed", errors)
    valid_input = case.opened is not None
    defined = valid_input and case.closed is not None
    truth_open = ((case.opened[0] == 0 and case.opened[1] == 0) if valid_input else None)
    truth_closed = ((case.closed[0] == 0 and case.closed[1] == 0) if defined else None)
    open_cert, closed_cert = result.get("open_loop_root_certificate"), result.get("closed_loop_root_certificate")
    if valid_input:
        pole_list = result.get("poles_real_imag")
        approx = result.get("rhp_poles_eigen")
        open_poles_valid = pole_rendering(result, "poles_real_imag", "poles_status", len(op)-1)
        approx_raw = ("rhp_poles_eigen" in result and result.get("rhp_poles_eigen_is_approximate") is True
                      and open_poles_valid and (
                          approx is None if pole_list is None else
                          type(approx) is int and approx == sum(pair[0] > 0 for pair in pole_list)))
        checks = {
            "returned": result.get("passed") is defined,
            "open_flag": result.get("hurwitz_stable") is truth_open,
            "closed_flag": ("closed_loop_hurwitz_stable" in result
                            and result["closed_loop_hurwitz_stable"] is truth_closed),
            "open_counts": (type(result.get("rhp_poles_routh")) is int
                            and type(result.get("jw_axis_poles")) is int
                            and (result["rhp_poles_routh"], result["jw_axis_poles"]) == case.opened[:2]),
            "open_certificate": check_certificate(open_cert, op, case.opened),
            "closed_certificate": check_certificate(closed_cert, cl, case.closed),
            "open_poles": open_poles_valid,
            "closed_poles": pole_rendering(result, "closed_loop_poles_real_imag", "closed_loop_poles_status",
                                           len(cl)-1 if cl else None),
            "approx_open_count": approx_raw,
            "reason_if_undefined": defined or bool(result.get("reason")),
            "bode_dc_present": all(k in result for k in BODE_DC),
        }
    else:
        checks = {"invalid_refusal": result.get("passed") is False and bool(result.get("reason"))}
    decisions, correct, false_approvals, false_refusals = [], 0, 0, 0
    for field, truth in (("hurwitz_stable", truth_open), ("closed_loop_hurwitz_stable", truth_closed)):
        for supplied in (False, True):
            expected = defined and supplied is truth
            checked = query({**spec, "claimed_field": field, "claimed_value": supplied},
                            f"{field}/{supplied}", errors)
            actual = checked.get("passed")
            decisions.append("A" if actual is True else "R" if actual is False else "E")
            false_approvals += int(actual is True and not expected)
            false_refusals += int(actual is False and expected)
            if defined:
                claim = checked.get("claim_check") or {}
                evidence = (checked.get(field) is truth and claim.get("field") == field
                            and claim.get("actual") == float(truth) and claim.get("claimed") == float(supplied)
                            and claim.get("within_tolerance") is expected)
            else:
                evidence = bool(checked.get("reason"))
            correct += int(actual is expected and evidence)
    return {"case": case.name, "failed_checks": [k for k, ok in checks.items() if not ok],
            "open_flag": result.get("hurwitz_stable"), "closed_flag": result.get("closed_loop_hurwitz_stable"),
            "open_rhp_jw": [result.get("rhp_poles_routh"), result.get("jw_axis_poles")],
            "expected_open": case.opened, "expected_closed": case.closed,
            "closed_counts": ([closed_cert.get(k) for k in ("rhp_poles", "jw_axis_poles", "lhp_poles")]
                              if isinstance(closed_cert, dict) else None),
            "reason": result.get("reason"), "decisions": "".join(decisions), "correct_claims": correct,
            "false_approvals": false_approvals, "false_refusals": false_refusals, "errors": errors,
            "bode_dc_sha256": digest({key: result[key] for key in BODE_DC if key in result})}


class TransferHurwitzTests(unittest.TestCase):
    def test_infinitesimal_shift_counts_and_claims(self):
        rows = [exercise(case) for case in cases()]
        for row in rows:
            print("TRANSFER_HURWITZ_CASE " + canonical(row))
        summary = {"cases": len(rows), "correct_unclaimed": sum(not row["failed_checks"] for row in rows),
                   "claims": sum(len(row["decisions"]) for row in rows),
                   "correct_claims": sum(row["correct_claims"] for row in rows),
                   "approvals": sum(row["decisions"].count("A") for row in rows),
                   "refusals": sum(row["decisions"].count("R") for row in rows),
                   "false_approvals": sum(row["false_approvals"] for row in rows),
                   "false_refusals": sum(row["false_refusals"] for row in rows),
                   "solver_errors": sum(e[1] == "solver" for row in rows for e in row["errors"]),
                   "json_errors": sum(e[1] == "json" for row in rows for e in row["errors"])}
        aggregate_digest = digest([[row["case"], row["bode_dc_sha256"]] for row in rows])
        print("TRANSFER_HURWITZ_TOTAL " + canonical({**summary, "bode_dc_sha256": aggregate_digest}))
        self.assertEqual(summary, {"cases": 20, "correct_unclaimed": 20, "claims": 80,
                                   "correct_claims": 80, "approvals": 36, "refusals": 44,
                                   "false_approvals": 0, "false_refusals": 0,
                                   "solver_errors": 0, "json_errors": 0})
