"""Independent integer comparison of accepted binary64 numbers on the Wave 5 control route."""
from dataclasses import dataclass
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()


@dataclass(frozen=True)
class Dyadic:
    """Integer units of 2^-bits, without using the production rational class."""
    units: int
    bits: int

    @classmethod
    def of(cls, x):
        n, d = float(x).as_integer_ratio()
        return cls(n, d.bit_length()-1)

    def aligned(self, bits):
        return self.units << (bits - self.bits)

    def distance(self, other):
        bits = max(self.bits, other.bits)
        return Dyadic(abs(self.aligned(bits) - other.aligned(bits)), bits)

    def scaled(self, other):
        return Dyadic(self.units * other.units, self.bits + other.bits)

    def at_most(self, other):
        bits = max(self.bits, other.bits)
        return self.aligned(bits) <= other.aligned(bits)

    def text(self):
        divisor = 1 << self.bits
        common = math.gcd(self.units, divisor)
        den = divisor // common
        n = self.units // common
        return str(n) if den == 1 else f"{n}/{den}"

    def display(self):
        try:
            scalar = self.units / (1 << self.bits)
        except OverflowError:
            return None, "above_float_range"
        if scalar == 0 and self.units:
            return None, "below_float_range"
        return scalar, "finite"


def expected(actual, claim, atol, rtol):
    e = Dyadic.of(actual).distance(Dyadic.of(claim))
    magnitude = Dyadic.of(max(abs(actual), 1e-12))
    relative = magnitude.scaled(Dyadic.of(rtol))
    absolute = Dyadic.of(atol)
    limit = relative if absolute.at_most(relative) else absolute
    return e, limit


@dataclass
class Case:
    name: str
    num: list
    den: list
    request: dict
    actual: object
    inside: object
    base_pass: bool = True

    @property
    def approved(self):
        return self.base_pass and self.inside is True


def numeric(name, a, c, atol, rtol, inside):
    return Case(name, [a], [1, 1],
                {"claimed_field": "dc_gain", "claimed_value": c, "atol": atol, "rtol": rtol}, a, inside)


def cases():
    e, d = 2.0**-54, 2.0**-27
    u = float.fromhex("0x0.0000000000001p-1022")
    for label, c, inside in (("outside", -e, False), ("equal", 0, True), ("inside", e, True)):
        yield numeric("absolute/" + label, 1, c, 1, 0, inside)
    for label, k, inside in (("outside", 4, False), ("equal", 3, True), ("inside", 2, True)):
        yield numeric("product/" + label, 1+3*d, -d-k*d*d, 0, 1+d, inside)
    for label, r, inside in (("outside", 1.9, False), ("equal", 2, True), ("inside", 2+2.0**-51, True)):
        yield numeric("overflow/" + label, 1e308, -1e308, 0, r, inside)
    for label, r, inside in (("zero", 0, False), ("rounds-up", 2.0**-1035, False),
                             ("inside", 2.0**-1034, True)):
        yield numeric("floor/" + label, 0, u, 0, r, inside)

    yield numeric("control/reflected-outside", -1, e, 1, 0, False)
    yield Case("control/rounded-third", [1], [3, 3], {"claimed_field": "dc_gain", "claimed_value": 1/3,
               "atol": 0, "rtol": 0}, 1/3, True)
    yield Case("control/boolean", [1], [1, 1], {"claimed_field": "hurwitz_stable", "claimed_value": True,
               "atol": 0, "rtol": 0}, True, True)
    # D+N vanishes identically. The available DC gain matches; overall refusal persists.
    yield Case("control/preserve-refusal", [-1, -1], [1, 1], {"claimed_field": "dc_gain", "claimed_value": -1,
               "atol": 0, "rtol": 0}, -1, True, base_pass=False)
    yield numeric("control/positive-limit-rounds-zero", 0, 0, 0, 2.0**-1036, True)

    # Existing tolerances combine by maximum; adding them would accept E=1.
    yield numeric("control/max-not-sum", 1, 0, .5, .75, False)

    valid = {"claimed_field": "dc_gain", "claimed_value": 1, "atol": 0, "rtol": 0}
    for label, request in (
        ("missing-value", {"claimed_field": "dc_gain"}),
        ("missing-field", {"claimed_value": 1}),
        ("unknown-field", {**valid, "claimed_field": "no_such_control_output"}),
        ("exact-evidence", {**valid, "claimed_field": "dc_gain_exact"}),
        ("structured-evidence", {**valid, "claimed_field": "open_loop_root_certificate"}),
        ("nonnumeric-claim", {**valid, "claimed_value": "not a number"}),
        ("infinite-claim", {**valid, "claimed_value": "Infinity"}),
        ("negative-atol", {**valid, "atol": -1}),
        ("negative-rtol", {**valid, "rtol": -1}),
        ("nan-rtol", {**valid, "rtol": "NaN"}),
    ):
        yield Case("refuse/" + label, [1], [1, 1], request, None, None)
    yield Case("refuse/unavailable-scalar", [1], [1, 0], valid, None, None)


def invoke(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(result)
    except (ValueError, TypeError, OverflowError) as exc:
        # Keep the raw decision observable even if its old diagnostics cannot travel in JSON.
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
    return result


def displayed(check, name, magnitude):
    value, status = magnitude.display()
    rendered = check.get(name)
    return (name in check and ((value is None and rendered is None) or
            (value is not None and type(rendered) is float and rendered == value))
            and check.get(name + "_float_status") == status and check.get(name + "_exact") == magnitude.text())


def exercise(case):
    base = {"mode": "transfer_function", "num": list(map(float, case.num)), "den": list(map(float, case.den))}
    errors = []
    ordinary = invoke(base, "unclaimed", errors)
    answer = invoke({**base, **case.request}, "claim", errors)
    ordinary_ok = ordinary.get("passed") is case.base_pass and "claim_check" not in ordinary
    failures = [] if ordinary_ok else ["unclaimed"]
    actual = answer.get("passed")
    if actual is not case.approved:
        failures.append("decision")
    if case.inside is None:
        if "claim_check" in answer or not answer.get("reason"):
            failures.append("explicit refusal")
    else:
        c = case.request["claimed_value"]
        e, limit = expected(case.actual, c, case.request["atol"], case.request["rtol"])
        ck = answer.get("claim_check") or {}
        if e.at_most(limit) is not case.inside:
            failures.append("integer comparison disagrees with analytic case")
        if (ck.get("field") != case.request["claimed_field"] or ck.get("actual") != float(case.actual)
                or ck.get("claimed") != float(c) or ck.get("within_tolerance") is not case.inside):
            failures.append("claim evidence")
        if not (displayed(ck, "abs_error", e) and displayed(ck, "tolerance_limit", limit)):
            failures.append("exact/range evidence")
        if not case.approved and not answer.get("reason"):
            failures.append("refusal reason")
    ordinary_hash = None if any(err[0] == "unclaimed" for err in errors) else fingerprint(ordinary)
    return {"case": case.name, "inside": case.inside, "expected": case.approved, "actual": actual,
            "unclaimed_pass": ordinary.get("passed"), "unclaimed_sha256": ordinary_hash,
            "failed_checks": failures, "errors": errors}


class ExactScalarComparisonTests(unittest.TestCase):
    def test_finite_comparisons_and_refusals(self):
        rows = [exercise(case) for case in cases()]
        for row in rows:
            print("SCALAR_TOLERANCE_CASE " + canonical(row))
        totals = {"cases": len(rows), "complete": sum(not r["failed_checks"] for r in rows),
                  "core_approved": sum(r["actual"] is True for r in rows[:12]),
                  "core_refused": sum(r["actual"] is False for r in rows[:12]),
                  "approved": sum(r["actual"] is True for r in rows), "refused": sum(r["actual"] is False for r in rows),
                  "false_approvals": sum(r["actual"] is True and not r["expected"] for r in rows),
                  "false_refusals": sum(r["actual"] is False and r["expected"] for r in rows),
                  "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        digest = fingerprint([[r["case"], r["unclaimed_sha256"]] for r in rows])
        print("SCALAR_TOLERANCE_TOTAL " + canonical({**totals, "unclaimed_sha256": digest}))
        self.assertEqual(totals, {"cases": 29, "complete": 29, "core_approved": 7, "core_refused": 5,
                                  "approved": 10, "refused": 19, "false_approvals": 0, "false_refusals": 0,
                                  "solver_errors": 0, "json_errors": 0})
