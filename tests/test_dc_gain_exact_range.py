"""Check known origin limits, binary64 rendering and exact-tolerance scalar claims."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


BODE = ("w_gc_rad_s", "phase_margin_deg", "w_pc_rad_s", "gain_margin_db",
        "gain_crossovers", "phase_crossovers", "unity_gain_intervals_rad_s",
        "negative_real_intervals_rad_s", "imaginary_axis_poles_rad_s", "removable_holes_rad_s")
DC = ("dc_gain", "dc_gain_status", "dc_gain_exact")


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class Limit:
    name: str
    num: list
    den: list
    exact: object
    displayed: object
    status: str
    refused_field: str
    refused_value: float


def systems():
    # In these formulas B=(s+1), so B(0)=1. Factors of s are explicit.
    # from_float retains the input's accepted binary64 value, including its sign.
    frac = Fraction.from_float
    u = float.fromhex("0x0.0000000000001p-1022")
    small = frac(1e-300) / frac(1e100)
    large = frac(1e300) / frac(1e-100)
    return [
        Limit("ordinary/third", [1], [3, 3], Fraction(1, 3), 1/3, "finite", "dc_gain_exact", 1/3),
        Limit("zero/identical", [0], [1, 1], 0, 0.0, "finite", "dc_gain_exact", 0),
        Limit("origin/extra-zero", [-3, 0, 0], [2, 2, 0], 0, 0.0, "finite_limit", "dc_gain", 1),
        Limit("origin/finite-negative", [-3, 0, 0], [2, 2, 0, 0], Fraction(-3, 2), -1.5,
              "finite_limit", "dc_gain_exact", -1.5),
        Limit("origin/pole", [-1, 0], [1, 1, 0, 0], None, None, "pole_at_zero", "dc_gain_exact", 0),
        Limit("underflow/positive", [1e-300], [1e100, 1e100], small, None,
              "outside_float_range", "dc_gain_exact", 0),
        Limit("underflow/negative-common-origin", [-1e-300, 0], [1e100, 1e100, 0], -small, None,
              "outside_float_range", "dc_gain_exact", 0),
        Limit("overflow/positive", [1e300], [1e-100, 1e-100], large, None,
              "outside_float_range", "dc_gain_exact", 0),
        Limit("overflow/negative-common-origin", [-1e300, 0], [1e-100, 1e-100, 0], -large, None,
              "outside_float_range", "dc_gain_exact", 0),
        Limit("subnormal/minimum-positive", [u], [1, 1], frac(u), u, "finite", "dc_gain", 0),
        Limit("subnormal/minimum-negative", [-u], [1, 1], -frac(u), -u, "finite", "dc_gain", 0),
        Limit("tie/half-positive", [u], [2, 2], frac(u)/2, None, "outside_float_range", "dc_gain_exact", 0),
        Limit("tie/half-negative", [-u], [2, 2], -frac(u)/2, None, "outside_float_range", "dc_gain_exact", 0),
        # Halfway between one and two subnormal units, the even result is two.
        Limit("tie/three-halves", [3*u], [2, 2], 3*frac(u)/2, 2*u, "finite", "dc_gain", 0),
    ]


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


def display_matches(result, oracle):
    if "dc_gain" not in result:
        return False
    displayed = result["dc_gain"]
    if oracle.displayed is None:
        return displayed is None
    return type(displayed) is float and displayed == oracle.displayed


def exercise(oracle):
    spec = {"mode": "transfer_function", "num": list(map(float, oracle.num)), "den": list(map(float, oracle.den))}
    errors = []
    result = query(spec, "unclaimed", errors)
    first_approved = oracle.displayed is not None
    claims = (("dc_gain", oracle.displayed if first_approved else 0, first_approved),
              (oracle.refused_field, oracle.refused_value, False))
    exact_text = str(oracle.exact) if oracle.exact is not None else None
    outcomes = []
    correct = false_approvals = false_refusals = 0
    for number, (name, value, should_pass) in enumerate(claims, 1):
        checked = query({**spec, "claimed_field": name, "claimed_value": value, "atol": 0, "rtol": 0},
                        f"claim{number}/{name}", errors)
        passed = checked.get("passed")
        outcomes.append("A" if passed is True else "R" if passed is False else "E")
        false_approvals += int(passed is True and not should_pass)
        false_refusals += int(passed is False and should_pass)
        if name == "dc_gain_exact":
            details = (name in checked and checked[name] == exact_text and bool(checked.get("reason"))
                       and name in checked["reason"] and "claim_check" not in checked)
        elif oracle.displayed is None:
            details = display_matches(checked, oracle) and bool(checked.get("reason"))
        else:
            claim_check = checked.get("claim_check") or {}
            details = (display_matches(checked, oracle) and claim_check.get("field") == name
                       and claim_check.get("actual") == oracle.displayed and claim_check.get("claimed") == value
                       and claim_check.get("within_tolerance") is should_pass
                       and (should_pass or bool(checked.get("reason"))))
        correct += int(passed is should_pass and details)

    non_dc = {key: val for key, val in result.items() if key not in DC}
    bode = {key: result[key] for key in BODE if key in result}
    old_dc = {key: result[key] for key in DC[:2] if key in result}
    checks = {"passed": result.get("passed") is True, "display": display_matches(result, oracle),
              "status": result.get("dc_gain_status") == oracle.status,
              "exact": "dc_gain_exact" in result and result["dc_gain_exact"] == exact_text,
              "bode_available": len(bode) == 10}
    return {"case": oracle.name, "failed_checks": [k for k, ok in checks.items() if not ok],
            "unclaimed_passed": result.get("passed"), "dc_gain": result.get("dc_gain"),
            "dc_gain_status": result.get("dc_gain_status"), "dc_gain_exact_present": "dc_gain_exact" in result,
            "exact_matches": checks["exact"], "outcomes": "".join(outcomes), "correct_claims": correct,
            "false_approvals": false_approvals, "false_refusals": false_refusals, "errors": errors,
            "legacy_dc_fields": len(old_dc), "legacy_dc_sha256": digest(old_dc),
            "bode_fields": len(bode), "bode_sha256": digest(bode),
            "non_dc_fields": len(non_dc), "non_dc_sha256": digest(non_dc)}


class ExactDcGainRangeTests(unittest.TestCase):
    def test_factored_origin_limits(self):
        rows = [exercise(system) for system in systems()]
        for row in rows:
            print("DC_RANGE_CASE " + canonical(row))
        totals = {"systems": len(rows), "complete": sum(not r["failed_checks"] for r in rows),
                  "unclaimed_approvals": sum(r["unclaimed_passed"] is True for r in rows),
                  "claims": sum(len(r["outcomes"]) for r in rows),
                  "correct_claims": sum(r["correct_claims"] for r in rows),
                  "approvals": sum(r["outcomes"].count("A") for r in rows),
                  "refusals": sum(r["outcomes"].count("R") for r in rows),
                  "false_approvals": sum(r["false_approvals"] for r in rows),
                  "false_refusals": sum(r["false_refusals"] for r in rows),
                  "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        projections = {key: digest([[r["case"], r[key]] for r in rows])
                       for key in ("legacy_dc_sha256", "bode_sha256", "non_dc_sha256")}
        print("DC_RANGE_TOTAL " + canonical({**totals, **projections}))
        self.assertEqual(totals, {"systems": 14, "complete": 14, "unclaimed_approvals": 14,
                                  "claims": 28, "correct_claims": 28, "approvals": 7, "refusals": 21,
                                  "false_approvals": 0, "false_refusals": 0, "solver_errors": 0, "json_errors": 0})
