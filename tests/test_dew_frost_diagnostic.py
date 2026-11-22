"""Separate independently specified pressure inversions from exact model endpoint conventions."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


DEW = "t_dp_c"
STATUS = "t_dp_status"
W = "w_kg_per_kgda"
D = Decimal


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def observed(value):
    try:
        canonical(value)
    except (TypeError, ValueError, OverflowError):
        return {"unserializable_type": type(value).__name__, "repr": repr(value)[:160]}
    return value


def pressure(temp_c, phase=None):
    # Published 2017 ASHRAE Fundamentals SI ch. 1, equations 5 and 6 (Pa).
    # Evaluate the model in decimal arithmetic, not the binary64 producer.
    # https://handbook.ashrae.org/Handbooks/F17/SI/f17_ch01/f17_ch01_si.aspx
    with localcontext() as ctx:
        ctx.prec = 80
        t = D(temp_c) + D("273.15")
        ice = temp_c < 0 if phase is None else phase == "ice"
        if ice:
            reciprocal, log_weight = D("-5674.5359"), D("4.1635019")
            coeff = ("6.3925247", "-0.009677843", "6.2215701e-7", "2.0747825e-9", "-9.484024e-13")
        else:
            reciprocal, log_weight = D("-5800.2206"), D("6.5459673")
            coeff = ("1.3914993", "-0.048640239", "4.1764768e-5", "-1.4452093e-8")
        poly = D(0)
        for c in reversed(coeff):
            poly = poly*t + D(c)
        return (reciprocal/t + poly + log_weight*t.ln()).exp()


def water_ratio(pw, total, scale=D(1)):
    with localcontext() as ctx:
        ctx.prec = 80
        pw = scale * pw
        return float(D(".621945") * pw / (D(total)-pw))


def state(db, p=101325, **inputs):
    return {"mode": "state", "t_db_c": db, "p_atm_pa": p, **inputs}


@dataclass
class Case:
    name: str
    spec: dict
    temperature: object = None   # closed reference interval for finite diagnostics
    status: object = None
    claims: tuple = ()           # (claimed temperature, absolute tolerance, expected verdict)
    expected_pw: object = None   # independent primary-equation or prescribed pressure
    preserve_full: bool = False


def cases():
    cold, hot = pressure(-90), pressure(170)
    extended_ice = ((-90, 1e-8, True), (-80, 1e-8, False))
    domain = ((-90-1e-8, -90+1e-8), (170-1e-8, 170+1e-8))
    yield Case("pressure/ice-minus90", state(20, **{W: water_ratio(cold, 101325)}), domain[0],
               claims=extended_ice, expected_pw=cold)
    yield Case("pressure/water170", state(185, 5000000, **{W: water_ratio(hot, 5000000)}), domain[1],
               claims=((170, 1e-8, True), (150, 1e-8, False)), expected_pw=hot)
    yield Case("roundtrip/explicit-minus90", state(20, t_dp_c=-90), domain[0], claims=extended_ice, expected_pw=cold)
    for name, spec, t in (("ice-lower", state(20, t_dp_c=-100), -100),
                          ("water-lower", state(20, t_dp_c=0), 0),
                          ("water-upper", state(200, 5000000, rh=1), 200)):
        yield Case("endpoint/" + name, spec, (t, t), claims=((t, 0, True),), expected_pw=pressure(t))
    for name, spec, status, wrong, atol in (
        ("dry-rh", state(20, rh=0), "dry_no_finite_point", -100, 0),
        ("dry-W", state(20, **{W: 0}), "dry_no_finite_point", -100, 0),
        ("below", state(20, **{W: 1e-16}), "below_correlation_range", -80, 0),
        ("gap", state(20, **{W: water_ratio(D("611.18"), 101325)}), "phase_gap", 0, 1e-8),
        # This state uses the existing humidity tolerance just above saturation.
        ("above-RH-tolerance", state(200, 5000000, **{W: water_ratio(pressure(200), 5000000, D("1.00001"))}),
         "above_correlation_range", 150, 0),
    ):
        yield Case("unavailable/" + name, spec, status=status, claims=((wrong, atol, False),))
    yield Case("ordinary/RH-half", state(20, rh=.5), (9, 10), preserve_full=True)
    yield Case("ordinary/frost-minus20", state(20, t_dp_c=-20), (-20-1e-8, -20+1e-8), preserve_full=True)
    yield Case("ordinary/dew-plus5", state(20, t_dp_c=5), (5-1e-8, 5+1e-8), preserve_full=True)
    yield Case("endpoint/ice-upper", state(20, t_dp_c=math.nextafter(0., -math.inf)),
               (-1e-8, math.nextafter(0., -math.inf)), expected_pw=pressure(0, phase="ice"))
    yield Case("process/mixing", {"mode": "adiabatic_mixing", "stream1": {"t_db_c": 20, "rh": .5, "m_da_kg_s": 1},
               "stream2": {"t_db_c": 30, "rh": .4, "m_da_kg_s": 1}}, preserve_full=True)
    yield Case("process/cooling", {"mode": "cooling_coil", "inlet": {"t_db_c": 20, "rh": .5},
               "outlet": {"t_db_c": 10, "rh": .5}, "m_da_kg_s": 1}, preserve_full=True)


def query(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "psychrometrics", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(result)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
    return result


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def pressure_regime(case, result):
    pw = result.get("pw_pa")
    if not finite(pw):
        return False
    if case.expected_pw is not None:
        return math.isclose(pw, float(case.expected_pw), rel_tol=2e-12, abs_tol=0)
    if case.status == "dry_no_finite_point":
        return pw == 0
    if case.status == "below_correlation_range":
        return 0 < pw < float(pressure(-100))
    if case.status == "above_correlation_range":
        # Assert that this pressure lies within the pre-existing RH input admission tolerance.
        return float(pressure(200)) < pw < 1.0001*float(pressure(200))
    if case.status == "phase_gap":
        return float(pressure(0, "ice")) < pw < float(pressure(0, "water"))
    return True


def exercise(case):
    errors = []
    result = query(case.spec, "unclaimed", errors)
    failed = []
    if result.get("passed") is not True:
        failed.append("ordinary admission")
    if case.spec["mode"] == "state":
        if not pressure_regime(case, result):
            failed.append("independent pressure")
        if case.status is None:
            value = result.get(DEW)
            if (not finite(value) or not case.temperature[0] <= value <= case.temperature[1]
                    or STATUS in result):
                failed.append("finite diagnostic")
        elif DEW not in result or result[DEW] is not None or result.get(STATUS) != case.status:
            failed.append("unavailable diagnostic")
    decisions, claim_failures = [], []
    for idx, (target, atol, approve) in enumerate(case.claims):
        answer = query({**case.spec, "claimed_field": DEW, "claimed_value": target, "rtol": 0, "atol": atol},
                       f"claim{idx+1}", errors)
        verdict = answer.get("passed")
        decisions.append("A" if verdict is True else "R" if verdict is False else "E")
        if case.status is not None:
            evidence = (DEW in answer and answer[DEW] is None and answer.get(STATUS) == case.status
                        and bool(answer.get("reason")) and "claim_check" not in answer)
        else:
            ck = answer.get("claim_check") or {}
            value = answer.get(DEW)
            evidence = (finite(value) and case.temperature[0] <= value <= case.temperature[1] and STATUS not in answer
                        and ck.get("field") == DEW and ck.get("actual") == value and ck.get("claimed") == target
                        and ck.get("within_tolerance") is approve and (approve or bool(answer.get("reason"))))
        if verdict is not approve or not evidence:
            claim_failures.append(idx+1)
    unclaimed_error = any(e[0] == "unclaimed" for e in errors)
    projection = {k: v for k, v in result.items() if k not in (DEW, STATUS)}
    return {"case": case.name, "passed": observed(result.get("passed")), "failed_checks": failed,
            "pw_pa": observed(result.get("pw_pa")), DEW: observed(result.get(DEW)), STATUS: observed(result.get(STATUS)),
            "decisions": "".join(decisions), "claim_failures": claim_failures, "errors": errors,
            "preserve_full": case.preserve_full, "full_sha256": None if unclaimed_error else digest(result),
            "non_dew_sha256": None if unclaimed_error else digest(projection)}


class DewFrostDiagnosticTests(unittest.TestCase):
    def test_supported_pressure_inverse(self):
        rows = [exercise(case) for case in cases()]
        for row in rows:
            print("DEW_DOMAIN_CASE " + canonical(row))
        totals = {"systems": len(rows), "unclaimed_approvals": sum(r["passed"] is True for r in rows),
                  "complete_unclaimed": sum(not r["failed_checks"] for r in rows),
                  "claims": sum(len(r["decisions"]) for r in rows),
                  "approvals": sum(r["decisions"].count("A") for r in rows),
                  "refusals": sum(r["decisions"].count("R") for r in rows),
                  "claim_failures": sum(len(r["claim_failures"]) for r in rows),
                  "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        print("DEW_DOMAIN_TOTAL " + canonical({**totals, "non_dew_sha256": digest(
            [[r["case"], r["non_dew_sha256"]] for r in rows])}))
        self.assertEqual(totals, {"systems": 17, "unclaimed_approvals": 17, "complete_unclaimed": 17,
                                  "claims": 14, "approvals": 6, "refusals": 8, "claim_failures": 0,
                                  "solver_errors": 0, "json_errors": 0})
