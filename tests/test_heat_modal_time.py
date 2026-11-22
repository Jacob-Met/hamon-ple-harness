"""Independent dimensionless time references for the registered heat route."""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


H = 2**40
G = 2**510
BIG = 2**537
SPACING = 2.0**-53
STEP_CENTER = 2**1021
STEP_ALLOWANCE = 2**976
FULL_CONTROLS = {2, 4, 7, 9}
EXACT_TEMPERATURES = {8, 9, 11, 12, 14}
MODE = "heat_1d_transient"
FIELDS = ("u_exact", "u_crank_nicolson", "query_abs_error", "linf_error")
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117")


@dataclass(frozen=True)
class Case:
    name: str
    # alpha, time, length, position
    rod: tuple
    # (integer harmonic, amplitude, tabulated exact scale, spatial sine)
    modes: tuple
    cells: int
    steps: int
    false_grid: float

    def spec(self):
        alpha, time, length, position = self.rod
        return {"mode": MODE, "alpha_m2_s": alpha, "t_eval_s": time, "length_m": length,
                "x_eval_m": position, "modes": [{"n": n, "amplitude": a} for n, a, _, _ in self.modes]}


def cases():
    p = lambda e: 2.0**e
    primary = ((H, 1, D(1)/16, 1),)
    four = ((1, 1, D(4), 1),)
    half = ((1, 1, D(1)/2, 1),)
    near_one_scale = D(2)**-55
    under = (p(-542), p(-542), p(-500), p(-541))
    return [
        Case("product/underflow", under, primary, 32*H, 1920, 1),
        Case("ordinary/high-mode-twin", (1, p(-84), 1, p(-41)), primary, 32*H, 1920, 1),
        Case("product/overflow", (p(512), p(512), p(511), p(510)), four, 120, 120, 0),
        Case("ordinary/four-twin", (4, 1, 1, .5), four, 120, 120, 0),
        Case("square/underflow", (1, p(-1074), p(-538), p(-539)), four, 120, 120, 0),
        Case("square/overflow", (p(512), p(511), p(512), p(511)), half, 120, 120, 0),
        Case("ordinary/half-twin", (.5, 1, 1, .5), half, 120, 120, 0),
        Case("quotient/below-minimum", (1, p(-1001), p(37), p(-474)),
             ((G, 1, near_one_scale, 1),), 32*G, 1920, 1),
        Case("ordinary/near-one-twin", (1, p(-135), 1, p(-41)),
             ((H, 1, near_one_scale, 1),), 32*H, 1920, 1),
        Case("modes/signed-distinct-scales", under, primary + ((-3*H, .5, D(9)/16, 1),), 24*H, 480, 1.5),
        Case("zero/time-before-large-square", (1, 0, p(512), p(511)), ((1, 1, D(0), 1),), 120, 0, 0),
        Case("zero/harmonic-before-large-product", (p(512), p(512), 1, .5), ((0, 1, D(0), 0),), 120, 120, 1),
        Case("range/unrepresentable-positive-rate", (p(1023), p(1023), 1, .5),
             ((1, 1, 2**2046, 1),), 0, 0, 1),
        Case("shared-steps/subnormal-ratio", (1, p(-56), 1, .5),
             ((1, 1, D(2)**-56, 1), (BIG, 1, 2**1018, 0)), 2*BIG, STEP_CENTER, 1-2*SPACING),
    ]


def small_sine(z):
    term = total = z
    for k in range(1, 100):
        term *= -z*z/((2*k)*(2*k+1))
        if abs(term) < D("1e-90"):
            break
        total += term
    return total


def reference(index, case):
    if index == 13:
        return {}
    if index == 14:
        # For every admissible M, K_low/h ≈ π²/8; the Cayley-log
        # correction and exp remainder leave the decrement in (1.23h,1.24h).
        # The other mode has zero spatial sine. The real estimator rounds
        # to zero (<0.31 minimum subnormals). PLAN gives the inequalities.
        return dict(zip(FIELDS, (1-SPACING, 1-SPACING, 0., 0.)))
    continuum = cn = spatial_bound = D(0)
    for n, amplitude, scale, spatial in case.modes:
        decay = grid_decay = D(1)
        angle = PI*abs(n)/case.cells
        if scale:
            d = D(case.cells)/abs(n)
            decay = (-PI*PI*scale).exp()
            k = 4*d*d*scale*small_sine(PI/(2*d))**2
            ratio = k/(2*case.steps)
            grid_decay = ((1-ratio)/(1+ratio))**case.steps
        continuum += D(amplitude)*spatial*decay
        cn += D(amplitude)*spatial*grid_decay
        spatial_bound += abs(D(amplitude))*(abs(grid_decay-decay) + decay*angle*angle/8)
    error = abs(continuum-cn)
    ref = dict(zip(FIELDS, (float(continuum), float(cn), float(error), float(max(error, spatial_bound)))))
    if index in (8, 9):
        # Decimal80 supplies the continuous/CN estimate and whole-domain
        # bound. The independent rounding proof fixes the two displayed
        # temperatures and hence the displayed pointwise discrepancy.
        ref.update(u_exact=1-2*SPACING, u_crank_nicolson=1-2*SPACING, query_abs_error=0.)
    return ref


def allowances(index, ref):
    if not ref:
        return {}
    diag = 2e-13 * max(abs(ref[k]) for k in ("u_exact", "u_crank_nicolson", "linf_error"))
    return {"u_exact": 0 if index in EXACT_TEMPERATURES else 2e-13*abs(ref["u_exact"]),
            "u_crank_nicolson": 0 if index in EXACT_TEMPERATURES else 2e-13*abs(ref["u_crank_nicolson"]),
            "query_abs_error": 0 if index in EXACT_TEMPERATURES else diag,
            "linf_error": 0 if index == 12 else diag}


def canonical(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def safe(value):
    try:
        canonical(value)
    except (ValueError, TypeError, OverflowError):
        return {"type": type(value).__name__, "repr": repr(value)[:160]}
    return value


def request(spec, label, errors):
    try:
        raw = run_router_conservation_cad_verifier({"mode": "pde", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:160]])
        return {}, None, False
    out = raw if isinstance(raw, dict) else {}
    verdict = out.get("passed")
    try:
        canonical(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:160]])
        return out, verdict, False
    return out, verdict, isinstance(raw, dict)


def collect(index, case):
    ref = reference(index, case)
    tol = allowances(index, ref)
    claim_atol = min(tol["u_exact"], abs(ref["u_crank_nicolson"]-case.false_grid)/8) if ref else 0
    arguments = [("u_exact", ref.get("u_exact", 0), bool(ref)), ("u_crank_nicolson", case.false_grid, False)]
    spec = case.spec()
    errors = []
    ordinary = request(spec, "ordinary", errors)
    challenges = []
    for field, value, expected in arguments:
        result = request({**spec, "claimed_field": field, "claimed_value": value, "atol": claim_atol, "rtol": 0},
                         field, errors)
        challenges.append((field, value, expected, result))
    return index, case, ref, tol, ordinary, challenges, claim_atol, errors


def within(value, reference_value, tol):
    return type(value) in (int, float) and math.isfinite(value) and abs(value-reference_value) <= tol


def physical(index, case, ref, tol, out):
    problems = []
    if out.get("mode") != MODE or out.get("domain") != "pde":
        problems.append("route")
    if not ref:
        if not isinstance(out.get("reason"), str) or not out["reason"].strip():
            problems.append("missing range refusal reason")
        if any(k in out for k in FIELDS + ("spatial_cells", "time_steps", "error_measure", "claim_check")):
            problems.append("partial solution or comparison on range refusal")
        return problems
    for field, target in ref.items():
        value = out.get(field)
        if not within(value, target, tol[field]):
            problems.append(field)
        elif field in ("linf_error", "query_abs_error") and value < 0:
            problems.append(field + " sign")
    cells = out.get("spatial_cells")
    steps = out.get("time_steps")
    if type(cells) is not int or cells != case.cells:
        problems.append("spatial_cells")
    step_tolerance = STEP_ALLOWANCE if index == 14 else 0
    if type(steps) is not int or abs(steps-case.steps) > step_tolerance:
        problems.append("time_steps")
    if out.get("error_measure") != "whole-domain upper bound":
        problems.append("error_measure")
    return problems


def inspect(index, case, ref, tol, ordinary, challenges, claim_atol, errors):
    out, passed, is_json = ordinary
    failed = physical(index, case, ref, tol, out)
    if "claim_check" in out:
        failed.append("unexpected ordinary claim comparison")
    claims = []
    for field, value, expected, (reply, raw_passed, ok_wire) in challenges:
        faults = physical(index, case, ref, tol, reply)
        ck = reply.get("claim_check")
        evidence = "claim_check" not in reply if not ref else (
            isinstance(ck, dict) and ck.get("field") == field and ck.get("claimed") == value
            and ck.get("actual") == reply.get(field) and ck.get("within_tolerance") is expected
            and (expected or bool(reply.get("reason"))))
        claims.append({"field": field, "claimed": value, "atol": claim_atol, "expected": expected,
                       "passed": safe(raw_passed), "evidence": evidence, "physical_failures": faults,
                       "complete": raw_passed is expected and ok_wire and evidence and not faults})
    return {"case": case.name, "passed": safe(passed), "expected": bool(ref), "expected_values": ref,
            "values": {k: safe(out.get(k)) for k in FIELDS},
            "spatial_cells": safe(out.get("spatial_cells")), "time_steps": safe(out.get("time_steps")),
            "reason": safe(out.get("reason")), "physical_failures": failed, "errors": errors,
            "complete_unclaimed": passed is bool(ref) and is_json and not failed, "claims": claims,
            "full_control": index in FULL_CONTROLS,
            "full_sha256": hashlib.sha256(canonical(out).encode()).hexdigest() if is_json else None}


def twin_failures(rows):
    faults = []
    for i, j in ((1, 2), (3, 4), (4, 5), (3, 5), (6, 7), (8, 9)):
        a, b = rows[i-1], rows[j-1]
        ta = allowances(i, a["expected_values"])
        tb = allowances(j, b["expected_values"])
        for key in FIELDS:
            left = a["values"][key]
            right = b["values"][key]
            # Two responses independently meeting the same bounded reference
            # may differ by the sum of the two allowed deviations.
            if type(right) not in (int, float) or not within(left, right, ta[key]+tb[key]):
                faults.append([i, j, key])
    return faults


class HeatModalTimeTests(unittest.TestCase):
    def test_dimensionless_decay_and_shared_steps(self):
        with localcontext() as ctx:
            ctx.prec = 80
            originals = [collect(i, case) for i, case in enumerate(cases(), 1)]
        # Every route has already returned before any evidence verdict or assertion.
        rows = [inspect(*record) for record in originals]
        for row in rows:
            print("HEAT_TIME_CASE " + canonical(row))
        checks = [claim for row in rows for claim in row["claims"]]
        errors = [e for row in rows for e in row["errors"]]
        totals = {"ordinary": len(rows), "approved": sum(r["passed"] is True for r in rows),
                  "refused": sum(r["passed"] is False for r in rows),
                  "complete_unclaimed": sum(r["complete_unclaimed"] for r in rows),
                  "claims": len(checks), "claim_approvals": sum(c["passed"] is True for c in checks),
                  "claim_refusals": sum(c["passed"] is False for c in checks),
                  "complete_claims": sum(c["complete"] for c in checks),
                  "false_approvals": sum(c["passed"] is True and not c["expected"] for c in checks),
                  "false_refusals": sum(c["passed"] is False and c["expected"] for c in checks),
                  "full_controls": sum(r["full_control"] for r in rows), "twin_failures": twin_failures(rows),
                  "solver_errors": sum(e[1] == "solver" for e in errors),
                  "json_errors": sum(e[1] == "json" for e in errors)}
        print("HEAT_TIME_TOTAL " + canonical(totals))
        self.assertEqual(totals, {"ordinary": 14, "approved": 13, "refused": 1, "complete_unclaimed": 14,
                                  "claims": 28, "claim_approvals": 13, "claim_refusals": 15,
                                  "complete_claims": 28, "false_approvals": 0, "false_refusals": 0,
                                  "full_controls": 4, "twin_failures": [], "solver_errors": 0, "json_errors": 0})
