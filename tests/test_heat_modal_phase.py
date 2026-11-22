"""Analytic spatial and time references for a Dirichlet rod at frozen virtual grids."""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


N, J = 2**53 - 2, 2**53 - 1
MODE = "heat_1d_transient"
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117")
VARYING = {"passed", "u_exact", "u_crank_nicolson", "query_abs_error", "linf_error"}
FULL_CONTROLS = {4, 8, 9, 11, 13, 14, 15, 16}
MODE_REASON = "Heat modes require integer harmonics and finite amplitudes"


@dataclass(frozen=True)
class Case:
    name: str
    modes: tuple
    x: object
    # At the two finite times, spatial_* are the signed initial samples.
    spatial_exact: object = 0
    spatial_grid: object = 0
    cells: int = 0
    steps: int = 0
    length: int = 1
    time: float = 0
    reason: str = ""

    def spec(self):
        spec = {"mode": MODE, "alpha_m2_s": 1, "length_m": self.length, "t_eval_s": self.time,
                "modes": [{"n": n, "amplitude": a} for n, a in self.modes]}
        if self.x is not None:
            spec["x_eval_m"] = self.x
        return spec


def cases():
    # J = 6k+1. C*x/L = 64k+10+2/3, so the two node angles, modulo 2π,
    # are 5π/16 and 11π/32. Half-angle identities give their positive sines.
    one, two = D(1), D(2)
    s = ((one - one/two.sqrt())/two).sqrt()
    left = ((one+s)/two).sqrt()
    right = ((one+((one-s)/two).sqrt())/two).sqrt()
    high = ((N, 1),)
    low = ((1, 1),)
    larger = 2**54 + 4
    return [
        Case("phase/safe-positive", high, .75, 1, 1, 32*N),
        Case("phase/signed-mode", ((-N, 1),), .75, -1, -1, 32*N),
        Case("phase/reflection", high, .25, -1, -1, 32*N),
        Case("boundary/left", high, 0, 0, 0, 32*N),
        Case("boundary/right", high, 1, 0, 0, 32*N),
        Case("interpolation/non-dyadic-position-ratio", ((J, 1),), 1,
             D(3).sqrt()/2, (left+2*right)/3, 32*J, length=3),
        Case("time/positive-high-mode", high, .75, 1, 1, 32*N, 1920, time=2.0**-110),
        Case("ordinary/midpoint", low, .5, 1, 1, 120),
        Case("ordinary/decaying-midpoint", low, .5, 1, 1, 120, 120, time=.125),
        Case("modes/signed-zero-duplicates", ((N, 1), (N, -.25), (-N, .5), (0, .125)),
             .75, D(".25"), D(".25"), 32*N),
        Case("defaults/empty-modes", (), None, 100, 100, 120),
        Case("phase/exact-large-authoring-witness", ((larger, 1),), .375, -1, -1, 32*larger),
        Case("invalid/nonintegral-mode", ((1.5, 1),), .5, reason=MODE_REASON),
        Case("invalid/boolean-mode", ((True, 1),), .5, reason=MODE_REASON),
        Case("invalid/unrepresentable-integer-mode", ((2**53+1, 1),), .5, reason=MODE_REASON),
        Case("invalid/outside-rod", low, 2, reason="Invalid 1D transient heat parameters"),
    ]


def sine_series(z):
    # The finite-time cases have |z| ≤ π/64. The first omitted term is
    # smaller than 1e-90, below the working precision.
    term = total = z
    for k in range(1, 100):
        term *= -z*z / ((2*k)*(2*k+1))
        if abs(term) < D("1e-90"):
            break
        total += term
    return total


def reference(case):
    modes = case.modes or ((1, 100),)
    allowance = 5e-13 * max(1, sum(abs(a) for _, a in modes))
    continuum, grid = D(case.spatial_exact), D(case.spatial_grid)
    bound = D(0)
    for harmonic, amplitude in modes:
        n, c = D(harmonic), D(case.cells)
        damp = discrete = D(1)
        if case.time:
            # The frozen finite-time cases each have one unit-amplitude mode.
            t = D.from_float(case.time)
            damp = (-(n*PI/case.length)**2*t).exp()
            k = 4*t*(c/case.length)**2*sine_series(n*PI/(2*c))**2
            q = k/(2*case.steps)
            discrete = ((1-q)/(1+q)) ** case.steps
            continuum *= damp
            grid *= discrete
        bound += abs(D(amplitude)) * (abs(damp-discrete) + damp*(PI*n/c)**2/8)
    err = abs(grid-continuum)
    return {"u_exact": float(continuum), "u_crank_nicolson": float(grid),
            "query_abs_error": float(err), "linf_error": float(max(bound, err))}, allowance


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def printable(value):
    try:
        canonical(value)
    except (TypeError, ValueError, OverflowError):
        return {"type": type(value).__name__, "repr": repr(value)[:180]}
    return value


def request(spec, label, errors):
    try:
        raw = run_router_conservation_cad_verifier({"mode": "pde", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:160]])
        return {}, None, False
    result = raw if isinstance(raw, dict) else {}
    verdict = result.get("passed")
    try:
        canonical(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:160]])
        return result, verdict, False
    return result, verdict, isinstance(raw, dict)


def claims(case, ref, allowance):
    if case.reason:
        return [("u_exact", 0, 0, False), ("u_crank_nicolson", 1, 0, False)]
    if case.name.startswith("boundary/"):
        wrong, atol = 1, 0
    else:
        wrong = ref["u_exact"] if case.name.startswith("interpolation/") else 0
        atol = min(allowance, abs(ref["u_crank_nicolson"]-wrong)/8)
    return [("u_exact", ref["u_exact"], atol, True), ("u_crank_nicolson", wrong, atol, False)]


def near(value, target, allowance):
    return type(value) in (int, float) and math.isfinite(value) and abs(value-target) <= allowance


def input_refusal(result, case):
    return (result.get("reason") == case.reason and "claim_check" not in result
            and not ({"u_exact", "u_crank_nicolson"} & result.keys()))


def physical_checks(case, result, ref, allowance):
    failures = []
    if result.get("domain") != "pde" or result.get("mode") != MODE:
        failures.append("route")
    if case.reason:
        if not input_refusal(result, case):
            failures.append("input refusal")
        return failures
    for field, target in ref.items():
        endpoint_zero = case.name.startswith("boundary/") and field != "linf_error"
        if not near(result.get(field), target, 0 if endpoint_zero else allowance):
            failures.append(field)
        elif field in ("query_abs_error", "linf_error") and result[field] < 0:
            failures.append(field + " sign")
    for field, value in (("spatial_cells", case.cells), ("time_steps", case.steps)):
        if type(result.get(field)) is not int or result[field] != value:
            failures.append(field)
    if result.get("error_measure") != "whole-domain upper bound":
        failures.append("error_measure")
    return failures


def exercise(index, case):
    spec = case.spec()
    ref, allowance = ({}, 0) if case.reason else reference(case)
    expected_claims = claims(case, ref, allowance)
    errors = []
    ordinary, verdict, valid = request(spec, "ordinary", errors)
    # Capture both actual requests before inspecting either result.
    observations = [
        request({**spec, "claimed_field": field, "claimed_value": value, "atol": atol, "rtol": 0},
                field, errors)
        for field, value, atol, _ in expected_claims
    ]
    problems = physical_checks(case, ordinary, ref, allowance)
    if "claim_check" in ordinary:
        problems.append("unexpected unclaimed comparison")
    admitted = not bool(case.reason)
    records = []
    for (field, value, atol, approved), (reply, raw_verdict, ok_wire) in zip(expected_claims, observations):
        ck = reply.get("claim_check")
        if case.reason:
            evidence = input_refusal(reply, case)
        else:
            evidence = (isinstance(ck, dict) and ck.get("field") == field and ck.get("claimed") == value
                        and ck.get("actual") == reply.get(field) and ck.get("within_tolerance") is approved
                        and (approved or bool(reply.get("reason"))))
        physical = physical_checks(case, reply, ref, allowance)
        records.append({"field": field, "value": value, "atol": atol, "expected": approved,
                        "passed": printable(raw_verdict), "evidence": evidence, "physical_failures": physical,
                        "complete": raw_verdict is approved and evidence and not physical and ok_wire})
    projection = {k: v for k, v in ordinary.items() if k not in VARYING}
    return {"case": case.name, "expected": admitted, "passed": printable(verdict),
            "unclaimed_complete": verdict is admitted and not problems and valid,
            "physical_failures": problems, "claims": records, "errors": errors,
            "values": {k: printable(ordinary.get(k)) for k in ref},
            "expected_values": ref, "full_control": index in FULL_CONTROLS,
            "full_sha256": digest(ordinary) if valid else None,
            "projection_sha256": digest(projection) if valid else None}


class HeatModalPhaseTests(unittest.TestCase):
    def test_rod_phase_and_virtual_grid(self):
        with localcontext() as context:
            context.prec = 80
            rows = [exercise(i, case) for i, case in enumerate(cases(), 1)]
        for row in rows:
            print("HEAT_PHASE_CASE " + canonical(row))
        checks = [c for row in rows for c in row["claims"]]
        errors = [e for row in rows for e in row["errors"]]
        totals = {
            "systems": len(rows), "approved": sum(r["passed"] is True for r in rows),
            "refused": sum(r["passed"] is False for r in rows),
            "complete_unclaimed": sum(r["unclaimed_complete"] for r in rows),
            "claims": len(checks), "claim_approvals": sum(c["passed"] is True for c in checks),
            "claim_refusals": sum(c["passed"] is False for c in checks),
            "complete_claims": sum(c["complete"] for c in checks),
            "false_approvals": sum(c["passed"] is True and not c["expected"] for c in checks),
            "false_refusals": sum(c["passed"] is False and c["expected"] for c in checks),
            "full_controls": sum(r["full_control"] for r in rows),
            "solver_errors": sum(e[1] == "solver" for e in errors),
            "json_errors": sum(e[1] == "json" for e in errors),
        }
        print("HEAT_PHASE_TOTAL " + canonical(totals))
        self.assertEqual(totals, {
            "systems": 16, "approved": 12, "refused": 4, "complete_unclaimed": 16,
            "claims": 32, "claim_approvals": 12, "claim_refusals": 20, "complete_claims": 32,
            "false_approvals": 0, "false_refusals": 0, "full_controls": 8,
            "solver_errors": 0, "json_errors": 0,
        })
