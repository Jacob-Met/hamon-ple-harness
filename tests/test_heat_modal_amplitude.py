"""First-cell heat amplitudes from exact dyadic scales and independent Decimal100 formulae."""
from dataclasses import dataclass
from decimal import Decimal as D, localcontext
from fractions import Fraction
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


MODE = "heat_1d_transient"
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647")
MINIMUM = 2.0**-1074
TEMPERATURES = ("u_exact", "u_crank_nicolson")
FIELDS = TEMPERATURES + ("query_abs_error", "linf_error")
EXACT = {3, 4, 11, 12, 13, 15}
FULL_CONTROLS = {2, 4, 6, 10, 11, 12, 13, 15}


@dataclass(frozen=True)
class Case:
    name: str
    length: float
    x: float
    amplitude: float
    # First-cell z=x/L and |b|=|A|z, tabulated independently of the solver.
    z_exponent: object = None
    b_exponent: object = None
    n: int = 1
    alpha: int = 1
    time: float = 0
    copies: int = 1
    false_value: float = 0

    def spec(self):
        return {"mode": MODE, "length_m": self.length, "x_eval_m": self.x,
                "alpha_m2_s": self.alpha, "t_eval_s": self.time,
                "modes": [{"n": self.n, "amplitude": self.amplitude} for _ in range(self.copies)]}


def models():
    return (
        Case("phase/large-scale-vanishing-ratio", 2.0**100, 2.0**-1022, 2.0**1022, -1122, -100),
        Case("ordinary/linear-twin-100", 1, 2.0**-100, 1, -100, -100),
        Case("subnormal/two-ulp-displays", 2, MINIMUM, 1, -1075, -1075),
        Case("ordinary/normal-factors-subnormal-result", 1, 2.0**-100, 2.0**-975, -100, -1075),
        Case("weight/vanishing-first-cell-weight", 256, MINIMUM, 2.0**1023, -1082, -59),
        Case("ordinary/linear-twin-59", 1, 2.0**-59, 1, -59, -59),
        Case("sign/negative-harmonic", 256, MINIMUM, 2.0**1023, -1082, -59, n=-1),
        Case("sign/negative-amplitude-before-weight-rounding", 2, MINIMUM, -2.0**1023,
             -1075, -52, false_value=-2.0**-50),
        Case("time/decaying-scaled-spatial-sample", 2, MINIMUM, 2.0**1023, -1075, -52, alpha=4, time=.125),
        Case("ordinary/decaying-linear-twin", 1, 2.0**-52, 1, -52, -52, time=.125),
        Case("boundary/left-with-large-amplitude", 256, 0, 2.0**1023, false_value=1),
        Case("boundary/right-with-large-amplitude", 256, 256, 2.0**1023, false_value=1),
        Case("range/true-final-underflow", 8, MINIMUM, 1, -1077, -1077, false_value=MINIMUM),
        Case("range/true-summed-overflow", 1, .5, 2.0**1023, copies=2, false_value=1),
        Case("ordinary/unit-midpoint", 1, .5, 1),
    )


def dyadic(exponent):
    q = Fraction(2)**exponent
    return D(q.numerator) / D(q.denominator)


def sinc(x):
    # All nonzero arguments here are at most pi/120. This alternating
    # series bounds truncation by the first omitted term.
    total = term = D(1)
    for k in range(1, 100):
        term *= -x*x / ((2*k)*(2*k+1))
        if abs(term) < D("1e-110"):
            break
        total += term
    return total


def reference(i, case):
    if i == 14:
        # The two exact peak contributions each equal 2^1023.
        return {}, {}
    c = 120
    damp = discrete = D(1)
    if case.time:
        damp = (-(PI**2)/8).exp()
        v = PI/(2*c)
        k = D(4)*c*c/8 * (v*sinc(v))**2
        q = k/240
        discrete = ((1-q)/(1+q))**120

    if i in (11, 12):
        continuum = grid = D(0)
    elif i == 15:
        continuum = grid = D(1)
    else:
        z = dyadic(case.z_exponent)
        b = dyadic(case.b_exponent)
        if (case.n < 0) != (case.amplitude < 0):
            b = -b
        continuum = b*PI*sinc(PI*z)*damp
        # 120 sin(pi/120) = pi sinc(pi/120).
        grid = b*PI*sinc(PI/c)*discrete

    error = abs(continuum-grid)
    bound = abs(D(case.amplitude)) * (abs(discrete-damp) + damp*(PI/c)**2/8)
    ref = {"u_exact": float(continuum), "u_crank_nicolson": float(grid),
           "query_abs_error": float(error), "linf_error": float(max(error, bound))}
    # Independent half-ULP proofs in PLAN determine each subnormal result.
    if i in (3, 4):
        ref.update(u_exact=2*MINIMUM, u_crank_nicolson=2*MINIMUM)
    elif i == 13:
        ref.update(u_exact=0., u_crank_nicolson=0.)
    if ref["u_exact"] == ref["u_crank_nicolson"]:
        ref["query_abs_error"] = 0.
    tol = {f: (0 if i in EXACT else 2e-13*abs(ref[f])) for f in TEMPERATURES}
    tol["query_abs_error"] = (0 if ref["query_abs_error"] == 0 else
                              2e-13*max(abs(ref[f]) for f in TEMPERATURES))
    tol["linf_error"] = (float(D("2e-13")*abs(D(case.amplitude))*max(damp, discrete))
                         if case.time else 2e-13*ref["linf_error"])
    return ref, tol


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def printable(value):
    try:
        canonical(value)
    except (TypeError, ValueError, OverflowError):
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
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:160]])
        return out, verdict, False
    return out, verdict, isinstance(raw, dict)


def collect(i, case):
    ref, tol = reference(i, case)
    spec = case.spec()
    errors = []
    ordinary = request(spec, "ordinary", errors)
    challenges = []
    for field, value, expected in (("u_exact", ref.get("u_exact", 0), bool(ref)),
                                   ("u_crank_nicolson", case.false_value, False)):
        atol = min(tol[field], abs(ref["u_crank_nicolson"]-case.false_value)/8) if ref else 0
        output = request({**spec, "claimed_field": field, "claimed_value": value, "atol": atol, "rtol": 0},
                         field, errors)
        challenges.append((field, value, expected, atol, output))
    return i, case, ref, tol, ordinary, challenges, errors


def within(value, target, tolerance):
    return type(value) in (int, float) and math.isfinite(value) and abs(value-target) <= tolerance


def physical(case, out, ref, tol):
    faults = []
    if out.get("domain") != "pde" or out.get("mode") != MODE:
        faults.append("route")
    if not ref:
        if not isinstance(out.get("reason"), str) or not out["reason"].strip():
            faults.append("missing final-range reason")
        if any(k in out for k in FIELDS + ("spatial_cells", "time_steps", "error_measure", "claim_check")):
            faults.append("partial solution or comparison")
        return faults
    for field in FIELDS:
        val = out.get(field)
        if not within(val, ref[field], tol[field]):
            faults.append(field)
        elif field in ("linf_error", "query_abs_error") and val < 0:
            faults.append(field + " sign")
    for field, expected in (("spatial_cells", 120), ("time_steps", 120 if case.time else 0)):
        if type(out.get(field)) is not int or out[field] != expected:
            faults.append(field)
    if out.get("error_measure") != "whole-domain upper bound":
        faults.append("error_measure")
    return faults


def inspect(i, case, ref, tol, ordinary, challenges, errors):
    out, raw_verdict, ok_wire = ordinary
    faults = physical(case, out, ref, tol)
    if "claim_check" in out:
        faults.append("unexpected ordinary comparison")
    claims = []
    for field, value, expected, atol, (result, verdict, valid) in challenges:
        ck = result.get("claim_check")
        claim_faults = physical(case, result, ref, tol)
        evidence = "claim_check" not in result if not ref else (
            isinstance(ck, dict) and ck.get("field") == field and ck.get("claimed") == value
            and ck.get("actual") == result.get(field) and ck.get("within_tolerance") is expected
            and (expected or bool(result.get("reason"))))
        claims.append({"field": field, "claimed": value, "expected": expected, "atol": atol,
                       "passed": printable(verdict), "evidence": evidence, "physical_failures": claim_faults,
                       "complete": verdict is expected and valid and evidence and not claim_faults})
    return {"case": case.name, "expected": bool(ref), "passed": printable(raw_verdict),
            "complete_unclaimed": raw_verdict is bool(ref) and ok_wire and not faults,
            "values": {k: printable(out.get(k)) for k in FIELDS}, "expected_values": ref, "tolerances": tol,
            "physical_failures": faults, "claims": claims, "errors": errors, "reason": printable(out.get("reason")),
            "full_control": i in FULL_CONTROLS,
            "full_sha256": hashlib.sha256(canonical(out).encode()).hexdigest() if ok_wire else None}


def comparison_failures(rows):
    failed = []
    for i, j in ((1, 2), (3, 4), (5, 6), (9, 10)):
        a, b = rows[i-1], rows[j-1]
        for field in TEMPERATURES:
            first, second = a["values"][field], b["values"][field]
            # The nonidentical analytic sines contribute a known negligible
            # reference separation. The first-cell grid references coincide.
            gap = abs(a["expected_values"][field]-b["expected_values"][field])
            tol = a["tolerances"][field] + b["tolerances"][field] + gap
            if type(second) not in (int, float) or not within(first, second, tol):
                failed.append([i, j, field])
    for field in TEMPERATURES:
        positive = rows[4]["values"][field]
        negative = rows[6]["values"][field]
        if type(positive) not in (int, float) or not within(negative, -positive, 0):
            failed.append([5, 7, field])
    return failed


class HeatModalAmplitudeTests(unittest.TestCase):
    def test_first_cell_spatial_amplitude(self):
        with localcontext() as ctx:
            ctx.prec = 100
            originals = [collect(i, case) for i, case in enumerate(models(), 1)]
        # All 45 registered requests have returned before any case verdict.
        rows = [inspect(*saved) for saved in originals]
        for row in rows:
            print("HEAT_AMPLITUDE_CASE " + canonical(row))
        claims = [c for row in rows for c in row["claims"]]
        errors = [e for row in rows for e in row["errors"]]
        totals = {"ordinary": len(rows), "approved": sum(r["passed"] is True for r in rows),
                  "refused": sum(r["passed"] is False for r in rows),
                  "complete_unclaimed": sum(r["complete_unclaimed"] for r in rows),
                  "claims": len(claims), "claim_approvals": sum(c["passed"] is True for c in claims),
                  "claim_refusals": sum(c["passed"] is False for c in claims),
                  "complete_claims": sum(c["complete"] for c in claims),
                  "false_approvals": sum(c["passed"] is True and not c["expected"] for c in claims),
                  "false_refusals": sum(c["passed"] is False and c["expected"] for c in claims),
                  "full_controls": sum(r["full_control"] for r in rows),
                  "comparison_failures": comparison_failures(rows),
                  "solver_errors": sum(e[1] == "solver" for e in errors),
                  "json_errors": sum(e[1] == "json" for e in errors)}
        print("HEAT_AMPLITUDE_TOTAL " + canonical(totals))
        self.assertEqual(totals, {"ordinary": 15, "approved": 14, "refused": 1, "complete_unclaimed": 15,
                                  "claims": 30, "claim_approvals": 14, "claim_refusals": 16,
                                  "complete_claims": 30, "false_approvals": 0, "false_refusals": 0,
                                  "full_controls": 8, "comparison_failures": [], "solver_errors": 0,
                                  "json_errors": 0})
