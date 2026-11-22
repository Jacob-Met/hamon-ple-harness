"""Closed-form modal, thermal, settlement and range checks; no stiffness-matrix oracle."""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


DISP, FORCE, STRESS = "displacements_xy_m", "axial_forces_N", "axial_stresses_Pa"
REACTION, U, WORK, RESIDUAL = "reactions_xy_N", "strain_energy_J", "external_work_J", "equilibrium_residual_N"
TRUSS_FIELDS = (DISP, FORCE, STRESS, REACTION, U, WORK, RESIDUAL)
BASIS = "represented_member_factors_before_float_rounding"
ZERO = (0, 0)
PLUS, MINUS = (3, 4), (4, -3)


def power(n):
    return Q(2) ** n


def scaled(s, v):
    return tuple(s*x for x in v)


def add(v, w):
    return tuple(x+y for x, y in zip(v, w))


def wire(value):
    if isinstance(value, Q):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [wire(x) for x in value]
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    return value


def canonical(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def safe(value):
    try:
        canonical(value)
    except (TypeError, ValueError, OverflowError):
        return {"unserializable": type(value).__name__, "repr": repr(value)[:160]}
    return value


@dataclass
class Case:
    name: str
    spec: dict
    expected: object = None
    refusal: str = ""
    range_field: str = ""
    range_index: tuple = ()
    exact: bool = False


def fixed(node, ux=0, uy=0):
    return {"node": node, "fix_x": True, "fix_y": True, "ux_m": ux, "uy_m": uy}


def two_bars(v, w, e_strong, e_weak, force):
    return {"mode": "truss_2d", "nodes": [scaled(-1, v), scaled(-1, w), ZERO],
            "elements": [{"i": 0, "j": 2, "E_pa": e_strong, "A_m2": 1},
                         {"i": 1, "j": 2, "E_pa": e_weak, "A_m2": 1}],
            "supports": [fixed(0), fixed(1)], "loads": [{"node": 2, "fx_N": force[0], "fy_N": force[1]}]}


def assembly_response(u, forces, reactions, energy, external):
    return {DISP: [ZERO, ZERO, u], FORCE: forces, STRESS: forces, REACTION: reactions, U: energy,
            WORK: external, RESIDUAL: 0}


def modal(label, a, b, *, e_weak=power(-60), flip=False, swap=False):
    force = add(scaled(a, PLUS), scaled(b, MINUS))
    spec = two_bars(PLUS, MINUS, 1, e_weak, force)
    if flip:
        for el in spec["elements"]:
            el["i"], el["j"] = el["j"], el["i"]
    if swap:
        spec["elements"].reverse()
    u = scaled(5, add(scaled(a, PLUS), scaled(b/e_weak, MINUS)))
    n = [5*a, 5*b]
    if swap:
        n.reverse()
    energy = Q(125, 2)*(a*a+b*b/e_weak)
    return Case(label, spec, assembly_response(u, n, [scaled(-a, PLUS), scaled(-b, MINUS), ZERO], energy, energy))


def horizontal(label, length, modulus, area, force, expected=None, field="", index=()):
    spec = {"mode": "truss_2d", "nodes": [ZERO, (length, 0)],
            "elements": [{"i": 0, "j": 1, "E_pa": modulus, "A_m2": area}],
            "supports": [fixed(0), {"node": 1, "fix_y": True}],
            "loads": [{"node": 1, "fx_N": force}]}
    if expected is None:
        return Case(label, spec, refusal="range", range_field=field, range_index=index)
    ux, stress, energy = expected
    return Case(label, spec, {DISP: [ZERO, (ux, 0)], FORCE: [force], STRESS: [stress],
                REACTION: [(-force, 0), ZERO], U: energy, WORK: energy, RESIDUAL: 0}, exact=True)


def cases():
    soft = power(-60)
    yield modal("pythagorean/pure-weak", Q(0), power(-70))
    yield modal("pythagorean/pure-strong", power(-10), Q(0))
    for label, flip, swap in (("mixed", False, False), ("reversed-endpoints", True, False),
                              ("reversed-members", False, True)):
        yield modal("pythagorean/" + label, power(-70), power(-69), flip=flip, swap=swap)
    for label, load in (("compatible", scaled(power(-10), PLUS)), ("incompatible", scaled(power(-70), MINUS))):
        spec = two_bars(PLUS, MINUS, 1, soft, load)
        spec["elements"].pop()
        yield Case("mechanism/" + label, spec, refusal="mechanism")
    strain = power(-12)
    motion = scaled(strain, MINUS)
    spec = two_bars(PLUS, MINUS, 1, soft, ZERO)
    spec["elements"][1].update(alpha_per_K=strain, dT_K=1)
    yield Case("thermal/free-expansion", spec, assembly_response(motion, [0, 0], [ZERO]*3, 0, 0))
    spec = two_bars(PLUS, MINUS, 1, soft, ZERO)
    spec["supports"].append(fixed(2, *motion))
    n = soft*strain
    apex = scaled(n/5, MINUS)
    yield Case("displacement/fully-prescribed", spec,
               assembly_response(motion, [0, n], [ZERO, scaled(-1, apex), apex], Q(5, 2)*soft*strain*strain, 0))
    yield modal("ordinary/equal-stiffness", power(-10), power(-11), e_weak=Q(1))

    v, w = (1, 1), (1, -1)
    # For the represented row w, ||w||²=2; the hypot length is a separate factor.
    length = Q.from_float(float.fromhex("0x1.6a09e667f3bcdp+0"))
    b = power(-70)
    spec = two_bars(v, w, 1, soft, scaled(b, w))
    u = scaled(b*length**3/(2*soft), w)
    energy = b*b*length**3/(2*soft)
    yield Case("irrational/rounded-length", spec,
               assembly_response(u, [0, b*length], [ZERO, scaled(-b, w), ZERO], energy, energy))
    spec = {"mode": "euler_bernoulli_beam_1d", "x_nodes_m": [0, 1], "EI_Nm2": 1,
            "supports": [{"node": 0, "fix_w": True, "fix_theta": True}],
            "nodal_loads": [{"node": 1, "fy_N": 1}]}
    yield Case("beam/cantilever", spec, {"deflections_m": [0, Q(1, 3)], "rotations_rad": [0, Q(1, 2)],
               "tip_deflection_m": Q(1, 3), "reactions_V_M": [(-1, -1), ZERO], U: Q(1, 6), WORK: Q(1, 6)})
    u_min = power(-1074)
    yield horizontal("wire/minimum-subnormal-force-energy", 1024, power(-1065), 1, u_min,
                     expected=(2, u_min, u_min))
    yield horizontal("wire/overflowed-EA", 1, power(1023), power(1023), power(972),
                     expected=(u_min, power(-51), power(-103)))
    yield horizontal("wire/energy-underflow", 1, 1, 1, u_min, field=U)
    yield horizontal("wire/displacement-overflow", 1, power(-1022), 1, power(100), field=DISP, index=(1, 0))


def leaves(item):
    if isinstance(item, (tuple, list)):
        return [v for x in item for v in leaves(x)]
    return [item]


def allowed_error(expected, zero_scale, exact):
    value = float(expected)
    if exact:
        return 0.0
    if value:
        return max(16*math.ulp(value), 2.0**-42*abs(value))
    return 2.0**-42*float(zero_scale)


def matches(actual, expected, scale, exact):
    if isinstance(expected, (tuple, list)):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(matches(v, e, scale, exact) for v, e in zip(actual, expected)))
    return (type(actual) in (float, int) and math.isfinite(actual)
            and abs(actual-float(expected)) <= allowed_error(expected, scale, exact))


def refusal_ok(case, output):
    reason = output.get("reason")
    if (output.get("passed") is not False or not isinstance(reason, str) or not reason
            or "claim_check" in output or any(k in output for k in TRUSS_FIELDS)):
        return False
    if case.refusal == "mechanism":
        return (output.get("response_status") != "outside_float_range" and
                any(x in reason.lower() for x in ("singular", "mechanism", "insufficient support")))
    required = {"passed", "domain", "mode", "reason", "response_status", "range_field", "range_index"}
    index = output.get("range_index")
    return (set(output) == required and output.get("domain") == "fem" and output.get("mode") == "truss_2d"
            and output.get("response_status") == "outside_float_range" and output.get("range_field") == case.range_field
            and isinstance(index, list) and all(type(i) is int for i in index) and index == list(case.range_index))


def model_checks(case, output):
    if case.refusal:
        return [] if refusal_ok(case, output) else [case.refusal + " refusal"]
    failed = []
    for key, truth in case.expected.items():
        scale = max((abs(v) for v in leaves(truth)), default=0)
        if key == RESIDUAL:
            applied = [abs(ld.get(tag, 0)) for ld in case.spec["loads"] for tag in ("fx_N", "fy_N")]
            scale = max([0, *applied, *(abs(v) for v in leaves(case.expected[REACTION]))])
        if not matches(output.get(key), truth, scale, case.exact or (key in (U, WORK) and truth == 0)):
            failed.append(key)
    if case.spec["mode"] == "truss_2d" and output.get("equilibrium_residual_basis") != BASIS:
        failed.append("equilibrium residual provenance")
    if case.name == "pythagorean/pure-strong" and DISP not in failed:
        ux, uy = output[DISP][2]
        scale = max(abs(v) for v in case.expected[DISP][2])
        if abs(4*ux-3*uy) > 2.0**-42*5*float(scale):
            failed.append("transverse displacement leakage")
    return failed


def query(spec, label, errors):
    try:
        output = run_router_conservation_cad_verifier({"mode": "fem", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(output)
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
    return output


def exercise(case):
    spec = wire(case.spec)
    errors = []
    ordinary = query(spec, "unclaimed", errors)
    key = "tip_deflection_m" if spec["mode"] == "euler_bernoulli_beam_1d" else U
    claims = [(0, 0., False), (1, 0., False)]
    if not case.refusal:
        target = float(case.expected[key])
        wrong = 2*target if target else 1.
        atol = min(allowed_error(target, 0, case.exact), abs(wrong-target)/4)
        claims = [(target, atol, True), (wrong, atol, False)]
    decisions, claim_failures = [], []
    false_approvals = false_refusals = 0
    for i, (value, atol, approved) in enumerate(claims, 1):
        output = query({**spec, "claimed_field": key, "claimed_value": value, "atol": atol, "rtol": 0},
                       f"claim{i}", errors)
        passed = output.get("passed")
        decisions.append("A" if passed is True else "R" if passed is False else "E")
        false_approvals += int(passed is True and not approved)
        false_refusals += int(passed is False and approved)
        if case.refusal:
            evidence = refusal_ok(case, output)
        else:
            ck = output.get("claim_check") or {}
            evidence = (matches(output.get(key), case.expected[key], 0, case.exact)
                        and ck.get("field") == key and ck.get("claimed") == value
                        and ck.get("actual") == output.get(key) and ck.get("within_tolerance") is approved
                        and (approved or bool(output.get("reason"))))
        if passed is not approved or not evidence:
            claim_failures.append(i)
    failed = model_checks(case, ordinary)
    if ordinary.get("passed") is not (not bool(case.refusal)):
        failed.append("unclaimed verdict")
    full_hash = None if any(e[0] == "unclaimed" for e in errors) else hashlib.sha256(canonical(ordinary).encode()).hexdigest()
    return {"case": case.name, "passed": safe(ordinary.get("passed")), "reason": safe(ordinary.get("reason")),
            "range_field": safe(ordinary.get("range_field")), "range_index": safe(ordinary.get("range_index")),
            "full_sha256": full_hash, "failed_checks": failed, "decisions": "".join(decisions),
            "claim_failures": claim_failures, "false_approvals": false_approvals, "false_refusals": false_refusals,
            "errors": errors}


class TrussMemberFactorTests(unittest.TestCase):
    def test_represented_member_response(self):
        rows = [exercise(case) for case in cases()]
        for row in rows:
            print("TRUSS_FACTOR_CASE " + canonical(row))
        counts = {"systems": len(rows), "admitted": sum(r["passed"] is True for r in rows),
                  "refused": sum(r["passed"] is False for r in rows),
                  "complete_unclaimed": sum(not r["failed_checks"] for r in rows),
                  "claims": sum(len(r["decisions"]) for r in rows),
                  "approvals": sum(r["decisions"].count("A") for r in rows),
                  "refusals": sum(r["decisions"].count("R") for r in rows),
                  "failed_claims": sum(len(r["claim_failures"]) for r in rows),
                  "false_approvals": sum(r["false_approvals"] for r in rows),
                  "false_refusals": sum(r["false_refusals"] for r in rows),
                  "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        print("TRUSS_FACTOR_TOTAL " + canonical(counts))
        self.assertEqual(counts, {"systems": 16, "admitted": 12, "refused": 4, "complete_unclaimed": 16,
                                  "claims": 32, "approvals": 12, "refusals": 20, "failed_claims": 0,
                                  "false_approvals": 0, "false_refusals": 0, "solver_errors": 0, "json_errors": 0})
