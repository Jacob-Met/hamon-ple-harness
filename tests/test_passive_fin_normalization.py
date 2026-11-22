"""Passive fin conductance from the unscaled, unit-drive ODE; compare both reported reference areas."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


ETA, EPS, META = "fin_efficiency", "fin_effectiveness", "fin_normalization"
METRICS = (ETA, EPS)
EVIDENCE = {"method": "passive_unit_drive", "efficiency_reference": "lateral_h_P_L",
            "effectiveness_reference": "bare_base_h_A", "length_reference": "provided_length_m",
            "zero_drive_policy": "undefined_heat_rate_ratio"}
MODES = ("adiabatic", "convective", "infinite")
U = float.fromhex("0x0.0000000000001p-1022")


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()


@dataclass
class System:
    name: str
    spec: dict
    exact_powers: object = None  # exact eta and eps are integer powers of two
    zero_flux: bool = False
    reject_old_pair: bool = False


def system(name, mode, drive, *, k=1, ac=1, p=4, h=1, length=.25, tip_h=8, tip_temp=0, **options):
    spec = {"mode": "convective_fin_1d", "tip_bc": mode, "k_w_mk": k, "area_m2": ac, "perimeter_m": p,
            "h_w_m2k": h, "length_m": length, "t_base_c": drive, "t_inf_c": 0, "x_eval_m": length/2}
    if mode == "convective":
        spec["h_tip_w_m2k"] = tip_h
    if mode == "prescribed_temp":
        spec["t_tip_c"] = tip_temp
    return System(name, spec, **options)


def systems():
    tiny = 2.0**-45
    for mode in MODES:
        for label, driver in (("ordinary", 8), ("tiny-positive", tiny), ("tiny-negative", -tiny), ("zero", 0)):
            yield system(f"drive/{mode}/{label}", mode, driver, reject_old_pair=label == "tiny-positive")
    for mode, label, threshold in (("adiabatic", "below", math.nextafter(1e-12, 0)),
                                   ("convective", "equal", 1e-12),
                                   ("infinite", "above", math.nextafter(1e-12, math.inf))):
        for sign, driver in (("positive", threshold), ("negative", -threshold)):
            yield system(f"threshold/{mode}/{label}/{sign}", mode, driver)
    for mode in MODES:
        for label, driver in (("positive", U), ("negative", -U)):
            # F=m=1/2 and H_tip=F for Robin; both matched and infinite give G=1/2.
            powers = (1, -1) if mode in ("convective", "infinite") else None
            yield system(f"zero-flux/{mode}/{label}", mode, driver, p=.25, length=1, tip_h=.5,
                         exact_powers=powers, zero_flux=True)
    yield system("matched/lateral-vs-full", "convective", 2, length=1, tip_h=2, exact_powers=(-1, 1))
    for name, k, ac, p, h, length, tip_h, drive, exponent in (
        ("below-range", -600, 600, -600, 600, 0, -600, -45, -1200),
        ("above-range", 600, -600, 600, -600, 0, 600, -45, 1200),
        ("minimum-subnormal", -537, 537, -537, 537, 0, -537, -45, -1074),
        ("finite-overflowed-product", 0, 600, -600, 600, 300, -300, -600, -900),
        ("near-maximum", 512, -512, 511, -511, 0, 512, -45, 1023),
    ):
        # In every row H_tip=F, z=1, hence G=F and eta=1. Bare-base eps=2^exponent.
        yield system("matched/" + name, "convective", 2.0**drive, k=2.0**k, ac=2.0**ac, p=2.0**p,
                     h=2.0**h, length=2.0**length, tip_h=2.0**tip_h, exact_powers=(0, exponent))
    yield system("prescribed/zero-base-hot-tip", "prescribed_temp", 0, length=1, tip_temp=1)
    yield system("prescribed/positive-base-hotter-tip", "prescribed_temp", 1, length=1, tip_temp=8)


def dyadic_target(exponent):
    # Powers of two have exactly known binary64 bit ranges, including 2^-1074.
    if exponent < -1074:
        return None, "below_float_range", 0.0
    if exponent > 1023:
        return None, "above_float_range", 0.0
    return math.ldexp(1.0, exponent), "finite", 0.0


def ratios(row):
    spec = row.spec
    if spec["t_base_c"] == 0:
        return [(None, "undefined_zero_base_drive", 0.0)] * 2
    if row.exact_powers is not None:
        return [dyadic_target(p) for p in row.exact_powers]

    with localcontext() as ctx:
        ctx.prec = 100
        def number(key):
            return Decimal.from_float(float(spec[key]))

        axial = number("k_w_mk") * number("area_m2")
        lateral = number("h_w_m2k") * number("perimeter_m")
        length = number("length_m")
        m = (lateral / axial).sqrt()
        flux_scale = axial * m
        z = m * length
        # General ODE solution theta(x)=C cosh(mx)+S sinh(mx), q_base=-kA*m*S.
        sh = (z.exp() - (-z).exp()) / 2
        ch = (z.exp() + (-z).exp()) / 2
        mode = spec["tip_bc"]
        if mode == "adiabatic":
            conductance = flux_scale * sh / ch
        elif mode == "convective":
            beta = number("h_tip_w_m2k") * number("area_m2") / flux_scale
            conductance = flux_scale * (sh + beta*ch) / (ch + beta*sh)
        elif mode == "infinite":
            conductance = flux_scale
        else:
            # The two prescribed endpoint temperatures drive independent heat.
            ratio = number("t_tip_c") / number("t_base_c")
            conductance = flux_scale * (ch - ratio) / sh
        values = [conductance / (lateral*length), conductance / (number("h_w_m2k")*number("area_m2"))]
        rendered = [float(v) for v in values]
    return [(v, "finite", 5e-13*abs(v)) for v in rendered]


def invoke(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "pde", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(result)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
    return result


def metric_ok(result, key, target, atol):
    if key not in result:
        return False
    if target is None:
        return result[key] is None
    actual = result[key]
    return type(actual) is float and math.isfinite(actual) and abs(actual-target) <= atol


def exercise(row):
    errors = []
    result = invoke(row.spec, "unclaimed", errors)
    targets = ratios(row)
    claims = [(key, v if v is not None else 0.0, atol, v is not None)
              for key, (v, status, atol) in zip(METRICS, targets)]
    if row.reject_old_pair:
        claims.extend(((ETA, 1, 0.0, False), (EPS, 0, 0.0, False)))
    decisions, claim_failures = [], []
    false_approvals = false_refusals = 0
    for idx, (key, value, atol, approval) in enumerate(claims):
        claimed = invoke({**row.spec, "claimed_field": key, "claimed_value": value, "rtol": 0, "atol": atol},
                         f"claim{idx+1}/{key}", errors)
        passed = claimed.get("passed")
        expected, status, tol = targets[METRICS.index(key)]
        decisions.append("A" if passed is True else "R" if passed is False else "E")
        false_approvals += int(passed is True and not approval)
        false_refusals += int(passed is False and approval)
        if expected is None:
            evidence = bool(claimed.get("reason")) and "claim_check" not in claimed
        else:
            ck = claimed.get("claim_check") or {}
            evidence = (ck.get("field") == key and ck.get("claimed") == value and
                        ck.get("actual") == claimed.get(key) and ck.get("within_tolerance") is approval)
        if passed is not approval or not evidence or not metric_ok(claimed, key, expected, tol):
            claim_failures.append(idx+1)

    physical_ok = result.get("passed") is True
    is_prescribed = row.spec["tip_bc"] == "prescribed_temp"
    failures = [] if physical_ok else ["underlying physical gate"]
    for key, (target, _, atol) in zip(METRICS, targets):
        if not metric_ok(result, key, target, atol):
            failures.append(key)
    if is_prescribed:
        q = result.get("q_fin_w")
        if META in result or type(q) is not float or not q < 0:
            failures.append("prescribed independence")
    else:
        meta = result.get(META) or {}
        for key, value in {**EVIDENCE, "efficiency_float_status": targets[0][1],
                           "effectiveness_float_status": targets[1][1]}.items():
            if meta.get(key) != value:
                failures.append(key)
    if row.zero_flux and result.get("q_fin_w") != 0.0:
        failures.append("physical q did not round to zero")
    if row.name == "matched/lateral-vs-full":
        # For drive two, G=2, q=4; full reference heat is 2*(4+2)=12.
        if result.get("q_fin_w") != 4 or result.get("q_max_ref_w") != 12:
            failures.append("full-area reference heat")
    projection = {k: v for k, v in result.items() if k not in (META, ETA, EPS)}
    has_unclaimed_error = any(e[0] == "unclaimed" for e in errors)
    return {"case": row.name, "unclaimed_passed": result.get("passed"), "failed_checks": failures,
            "expected_status": [v[1] for v in targets], "decisions": "".join(decisions),
            "claim_failures": claim_failures, "false_approvals": false_approvals, "false_refusals": false_refusals,
            "physical_fields": len(projection), "physical_sha256": None if has_unclaimed_error else digest(projection),
            "prescribed_full_sha256": digest(result) if is_prescribed and not has_unclaimed_error else None,
            "errors": errors}


class PassiveFinNormalizationTests(unittest.TestCase):
    def test_unit_drive_conductance(self):
        rows = [exercise(row) for row in systems()]
        for row in rows:
            print("PASSIVE_NORM_CASE " + canonical(row))
        totals = {"systems": len(rows), "unclaimed_passed": sum(r["unclaimed_passed"] is True for r in rows),
                  "complete": sum(not r["failed_checks"] for r in rows),
                  "claims": sum(len(r["decisions"]) for r in rows),
                  "claim_failures": sum(len(r["claim_failures"]) for r in rows),
                  "approvals": sum(r["decisions"].count("A") for r in rows),
                  "refusals": sum(r["decisions"].count("R") for r in rows),
                  "false_approvals": sum(r["false_approvals"] for r in rows),
                  "false_refusals": sum(r["false_refusals"] for r in rows),
                  "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        projection = digest([[r["case"], r["physical_sha256"]] for r in rows])
        print("PASSIVE_NORM_TOTAL " + canonical({**totals, "physical_sha256": projection}))
        self.assertEqual(totals, {"systems": 32, "unclaimed_passed": 32, "complete": 32, "claims": 70,
                                  "claim_failures": 0, "approvals": 54, "refusals": 16, "false_approvals": 0,
                                  "false_refusals": 0, "solver_errors": 0, "json_errors": 0})
