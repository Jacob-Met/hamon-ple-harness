"""Analytic frequency geometry for four rational transfer families and a constant control."""
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


GAIN = "gain_crossovers"
PHASE = "phase_crossovers"
UNITY = "unity_gain_intervals_rad_s"
NEGATIVE = "negative_real_intervals_rad_s"
POLES = "imaginary_axis_poles_rad_s"
HOLES = "removable_holes_rad_s"
FIELDS = ("w_gc_rad_s", "phase_margin_deg", "w_pc_rad_s", "gain_margin_db",
          GAIN, PHASE, UNITY, NEGATIVE, POLES, HOLES, "dc_gain", "dc_gain_status")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class System:
    name: str
    num: list
    den: list
    dc: float
    ga: list = field(default_factory=list)  # (exact squared frequency, phase margin)
    ph: list = field(default_factory=list)  # (exact squared frequency, gain margin)
    unity: list = field(default_factory=list)
    negative: list = field(default_factory=list)
    poles: list = field(default_factory=list)
    holes: list = field(default_factory=list)
    defined: bool = True


def systems():
    # Bode interprets the decimal strings of the transported floats as rationals.
    scales = [(Decimal("1e-140"), Decimal("1e200"), "high"),
              (Decimal("1e200"), Decimal("1e-140"), "low"),
              (Decimal(1), Decimal(1), "unit")]
    target_pm = 180 - math.degrees(math.atan2(4, 3))
    for a, b, label in scales:
        x = (Fraction(b) / Fraction(a)) ** 2
        yield System("one-pole/" + label, [5*b], [4*a, 3*b], 5/3, ga=[(x, target_pm)])
    for (a, b, label), pm, gm in zip(scales, (90, -90, 0), (-6800, 6800, 0)):
        # |G|=w; at w=B/A, G=-B/A. At w=1 the extreme angles round to +/-90.
        x = (Fraction(b) / Fraction(a)) ** 2
        yield System("phase/" + label, [a, -b, 0], [a, b], 0,
                     ga=[(Fraction(1), pm)], ph=[(x, gm)])
    for a, b, label in ((Decimal("1e-320"), Decimal("1e308"), "high"),
                        (Decimal("1e308"), Decimal("1e-320"), "subnormal")):
        # H(jw)=(B/2)/(B-A*x); H=+1,-1 at x=p/2,3p/2 respectively.
        p = Fraction(b) / Fraction(a)
        yield System("resonance/" + label, [b/2], [a, 0, b], .5,
                     ga=[(p/2, -180), (3*p/2, 0)], negative=[(p, None)], poles=[p])
    p = Fraction(10) ** 628
    split = [(Fraction(0), p), (p, None)]
    yield System("removable/undefined-feedback", [Decimal("-1e-320"), 0, Decimal("-1e308")],
                 [Decimal("1e-320"), 0, Decimal("1e308")], -1, unity=split, negative=split,
                 holes=[p], defined=False)
    yield System("constant/absent", [2, 2], [1, 1], 2)
    # G(jw)=w²(B-iAw)/(B+iAw). For positive w its angle is strictly
    # -2*atan(Aw/B), approaching but never attaining 0 or -pi.
    # Its one gain crossing is w=1; the tiny negative angle gives +180
    # after rounding the margin, even if its binary64 imaginary part is -0.
    a, b = Decimal("1e-140"), Decimal("1e200")
    yield System("phase-wrap/approach-from-below", [a, -b, 0, 0], [a, b], 0,
                 ga=[(Fraction(1), 180)])


def close_margin(value, expected):
    return type(value) in (float, int) and math.isclose(value, expected, rel_tol=1e-13, abs_tol=1e-10)


def event(expected_x, rendered, evidence):
    if not isinstance(evidence, dict) or "squared_interval" not in evidence:
        raise ValueError("missing frequency evidence")
    status = evidence.get("status")
    bounds = evidence["squared_interval"]
    if expected_x is None:
        if status != "unbounded" or bounds is not None or rendered is not None:
            raise ValueError("unbounded endpoint")
        return None
    if expected_x == 0:
        if status != "zero" or bounds != ["0", "0"] or type(rendered) is not float or rendered != 0:
            raise ValueError("zero endpoint")
        return None
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not str for v in bounds):
        raise ValueError("positive squared interval format")
    lo, hi = map(Fraction, bounds)
    if not 0 < lo <= expected_x <= hi:
        raise ValueError("squared interval misses analytic root")
    with localcontext() as ctx:
        ctx.prec = 100
        w = float((Decimal(expected_x.numerator) / Decimal(expected_x.denominator)).sqrt())
    reference_status = "above_float_range" if math.isinf(w) else "below_float_range" if w == 0 else "finite"
    if status != reference_status:
        raise ValueError("wrong range status")
    if status == "finite":
        if (type(rendered) not in (int, float) or not math.isfinite(rendered) or rendered <= 0
                or abs(rendered-w) > math.ulp(w)):
            raise ValueError("frequency differs by more than one binary64 ULP")
    elif rendered is not None:
        raise ValueError("out-of-range frequency must render null")
    return lo, hi


def inspect_system(system, result):
    failures = []
    evidence = result.get("bode_frequency_evidence") or {}
    if (evidence.get("coordinate") != "omega_squared_rad2_per_s2"
            or evidence.get("rendering_basis") != "isolating_interval_midpoint"):
        failures.append("coordinate/rendering basis")
    if result.get("passed") is not system.defined or (not system.defined and not result.get("reason")):
        failures.append("unclaimed decision")
    if (result.get("dc_gain_status") != "finite" or not close_margin(result.get("dc_gain"), system.dc)):
        failures.append("dc")
    for key, expected, margin, scalar_w, index_key in (
        (GAIN, system.ga, "phase_margin_deg", "w_gc_rad_s", "selected_gain_index"),
        (PHASE, system.ph, "gain_margin_db", "w_pc_rad_s", "selected_phase_index"),
        (POLES, system.poles, None, None, None), (HOLES, system.holes, None, None, None),
        (UNITY, system.unity, None, None, None), (NEGATIVE, system.negative, None, None, None),
    ):
        try:
            rows, sidecar = result[key], evidence[key]
            if not (isinstance(rows, list) and isinstance(sidecar, list)
                    and len(rows) == len(sidecar) == len(expected)):
                raise ValueError("row cardinality")
            last_hi = None
            for oracle, row, meta in zip(expected, rows, sidecar):
                if key in (UNITY, NEGATIVE):
                    if not (isinstance(row, list) and isinstance(meta, list) and len(row) == len(meta) == 2):
                        raise ValueError("interval endpoint cardinality")
                    for x, value, entry in zip(oracle, row, meta):
                        event(x, value, entry)
                    continue
                if margin:
                    x, truth = oracle
                    if not close_margin(row[margin], truth):
                        raise ValueError("incorrect margin")
                    bounds = event(x, row["w_rad_s"], meta)
                else:
                    bounds = event(oracle, row, meta)
                lo, hi = bounds
                if last_hi is not None and last_hi >= lo:
                    raise ValueError("events not in certified disjoint order")
                last_hi = hi
            if margin:
                selected = min(range(len(expected)), key=lambda i: abs(expected[i][1]), default=None)
                actual_index = evidence[index_key]
                if actual_index != selected or (selected is not None and type(actual_index) is not int):
                    raise ValueError("wrong selected row")
                expected_w = rows[selected]["w_rad_s"] if selected is not None else None
                expected_margin = rows[selected][margin] if selected is not None else None
                if result[scalar_w] != expected_w or result[margin] != expected_margin:
                    raise ValueError("selected scalar mismatch")
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            failures.append(key + ": " + str(exc))
    return failures


def call(spec, label, errors):
    try:
        result = run_router_conservation_cad_verifier({"mode": "control", "spec": spec})
    except Exception as exc:
        errors.append([label, "solver", type(exc).__name__, str(exc)[:120]])
        return {}
    try:
        canonical(result)
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append([label, "json", type(exc).__name__, str(exc)[:120]])
        return {}
    return result


def exercise(system):
    spec = {"mode": "transfer_function", "num": list(map(float, system.num)),
            "den": list(map(float, system.den))}
    errors = []
    result = call(spec, "unclaimed", errors)
    claims = [("w_gc_rad_s", 0, False), ("w_pc_rad_s", 0, False)]
    if system.ga:
        claims.append(("phase_margin_deg", min((m for _, m in system.ga), key=abs), True))
    if system.ph:
        claims.append(("gain_margin_db", system.ph[0][1], True))
    outcomes, correct = [], 0
    for name, target, should_pass in claims:
        claimed = call({**spec, "claimed_field": name, "claimed_value": target,
                        "atol": 1e-10 if should_pass else 0, "rtol": 0}, name, errors)
        decision = claimed.get("passed")
        outcomes.append("A" if decision is True else "R" if decision is False else "E")
        value = claimed.get(name)
        check = claimed.get("claim_check") or {}
        if value is None:
            details = name in claimed and not should_pass and bool(claimed.get("reason"))
        else:
            details = (check.get("field") == name and check.get("claimed") == float(target)
                       and check.get("actual") == value and check.get("within_tolerance") is should_pass
                       and (close_margin(value, target) if should_pass else value > 0 and bool(claimed.get("reason"))))
        correct += int(decision is should_pass and details)
    fields = {key: result[key] for key in FIELDS if key in result}
    failures = inspect_system(system, result)
    return {"case": system.name, "failed_checks": failures, "passed": result.get("passed"),
            "outcomes": "".join(outcomes), "correct_claims": correct, "errors": errors,
            "projection_fields": len(fields), "bode_dc_sha256": digest(fields)}


class BodeFrequencyRangeTests(unittest.TestCase):
    def test_analytic_frequency_geometry(self):
        rows = [exercise(system) for system in systems()]
        for row in rows:
            print("BODE_RANGE_CASE " + canonical(row))
        summary = {"systems": len(rows), "complete": sum(not r["failed_checks"] for r in rows),
                   "unclaimed_passed": sum(r["passed"] is True for r in rows),
                   "correct_claims": sum(r["correct_claims"] for r in rows),
                   "claims": sum(len(r["outcomes"]) for r in rows),
                   "approved": sum(r["outcomes"].count("A") for r in rows),
                   "refused": sum(r["outcomes"].count("R") for r in rows),
                   "solver_errors": sum(e[1] == "solver" for r in rows for e in r["errors"]),
                   "json_errors": sum(e[1] == "json" for r in rows for e in r["errors"])}
        print("BODE_RANGE_TOTAL " + canonical({**summary, "projection_sha256": digest(
            [[row["case"], row["projection_fields"], row["bode_dc_sha256"]] for row in rows])}))
        self.assertEqual(summary, {"systems": 11, "complete": 11, "unclaimed_passed": 10, "correct_claims": 34,
                                   "claims": 34, "approved": 12, "refused": 22, "solver_errors": 0, "json_errors": 0})
