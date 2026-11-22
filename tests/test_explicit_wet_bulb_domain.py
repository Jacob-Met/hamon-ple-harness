"""Separate moist-air energy feasibility from the public inverse's input-resolution convention."""
import hashlib
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


W = "w_kg_per_kgda"
TAG = "wet_bulb_dry_bracket_c"
# ASHRAE Fundamentals 2025 SI, chapter 1, Table 3: 103.24, 259.87 and 872.57 Pa.
# Brackets allow for differences between the tabulation and the maintained saturation model.
# https://handbook.ashrae.org/Handbooks/F25/SI/F25_Ch01/F25_Ch01_si.aspx
SAT_PA = {-20: (102, 105), -10: (258, 262), 5: (870, 875)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def observed(value):
    """Retain readable evidence when the original result fails strict JSON."""
    try:
        canonical(value)
    except (TypeError, ValueError, OverflowError):
        return {"unserializable_type": type(value).__name__, "repr": repr(value)[:160]}
    return value


def state(db, **inputs):
    return {"mode": "state", "t_db_c": db, **inputs}


def energy_bounds(db, wb, pressure):
    # Adiabatic saturation: h_in + (Ws - W)*h_cond(wb) = h_sat(wb).
    # h_v(T)=2501+1.86T; the specified ice model is h_i=-329+2.1T.
    condensate = 4.186*wb if wb >= 0 else -329+2.1*wb
    hv_w, hv_db = 2501+1.86*wb, 2501+1.86*db
    return [((.621945*p/(pressure-p))*(hv_w-condensate) - 1.006*(db-wb)) / (hv_db-condensate)
            for p in SAT_PA[wb]]


def call(name, spec, *, near=False):
    errors = []
    try:
        result = run_router_conservation_cad_verifier({"mode": "psychrometrics", "spec": spec})
    except Exception as exc:
        errors.append(["solver", type(exc).__name__, str(exc)[:120]])
        result = {}
    try:
        full = digest(result) if not errors else None
        projected = digest({k: v for k, v in result.items() if k != TAG}) if near and not errors else full
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append(["json", type(exc).__name__, str(exc)[:120]])
        full = projected = None
    row = {"case": name, "passed": observed(result.get("passed")), "errors": errors, "failed_checks": [],
           "full_sha256": full, "projection_sha256": projected, "has_bracket": TAG in result,
           "bracket": observed(result.get(TAG)), "w": observed(result.get(W)), "reason": observed(result.get("reason"))}
    return result, row


def refused(result):
    return result.get("passed") is False and bool(result.get("reason"))


def check(row, flag, label):
    if not flag:
        row["failed_checks"].append(label)


def process_cases():
    impossible = {"t_db_c": 40, "t_wb_c": 5}
    possible = {"t_db_c": 21, "t_wb_c": 5}
    second = {"t_db_c": 20, "rh": .25}
    for label, inlet, pressure, admitted in (("impossible", impossible, 101325, False),
                                            ("valid", possible, 70000, True)):
        yield (f"mix/{label}", {"mode": "adiabatic_mixing", "p_atm_pa": pressure,
               "stream1": {**inlet, "m_da_kg_s": 1}, "stream2": {**second, "m_da_kg_s": 1}}, admitted)
    # Dry outlet: the previously clipped dry inlet would incorrectly make this sensible-only coil admissible.
    for label, inlet, pressure, admitted in (("impossible", impossible, 101325, False),
                                            ("valid", possible, 70000, True)):
        yield (f"coil/{label}", {"mode": "cooling_coil", "p_atm_pa": pressure,
               "inlet": inlet, "outlet": {"t_db_c": 15, "rh": 0}, "m_da_kg_s": 1}, admitted)


def fixed_rows():
    rows = []
    claim_specs = []
    for label, db, wb, p, admitted in (("liquid/strong-negative", 40, 5, 101325, False),
                                        ("ice/strong-negative", 0, -20, 101325, False),
                                        ("pressure/sea-level", 21, 5, 101325, False),
                                        ("pressure/lower", 21, 5, 70000, True),
                                        ("ice/positive", -8, -10, 101325, True),
                                        ("liquid/saturated", 5, 5, 101325, True)):
        lo, hi = energy_bounds(db, wb, p)
        spec = state(db, t_wb_c=wb, p_atm_pa=p)
        result, row = call(label, spec)
        check(row, lo > 0 if admitted else hi < 0, "independent energy sign")
        if admitted:
            w = result.get(W)
            check(row, result.get("passed") is True and type(w) is float and lo <= w <= hi,
                  "humidity within tabulated energy bounds")
            if label == "liquid/saturated":
                rh = result.get("rh")
                check(row, type(rh) in (int, float) and math.isfinite(rh)
                      and math.isclose(rh, 1, abs_tol=1e-12, rel_tol=0),
                      "saturated explicit state")
        else:
            check(row, refused(result), "impossible state must refuse")
            claim_specs.append((label, spec, W, 0, False))
        check(row, TAG not in result, "unexpected dry bracket")
        rows.append(row)
    for label, spec in (("alternate/rh", state(20, rh=.5)), ("alternate/saturated", state(20, rh=1)),
                        ("alternate/dry-W", state(20, w_kg_per_kgda=0)),
                        ("alternate/dew-point", state(20, t_dp_c=5)),
                        ("alternate/RH-priority", state(20, rh=.5, t_wb_c=-40))):
        result, row = call(label, spec)
        check(row, result.get("passed") is True and TAG not in result, "alternate input")
        if label == "alternate/dry-W":
            check(row, result.get(W) == 0.0, "dry state")
            claim_specs.append((label, spec, W, 0, True))
        if label == "alternate/saturated":
            check(row, result.get("rh") == 1.0, "saturated state")
            claim_specs.append((label, spec, "rh", 1, True))
        if label == "alternate/RH-priority":
            ordinary = next(r for r in rows if r["case"] == "alternate/rh")
            check(row, row["full_sha256"] is not None and row["full_sha256"] == ordinary["full_sha256"],
                  "RH precedence")
        rows.append(row)
    for label, spec, admitted in process_cases():
        result, row = call(label, spec)
        check(row, result.get("passed") is True if admitted else refused(result), "process propagation")
        rows.append(row)
    for label, spec, field, value, approved in claim_specs:
        result, row = call("claim/" + label, {**spec, "claimed_field": field, "claimed_value": value,
                                              "rtol": 0, "atol": 0})
        ck = result.get("claim_check") or {}
        if approved:
            check(row, result.get("passed") is True and ck.get("field") == field and ck.get("actual") == value
                  and ck.get("claimed") == value and ck.get("within_tolerance") is True, "scalar claim")
        else:
            check(row, refused(result) and "claim_check" not in result, "invalid input precedes scalar comparison")
        rows.append(row)
    return rows


def boundary_rows(db, phase):
    prefix = "resolution/" + phase
    anchor, first = call(prefix + "/RH-zero", state(db, rh=0))
    t = anchor.get("t_wb_c")
    valid_anchor = (anchor.get("passed") is True and anchor.get(W) == 0.0 and type(t) is float
                    and math.isfinite(t) and ((1 < t < db) if db > 0 else (-60 < t < db)))
    check(first, valid_anchor and TAG not in anchor, "public dry inverse")
    first["inverse_wet_hex"] = t.hex() if valid_anchor else None
    if not valid_anchor:
        # Keep all five requests observable; this group still fails its anchor requirement.
        t = 15.0 if db > 0 else -12.0
    rows, answers = [first], {}
    probes = (("far-lower", t-1e-4), ("predecessor", math.nextafter(t, -math.inf)), ("center", t),
              ("successor", math.nextafter(t, math.inf)), ("far-upper", t+1e-4))
    for label, wet in probes:
        near = label in ("predecessor", "center", "successor")
        result, row = call(prefix + "/" + label, state(db, t_wb_c=wet), near=near)
        row["wet_input_hex"] = wet.hex()
        if TAG in result:
            w = result.get(W)
            check(row, near and result.get("passed") is True and type(w) is float and w == 0
                  and math.copysign(1, w) == 1 and type(result[TAG]) is list
                  and all(type(v) is float for v in result[TAG])
                  and result[TAG] == [wet, math.nextafter(wet, math.inf)],
                  "one-ULP bracket and positive zero")
        if label == "far-lower":
            check(row, refused(result) and TAG not in result, "farther-below refusal")
        if label == "far-upper":
            check(row, result.get("passed") is True and type(result.get(W)) is float and result[W] > 0
                  and TAG not in result, "farther-above positive humidity")
        if near:
            check(row, (result.get("passed") is True and type(result.get(W)) is float and result[W] >= 0)
                  or refused(result), "admission or explicit refusal")
        answers[label] = result
        rows.append(row)
    tagged = [label for label in ("predecessor", "center") if TAG in answers[label]]
    check(first, len(tagged) == 1, "exactly one lower endpoint among predecessor and center")
    if len(tagged) == 1:
        upper = answers["center" if tagged[0] == "predecessor" else "successor"]
        check(first, upper.get("passed") is True and TAG not in upper and type(upper.get(W)) is float
              and upper[W] >= 0, "immediate upper endpoint is admitted without a bracket")
    return rows


class ExplicitWetBulbDomainTests(unittest.TestCase):
    def test_energy_feasibility_and_input_resolution(self):
        static = fixed_rows()
        resolution = boundary_rows(40, "water") + boundary_rows(-10, "ice")
        rows = static + resolution
        for row in rows:
            print("WET_DOMAIN_CASE " + canonical(row))
        counts = {"fixed": len(static), "fixed_approvals": sum(r["passed"] is True for r in static),
                  "fixed_refusals": sum(r["passed"] is False for r in static),
                  "resolution": len(resolution), "calls": len(rows),
                  "failed_rows": sum(bool(r["failed_checks"]) for r in rows),
                  "solver_errors": sum(e[0] == "solver" for r in rows for e in r["errors"]),
                  "json_errors": sum(e[0] == "json" for r in rows for e in r["errors"])}
        print("WET_DOMAIN_TOTAL " + canonical({**counts,
              "resolution_approvals": sum(r["passed"] is True for r in resolution),
              "resolution_refusals": sum(r["passed"] is False for r in resolution)}))
        self.assertEqual(counts, {"fixed": 20, "fixed_approvals": 12, "fixed_refusals": 8,
                                  "resolution": 12, "calls": 32, "failed_rows": 0,
                                  "solver_errors": 0, "json_errors": 0})
