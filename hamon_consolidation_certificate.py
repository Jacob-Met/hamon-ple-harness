"""Certify reported 1-D consolidation quantities against two physical solutions."""
from fractions import Fraction
import math


CLAIMS = (
    ("proposed_settlement_m", "settlement_at_time_m", "settlement_reporting_increment_m", "m"),
    ("proposed_mean_excess_pore_pressure_Pa", "mean_excess_pore_pressure_Pa", "pressure_reporting_increment_Pa", "Pa"),
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _reports(value, computed, increment):
    if not _finite(value) or not _finite(computed):
        return False
    if increment is None:
        return value == computed or abs(value-computed)/max(abs(value), abs(computed)) < 1e-6
    # Use exact decimal grid arithmetic for ordinary engineering reporting. At
    # half-increments Python's round chooses the even multiple (ties to even).
    step = Fraction(str(increment))
    return Fraction(str(value)) == round(Fraction(str(computed))/step)*step


def _single_layer(spec, dispatch):
    result = dispatch("hamon_conservation_cad_verifier", {"mode": "civil_geotech", "spec": spec})
    if (result.get("calculation") != "consolidation_history_1d" or result.get("ok") is not True
            or result.get("conservation_verified") is not True
            or not _finite(result.get("dual_solver_rel_residual"))
            or abs(result["dual_solver_rel_residual"]) >= 1e-6):
        return None
    strain = spec["mv_per_Pa"] * result["finite_volume_mean_effective_increment_Pa"]
    return {**result, "finite_volume_settlement_at_time_m": strain*spec["H_clay_m"]}


def _drained_column(spec, dispatch):
    layers = spec["drained_layers"]
    top, bottom = spec["top_boundary"], spec["bottom_boundary"]
    if (spec["interface_condition"] != "fully_drained" or not 2 <= len(layers) <= 8
            or top not in ("drained", "sealed") or bottom not in ("drained", "sealed")):
        return None
    rows = []
    for i, layer in enumerate(layers):
        sealed_face = (i == 0 and top == "sealed") or (i == len(layers)-1 and bottom == "sealed")
        drainage = "single" if sealed_face else "double"
        result = _single_layer({**layer, "calculation": "consolidation_history_1d", "drainage": drainage,
                                "time_s": spec["time_s"], "load_history": spec["load_history"]}, dispatch)
        if result is None:
            return None
        row = {key: result[key] for key in ("settlement_at_time_m", "finite_volume_settlement_at_time_m",
               "mean_excess_pore_pressure_Pa", "finite_volume_mean_excess_pore_pressure_Pa", "dual_solver_rel_residual")}
        rows.append({**row, "thickness_m": layer["H_clay_m"], "drainage": drainage})
    height = math.fsum(row["thickness_m"] for row in rows)
    result = {"ok": True, "conservation_verified": True, "mode": "civil_geotech", "calculation": spec["calculation"],
              "interface_condition": "fully_drained", "layers": rows,
              "dual_solver_rel_residual": max(abs(row["dual_solver_rel_residual"]) for row in rows)}
    for prefix in ("", "finite_volume_"):
        pressure = prefix + "mean_excess_pore_pressure_Pa"
        settlement = prefix + "settlement_at_time_m"
        result[pressure] = math.fsum(row[pressure]*(row["thickness_m"]/height) for row in rows)
        result[settlement] = math.fsum(row[settlement] for row in rows)
    return result


def history_certificate(spec, dispatch):
    """Verify the physical response first, then each requested value in its stated SI unit."""
    report_keys = {key for claim, _, increment, _ in CLAIMS for key in (claim, increment)}
    physics = {key: value for key, value in spec.items() if key not in report_keys}
    if spec["calculation"] == "consolidation_history_1d":
        result = _single_layer(physics, dispatch)
    else:
        result = _drained_column(physics, dispatch)
    if result is None:
        return None
    verified = {}
    for claim, output, increment_key, unit in CLAIMS:
        increment = spec.get(increment_key)
        if increment_key in spec and (claim not in spec or not _finite(increment) or increment <= 0):
            return None
        if claim not in spec:
            continue
        value = spec[claim]
        if not all(_reports(value, result[prefix+output], increment) for prefix in ("", "finite_volume_")):
            return None
        verified[claim] = {"value": value, "unit": unit,
                           **({"reporting_increment": increment} if increment is not None else {"relative_tolerance": 1e-6})}
    if verified:
        result["verified_claims"] = verified
    return result
