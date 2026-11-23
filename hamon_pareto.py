"""Stage 2 Accuracy-vs-Cost Pareto Router & Strict Deliverable Semantic/Format Verifier.

Routes per-task reasoning effort (`low` / `medium` / `high` / `max`) and delivery
reserve windows strictly from task-agnostic structural observables (file counts,
CAD/IFC/PDF modalities, MCP server count, deliverable count, and deterministic
symbolic/CAD pre-solve verification) — zero task-ID or filename shortcuts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from engineering_toolkit import audit_local_deliverable

CAD_MODALITIES = frozenset({".ifc", ".ifczip", ".dxf", ".dwg", ".step", ".stp", ".rvt"})
DOC_MODALITIES = frozenset({".pdf", ".xlsx", ".xlsm", ".docx"})

# Reference token-cost multipliers relative to fixed THINKING="max" ($14.87/task baseline)
TIER_COST_MULTIPLIER = {
    "low": 0.22,
    "medium": 0.36,
    "high": 0.58,
    "max": 0.92,  # prompt-cached + deterministic pre-computation savings
}

def route_task_budget(
    *,
    by_extension: dict[str, int] | None,
    engineering_file_count: int = 0,
    mcp_server_count: int = 0,
    deliverable_count: int = 0,
    agent_timeout_sec: int | None = None,
    model_api: str = "google-cloud/claude-opus-5-5",
    presolve_verified: bool = False,
) -> dict[str, Any]:
    """Select Pareto-optimal Pi `--thinking` tier and delivery reserve budget."""
    exts = {k.lower(): int(v) for k, v in (by_extension or {}).items() if int(v) > 0}
    has_3d_bim = any(ext in exts for ext in (".ifc", ".ifczip", ".step", ".stp", ".rvt"))
    pdf_count = exts.get(".pdf", 0)
    xlsx_count = exts.get(".xlsx", 0) + exts.get(".xlsm", 0)
    dxf_count = exts.get(".dxf", 0) + exts.get(".dwg", 0)

    complexity_points = 0
    if has_3d_bim:
        complexity_points += 4
    if pdf_count >= 3 or dxf_count >= 2:
        complexity_points += 3
    elif pdf_count >= 1 or dxf_count >= 1:
        complexity_points += 1
    if xlsx_count >= 2:
        complexity_points += 2
    elif xlsx_count == 1:
        complexity_points += 1
    if mcp_server_count > 0:
        complexity_points += 2
    if engineering_file_count > 25 or deliverable_count >= 3:
        complexity_points += 2

    if complexity_points >= 5:
        if presolve_verified and complexity_points <= 5 and mcp_server_count == 0 and deliverable_count <= 2:
            tier = "medium"
            reserve_sec = 12 * 60
        else:
            tier = "max"
            reserve_sec = 25 * 60
    elif complexity_points >= 2:
        if presolve_verified and not has_3d_bim and complexity_points <= 3 and mcp_server_count == 0:
            tier = "medium"
            reserve_sec = 10 * 60
        else:
            tier = "high"
            reserve_sec = 16 * 60
    elif presolve_verified and not has_3d_bim and mcp_server_count == 0:
        tier = "low"
        reserve_sec = 6 * 60
    else:
        tier = "medium"
        reserve_sec = 10 * 60

    # Gemini 4 Argon supports low/medium/high (high thinks deeper than MAX per PLE docs/models.md)
    if "gemini-4-argon" in model_api.lower() and tier == "max":
        tier = "high"

    if agent_timeout_sec is not None and agent_timeout_sec > 0:
        reserve_sec = min(reserve_sec, max(90, agent_timeout_sec // 4))

    est_cost_usd = round(14.87 * TIER_COST_MULTIPLIER[tier], 4)
    return {
        "thinking_tier": tier,
        "delivery_reserve_sec": reserve_sec,
        "complexity_points": complexity_points,
        "presolve_verified": bool(presolve_verified),
        "estimated_cost_usd": est_cost_usd,
        "cost_reduction_factor": round(14.87 / est_cost_usd, 3),
    }


def sanitize_scalar_or_json_answer(raw_text: str, expected_format: str = "auto") -> str:
    """Strip accidental Markdown code fences or prose prefix when a strict format is required."""
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)\n```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if expected_format == "json":
        obj = json.loads(text)
        return json.dumps(obj, sort_keys=True)
    if expected_format == "number":
        m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text.replace(",", ""))
        if not m:
            raise ValueError("no finite numeric token found")
        val = float(m.group(0))
        if not math.isfinite(val):
            raise ValueError("non-finite number")
        return m.group(0)
    return text


# These schemas are the self-contained part of the public conservation tool.
# Inputs outside them can still use the full solver during the normal model turn.
_WAVE8_INPUTS = {
    "pipe_network": (
        "thermal_fluid",
        "nodes pipes kinematic_viscosity_m2_s gravity_m_s2",
        "max_mass_residual_m3_s max_energy_residual_m max_nodal_cycle_flow_difference_m3_s",
    ),
    "ac_power_flow": (
        "circuit",
        "buses lines",
        "max_rectangular_mismatch_pu max_voltage_disagreement_pu tellegen_residual_pu max_terminal_current_mismatch_pu",
    ),
    "symmetrical_fault": (
        "circuit",
        "fault_type faulted_phases sequence_impedances_pu prefault_positive_voltage_pu fault_impedance_pu",
        "max_phase_sequence_difference_pu max_terminal_boundary_residual_pu",
    ),
    "euler_rotor": (
        "thermal_fluid",
        "machine_kind rotation_rpm inlet outlet mass_flow_kg_s measured_fluid_torque_N_m measured_rotor_static_enthalpy_change_J_kg isentropic_specific_work_J_kg stage_boundary_speeds_m_s suction_flange",
        "energy_residual_J_kg",
    ),
    "layered_footing_settlement": (
        "civil_geotech", "settlement_column rectangular_loads settlement_layers proposed_settlement_m",
        "dual_solver_rel_residual",
    ),
    "consolidation_1d": (
        "civil_geotech",
        "H_clay_m e0 Cc sigma_v0_eff_Pa delta_sigma_v_Pa cv_m2_s time_s drainage sigma_preconsolidation_Pa Cr",
        "dual_solver_abs_residual",
    ),
    "layered_consolidation_transient": (
        "civil_geotech",
        "hydraulic_layers top_boundary bottom_boundary time_s delta_sigma_v_Pa rectangular_loads settlement_column proposed_settlement_m",
        "dual_solver_rel_residual",
    ),
    "layered_consolidation_1d": (
        "civil_geotech",
        "consolidation_layers time_s drainage load_history proposed_settlement_m proposed_mean_excess_pore_pressure_Pa",
        "dual_solver_rel_residual component_rel_residual inversion_rel_residual load_balance_rel_residual",
    ),
    "drained_layer_consolidation_history": (
        "civil_geotech",
        "drained_layers interface_condition top_boundary bottom_boundary time_s load_history proposed_settlement_m proposed_mean_excess_pore_pressure_Pa settlement_reporting_increment_m pressure_reporting_increment_Pa",
        "dual_solver_rel_residual",
    ),
    "consolidation_history_1d": (
        "civil_geotech",
        "H_clay_m cv_m2_s mv_per_Pa time_s drainage load_history proposed_settlement_m proposed_mean_excess_pore_pressure_Pa settlement_reporting_increment_m pressure_reporting_increment_Pa",
        "dual_solver_rel_residual",
    ),
    "rankine_layers": (
        "civil_geotech",
        "pressure_state layers front_layers front_surface_depth_m water_table_depth_m front_water_table_depth_m gamma_water_N_m3 surcharge_Pa front_surcharge_Pa",
        "dual_solver_rel_residual",
    ),
    "bearing_capacity": (
        "civil_geotech",
        "footing_B_m footing_L_m footing_shape depth_Df_m c_prime_Pa phi_deg gamma_N_m3 gamma_sat_N_m3 FS bearing_method water_table_depth_m applied_gross_pressure_Pa column_load_N",
        "stress_integration_residual_Pa",
    ),
}
_RANKINE_LAYER_KEYS = "thickness_m phi_deg gamma_above_water_N_m3 gamma_submerged_N_m3 gamma_saturated_N_m3"
_WAVE8_ROW_KEYS = {
    "rectangular_loads": "center_x_m center_y_m width_m length_m net_pressure_Pa",
    "hydraulic_layers": "top_depth_m bottom_depth_m mv_per_Pa cv_m2_s",
    "settlement_layers": "top_depth_m bottom_depth_m e0 Cc Cr sigma_v0_eff_Pa sigma_preconsolidation_Pa",
    "consolidation_layers": "thickness_m cv_m2_s mv_per_Pa",
    "load_history": "time_s surcharge_Pa",
    "drained_layers": "H_clay_m cv_m2_s mv_per_Pa",
    "nodes": "id demand_m3_s fixed_head_m",
    "pipes": "id from to length_m diameter_m roughness_m minor_loss_K",
    "buses": "id type p_injection_pu q_injection_pu vm_pu va_deg q_min_injection_pu q_max_injection_pu shunt_g_pu shunt_b_pu",
    "lines": "id from to r_pu x_pu tap_ratio phase_shift_deg line_charging_b_pu",
    "layers": _RANKINE_LAYER_KEYS,
    "front_layers": _RANKINE_LAYER_KEYS,
}
_STATION_KEYS = "radius_m meridional_velocity_m_s absolute_tangential_velocity_m_s absolute_flow_angle_deg relative_flow_angle_deg blade_metal_angle_deg"
_WAVE8_RECORD_KEYS = {
    "settlement_column": "x_m y_m",
    "inlet": _STATION_KEYS,
    "outlet": _STATION_KEYS,
    "stage_boundary_speeds_m_s": "inlet outlet",
    "suction_flange": "density_kg_m3 suction_absolute_pressure_Pa vapor_pressure_Pa suction_velocity_m_s required_npsh_m gauge_elevation_above_npsh_datum_m loss_to_npsh_datum_m",
    "fault_impedance_pu": "real imag",
    "prefault_positive_voltage_pu": "real imag",
}


def _unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Repeated specification field")
    return result


def _scalar_record(record, keys):
    return (
        isinstance(record, dict)
        and record.keys() <= set(keys.split())
        and all(type(v) in (int, float, str) for v in record.values())
    )


def _bounded_wave8_field(key, value):
    if key in _WAVE8_ROW_KEYS:
        return (
            isinstance(value, list)
            and 0 < len(value) <= 32
            and all(_scalar_record(row, _WAVE8_ROW_KEYS[key]) for row in value)
        )
    if key in _WAVE8_RECORD_KEYS:
        return _scalar_record(value, _WAVE8_RECORD_KEYS[key])
    if key == "faulted_phases":
        return (
            isinstance(value, list)
            and 1 <= len(value) <= 3
            and all(x in ("A", "B", "C") for x in value)
        )
    if key == "sequence_impedances_pu":
        return (
            isinstance(value, dict)
            and set(value) == {"zero", "positive", "negative"}
            and all(_scalar_record(x, "real imag") for x in value.values())
        )
    return type(value) in (int, float, str)


# Only these delivery-neutral covers are part of the certified instruction grammar.
# Additional prose can change units, requested comparisons or output obligations.
_WAVE8_COVERS = {
    "Run the calculation below and return its JSON result.": "single",
    "Run the calculations below and return their JSON results in the same order.": "batch",
}


def _wave8_request(instruction):
    text = instruction.strip()
    lines = text.splitlines()
    cover = _WAVE8_COVERS.get(lines[0]) if lines else None
    if cover:
        text = "\n".join(lines[1:]).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (len(lines) < 3 or lines[0] not in ("```", "```json")
                or lines[-1] != "```"):
            return None
        text = "\n".join(lines[1:-1])
    value = json.loads(text, object_pairs_hook=_unique_keys)
    if isinstance(value, list):
        if cover == "single" or not 1 <= len(value) <= 2:
            return None
    elif cover == "batch" or not isinstance(value, dict):
        return None
    requests = value if isinstance(value, list) else [value]
    activity_requests = [i for i, req in enumerate(requests)
                         if isinstance(req, dict) and req.get("mode") == "electrochem_separations"
                         and isinstance(req.get("spec"), dict)
                         and req["spec"].get("subkind") == "electrochemistry"]
    if activity_requests:
        # The text is the source of the certificate. Refuse a numeric lexeme
        # rounded to a different decimal value by the ordinary JSON decoder.
        literal = json.loads(text, parse_float=Decimal)
        literal_requests = literal if isinstance(literal, list) else [literal]
        for i in activity_requests:
            parsed = json.loads(json.dumps(requests[i], allow_nan=False), parse_float=Decimal)
            if parsed != literal_requests[i]:
                return None
    return value


def _nernst_equilibrium_result(req):
    """Qualify a complete open-circuit request against an exact activity product."""
    spec = req["spec"]
    if set(spec) != {"subkind", "E0_V", "n_electrons", "T_K", "activity_terms"}:
        return None
    if spec["subkind"] != "electrochemistry":
        return None
    if any(type(spec[key]) not in (int, float) or not math.isfinite(spec[key]) for key in ("E0_V", "T_K")):
        return None
    n, terms = spec["n_electrons"], spec["activity_terms"]
    if type(n) is not int or n <= 0 or spec["T_K"] <= 0 or not isinstance(terms, list) or not 1 <= len(terms) <= 16:
        return None
    product = Fraction(1)
    species = set()
    for row in terms:
        if not isinstance(row, dict) or set(row) != {"species", "activity", "stoich"}:
            return None
        name = row["species"]
        if not isinstance(name, str) or not name.strip() or name.strip() in species:
            return None
        species.add(name.strip())
        activity, power = row["activity"], row["stoich"]
        if type(power) is not int or not 1 <= abs(power) <= 16:
            return None
        if type(activity) not in (float, int) or not math.isfinite(activity) or activity <= 0:
            return None
        product *= Fraction(str(activity)) ** power
    result = dispatch_ple_tool("hamon_conservation_cad_verifier", req)
    if result.get("verified") is not True or result.get("subkind") != "electrochemistry":
        return None
    with localcontext() as ctx:
        ctx.prec = 70
        quotient = Decimal(product.numerator) / Decimal(product.denominator)
        log_q = quotient.ln()
        if product != 1 and log_q == 0:
            return None
        shift = Decimal("8.314462618") * Decimal(str(spec["T_K"])) * log_q / (n * Decimal("96485.33212"))
        voltage = Decimal(str(spec["E0_V"])) - shift
        nf = n * Decimal("96485.33212")
        rt = Decimal("8.314462618") * Decimal(str(spec["T_K"]))
        e0 = Decimal(str(spec["E0_V"]))
        expected_fields = {
            "ln_Q_reaction": log_q, "E_nernst_V": voltage, "V_cell_V": voltage,
            "delta_G0_J_mol": -nf * e0, "delta_G_J_mol": -nf * voltage,
            "ln_K_eq": nf * e0 / rt, "n_electrons": n, "product_stoich": 1,
            "charge_passed_C": 0, "reaction_charge_C": 0, "reaction_extent_mol": 0,
            "moles_reacted": 0, "product_moles": 0, "mass_deposited_g": 0,
        }
    residual = result.get("residual")
    if type(residual) not in (int, float) or not 0 <= residual < 1e-11:
        return None
    for field, expected in expected_fields.items():
        observed = result.get(field)
        if type(observed) not in (float, int) or not math.isfinite(observed):
            return None
        reference = float(expected)
        if (expected and reference == 0) or not math.isclose(observed, reference, rel_tol=1e-12, abs_tol=0):
            return None
    return result


def _distillation_result(req):
    """Admit only closed binary-column calculations with consistent stream flows."""
    spec = req["spec"]
    required = {"z_F", "x_D", "x_B"}
    numeric = required | {"alpha", "F_mol_s", "q", "R", "R_mult"}
    tabulated = "equilibrium_table" in spec
    if (spec.get("subkind") != "distillation"
            or spec.get("domain", req["mode"]) != req["mode"]
            or not required <= spec.keys() <= numeric | {"subkind", "domain", "equilibrium_table"}
            or tabulated == ("alpha" in spec)
            or {"R", "R_mult"} <= spec.keys()
            or not all(type(spec[k]) in (float, int) and math.isfinite(spec[k])
                       for k in numeric & spec.keys())):
        return None

    result = dispatch_ple_tool("hamon_conservation_cad_verifier", req)
    if (result.get("verified") is not True or result.get("domain") != req["mode"]
            or result.get("subkind") != "distillation" or not 0 <= result["residual"] < 1e-10):
        return None

    if tabulated:
        if result.get("equilibrium_model") != "piecewise_linear_table":
            return None
        count = result["N_stages_integer"]
        fractional = result["N_theoretical"]
        feed_stage = result["feed_stage"]
        if (type(count) is not int or type(feed_stage) is not int
                or type(fractional) not in (int, float)
                or not 1 <= feed_stage <= count or not count - 1 < fractional <= count):
            return None
        circulation = result["L_rect_mol_s"] > 0, result["V_strip_mol_s"] > 0
        section = {(True, True): "two_section", (True, False): "rectifying_only",
                   (False, True): "stripping_only", (False, False): "single_equilibrium_flash"}
        if result["section_mode"] != section[circulation]:
            return None
        if not any(circulation):
            if count != 1 or fractional != 1:
                return None
        else:
            gap = result["min_operating_gap"]
            if type(gap) not in (int, float) or not math.isfinite(gap) or gap <= 0:
                return None

    feed = spec.get("F_mol_s", 100.0)
    q = spec.get("q", 1.0)
    d, b = result["D_mol_s"], result["B_mol_s"]
    lr, vr = result["L_rect_mol_s"], result["V_rect_mol_s"]
    ls, vs = result["L_strip_mol_s"], result["V_strip_mol_s"]
    if not (feed > 0 and d > 0 and b > 0 and min(lr, vr, ls, vs) >= 0):
        return None
    # Normalize by the specified feed so the same gate works across flow units.
    # Check the product split, condenser, reboiler and both sides of the feed tray.
    reflux = spec.get("R", spec.get("R_mult", 1.25) * result["R_min"])
    if not math.isclose(result["R_operating"], reflux, rel_tol=1e-12):
        return None
    balances = (
        (d + b, feed),
        (d * spec["x_D"] + b * spec["x_B"], feed * spec["z_F"]),
        (lr, reflux * d), (vr, lr + d), (ls, vs + b),
        (ls, lr + q * feed), (vr, vs + (1 - q) * feed),
    )
    if not all(math.isfinite(left) and math.isfinite(right)
               and abs(left - right) / feed < 1e-10 for left, right in balances):
        return None
    return result


def _jacketed_cstr_certificate(spec):
    """Admit a complete first-order liquid jacket model and check every reported balance."""
    required = {"reactor_type", "order", "C_A0", "v0", "T0", "delta_T_adiabatic",
                "jacket_to_feed_heat_capacity_ratio", "coolant_T_K", "volume"}
    optional = {"steady_state_branch", "gas_phase"}
    rate_laws = ({"k"}, {"k_ref", "T_ref", "E_a"}, {"A_preexp", "E_a"})
    if (not any(required | law == spec.keys() - optional for law in rate_laws)
            or spec["reactor_type"] != "cstr" or spec["order"] != 1
            or spec.get("gas_phase", False) is not False):
        return None
    for key in spec.keys() - {"reactor_type", "gas_phase", "steady_state_branch"}:
        if type(spec[key]) not in (int, float) or not math.isfinite(spec[key]):
            return None
    branch = spec.get("steady_state_branch")
    if "steady_state_branch" in spec and not (branch == "all" or type(branch) is int and branch >= 0):
        return None
    result = dispatch_ple_tool("hamon_conservation_cad_verifier", {"mode": "reactor_kinetics", "spec": spec})
    if not isinstance(result, dict):
        return None
    states = result.get("steady_states")
    if (result.get("verified") is not True or not isinstance(states, list)
            or not states or len(states) != result.get("steady_state_count")):
        return None
    for row in states:
        if not isinstance(row, dict) or row.get("verified") is not True:
            return None
        for key in ("mass_balance_residual", "energy_balance_residual"):
            value = row.get(key)
            if type(value) not in (int, float) or not 0 <= value < 1e-9:
                return None
    return result


def _wave8_result(req):
    """Check one existing operation and its physical residuals."""
    if (
        not isinstance(req, dict)
        or set(req) != {"mode", "spec"}
        or not isinstance(req["spec"], dict)
    ):
        return None
    spec = req["spec"]
    if req["mode"] == "reactor_kinetics":
        return _jacketed_cstr_certificate(spec)
    if req["mode"] == "electrochem_separations":
        if spec.get("subkind") == "electrochemistry":
            return _nernst_equilibrium_result(req)
        return _distillation_result(req)
    calculation = spec["calculation"]
    mode, fields, required = _WAVE8_INPUTS[calculation]
    if req["mode"] not in ("auto", mode) or not spec.keys() <= {
        "calculation",
        *fields.split(),
    }:
        return None
    if not all(_bounded_wave8_field(k, v) for k, v in spec.items()):
        return None
    if calculation in ("consolidation_history_1d", "drained_layer_consolidation_history"):
        from hamon_consolidation_certificate import history_certificate
        result = history_certificate(spec, dispatch_ple_tool)
    else:
        result = dispatch_ple_tool("hamon_conservation_cad_verifier", req)
    if result is None:
        return None
    if (
        result.get("calculation") != calculation
        or result.get("ok") is not True
        or not result.get("conservation_verified")
    ):
        return None
    names = required.split()
    if calculation == "euler_rotor":
        for measured, residual in (
            ("measured_fluid_torque_N_m", "measured_torque_residual_N_m"),
            (
                "measured_rotor_static_enthalpy_change_J_kg",
                "measured_rothalpy_residual_J_kg",
            ),
        ):
            if measured in spec:
                names.append(residual)
        if len(names) == 1:
            return None
    if not all(
        type(result[k]) in (float, int)
        and math.isfinite(result[k])
        and abs(result[k]) < 1e-6
        for k in names
    ):
        return None
    return result


def _empty_inline_context(inventory, mcp_server_count, deliverable_count):
    return not (
        mcp_server_count
        or deliverable_count
        or inventory.get("schema") != "hamon.ple.intake.v1"
        or type(inventory.get("scanned_files")) is not int
        or type(inventory.get("engineering_file_count")) is not int
        or inventory.get("metadata", {}) != {}
        or inventory.get("scanned_files") != 0
        or inventory.get("engineering_file_count") != 0
        or inventory.get("files") != []
        or inventory.get("warnings") != []
        or inventory.get("by_extension") != {}
    )


_FIN_NUMBERS = {
    "k_w_mk", "area_m2", "perimeter_m", "h_w_m2k", "length_m", "t_base_c", "t_inf_c"
}


def _fin_presolve(instruction, inventory, mcp_server_count, deliverable_count):
    """Check a complete passive-fin request against the independent finite-volume solver."""
    if not _empty_inline_context(inventory, mcp_server_count, deliverable_count):
        return None, "additional task inputs or delivery obligations"
    lines = instruction.strip().splitlines()
    if len(instruction) > 12000 or len(lines) < 3 or lines[0] != "```wave5_spec" or lines[-1] != "```":
        return None, "expected one complete wave5_spec fence"
    try:
        spec = json.loads("\n".join(lines[1:-1]), object_pairs_hook=_unique_keys)
        if not isinstance(spec, dict) or spec.get("mode") != "convective_fin_1d":
            return None, "unsupported structured calculation"
        if spec.get("tip_bc") not in ("adiabatic", "convective"):
            return None, "unsupported or missing fin tip boundary"
        optional = {"x_eval_m"}
        if spec["tip_bc"] == "convective":
            optional.add("h_tip_w_m2k")
        if not _FIN_NUMBERS <= spec.keys() <= _FIN_NUMBERS | optional | {"mode", "tip_bc"}:
            return None, "unexpected or missing fin input"
        for key in _FIN_NUMBERS | (optional & spec.keys()):
            if type(spec[key]) not in (int, float) or not math.isfinite(spec[key]):
                return None, "expected finite numeric fin inputs"
        if not 0 <= spec.get("x_eval_m", spec["length_m"]) <= spec["length_m"]:
            return None, "evaluation point outside fin"
        result = verify_wave5_engineering(spec)
        if result.get("passed") is not True:
            return None, "physical solver or finite-volume cross-check refused"
        q, fv = result["q_fin_w"], result["q_fvm_w"]
        # Do not normalize tiny fluxes by the verifier's absolute near-zero floor.
        if not math.isclose(q, fv, rel_tol=1e-3, abs_tol=0.0):
            return None, "analytic and finite-volume flux disagree"
        return json.dumps(result, allow_nan=False, separators=(",", ":")), None
    except (ArithmeticError, ValueError, KeyError, TypeError, RecursionError):
        return None, "malformed or numerically unverifiable fin request"


def _wave8_certificate(instruction, inventory, mcp_server_count, deliverable_count):
    """Certify every requested calculation in the original order, or abstain."""
    if (not isinstance(instruction, str) or len(instruction) > 12000
            or not _empty_inline_context(inventory, mcp_server_count, deliverable_count)):
        return None
    try:
        request = _wave8_request(instruction)
        if request is None:
            return None
        if isinstance(request, list):
            result = []
            for case in request:
                verified = _wave8_result(case)
                if verified is None:
                    return None
                result.append(verified)
        else:
            result = _wave8_result(request)
            if result is None:
                return None
        # Reject non-JSON results and keep the additional model context bounded.
        encoded = json.dumps(result, allow_nan=False, separators=(",", ":"))
        return encoded if len(encoded) <= 12000 else None
    except (
        ArithmeticError,
        ImportError,
        KeyError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None


def route_from_intake_prompt(
    engineering_inventory: str,
    *,
    mcp_server_count: int = 0,
    deliverable_count: int = 0,
    model_api: str = "google-cloud/claude-opus-5-5",
    presolve_verified: bool = False,
    instruction: str | None = None,
) -> dict[str, Any] | None:
    """Parse hamon.ple.intake.v1 metadata from intake_for_prompt output if present (zero extra exec calls)."""
    if not engineering_inventory:
        return None
    for line in engineering_inventory.splitlines():
        s = line.strip()
        if s.startswith("{") and '"hamon.ple.intake.v1"' in s:
            try:
                rec = json.loads(s)
            except ValueError:
                return None
            exts = rec.get("by_extension") if isinstance(rec.get("by_extension"), dict) else {}
            fcount = rec.get("engineering_file_count")
            if not isinstance(fcount, int) or fcount < 0:
                fcount = sum(int(v) for v in exts.values()) if exts else len(rec.get("files") or [])
            fin_request = isinstance(instruction, str) and instruction.lstrip().startswith("```wave5_spec")
            if fin_request:
                certificate, abstention = _fin_presolve(instruction, rec, mcp_server_count, deliverable_count)
            else:
                certificate = _wave8_certificate(instruction, rec, mcp_server_count, deliverable_count)
            # Legacy callers can supply a prior presolve. Live participant requests
            # obtain a certificate from their own inline problem and observed inventory.
            pre_ok = (bool(presolve_verified or rec.get("presolve_verified", False))
                      if instruction is None else certificate is not None)
            plan = route_task_budget(
                by_extension=exts,
                engineering_file_count=fcount,
                mcp_server_count=mcp_server_count,
                deliverable_count=deliverable_count,
                model_api=model_api,
                presolve_verified=pre_ok,
            )
            if certificate is not None:
                plan["presolve_result_json"] = certificate
                plan["presolve_source_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
            elif fin_request:
                plan["presolve_abstention"] = abstention
            return plan
    return None


PLE_DOMAIN_TOOLS = (
    "hamon_ifc_spatial_clash",
    "hamon_drawing_reader",
    "hamon_conservation_cad_verifier",
    "check_deliverable_preflight",
)

_WAVE4_EXTS = (".step", ".stp", ".p21", ".cir", ".sp", ".spice", ".net")
_WAVE4_TERMS = ('butler-volmer', 'butler–volmer', 'exchange current', 'overpotential', 'nernst', 'step', 'stp', 'brep', 'mass moment', 'inertia', 'diaphragm', 'stress', 'deflection', 'buckling', 'von mises', 'kcl', 'kvl', 'spice', 'nodal', 'tellegen', 'phasor', 'conservation', 'enthalpy', 'entropy', 'mass balance', 'energy balance', 'species balance', 'head loss', 'colebrook', 'pipe network', 'hardy cross', 'hardy-cross', 'symmetrical components', 'sequence impedance', 'ac power flow', 'load flow', 'turbomachinery', 'velocity triangle', 'rothalpy', 'bearing capacity', 'consolidation', 'earthwork', 'seepage', 'dimensional')


def select_domain_tools(prompt: str = "", file_paths=()) -> tuple[str, ...]:
    p_low = (prompt or "").lower()
    f_low = [str(f).lower() for f in (file_paths or ())]
    selected = []
    if any(f.endswith(".ifc") for f in f_low) or any(k in p_low for k in ("ifc", "bim", "clash", "clearance", "takeoff", "solid")):
        selected.append("hamon_ifc_spatial_clash")
    if any(f.endswith(".pdf") for f in f_low) or any(k in p_low for k in ("pdf", "drawing", "sheet", "plan", "scale", "title-block", "vector")):
        selected.append("hamon_drawing_reader")
    if any(f.endswith(_WAVE4_EXTS) for f in f_low) or any(k in p_low for k in _WAVE4_TERMS):
        selected.append("hamon_conservation_cad_verifier")
    selected.append("check_deliverable_preflight")
    return tuple(selected)


def dispatch_ple_tool(tool_name: str, payload: dict) -> dict:
    import hamon_conservation_cad_verifier as _ccv
    import hamon_drawing_reader as _dr
    import hamon_ifc_spatial_clash as _ifc

    if tool_name == "hamon_ifc_spatial_clash":
        return _ifc.run_router_ifc_clash(payload)
    if tool_name == "hamon_drawing_reader":
        return _dr.run_router_drawing_reader(payload)
    if tool_name == "hamon_conservation_cad_verifier":
        return _ccv.run_router_conservation_cad_verifier(payload)
    if tool_name == "check_deliverable_preflight":
        return _ifc.run_router_preflight(payload)
    raise ValueError(f"Unknown PLE domain tool: {tool_name}")


# --- Wave-5 PDE / Control / Psychrometrics / Structural FEM Verifier Hook ---
from hamon_pde_control_psychro_fem import (
    detect_wave5_domain,
    presolve_wave5_prompt,
    verify_wave5_engineering,
)

from hamon_wave6_physics_verifiers import (
    WAVE6_DOMAINS,
    dispatch_wave6_verifier,
    infer_wave6_domain,
    prune_deterministic_fast_path,
    solve_compressible_gas_acoustics,
    solve_mohr_vessel_buckling,
    solve_reactor_kinetics,
    solve_transmission_line_rlc,
)

from hamon_wave7_engineering_verifiers import (
    WAVE7_DOMAINS,
    dispatch_wave7_verifier,
    infer_wave7_domain,
)

