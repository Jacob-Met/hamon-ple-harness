"""Stage 2 Accuracy-vs-Cost Pareto Router & Strict Deliverable Semantic/Format Verifier.

Routes per-task reasoning effort (`low` / `medium` / `high` / `max`) and delivery
reserve windows strictly from task-agnostic structural observables (file counts,
CAD/IFC/PDF modalities, MCP server count, deliverable count, and deterministic
symbolic/CAD pre-solve verification) — zero task-ID or filename shortcuts.
"""
from __future__ import annotations

import json
import math
import re
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
    "consolidation_1d": (
        "civil_geotech",
        "H_clay_m e0 Cc sigma_v0_eff_Pa delta_sigma_v_Pa cv_m2_s time_s drainage sigma_preconsolidation_Pa Cr",
        "dual_solver_abs_residual",
    ),
    "rankine_layers": (
        "civil_geotech",
        "pressure_state layers water_table_depth_m gamma_water_N_m3 surcharge_Pa",
        "dual_solver_rel_residual",
    ),
    "bearing_capacity": (
        "civil_geotech",
        "footing_B_m footing_L_m footing_shape depth_Df_m c_prime_Pa phi_deg gamma_N_m3 gamma_sat_N_m3 FS bearing_method water_table_depth_m applied_gross_pressure_Pa column_load_N",
        "stress_integration_residual_Pa",
    ),
}
_WAVE8_ROW_KEYS = {
    "nodes": "id demand_m3_s fixed_head_m",
    "pipes": "id from to length_m diameter_m roughness_m minor_loss_K",
    "buses": "id type p_injection_pu q_injection_pu vm_pu va_deg q_min_injection_pu q_max_injection_pu shunt_g_pu shunt_b_pu",
    "lines": "id from to r_pu x_pu tap_ratio phase_shift_deg line_charging_b_pu",
    "layers": "thickness_m phi_deg gamma_above_water_N_m3 gamma_submerged_N_m3 gamma_saturated_N_m3",
}
_STATION_KEYS = "radius_m meridional_velocity_m_s absolute_tangential_velocity_m_s absolute_flow_angle_deg relative_flow_angle_deg blade_metal_angle_deg"
_WAVE8_RECORD_KEYS = {
    "inlet": _STATION_KEYS,
    "outlet": _STATION_KEYS,
    "stage_boundary_speeds_m_s": "inlet outlet",
    "suction_flange": "density_kg_m3 suction_absolute_pressure_Pa vapor_pressure_Pa suction_velocity_m_s required_npsh_m",
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


def _wave8_certificate(instruction, inventory, mcp_server_count, deliverable_count):
    """Solve only a complete inline tool request with a confirmed empty workspace."""
    if (
        not isinstance(instruction, str)
        or len(instruction) > 12000
        or not instruction.lstrip().startswith("{")
        or mcp_server_count
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
    ):
        return None
    try:
        req = json.loads(instruction, object_pairs_hook=_unique_keys)
        if (
            not isinstance(req, dict)
            or set(req) != {"mode", "spec"}
            or not isinstance(req["spec"], dict)
        ):
            return None
        spec = req["spec"]
        calculation = spec["calculation"]
        mode, fields, required = _WAVE8_INPUTS[calculation]
        if req["mode"] not in ("auto", mode) or not spec.keys() <= {
            "calculation",
            *fields.split(),
        }:
            return None
        if not all(_bounded_wave8_field(k, v) for k, v in spec.items()):
            return None
        result = dispatch_ple_tool("hamon_conservation_cad_verifier", req)
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
            return plan
    return None


PLE_DOMAIN_TOOLS = (
    "hamon_ifc_spatial_clash",
    "hamon_drawing_reader",
    "hamon_conservation_cad_verifier",
    "check_deliverable_preflight",
)

_WAVE4_EXTS = (".step", ".stp", ".p21", ".cir", ".sp", ".spice", ".net")
_WAVE4_TERMS = ('step', 'stp', 'brep', 'mass moment', 'inertia', 'diaphragm', 'stress', 'deflection', 'buckling', 'von mises', 'kcl', 'kvl', 'spice', 'nodal', 'tellegen', 'phasor', 'conservation', 'enthalpy', 'entropy', 'mass balance', 'energy balance', 'species balance', 'head loss', 'colebrook', 'pipe network', 'hardy cross', 'hardy-cross', 'symmetrical components', 'sequence impedance', 'ac power flow', 'load flow', 'turbomachinery', 'velocity triangle', 'rothalpy', 'bearing capacity', 'consolidation', 'earthwork', 'seepage', 'dimensional')


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
