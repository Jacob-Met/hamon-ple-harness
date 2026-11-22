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


def route_from_intake_prompt(
    engineering_inventory: str,
    *,
    mcp_server_count: int = 0,
    deliverable_count: int = 0,
    model_api: str = "google-cloud/claude-opus-5-5",
    presolve_verified: bool = False,
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
            pre_ok = bool(presolve_verified or rec.get("presolve_verified", False))
            return route_task_budget(
                by_extension=exts,
                engineering_file_count=fcount,
                mcp_server_count=mcp_server_count,
                deliverable_count=deliverable_count,
                model_api=model_api,
                presolve_verified=pre_ok,
            )
    return None


PLE_DOMAIN_TOOLS = (
    "hamon_ifc_spatial_clash",
    "hamon_drawing_reader",
    "hamon_conservation_cad_verifier",
    "check_deliverable_preflight",
)

_WAVE4_EXTS = (".step", ".stp", ".p21", ".cir", ".sp", ".spice", ".net")
_WAVE4_TERMS = ('step', 'stp', 'brep', 'mass moment', 'inertia', 'stress', 'deflection', 'buckling', 'von mises', 'kcl', 'kvl', 'spice', 'nodal', 'tellegen', 'phasor', 'conservation', 'enthalpy', 'entropy', 'mass balance', 'energy balance', 'species balance', 'head loss', 'colebrook', 'bearing capacity', 'consolidation', 'earthwork', 'seepage', 'dimensional')


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
