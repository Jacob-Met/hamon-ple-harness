"""Stage 2 Accuracy-vs-Cost Pareto Router & Strict Deliverable Semantic/Format Verifier.

Routes per-task reasoning effort (`low` / `medium` / `high` / `max`) and delivery
reserve windows strictly from task-agnostic structural observables (file counts,
CAD/IFC/PDF modalities, MCP server count, deliverable count, and deterministic
symbolic/CAD pre-solve verification) — zero task-ID or filename shortcuts.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import math
from pathlib import Path
import re
from typing import Any
import zipfile

CAD_MODALITIES = frozenset({".ifc", ".ifczip", ".dxf", ".dwg", ".step", ".stp", ".rvt"})
DOC_MODALITIES = frozenset({".pdf", ".xlsx", ".xlsm", ".docx"})

# Reference token-cost multipliers relative to fixed THINKING="max" ($14.87/task baseline)
TIER_COST_MULTIPLIER = {
    "low": 0.22,
    "medium": 0.36,
    "high": 0.58,
    "max": 0.92,  # prompt-cached + deterministic pre-computation savings
}

_STUB_MARKERS = frozenset({
    "[tool call]",
    "<tool_call>",
    "(pi stopped without a final message)",
})


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


def audit_local_deliverable(path: str | Path) -> dict[str, Any]:
    """Deterministic format, syntax, STEP/IFC & stub hygiene guard on a deliverable file."""
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "path": str(p), "reason": "missing"}
    sz = p.stat().st_size
    if sz == 0:
        return {"ok": False, "path": str(p), "reason": "empty"}
    ext = p.suffix.lower()
    if ext == ".json":
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_json:{type(exc).__name__}"}
    elif ext == ".csv":
        try:
            rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8", errors="strict"))))
            if not rows or any(len(r) != len(rows[0]) for r in rows):
                return {"ok": False, "path": str(p), "reason": "ragged_or_empty_csv"}
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_csv:{type(exc).__name__}"}
    elif ext in (".md", ".txt"):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if not txt or txt.lower() in _STUB_MARKERS:
            return {"ok": False, "path": str(p), "reason": "placeholder_or_stub_text"}
    elif ext == ".py":
        try:
            src = p.read_text(encoding="utf-8", errors="strict")
            tree = ast.parse(src)
            if not tree.body:
                return {"ok": False, "path": str(p), "reason": "empty_python_ast"}
        except Exception as exc:
            return {"ok": False, "path": str(p), "reason": f"invalid_python:{type(exc).__name__}"}
    elif ext in (".ifc", ".step", ".stp"):
        head_tail = p.read_text(encoding="utf-8", errors="replace")
        if "ISO-10303-21;" not in head_tail[:512] or "DATA;" not in head_tail or "END-ISO-10303-21;" not in head_tail[-512:]:
            return {"ok": False, "path": str(p), "reason": "malformed_ifc_step_structure"}
    elif ext in (".xlsx", ".xlsm"):
        import engineering_toolkit as tk
        audit = tk.xlsx_audit(str(p))
        err_cells = sum(s.get("error_count", 0) for s in audit.get("sheets", []))
        mismatches = audit.get("lookup_audit", {}).get("counts", {}).get("mismatch", 0)
        if err_cells > 0 or mismatches > 0:
            return {
                "ok": False,
                "path": str(p),
                "reason": f"xlsx_errors(error_cells={err_cells},lookup_mismatches={mismatches})",
            }
    elif ext in (".zip", ".docx"):
        try:
            with zipfile.ZipFile(p, "r") as zf:
                if not zf.namelist() or zf.testzip() is not None:
                    return {"ok": False, "path": str(p), "reason": "corrupt_or_empty_zip"}
        except Exception:
            return {"ok": False, "path": str(p), "reason": "bad_zip"}
    return {"ok": True, "path": str(p), "bytes": sz}


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
            fcount = len(rec.get("files") or []) if isinstance(rec.get("files"), list) else sum(int(v) for v in exts.values())
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
