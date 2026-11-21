"""Stage 2 Accuracy-vs-Cost Pareto Router & Strict Deliverable Format Verifier.

Routes per-task reasoning effort (`medium` / `high` / `max`) and delivery reserve
windows strictly from task-agnostic structural observables (file counts, CAD/IFC/PDF
modalities, MCP server count, deliverable count) — zero task-ID or filename shortcuts.
"""
from __future__ import annotations

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


def route_task_budget(
    *,
    by_extension: dict[str, int] | None,
    engineering_file_count: int = 0,
    mcp_server_count: int = 0,
    deliverable_count: int = 0,
    agent_timeout_sec: int | None = None,
    model_api: str = "google-cloud/claude-opus-5-5",
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
        tier = "max"
        reserve_sec = 25 * 60
    elif complexity_points >= 2:
        tier = "high"
        reserve_sec = 16 * 60
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
    """Deterministic format & calculation hygiene guard on a deliverable file."""
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
            return route_task_budget(
                by_extension=exts,
                engineering_file_count=fcount,
                mcp_server_count=mcp_server_count,
                deliverable_count=deliverable_count,
                model_api=model_api,
            )
    return None
