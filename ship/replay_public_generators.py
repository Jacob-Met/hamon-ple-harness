"""Offline falsification of public PLE model-generator candidates.

This is an *audit*, not a live harness, benchmark score or submission.
Generator scripts are sourced ONLY from historical, public template exports
and run in a fresh temporary directory with a deliberately small environment.
Nothing generated is written to artifacts/ or committed to Git.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def replay(source: Path, limit_sec: int = 75) -> dict:
    """Try to turn a public historical generator into an IFC4 deliverable."""
    with tempfile.TemporaryDirectory(prefix="ple-mep-replay-") as td:
        workspace = Path(td)
        script = workspace / "solution.py"
        shutil.copy2(source, script)
        target = workspace / "model.ifc"
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
        }
        started = time.monotonic()
        try:
            trial = subprocess.run(
                [sys.executable, str(script), str(target)],
                cwd=workspace,
                env=env,
                timeout=limit_sec,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            elapsed = time.monotonic() - started
        except subprocess.TimeoutExpired:
            return {
                "state": "timed_out",
                "wall_s": round(time.monotonic() - started, 2),
                "reason": "generator exceeded bounded trial",
            }
        if trial.returncode != 0:
            return {
                "state": "failed",
                "exit_code": trial.returncode,
                "wall_s": round(elapsed, 2),
                # Last line names the violated assertion; no massive logs.
                "reason_tail": trial.stderr[-1200:],
            }
        if not target.is_file():
            return {"state": "missing", "wall_s": round(elapsed, 2)}
        size = target.stat().st_size
        if size == 0:
            return {"state": "empty", "wall_s": round(elapsed, 2)}
        try:
            import ifcopenshell
            model = ifcopenshell.open(str(target))
            products = len(model.by_type("IfcProduct"))
            schema = model.schema
            if schema != "IFC4" or products == 0:
                state = "parsed_but_invalid_contract"
            else:
                state = "materialized_parseable_ifc4"
            return {
                "state": state, "wall_s": round(elapsed, 2),
                "bytes": size, "schema": schema, "products": products,
                "reconciliation_created": (workspace / "reconciliation.json").is_file(),
            }
        except Exception as exc:
            return {
                "state": "parse_failed",
                "wall_s": round(elapsed, 2),
                "bytes": size, "reason": type(exc).__name__,
            }


def main() -> int:
    root = ROOT / "artifacts"
    # Generic corpus discovery. No task-ID hardcoding, no runtime use.
    paths = sorted(root.glob("*/files/**/output/solution.py"))
    rows = []
    for p in paths:
        row = replay(p)
        rows.append({"task": p.relative_to(root).parts[0], **row})
    print(json.dumps({
        "schema": "hamon.ple.historical-generator-replay.v1",
        "provenance": "public_inherited_template_artifacts",
        "current_model_score": None,
        "organizer_grading": False,
        "cases_found": len(paths),
        "rows": rows,
    }, indent=2, sort_keys=True))
    # A zero-case audit is a discovery failure, not proof of success.
    return 2 if not paths else 0


if __name__ == "__main__":
    sys.exit(main())
