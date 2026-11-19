"""Cold zero-model end-to-end proof of the production PLE materializer.

Two genuinely archived public failed exports contain standalone IFC generators,
not IFC deliverables. Replays the EXACT production preflight -> changed-script
selection -> task-native command -> output probe path, then parses generated
IFC4 with IfcOpenShell. Runs only on ephemeral CI /filesystem; cleans all files.
This proves output materialization, NOT engineering correctness or model score.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import delivery_guard
import delivery_materialize


class LocalTaskProtocol:
    """JSON echo of the official task exec result, using a fresh CI sandbox."""

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def time_left(self) -> int:
        return 900

    async def exec(self, command: str, timeout_sec: int = 120) -> dict:
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=str(self.workdir),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return {"exit_code": -1, "output": "", "timed_out": True}
        return {
            "exit_code": process.returncode,
            "output": stdout.decode(errors="replace"),
            "stderr_present": bool(stderr),
            "truncated": False, "timed_out": False,
        }


async def replay_source(source: Path) -> dict:
    import ifcopenshell

    with tempfile.TemporaryDirectory(
        prefix="ple-materialize-", dir="/filesystem"
    ) as raw:
        root = Path(raw)
        output = root / "output"
        output.mkdir()
        session = LocalTaskProtocol(root)
        desired = [
            {"path": str(output / "model.ifc"), "kind": "file"},
            {"path": str(output / "reconciliation.json"), "kind": "file"},
        ]
        candidates = delivery_materialize.candidates(desired)
        scripts = delivery_materialize.snapshot_targets(candidates)
        before_outputs = await delivery_guard.snapshot(session, desired)
        before_scripts = await delivery_guard.snapshot(session, scripts)
        if before_outputs is None or before_scripts is None:
            raise RuntimeError("official preflight-probe contract unavailable")
        shutil.copy2(source, output / "solution.py")
        after_scripts = await delivery_guard.snapshot(session, scripts)
        still = await delivery_guard.snapshot(session, desired)
        issues = delivery_guard.compare(before_outputs, still)
        selected = delivery_materialize.plan(
            candidates, issues, before_scripts, after_scripts,
        )
        if len(selected) != 1 or not selected[0]["script"].endswith("solution.py"):
            raise AssertionError("generator selection failed changed-code proof")
        started = time.monotonic()
        observations = await delivery_materialize.materialize(session, selected)
        final_outputs = await delivery_guard.snapshot(session, desired)
        residual = delivery_guard.compare(before_outputs, final_outputs)
        if residual:
            raise AssertionError(f"exact output contract incomplete: {residual}")
        model = ifcopenshell.open(str(output / "model.ifc"))
        products = len(model.by_type("IfcProduct"))
        if model.schema != "IFC4" or products < 100:
            raise AssertionError("not a meaningful IFC4 BIM model")
        recon = json.loads((output / "reconciliation.json").read_text())
        if not isinstance(recon, dict) or not recon:
            raise AssertionError("reconciliation payload is empty")
        return {
            "task": source.parts[source.parts.index("artifacts") + 1],
            "materializer": observations[0]["state"],
            "exact_outputs_ready": 2,
            "ifc_bytes": (output / "model.ifc").stat().st_size,
            "ifc_schema": model.schema, "products": products,
            "wall_s": round(time.monotonic() - started, 2),
            "current_model_score": None,
        }


async def main() -> int:
    if not Path("/filesystem").is_dir():
        raise RuntimeError("CI must provision ephemeral /filesystem first")
    src = sorted(
        (ROOT / "artifacts").glob("*/files/**/output/solution.py")
    )
    if len(src) < 2:
        raise AssertionError("public corpus generators missing or incomplete")
    rows = [await replay_source(source) for source in src]
    print(json.dumps({
        "schema": "hamon.ple.materializer-e2e.v1",
        "task_cases": len(rows),
        "model_calls": 0, "submitted": False,
        "rows": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
