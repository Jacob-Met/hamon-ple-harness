"""Offline PLE output-delivery audit on inherited public-template traces.

No inherited artifact is a score for current source. No model calls, grading,
task submission, hidden data access, or paid endpoint use occurs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import delivery_guard


def exported_path(key: str) -> str | None:
    if not isinstance(key, str) or ".." in key.split("/"):
        return None
    if key.startswith("filesystem/"):
        return "/" + key
    if key.startswith("app/files/"):
        return "/" + key
    if key.startswith("app/"):
        return "/" + key
    return None


def task_workdir(instruction: str, exported: list[str]) -> str:
    """Replay-only inference. The real harness uses session.workdir."""
    m = re.search(r"working in .(/filesystem/[A-Za-z0-9_.+\-/]+).",
                  instruction, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".")
    # Replay infers the task root from an observed export suffix. This must
    # NEVER be used by the live harness: only session.workdir is authoritative.
    relative = re.findall(
        r"(?<![A-Za-z0-9_.])([A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)+"
        r"\.(?:zip|docx|xlsx|xlsm|pdf|ifc|dwg|dxf|csv|json|md|txt|png))",
        instruction, flags=re.I,
    )
    for path in exported:
        for suffix in relative:
            if path.endswith("/" + suffix):
                prefix = path[:-(len(suffix) + 1)]
                if prefix.startswith("/app/files"):
                    return prefix
        if "/output/" in path:
            return path.split("/output/", 1)[0]
        if "/submission/" in path:
            return path.split("/submission/", 1)[0]
    return "/app/files" if any(p.startswith("/app/files/") for p in exported) else "/filesystem"


def evaluate_task(name: str, prompt: str, files: dict[str, str],
                  reward: float | None = None) -> dict:
    """Static output observation, not task success or an organizer score."""
    exported = sorted(filter(None, (exported_path(k) for k in files)))
    cwd = task_workdir(prompt, exported)
    targets = delivery_guard.extract_targets(prompt, cwd)
    matched, unmatched = [], []
    for target in targets:
        path, kind = target["path"], target["kind"]
        observed = path in exported if kind == "file" else any(
            e.startswith(path.rstrip("/") + "/") for e in exported
        )
        (matched if observed else unmatched).append(path)
    if not targets:
        state = "no_explicit_output_contract"
    elif unmatched and not matched:
        state = "no_detected_target_export"
    elif unmatched:
        state = "partial_target_export"
    else:
        state = "all_detected_targets_exported"
    return {
        "task": name, "state": state,
        "instruction_bytes": len(prompt.encode()), "cwd_inferred": cwd,
        "targets": targets, "matched": matched, "unmatched": unmatched,
        "exported_files": len(exported),
        "historical_template_reward": reward,
    }


def run(root: Path) -> dict:
    manifest = json.loads((root / "artifacts" / "manifest.json").read_text())
    if not isinstance(manifest.get("tasks"), dict):
        raise ValueError("public task map missing")
    rows, unavailable = [], []
    for name, record in sorted(manifest["tasks"].items()):
        if not isinstance(record, dict) or not re.fullmatch(r"[a-z0-9-]+", name):
            raise ValueError("invalid public task identifier")
        trace = root / "artifacts" / name / "trajectory.json"
        try:
            data = json.loads(trace.read_text())
            first = next(step.get("message", "") for step in data["steps"]
                         if step.get("source") == "user")
            if not isinstance(first, str):
                raise ValueError("first user step is not text")
            files = record["files"]
            if not isinstance(files, dict):
                raise ValueError("historical files map malformed")
            rows.append(evaluate_task(name, first, files, record.get("reward")))
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
            unavailable.append({"task": name, "reason": type(exc).__name__})
    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    return {
        "schema": "hamon.ple.delivery-corpus.v1",
        "source": "inherited_public_template_artifacts_only",
        "current_harness_scored": False, "organizer_score_verified": False,
        "template_manifest_commit": manifest.get("harness", {}).get("commit"),
        "public_tasks": len(manifest["tasks"]),
        "evaluated_traces": len(rows),
        "unavailable": unavailable, "states": states, "rows": rows,
        "review_cases": [
            {"task": row["task"], "state": row["state"],
             "targets": [t["path"] for t in row["targets"]],
             "unmatched": row["unmatched"],
             "exported_files": row["exported_files"]}
            for row in rows if row["state"] != "all_detected_targets_exported"
        ],
        "caveats": [
            "A path match does not establish semantic or engineering correctness.",
            "Workdirs without an explicit instruction are inferred only for this replay.",
            "Historical rewards cannot be attributed to the current harness.",
            "No hidden tasks, model inference, credits or contest submissions used.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--json", action="store_true", help="include task rows")
    opts = ap.parse_args(argv)
    try:
        result = run(opts.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "hamon.ple.delivery-corpus.v1",
                          "status": "error", "reason": type(exc).__name__}))
        return 2
    if not opts.json:
        result.pop("rows")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result["unavailable"] else 0


if __name__ == "__main__":
    sys.exit(main())
