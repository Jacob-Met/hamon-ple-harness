"""Offline PLE output-delivery audit on inherited public-template traces.

No inherited artifact is a score for current source. No model calls, grading,
task submission, hidden data access, or paid endpoint use occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import delivery_guard
from delivery_guard import named_return


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


def verify_export(root: Path, key: str, expected: str) -> dict:
    """Join the frozen manifest to actual exported bytes by path AND Git blob ID."""
    if not exported_path(key) or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{40}", expected):
        return {"state": "invalid_manifest_entry"}
    path = root / key
    try:
        resolved = path.resolve(strict=True)
        if (not resolved.is_relative_to(root.resolve()) or not resolved.is_file()
                or any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root))):
            return {"state": "unavailable"}
        size = resolved.stat().st_size
        git = hashlib.sha1(f"blob {size}\0".encode())
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                git.update(block)
                digest.update(block)
        actual = git.hexdigest()
    except OSError:
        return {"state": "unavailable"}
    return {"state": "verified" if actual == expected else "blob_mismatch",
            "path": key, "expected_blob": expected, "actual_blob": actual,
            "sha256": digest.hexdigest(), "bytes": size}


def custody(prompt: str, row: dict, verified: dict) -> list[dict]:
    """Report exactly what the archived export proves about each destination."""
    results = []
    named = named_return(prompt)
    named_path = None
    if named:
        name, destination = named["name"], named["destination"]
        local = [value for path, value in verified.items() if Path(path).name == name]
        if destination and destination.startswith(("/filesystem/", "/app/")):
            expected = destination.rstrip("/") + "/" + name
            named_path = expected
            record = verified.get(expected.lstrip("/"))
            local = [record] if record else []
            state = record["state"] if record else "missing_export"
        elif destination:
            state = "destination_receipt_unverified"
        elif len(local) == 1:
            state = local[0]["state"]
        else:
            state = "ambiguous_exports" if local else "missing_export"
        results.append({"kind": "named_filing" if destination else "named_return",
                        "name": name, "destination": destination, "state": state, "exports": local})
    for target in row["targets"]:
        path = target["path"]
        if path == named_path:
            continue
        if target["kind"] == "dir":
            local = [value for key, value in verified.items() if ("/"+key).startswith(path.rstrip("/")+"/")]
            state = "missing_export"
            if local:
                state = "verified" if all(v["state"] == "verified" for v in local) else "unverified_members"
        else:
            local = [verified[path.lstrip("/")]] if path.lstrip("/") in verified else []
            state = local[0]["state"] if local else "missing_export"
        results.append({"kind": target["kind"], "destination": path, "state": state, "exports": local})
    return results


def delivery_status(evidence: list[dict]) -> str:
    """Only verified bytes at every declared destination establish delivery."""
    states = {item["state"] for item in evidence}
    if states == {"verified"}:
        return "verified"
    if states.intersection({"missing_export", "blob_mismatch"}):
        return "incomplete"
    return "unverified"


def terminal_step(steps: list[dict]) -> dict:
    """A final tool request and its associated tool result are different evidence."""
    step = steps[-1]
    calls = [c["tool_call_id"] for c in step.get("tool_calls", []) if c.get("tool_call_id")]
    observed = {v.get("source_call_id") for v in (step.get("observation") or {}).get("results", [])}
    return {"step_id": step.get("step_id"), "tool_requests": len(calls),
            "without_observation": [c for c in calls if c not in observed]}


def run(root: Path) -> dict:
    manifest = json.loads((root / "artifacts" / "manifest.json").read_text())
    if not isinstance(manifest.get("tasks"), dict):
        raise ValueError("public task map missing")
    rows, unavailable = [], []
    export_identity_counts = {}
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
            row = evaluate_task(name, first, files, record.get("reward"))
            verified = {key: verify_export(trace.parent / "files", key, blob)
                        for key, blob in files.items()}
            for identity in verified.values():
                state = identity["state"]
                export_identity_counts[state] = export_identity_counts.get(state, 0) + 1
            row["custody"] = custody(first, row, verified)
            row["delivery_status"] = delivery_status(row["custody"])
            row["terminal_step"] = terminal_step(data["steps"])
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
            unavailable.append({"task": name, "reason": type(exc).__name__})
    states = {}
    delivery_counts = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
        status = row["delivery_status"]
        delivery_counts[status] = delivery_counts.get(status, 0) + 1
    return {
        "schema": "hamon.ple.delivery-corpus.v1",
        "source": "inherited_public_template_artifacts_only",
        "current_harness_scored": False, "organizer_score_verified": False,
        "template_manifest_commit": manifest.get("harness", {}).get("commit"),
        "public_tasks": len(manifest["tasks"]),
        "evaluated_traces": len(rows),
        "unavailable": unavailable, "states": states, "rows": rows,
        "export_identity_counts": export_identity_counts, "delivery_counts": delivery_counts,
        "review_cases": [
            {"task": row["task"], "state": row["state"],
             "delivery_status": row["delivery_status"], "custody": row["custody"],
             "terminal_step": row["terminal_step"],
             "targets": [t["path"] for t in row["targets"]],
             "unmatched": row["unmatched"],
             "exported_files": row["exported_files"]}
            for row in rows if row["delivery_status"] != "verified"
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
    ap.add_argument("--require-delivery", metavar="TASK",
                    help="check one archived task's declared delivery (0 verified, 1 incomplete, 3 unverified)")
    opts = ap.parse_args(argv)
    try:
        result = run(opts.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "hamon.ple.delivery-corpus.v1",
                          "status": "error", "reason": type(exc).__name__}))
        return 2
    if opts.require_delivery:
        row = next((r for r in result["rows"] if r["task"] == opts.require_delivery), None)
        if row is None:
            unavailable = any(r["task"] == opts.require_delivery for r in result["unavailable"])
            print(json.dumps({"task": opts.require_delivery,
                              "delivery_status": "unverified" if unavailable else "unknown_task"}))
            return 3 if unavailable else 2
        print(json.dumps({"task": row["task"], "delivery_status": row["delivery_status"],
                          "custody": row["custody"], "terminal_step": row["terminal_step"],
                          "source": result["source"], "organizer_score_verified": False}, sort_keys=True))
        return {"verified": 0, "incomplete": 1, "unverified": 3}[row["delivery_status"]]
    if not opts.json:
        result.pop("rows")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result["unavailable"] else 0


if __name__ == "__main__":
    sys.exit(main())
