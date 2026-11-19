"""Materialize an agent-written generator before spending another model pass.

Converts only explicit, still-incomplete deliverables whose SAME-DIRECTORY
generator script was newly created/changed by the current task session.
The generator runs inside the isolated task container over the official exec
protocol (not in the harness container), with a deadline, no shell evaluation
of user paths, reduced environment, and no access to hidden benchmarks or
historical dev artifacts. The caller rechecks the delivered files independently.
"""
from __future__ import annotations

import asyncio
import json
import posixpath
import shlex
from typing import Any

# Output scripts are conventions, not task-specific answer keys.
GENERATOR_NAMES = ("solution.py", "generate.py", "build.py")
# These may be built from scripts; an explicit user-named target is mandatory.
GENERATABLE_EXTENSIONS = frozenset((
    ".ifc", ".pdf", ".zip", ".docx", ".xlsx", ".xlsm", ".csv",
    ".json", ".png", ".dxf", ".dwg",
))
MAX_GENERATORS = 6
PER_SCRIPT_TIMEOUT = 55


def _safe_absolute(path: Any) -> bool:
    return (
        isinstance(path, str)
        and path.startswith(("/app/", "/filesystem/"))
        and posixpath.normpath(path) == path
        and not any(x in (".", "..") for x in path.split("/"))
        and all(ch not in path for ch in ('\x00', '\n', '\r', '\\'))
    )


def candidates(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    """One execution candidate per (output directory, generator convention).

    Prefer .ifc (the published cold failure), then other exact target formats.
    No script path can originate from the instruction except the target's
    declared parent directory. Directory-only tasks are never executed.
    """
    groups: dict[str, list[str]] = {}
    for target in targets:
        path = target.get("path")
        if (target.get("kind") != "file"
                or not _safe_absolute(path)
                or posixpath.splitext(path)[1].lower() not in GENERATABLE_EXTENSIONS):
            continue
        groups.setdefault(posixpath.dirname(path), []).append(path)
    result = []
    for folder, outputs in groups.items():
        # Output should be chosen deterministically without baking task IDs in.
        first = sorted(outputs, key=lambda p: (
            0 if p.lower().endswith(".ifc") else 1, p
        ))[0]
        for name in GENERATOR_NAMES:
            script = posixpath.join(folder, name)
            result.append({"script": script, "target": first})
            if len(result) >= MAX_GENERATORS * len(groups):
                break
    return result


def snapshot_targets(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """Use delivery_guard.snapshot for pre- and post-run code fingerprints."""
    return [
        {"path": path, "kind": "file"}
        for path in dict.fromkeys(i["script"] for i in items)
    ]


def newly_written(before: dict | None, after: dict | None,
                  script: str) -> bool:
    """Never execute inherited unchanged input code or unknown observations."""
    if before is None or after is None:
        return False
    initial, latest = before.get(script), after.get(script)
    if not isinstance(initial, dict) or not isinstance(latest, dict):
        return False
    if latest.get("state") != "ready":
        return False
    if initial.get("state") in ("unknown", "invalid"):
        return False
    if initial.get("state") == "ready":
        return (initial.get("fingerprint")
                and initial.get("fingerprint") != latest.get("fingerprint"))
    return initial.get("state") in ("missing", "empty")


def plan(items: list[dict[str, str]], issues: list[dict[str, str]],
         before: dict | None, after: dict | None) -> list[dict[str, str]]:
    """Select new generators only for still-failed user-declared outputs."""
    failed = {i["path"] for i in issues}
    chosen: list[dict[str, str]] = []
    folders = set()
    for item in items:
        path, script = item["target"], item["script"]
        folder = posixpath.dirname(script)
        if (path not in failed or folder in folders
                or not _safe_absolute(path) or not _safe_absolute(script)):
            continue
        if newly_written(before, after, script):
            chosen.append(item)
            folders.add(folder)
        if len(chosen) >= MAX_GENERATORS:
            break
    return chosen


# Source is passed as a Python -c argument via shlex, and the exact parsed
# target/script values are passed as JSON. The script itself runs without a
# shell, in the project's task container, with output redacted.
REMOTE_MATERIALIZE = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

spec = json.loads(sys.argv[1])
script, target = Path(spec["script"]), Path(spec["target"])
timeout = max(1, min(55, int(spec.get("timeout", 55))))

def regular(path):
    for p in (path, *path.parents):
        if p.is_symlink():
            return False
    return True

def allowed(path):
    s = str(path)
    return (s.startswith("/app/") or s.startswith("/filesystem/")) \
        and ".." not in path.parts and "." not in path.parts \
        and path.is_absolute() and regular(path)

result = {"state":"skipped"}
if (allowed(script) and allowed(target) and script.is_file()
        and script.suffix == ".py" and script.parent == target.parent
        and not target.is_symlink()):
    env = {
        k: os.environ[k] for k in ("PATH","LANG","LC_ALL","TMPDIR")
        if k in os.environ
    }
    env["HOME"] = "/tmp"
    env["PYTHONNOUSERSITE"] = "1"
    started = time.monotonic()
    try:
        run = subprocess.run(
            [sys.executable, str(script), str(target)],
            cwd=str(script.parent),
            env=env,
            capture_output=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        result = {
            "state": "executed" if run.returncode == 0 else "nonzero",
            "exit_code": run.returncode,
            "wall_ms": round(1000*(time.monotonic()-started)),
            "target_created": target.is_file() and target.stat().st_size > 0,
        }
    except subprocess.TimeoutExpired:
        result = {"state": "timeout", "wall_ms": round(1000*(time.monotonic()-started))}
    except OSError:
        result = {"state": "exec_error"}
print(json.dumps(result, separators=(",",":")))
'''


async def materialize(session: Any, selected: list[dict[str, str]]) -> list[dict]:
    """Bounded task-native executions. Never assert engineering validity.

    session.exec is the competition action protocol, not host subprocesses.
    asyncio.CancelledError propagates so task cancellation remains reliable.
    """
    observations: list[dict] = []
    for item in selected[:MAX_GENERATORS]:
        left = session.time_left()
        if left is not None and left < 160:
            break
        timeout = PER_SCRIPT_TIMEOUT if left is None else min(
            PER_SCRIPT_TIMEOUT, max(1, int(left - 100))
        )
        spec = {"script": item["script"], "target": item["target"],
                "timeout": timeout}
        command = ("python3 -c " + shlex.quote(REMOTE_MATERIALIZE)
                   + " " + shlex.quote(json.dumps(spec, separators=(",",":"))))
        try:
            reply = await asyncio.wait_for(
                session.exec(command, timeout_sec=timeout + 8),
                timeout=timeout + 13,
            )
            if (reply.get("exit_code") != 0 or reply.get("timed_out")
                    or reply.get("truncated")):
                state = "probe_error"
            else:
                data = json.loads(reply.get("output") or "")
                state = data.get("state", "probe_error")
            observations.append({
                "target": item["target"], "state": state
            })
        except (OSError, KeyError, ValueError, RuntimeError, asyncio.TimeoutError):
            observations.append({"target": item["target"], "state": "unknown"})
    return observations
