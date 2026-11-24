"""Materialize fresh agent-written generators within the task's remaining time.

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
from time import monotonic
from typing import Any

from taskbridge import Cancelled

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


def candidates(targets: list[dict[str, str]], *, archive_members=(), witness=False) -> list[dict[str, Any]]:
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
    # Read-only observation is bounded by delivery_guard.REMOTE_PROBE's 24
    # target cap. Allocate the first convention across all target directories
    # before spending probe slots on alternative script conventions.
    result = []
    grouped_outputs = {
        folder: sorted(set(outputs), key=lambda p: (
            0 if p.lower().endswith(".ifc") else 1, p
        ))
        for folder, outputs in groups.items()
    }
    for name in GENERATOR_NAMES:
        for folder, outputs in grouped_outputs.items():
            if len(result) >= 24:
                return result
            result.append({
                "script": posixpath.join(folder, name),
                "target": outputs[0], "outputs": outputs,
            })
    if len(targets) == 1 and groups and len(result) < 24:
        target = targets[0]["path"]
        supported = {".json", ".md", ".txt", ".csv", ".xlsx"}
        archive = archive_members and all(
            posixpath.splitext(name)[1].lower() in supported for name in archive_members
        )
        if (archive and target.lower().endswith(".zip")) or (witness and target.lower().endswith(".docx")):
            result.append({"script": target + ".draft.json", "target": target,
                           "outputs": [target], "draft_members": list(archive_members)})
    return result


def draft_note(items):
    draft = next((item for item in items if "draft_members" in item), None)
    if not draft:
        return ""
    prefix = ("\\nFor interrupted delivery, you may save the completed project data to "
              + draft["script"] + ". ")
    if not draft["draft_members"]:
        return prefix + ("Use the witness-package JSON fields for the actual dated roster, "
                         "decisions, source references and standing case items. The controller "
                         "can export a reconciled record to the local Word working copy.")
    return prefix + ('Use {"members":{...}} keyed by every required root filename. '
                     'For JSON use {"json": value}, for text use {"text": "contents"}, '
                     'and for XLSX use {"template": "/filesystem/.../owner.xlsx", '
                     '"cells": {"Actual sheet": {"B3": 12.5}}}. Use your project-derived '
                     'values and the supplied workbook templates.')


def snapshot_targets(items: list[dict[str, Any]]) -> list[dict[str, str]]:
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
        return bool(initial.get("fingerprint") and (
            initial.get("fingerprint") != latest.get("fingerprint")
        ))
    return initial.get("state") in ("missing", "empty")


def plan(items: list[dict[str, Any]], issues: list[dict[str, str]],
         before: dict | None, after: dict | None, *,
         attempted: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Select new generators only for still-failed user-declared outputs."""
    failed = {i["path"] for i in issues}
    chosen: list[dict[str, Any]] = []
    folders = set()
    for item in items:
        path, script = item["target"], item["script"]
        folder = posixpath.dirname(script)
        # A generator can have completed its preferred target before stopping
        # short of a sibling. Keep the original invocation target, but consider
        # every declared output in that directory when deciding to rerun it.
        if (not any(output in failed for output in item.get("outputs", [path]))
                or folder in folders
                or not _safe_absolute(path) or not _safe_absolute(script)):
            continue
        latest = (after or {}).get(script)
        fingerprint = latest.get("fingerprint") if isinstance(latest, dict) else None
        if (newly_written(before, after, script)
                and (attempted is None or (script, fingerprint) not in attempted)):
            chosen.append({**item, "fingerprint": fingerprint} if "draft_members" in item else item)
            folders.add(folder)
        if len(chosen) >= MAX_GENERATORS:
            break
    return chosen


# Source is passed as a Python -c argument via shlex, and the exact parsed
# target/script values are passed as JSON. The script itself runs without a
# shell, in the project's task container, with output redacted.
REMOTE_MATERIALIZE = r'''
import contextlib
import json
import os
from pathlib import Path
import signal
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

def observe_exit(process, seconds):
    # Keep the exited leader unreaped until its group has been signalled, so
    # that its numeric ID cannot be reused for an unrelated process group.
    deadline = time.monotonic() + seconds
    while True:
        status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is not None:
            return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, seconds)
        time.sleep(min(.02, remaining))

def stop_generator(process):
    # The generator owns its session. Stop its group before observing outputs.
    # A one-second grace fits within the outer eight-second exec allowance.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        observe_exit(process, 1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()

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
        process = subprocess.Popen(
            [sys.executable, str(script), str(target)],
            cwd=str(script.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            shell=False,
        )
        try:
            code = observe_exit(process, timeout)
            state = "executed" if code == 0 else "nonzero"
        except subprocess.TimeoutExpired:
            code, state = None, "timeout"
        finally:
            stop_generator(process)
        result = {"state": state, "wall_ms": round(1000*(time.monotonic()-started))}
        if code is not None:
            result.update(exit_code=code,
                          target_created=target.is_file() and target.stat().st_size > 0)
    except OSError:
        result = {"state": "exec_error"}
print(json.dumps(result, separators=(",",":")))
'''


async def materialize(session: Any, selected: list[dict[str, Any]], *,
                      reserve_sec: float = 100,
                      deadline: float | None = None) -> list[dict]:
    """Bounded task-native executions. Never assert engineering validity.

    session.exec is the competition action protocol, not host subprocesses.
    Evaluator and asyncio cancellation propagate to the caller. Keep the caller's
    verification/repair reserve plus the entire remote and transport allowance.
    A shared monotonic deadline bounds a final pass even if time_left is absent.
    """
    observations: list[dict] = []
    for item in selected[:MAX_GENERATORS]:
        left = session.time_left()
        if deadline is not None:
            left = deadline - monotonic() if left is None else min(left, deadline - monotonic())
        timeout = PER_SCRIPT_TIMEOUT if left is None else min(
            PER_SCRIPT_TIMEOUT, int(left - reserve_sec - 13)
        )
        if timeout < 1:
            break
        spec = {"script": item["script"], "target": item["target"],
                "timeout": timeout}
        if "draft_members" in item:
            command = shlex.join(["python3", "/tmp/hamon-engineering-toolkit.py",
                                  "materialize-draft", item["script"], item["target"],
                                  item["fingerprint"], json.dumps(item["draft_members"])])
        else:
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
        except Cancelled:
            raise
        except Exception:
            # The official protocol raises its own TaskError, which is not
            # an OSError or RuntimeError. Unavailable observation must never
            # abort the user's original deliverable task. Both evaluator
            # cancellation and asyncio cancellation still propagate.
            observations.append({"target": item["target"], "state": "unknown"})
    return observations

