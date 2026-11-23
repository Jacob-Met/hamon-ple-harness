"""Source-neutral PLE deliverable completion gate.

Detects only explicit output destinations in the participant's instruction.
Checks actual task-container files over the official WebSocket exec protocol,
never reads model keys, never touches the submission quota, and never mistakes
an inherited template file for a newly delivered answer. A failed inspection
is unknown, not evidence that the model failed.
"""
from __future__ import annotations

import asyncio
import csv
import inspect
import io
import json
import posixpath
import re
import shlex
from time import monotonic
from typing import Any
from xml.etree.ElementTree import ParseError
import zipfile
import zlib

import archive_contract
import office_package
from archive_checkpoint import verify_typed_members
from office_package import check_office_package
from taskbridge import Cancelled

FILE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<path>(?:/(?:filesystem|app)/)?"
    r"(?:[A-Za-z0-9_.+\-]+/)*[A-Za-z0-9_.+\-]+"
    r"\.(?:zip|docx|xlsx|xlsm|pdf|ifc|dwg|dxf|csv|json|md|txt|png))"
    r"(?=\.(?:\s|$)|[^A-Za-z0-9_.]|$)", re.I
)
DIRECTORY = re.compile(r"(?<![A-Za-z0-9_])(?P<path>output/)(?![A-Za-z0-9_])", re.I)
ACTION = re.compile(
    r"\b(?:save(?:d|s)?\b.{0,90}?\b(?:as|to|under)\b"
    r"|write\b.{0,90}?\b(?:to|as|under)\b"
    r"|(?:final\s+)?deliverable\b.{0,90}?\b(?:is|to|as)\b"
    r"|(?:produce|export|submit|place|put)\b.{0,90}?\b(?:in|to|as|under)\b"
    r"|(?:files?|outputs?)\b.{0,70}?\b(?:under|to)\b"
    r"|expected\s+output\s*submit\b)",
    re.I | re.S,
)
# An imperative may name its object without "to"/"as". Anchor the whole
# clause so an input mentioned later in the sentence cannot inherit the verb.
DIRECT_FILE = re.compile(
    r"(?:^|[.!?;]\s+|\n)\s*(?:please\s+)?(?:write|create|save|submit|export)\s+"
    r"(?:(?:(?:a|the)\s+)?file\s+(?:named|called)\s+"
    r"|(?:a|the)\s+first\s+complete\s+version\s+of\s+)?[`'\x22*]*$",
    re.I,
)
SCRIPT_DELIVERY = re.compile(
    r"\bexpected\s+output\b[\s:#*-]*(?:one|a single)\s+(?:self[- ]contained\s+)?"
    r"python\s+script\s+at\s+[`'\x22]*(?P<path>[A-Za-z0-9_./+\-]+\.py)"
    r"[`'\x22]*\s+that\s+(?:builds|generates|writes|produces)\b", re.I,
)
MAX_TARGETS = 24


def _safe_path(raw: str, cwd: str) -> str | None:
    """Normalize a literal task path, rejecting traversal and host paths."""
    path = raw.strip(" \t\r\n`'\"*.,;:")
    if not path or "\x00" in path or "\\" in path:
        return None
    if any(part in ("..", "~") for part in path.split("/")):
        return None
    if re.search(r"[$*{}\[\]?]", path):
        return None
    if path.startswith("/"):
        absolute = posixpath.normpath(path)
    else:
        absolute = posixpath.normpath(posixpath.join(cwd, path))
    if not (absolute.startswith("/filesystem/") or absolute == "/filesystem"
            or absolute.startswith("/app/") or absolute == "/app"):
        return None
    return absolute


def extract_targets(instruction: str, workdir: str) -> list[dict[str, str]]:
    """Extract high-precision file/dir deliverables; no guessed paths."""
    if not isinstance(instruction, str) or not isinstance(workdir, str):
        return []
    found: dict[str, dict[str, str]] = {}
    script = SCRIPT_DELIVERY.search(instruction)
    if script:
        script_path = _safe_path(script.group("path"), workdir)
        if script_path:
            found[script_path] = {"path": script_path, "kind": "file"}
    for match in FILE.finditer(instruction):
        context = instruction[max(0, match.start() - 180):match.start()]
        # Do not silently strip directory traversal into a valid-looking path.
        if context.endswith("../"):
            continue
        # A path is an output only when a nearby affirmative write clause says so.
        # Keep the last statement to prevent an earlier "write" marking inputs.
        clause = re.split(r"(?<=[.;])\s+|\n{2,}", context)[-1]
        if script:
            # The submitted code is the deliverable. A default runtime filename
            # and an explicitly optional side file are not additional obligations.
            if (re.search(r"\bdefault\s+[`'\x22]*$", context, re.I)
                    or re.search(r"\boptional side file\b", clause, re.I)):
                continue
        direct = DIRECT_FILE.search(instruction, max(0, match.start() - 180), match.start())
        if not (ACTION.search(clause) or direct):
            continue
        path = _safe_path(match.group("path"), workdir)
        if path:
            found[path] = {"path": path, "kind": "file"}
        if len(found) >= MAX_TARGETS:
            break
    if not found:
        for match in DIRECTORY.finditer(instruction):
            context = instruction[max(0, match.start() - 160):match.start()]
            clause = re.split(r"(?<=[.;])\s+|\n{2,}", context)[-1]
            if not ACTION.search(clause):
                continue
            path = _safe_path(match.group("path"), workdir)
            if path:
                found[path] = {"path": path, "kind": "dir"}
                break
    filing = named_return(instruction)
    if filing and filing["destination"]:
        destination = filing["destination"]
        if destination.startswith(("/filesystem/", "/app/")):
            target = _safe_path(posixpath.join(destination, filing["name"]), workdir)
            if target:
                found[target] = {"path": target, "kind": "file"}
    return list(found.values())


def named_return(prompt: str) -> dict | None:
    """A required returned filename need not identify a task-container path."""
    for match in FILE.finditer(prompt):
        sentence = re.split(r"[.!?]\s+|\n", prompt[max(0, match.start()-240):match.start()])[-1]
        if not re.search(r"\breturn\b[^.;!?]*\b(?:named|called)\s+[`'\x22*]*$", sentence, re.I):
            continue
        name = match.group("path")
        # Explicit paths are handled by the ordinary local-path check below.
        if "/" in name:
            continue
        suffix = re.split(r"[.;!?]\s+|\n", prompt[match.end():match.end()+300], maxsplit=1)[0]
        filing = re.match(r"^[`'\x22*]*\s*,?\s*(?:to be\s+)?(?:filed|uploaded|posted|submitted)\s+(?:in|to|under)\s+([^,.;!?\n]+)", suffix, re.I)
        return {"name": name, "destination": filing.group(1).strip() if filing else None}
    return None


class TerminalDelivery:
    """Keep file observations separate from the participant's delivery claims."""
    def __init__(self, instruction, targets):
        self.paths = [target["path"] for target in targets]
        named = self.named = named_return(instruction)
        self.filing = named if named and named["destination"] and not named["destination"].startswith(("/app/", "/filesystem/")) else None
        self.copy_path = None
        self.before = None
        self.after = None
        self.observed = False

    def add_working_copy(self, workdir):
        """Reserve an observable copy for a named return without a literal file path."""
        if self.paths or not self.named:
            return None
        base = workdir if workdir == "/filesystem" or workdir.startswith("/filesystem/") else "/app/output"
        self.copy_path = _safe_path(posixpath.join(base, ".ple-delivery", self.named["name"]), base)
        if self.copy_path:
            self.paths.append(self.copy_path)
        return self.copy_path

    def copy_instruction(self, path):
        note = (f"\n\nKeep a working copy of {self.named['name']} at {path}. "
                "Create the parent directory if needed. ")
        if not self.filing:
            return note + "Use the task records to complete the named file and verify its requested format."
        return (note
                + f"The requested final destination is {self.filing['destination']}. "
                "Check the destination first. If already filed, read the existing artifact "
                "back into the working copy. Otherwise create the document from the task records, "
                "validate it, file it using the task's document-store interface, and read it back. "
                "If you build the document with a script, save the actual generator as generate.py "
                "beside the working copy so an interrupted build can resume. "
                "Keep the working copy for verification. Inspect existing records before any "
                "related case updates and avoid duplicating completed actions.")

    def invalidate(self):
        self.after = None
        self.observed = False

    def observe(self, snapshot):
        self.after = snapshot
        self.observed = True

    def summary(self, participant):
        missing = []
        unknown = []
        for path in self.paths:
            row = (self.after or {}).get(path) or {}
            prior = (self.before or {}).get(path) or {}
            state = row.get("state")
            if state in ("missing", "empty", "invalid"):
                missing.append(f"{path} ({row.get('reason') or state})")
            elif state != "ready" or prior.get("state") in (None, "unknown"):
                unknown.append(path)
            elif prior.get("state") == "ready" and not (prior.get("fingerprint") and row.get("fingerprint")):
                unknown.append(path)
            elif prior.get("state") == "ready" and prior.get("fingerprint") == row.get("fingerprint"):
                missing.append(f"{path} (unchanged starter)")
        notes = []
        if missing:
            label = "Incomplete working copy: " if self.copy_path else "Incomplete delivery: "
            notes.append(label + ", ".join(missing) + ".")
        if unknown:
            notes.append("Current delivery could not be verified for " + ", ".join(unknown) + ".")
        if self.filing:
            notes.append(f"External filing of {self.filing['name']} in {self.filing['destination']} was not independently verified.")
        if not notes:
            return participant
        if participant and not missing:
            notes.append("Participant report: " + participant)
        return "\n".join(notes)



def file_format_state(p, size, root_members=None):
    """Apply the same physical format checks to live and archived deliverables."""
    if size == 0:
        return {"state": "empty"}
    ext = p.suffix.lower()
    if ext in (".zip",".docx",".xlsx",".xlsm"):
        zip_errors = (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error)
        try:
            from lzma import LZMAError
        except ImportError:
            pass  # Python can run without the optional LZMA codec.
        else:
            zip_errors += (LZMAError,)
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                if not names:
                    return {"state":"invalid","reason":"zip_empty"}
                if root_members:
                    if sorted(names) != sorted(root_members):
                        return {"state":"invalid","reason":"zip_root_members_mismatch"}
                    if any(member.file_size == 0 for member in z.infolist()):
                        return {"state":"invalid","reason":"zip_empty_root_member"}
                    try:
                        if size > 32_000_000 or not verify_typed_members(z, root_members, 32_000_000):
                            return {"state":"unknown"}
                    except (ValueError, EOFError, csv.Error, zipfile.BadZipFile, RuntimeError, zlib.error):
                        return {"state":"invalid","reason":"zip_invalid_member_format"}
                if ext in (".docx", ".xlsx", ".xlsm"):
                    if size > 32_000_000 or sum(m.file_size for m in z.infolist()) > 32_000_000:
                        return {"state":"unknown"}
                    try:
                        check_office_package(z, ext)
                    except (ValueError, ParseError, LookupError):
                        return {"state":"invalid","reason":"office_invalid_package"}
                if size <= 32_000_000 and z.testzip() is not None:
                    return {"state":"invalid","reason":"zip_crc"}
        except zip_errors:
            return {"state":"invalid","reason":"zip_invalid"}
    elif ext == ".pdf":
        with p.open("rb") as f:
            if f.read(5) != b"%PDF-":
                return {"state":"invalid","reason":"pdf_header"}
    elif ext == ".ifc":
        with p.open("rb") as f:
            if b"ISO-10303-21;" not in f.read(512):
                return {"state":"invalid","reason":"ifc_header"}
    return {"state": "ready"}

# Executed in the evaluator's task container. The only transport payload is
# file status, size and a fingerprint (no task document bodies or metadata).
REMOTE_PROBE = r'''
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import zlib

from xml.etree.ElementTree import ParseError

''' + inspect.getsource(office_package) + '\n' + inspect.getsource(verify_typed_members) + '\n' + inspect.getsource(file_format_state) + r'''

targets = json.loads(sys.argv[1])
result = {}

def regular(path):
    # Do not follow file or ancestor symlinks.
    for p in (path, *path.parents):
        try:
            if p.is_symlink():
                return False
        except OSError:
            return False
    return True

def scan_file(p, root_members):
    try:
        if not p.is_file():
            return {"state":"missing"}
        st = p.stat()
        if st.st_size == 0:
            return {"state":"empty"}
        h = hashlib.sha256()
        h.update(str(st.st_size).encode())
        # Equal-sized CAD edits can occur anywhere in a large deliverable.
        # Stream the complete file under the task protocol's existing deadline.
        with p.open("rb") as f:
            while block := f.read(1_048_576):
                h.update(block)
        shape = file_format_state(p, st.st_size, root_members)
        if shape["state"] != "ready":
            return shape
        return {"state":"ready","fingerprint":h.hexdigest(),"size":st.st_size}
    except (OSError,ValueError):
        return {"state":"unknown"}

def scan_dir(p):
    if not p.is_dir():
        return {"state":"missing"}
    h = hashlib.sha256()
    count = 0

    def raise_walk_error(error):
        raise error

    try:
        for root, dirs, files in os.walk(p, followlinks=False, onerror=raise_walk_error):
            folder = Path(root)
            dirs[:] = sorted(d for d in dirs if not (folder / d).is_symlink())
            for name in sorted(files):
                child = folder / name
                if not regular(child) or not child.is_file():
                    continue
                member = scan_file(child, None)
                if member["state"] == "empty":
                    continue
                if member["state"] != "ready":
                    return member
                # Frame each pathname and fixed-width content digest separately.
                relative = os.fsencode(child.relative_to(p))
                h.update(len(relative).to_bytes(8, "big"))
                h.update(relative)
                h.update(bytes.fromhex(member["fingerprint"]))
                count += 1
        return {"state":"ready" if count else "empty",
                "fingerprint":h.hexdigest(),"files":count}
    except (OSError,ValueError):
        return {"state":"unknown"}

for item in targets[:24]:
    path = item.get("path")
    kind = item.get("kind")
    if not isinstance(path,str) or not (path.startswith("/filesystem/") or
       path.startswith("/app/")) or ".." in path.split("/"):
        continue
    p = Path(path)
    if not regular(p):
        result[path] = {"state":"invalid","reason":"symlink_path"}
    elif kind == "dir":
        result[path] = scan_dir(p)
    else:
        result[path] = scan_file(p, item.get("root_members"))
print(json.dumps(result,separators=(",",":")))
'''


async def snapshot(session: Any, targets: list[dict[str, str]], *,
                   deadline: float | None = None) -> dict | None:
    """One bounded, read-only remote observation; None means unavailable."""
    if not targets:
        return {}
    members = archive_contract.declared_members(getattr(session, "instruction", ""), targets)
    probe_targets = [{**targets[0], "root_members": members}] if members else targets
    argv = shlex.quote(json.dumps(probe_targets, separators=(",",":")))
    cmd = "python3 -c " + shlex.quote(REMOTE_PROBE) + " " + argv
    try:
        execution_seconds, wait_seconds = 25, 29.0
        if deadline is not None:
            remaining = deadline - monotonic()
            # Taskbridge clamps remote exec limits to at least one second.
            if remaining < 1:
                return None
            execution_seconds = min(execution_seconds, int(remaining))
            wait_seconds = min(wait_seconds, remaining)
        reply = await asyncio.wait_for(
            session.exec(cmd, timeout_sec=execution_seconds), timeout=wait_seconds
        )
        if reply.get("timed_out") or reply.get("truncated") or reply.get("exit_code") != 0:
            return None
        data = json.loads(reply.get("output") or "")
        if not isinstance(data, dict):
            return None
        return data
    except Cancelled:
        raise
    except Exception:
        # Any task-protocol observation failure is unknown, never model failure.
        # Both evaluator cancellation and asyncio cancellation propagate.
        return None


def compare(before: dict | None, after: dict | None) -> list[dict[str, str]]:
    """Only known evidence can trigger a retry; no fabricated failures."""
    if before is None or after is None:
        return []
    issues = []
    for path, initial in before.items():
        if not isinstance(initial, dict) or initial.get("state") == "unknown":
            continue
        latest = after.get(path)
        if not isinstance(latest, dict) or latest.get("state") == "unknown":
            continue
        state = latest.get("state")
        if state != "ready":
            issues.append({"path": path, "reason": latest.get("reason") or state})
        elif initial.get("state") == "ready" and (
                initial.get("fingerprint") == latest.get("fingerprint")):
            issues.append({"path": path, "reason": "unchanged"})
    return issues


def repair_instruction(issues: list[dict[str, str]]) -> str:
    targets = "\n".join(
        "- " + i["path"] + " (" + i["reason"] + ")" for i in issues[:MAX_TARGETS]
    )
    return (
        "\n\nSECOND PASS — REQUIRED DELIVERY REPAIR.\n"
        "The previous pass ended, but the task-container output check found "
        "required deliverables missing, invalid, empty, or unchanged from "
        "the original starter. The requested exact paths are:\n" + targets +
        "\nInspect the current task files, preserve valid work, and repair these "
        "destinations before refining anything else. Verify the actual files "
        "with shell commands, not just a written claim. This check proves only "
        "file delivery and container integrity, not engineering correctness. "
        "Finish with a concise summary identifying final paths."
    )
