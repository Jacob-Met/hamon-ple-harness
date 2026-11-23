"""Enable two-pass ZIP checkpoints only for an explicitly enumerated early-delivery contract."""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import re
import secrets
import shlex
from time import monotonic
import zipfile

from taskbridge import Cancelled

NUMBERS = "zero one two three four five six seven eight nine ten eleven twelve".split()
EARLY_RE = re.compile(
    r"\b(?:early.{0,65}improv|first.{0,60}complete.{0,120}(?:then\s+)?(?:iterat|improv))",
    re.I | re.S,
)
ZIP_OUT_RE = re.compile(r"\b[\w./-]+\.zip\b", re.I)
PARAMETRIC_ZIP_RE = re.compile(
    r"\.zip`?\*{0,2}\s+containing,\s+for\s+\*{0,2}each\*{0,2}\b[^:\n]{0,90}\bin\s*\{",
    re.I,
)
COUNT_HEADERS = (
    re.compile(
        r"containing\s+(?:exactly\s+(?:these\s+)?|these\s+|the\s+)?(\d+|"
        + "|".join(NUMBERS)
        + r")\s+files\b[^:]{0,160}?(?:archive root|top level|below|templates)[^:]{0,110}[:.]",
        re.I | re.S,
    ),
    re.compile(
        r"(?:write|produce)\s+(?:exactly\s+)?(?:these\s+|the\s+)(\d+|"
        + "|".join(NUMBERS)
        + r")\s+files\b[^:.]{0,160}[.:]",
        re.I | re.S,
    ),
    re.compile(
        r"(?m)^##+\s+Deliverables?\s*\n+(\d+|"
        + "|".join(NUMBERS)
        + r")\s+files\s*[.:]",
        re.I,
    ),
)
ROOT_HEADING_CONTRACT_RE = re.compile(
    r"Write\s+one\s+file,\s*`[^`\n]+\.zip`,\s*containing\s+at\s+its\s+root\s*:\s*\n",
    re.I,
)
BULLET_ZIP_CONTRACT_RE = re.compile(
    r"Hand\s+back\s+`[^`\n]+\.zip`\s+containing\s*:?\s*\n+",
    re.I,
)
NUMBERED_ITEM_RE = re.compile(
    r"(?m)^\s*\*{0,2}(\d+)\.\s+\*{0,2}`([\w+ .-]+\.[A-Za-z0-9]+)`(?!`*/)"
)
HEADING_ITEM_RE = re.compile(
    r"(?m)^###\s+(\d+)\.\s+\*{0,2}`([\w+ .-]+\.[A-Za-z0-9]+)`(?!`*/)"
)
TABLE_ROW_RE = re.compile(
    r"(?m)^\|\s*`([\w+ .-]+\.[A-Za-z0-9]+)`\s*\|"
)
BULLET_ITEM_RE = re.compile(
    r"^\s*-\s+`([\w+ .-]+\.[A-Za-z0-9]+)`(?:\s+--|\s+[-\u2013\u2014:])"
)
STOP_SECTION_RE = re.compile(
    r"(?m)^##\s+(?:How\b|Grading\b|Verifier\b|Section\s+format\b|Deliverable\s+schema\b|Attribution\b)"
)


def _parse_count(token: str) -> int:
    low = token.lower()
    return int(low) if low.isdecimal() else NUMBERS.index(low)


def _valid_distinct_root_names(names: list[str], expected_count: int | None = None) -> list[str]:
    if not names or len(set(names)) != len(names):
        return []
    if any("/" in n or "\\" in n or n in {".", ".."} for n in names):
        return []
    if expected_count is not None and len(names) != expected_count:
        return []
    return names


def _bounded_tail(instruction: str, start: int) -> str:
    stop = STOP_SECTION_RE.search(instruction, start)
    return instruction[start : stop.start()] if stop else instruction[start:]


def member_names(instruction: str) -> list[str]:
    if not EARLY_RE.search(instruction) or not ZIP_OUT_RE.search(instruction):
        return []
    if PARAMETRIC_ZIP_RE.search(instruction):
        return []

    for header_re in COUNT_HEADERS:
        for h in header_re.finditer(instruction):
            count = _parse_count(h[1])
            if not 1 <= count <= 32:
                continue
            scope = _bounded_tail(instruction, h.end())
            for item_re in (NUMBERED_ITEM_RE, HEADING_ITEM_RE):
                found = list(item_re.finditer(scope))[:count]
                if [int(m[1]) for m in found] == list(range(1, count + 1)):
                    valid = _valid_distinct_root_names([m[2] for m in found], count)
                    if valid:
                        return valid
            table_window = scope[:1600]
            trows = [m[1] for m in TABLE_ROW_RE.finditer(table_window)]
            if len(trows) == count:
                valid = _valid_distinct_root_names(trows, count)
                if valid:
                    return valid

    m_root = ROOT_HEADING_CONTRACT_RE.search(instruction)
    if m_root:
        scope = _bounded_tail(instruction, m_root.end())
        found = list(HEADING_ITEM_RE.finditer(scope))
        if len(found) >= 2 and [int(m[1]) for m in found] == list(range(1, len(found) + 1)):
            valid = _valid_distinct_root_names([m[2] for m in found])
            if valid:
                return valid

    m_bul = BULLET_ZIP_CONTRACT_RE.search(instruction)
    if m_bul:
        scope = _bounded_tail(instruction, m_bul.end())
        bullets: list[str] = []
        started = False
        for line in scope.splitlines():
            bm = BULLET_ITEM_RE.match(line)
            if bm:
                bullets.append(bm[1])
                started = True
            elif started:
                if not line.strip() or not line.startswith("  "):
                    break
        if len(bullets) >= 2:
            valid = _valid_distinct_root_names(bullets)
            if valid:
                return valid

    return []


def declared_members(instruction, targets):
    if len(targets) != 1 or targets[0]["kind"] != "file" or not targets[0]["path"].lower().endswith(".zip"):
        return []
    return member_names(instruction)


async def provision(session, targets, *, deadline: float | None = None):
    names = declared_members(getattr(session, "instruction", ""), targets)
    if not names:
        return None
    if deadline is not None and deadline <= monotonic():
        return None
    tag = secrets.token_hex(8)
    helper, stage = f"/tmp/ple-archive-{tag}.pyz", f"/tmp/ple-members-{tag}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(Path(__file__).with_name("archive_checkpoint.py"), "__main__.py")
        bundle.write(Path(__file__).with_name("office_package.py"), "office_package.py")
    body = buffer.getvalue()
    try:
        timeout = 12.0 if deadline is None else min(12.0, deadline - monotonic())
        if timeout <= 0:
            return None
        if await asyncio.wait_for(session.write_file(helper, body), timeout) != len(body):
            return None
    except Cancelled:
        raise
    except Exception:
        return None
    target = targets[0]["path"]
    command = "python3 " + shlex.quote(helper) + " " + shlex.quote(json.dumps([stage, target, names]))
    return stage, command


def first_instruction(stage, command):
    return (
        "\n\nFIRST COMPLETE VERSION. Use the project inputs to produce the first feasible complete version. "
        f"Create {stage} and write every required archive member there. Validate the engineering contents. "
        "Publish all members together with this exact command:\n" + command +
        "\nStop after publication. Optional optimization belongs to a subsequent pass. "
        "Always revise the staged members and use the publisher to replace the graded archive atomically."
    )


async def _command_result(session, command):
    try:
        reply = await asyncio.wait_for(session.exec(command, timeout_sec=30), 33)
        if (reply.get("exit_code") == 0 and not reply.get("timed_out")
                and not reply.get("truncated")):
            return reply
    except Cancelled:
        raise
    except Exception:
        pass
    return None


async def run_command(session, command):
    return await _command_result(session, command) is not None


async def recover_after_refinement(session, command):
    reply = await _command_result(session, command + " --recover")
    if reply is not None:
        try:
            outcome = json.loads(reply.get("output", ""))
        except (TypeError, ValueError):
            return None
        action = outcome.get("action") if isinstance(outcome, dict) else None
        if isinstance(action, str) and action in {"kept", "published", "restored"}:
            return action
    return None


class FirstPublication:
    """Publish the first complete staged set at an acknowledged task-tool boundary."""

    def __init__(self, plan, *, target=None, members=()):
        self.stage, self.command = plan
        self.required_outputs = {str(Path(target).parent / name) for name in members} if target else set()
        self.written_outputs: set[str] = set()
        self.pending_outputs: dict[str, tuple[str, str]] = {}
        self.pending: dict[str, bool] = {}
        self.published = False

    async def observe(self, session, event):
        if self.published:
            return
        if event.get("type") == "tool_execution_start":
            call_id = event.get("toolCallId")
            args = event.get("args") or {}
            tool = event.get("toolName")
            if (isinstance(call_id, str) and isinstance(args, dict) and tool in {"write", "edit"}
                    and isinstance(args.get("path"), str) and args["path"] in self.required_outputs):
                self.pending_outputs[call_id] = (tool, args["path"])
            if (isinstance(call_id, str) and isinstance(args, dict)
                    and event.get("toolName") in {"write", "edit", "bash"}
                    and any(isinstance(arg, str) and self.stage in arg for arg in args.values())):
                self.pending[call_id] = (event.get("toolName") == "bash"
                                         and args.get("command") == self.command)
        elif event.get("type") == "message_end":
            msg = event.get("message") or {}
            call_id = msg.get("toolCallId")
            if msg.get("role") == "toolResult" and call_id in self.pending_outputs:
                tool, path = self.pending_outputs.pop(call_id)
                if msg.get("toolName") == tool and msg.get("isError") is False:
                    self.written_outputs.add(path)
                    if self.written_outputs == self.required_outputs and await run_command(session, self.command):
                        self.published = True
                        await session.log("delivery_gate: complete archive published from acknowledged output members")
                return
            if msg.get("role") == "toolResult" and call_id in self.pending:
                explicit_publisher = self.pending.pop(call_id)
                if msg.get("isError"):
                    return
                # The exact provisioned publisher emits this acknowledgement only
                # after format/CRC checks, change detection and atomic publication.
                # A staging command or an ambiguous result still needs our publisher.
                acknowledged = (explicit_publisher and msg.get("toolName") == "bash"
                                and msg.get("isError") is False
                                and msg.get("content") == [{"type": "text", "text": "PLE_ARCHIVE_PUBLISHED"}])
                if acknowledged or await run_command(session, self.command):
                    self.published = True
                    await session.log("delivery_gate: first complete archive published during initial pass")
