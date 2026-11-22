"""Enable two-pass ZIP checkpoints only for an explicitly enumerated early-delivery contract."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import secrets
import shlex

NUMBERS = "zero one two three four five six seven eight nine ten eleven twelve".split()
HEADER = re.compile(
    r"containing\s+exactly\s+(?:these\s+)?(\d+|" + "|".join(NUMBERS) +
    r")\s+files\b[^:]{0,100}?(?:archive root|top level)\b[^:]{0,90}:", re.I)
MEMBER = re.compile(r"(?m)^\s*(\d+)\.\s+\*{0,2}`([\w+ .-]+\.[\w]+)`")


def member_names(instruction):
    if not re.search(r"\b(?:early.{0,65}improv|first.{0,45}complete.{0,90}then iterat)", instruction, re.I | re.S):
        return []
    h = HEADER.search(instruction)
    if not h:
        return []
    raw = h[1].lower()
    count = int(raw) if raw.isdecimal() else NUMBERS.index(raw)
    if not 1 <= count <= 32:
        return []
    found = list(MEMBER.finditer(instruction, h.end()))[:count]
    if [int(m[1]) for m in found] != list(range(1,count+1)):
        return []
    names = [m[2] for m in found]
    return names if len(set(names)) == count else []


async def provision(session, targets):
    if len(targets) != 1 or targets[0]["kind"] != "file" or not targets[0]["path"].lower().endswith(".zip"):
        return None
    names = member_names(session.instruction)
    if not names:
        return None
    tag = secrets.token_hex(8)
    helper, stage = f"/tmp/ple-archive-{tag}.py", f"/tmp/ple-members-{tag}"
    body = Path(__file__).with_name("archive_checkpoint.py").read_bytes()
    try:
        if await asyncio.wait_for(session.write_file(helper, body), 12) != len(body):
            return None
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


async def run_command(session, command):
    try:
        reply = await asyncio.wait_for(session.exec(command, timeout_sec=30), 33)
        return (reply.get("exit_code") == 0 and not reply.get("timed_out")
                and not reply.get("truncated"))
    except Exception:
        return False


class FirstPublication:
    """Publish the first complete staged set at an acknowledged task-tool boundary."""

    def __init__(self, plan):
        self.stage, self.command = plan
        self.pending = set()
        self.published = False

    async def observe(self, session, event):
        if self.published:
            return
        if event.get("type") == "tool_execution_start":
            call_id = event.get("toolCallId")
            args = event.get("args") or {}
            if (isinstance(call_id, str) and isinstance(args, dict)
                    and event.get("toolName") in {"write", "edit", "bash"}
                    and any(isinstance(arg, str) and self.stage in arg for arg in args.values())):
                self.pending.add(call_id)
        elif event.get("type") == "message_end":
            msg = event.get("message") or {}
            call_id = msg.get("toolCallId")
            if msg.get("role") == "toolResult" and call_id in self.pending:
                self.pending.remove(call_id)
                if not msg.get("isError") and await run_command(session, self.command):
                    self.published = True
                    await session.log("delivery_gate: first complete archive published during initial pass")
