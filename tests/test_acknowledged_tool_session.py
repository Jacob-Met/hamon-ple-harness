"""Exercise the actual solve boundary with matched and unresolved tool results."""
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import harness


class AcknowledgedToolSession(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, mode, *, local=False, remaining=240):
        prompt = ("Write the report to output/report.pdf." if local
                  else "File the report in the team's document store.")
        session = SimpleNamespace(instruction=prompt, workdir="/filesystem", mcp_servers=[],
                                  log=mock.AsyncMock(), usage=mock.AsyncMock(),
                                  time_left=lambda: remaining, final_message="")
        assistant = {"role": "assistant", "stopReason": "toolUse", "content": [
            {"type": "text", "text": "I will check the document."},
            {"type": "toolCall", "id": "a", "name": "read", "arguments": {"path": "report"}},
            {"type": "toolCall", "id": "b", "name": "read", "arguments": {"path": "index"}},
        ]}
        results = [{"role": "toolResult", "toolCallId": i, "toolName": "read",
                    "content": [{"type": "text", "text": "observed " + i}],
                    "isError": i == "b"} for i in ("b", "a")]
        calls = []
        async def process(argv, s, *, on_line, **kwargs):
            calls.append((list(argv), kwargs["stdin"]))
            async def event(message):
                await on_line(json.dumps({"type": "message_end", "message": message}).encode())
            if len(calls) == 1:
                directory = Path(argv[argv.index("--session-dir") + 1])
                directory.mkdir()
                header = {"type": "session", "id": "test", "version": 3, "cwd": "/work"}
                user = {"role": "user", "content": prompt}
                await on_line(json.dumps(header).encode())
                await event(user)
                await event(assistant)
                live = results if mode != "partial" else results[:1]
                for value in live:
                    await event(value)
                persisted = list(live)
                if mode == "changed":
                    persisted[0] = dict(results[0], isError=False)
                if mode == "omitted":
                    persisted = persisted[:1]
                rows = [header]
                if mode == "earlier":
                    user = dict(user, content="different earlier history")
                for i, msg in enumerate([user, assistant, *persisted]):
                    rows.append({"type": "message", "id": str(i),
                                 "parentId": str(i-1) if i else None, "message": msg})
                if mode == "branch":
                    rows[-1]["parentId"] = None
                if mode == "ambiguous":
                    (directory / "other.jsonl").touch()
                (directory / "single.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            else:
                self.assertEqual(len(calls), 2)
                if not local:
                    file = Path(argv[argv.index("--session") + 1])
                    self.assertEqual(file.name, "single.jsonl")
                    self.assertTrue(file.exists())
                await event({"role": "assistant", "stopReason": "stop", "content": [
                    {"type": "text", "text": "The requested status was recorded."}]})
            return SimpleNamespace(code=0, stderr="")
        relay = mock.MagicMock()
        relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
        with contextlib.ExitStack() as stack:
            def replace(obj, name, replacement):
                stack.enter_context(mock.patch.object(obj, name, replacement))
            replace(harness.taskbridge, "Relay", lambda _: relay)
            replace(harness.taskbridge, "run_process", process)
            replace(harness, "configure", lambda *a: ("model", "fixture"))
            replace(harness.engineering_intake, "intake_for_prompt", mock.AsyncMock(return_value=""))
            replace(harness.engineering_intake, "provision_engineering_cli", mock.AsyncMock(return_value=""))
            replace(harness.delivery_guard, "snapshot", mock.AsyncMock(return_value={
                "/filesystem/output/report.pdf": {"state": "missing"}} if local else {}))
            result = await harness.solve(session)
        return calls, result

    async def test_acknowledged_remote_tool_turn_continues_original_session_once(self):
        calls, result = await self.exercise("complete")
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, "The requested status was recorded.")

    async def test_pending_or_unverified_tool_results_never_continue(self):
        for mode in ("partial", "changed", "omitted", "ambiguous", "branch", "earlier"):
            with self.subTest(mode=mode):
                calls, result = await self.exercise(mode)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("I will check", result)
        calls, _ = await self.exercise("partial", local=True)
        self.assertEqual(len(calls), 1)

    async def test_empty_error_cannot_promote_previous_tool_progress_to_final(self):
        session = SimpleNamespace(final_message="", log=mock.AsyncMock(), usage=mock.AsyncMock())
        run = harness.Run(session, "fixture")
        for message in [
            {"role": "assistant", "stopReason": "toolUse", "content": [
                {"type": "text", "text": "I will inspect the report."},
                {"type": "toolCall", "id": "a", "name": "read", "arguments": {}}]},
            {"role": "toolResult", "toolCallId": "a", "content": []},
            {"role": "assistant", "stopReason": "error", "content": []},
        ]:
            await run.on_event({"type": "message_end", "message": message})
        self.assertEqual(session.final_message, "")

    async def test_insufficient_budget_keeps_only_the_original(self):
        calls, _ = await self.exercise("complete", remaining=35)
        self.assertEqual(len(calls), 1)
