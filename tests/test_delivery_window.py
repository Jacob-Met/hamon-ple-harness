"""Exercise the deadline transition through the harness' normal solve entry point."""
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import harness


class DeliveryWindow(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, persisted, finish_first=False):
        instruction = "Calculate the beam reactions. Save the report to output/report.pdf."
        target = "/filesystem/example/output/report.pdf"
        session = SimpleNamespace(
            workdir="/filesystem/example", instruction=instruction, mcp_servers=[],
            final_message="", log=mock.AsyncMock(), usage=mock.AsyncMock(),
        )
        left = 1200.0
        session.time_left = lambda: left
        ready = False
        calls = []

        async def snapshot(_, targets):
            if not targets:
                return {}
            return {target: {"state": "ready" if ready else "missing", "fingerprint": "done"}}

        async def process(argv, _, *, stdin, stop_before_sec, on_line, **kwargs):
            nonlocal ready, left
            calls.append((argv, stdin, left, stop_before_sec))
            if len(calls) == 1:
                self.assertNotIn("--continue", argv)
                self.assertNotIn("--no-session", argv)
                if persisted:
                    directory = Path(argv[argv.index("--session-dir") + 1])
                    directory.mkdir()
                    (directory / "case.jsonl").write_text('{"type":"session"}\n')
                left -= harness.taskbridge.time_share(left, stop_before_sec)
                ready = finish_first
            else:
                ready = True
            await on_line(json.dumps({"type": "message_end", "message": {
                "role": "assistant", "content": [{"type": "text", "text": "Finished phase"}],
            }}).encode())
            return SimpleNamespace(code=0, stderr="")

        relay = mock.MagicMock()
        relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(harness.taskbridge, "Relay", return_value=relay))
            stack.enter_context(mock.patch.object(harness.taskbridge, "run_process", side_effect=process))
            stack.enter_context(mock.patch.object(harness, "configure", return_value=("model", "test")))
            stack.enter_context(mock.patch.object(harness.engineering_intake, "intake_for_prompt", return_value=""))
            stack.enter_context(mock.patch.object(harness.engineering_intake, "provision_engineering_cli", return_value=""))
            stack.enter_context(mock.patch.object(harness.delivery_guard, "snapshot", side_effect=snapshot))
            await harness.solve(session)
        self.assertEqual(len(calls), 1 if finish_first else 2)
        directory = Path(calls[0][0][calls[0][0].index("--session-dir") + 1])
        self.assertFalse(directory.exists())
        if not finish_first:
            argv, prompt, left, reserve = calls[1]
            self.assertEqual(left, 300)
            self.assertEqual(reserve, harness.STOP_BEFORE_SEC)
            self.assertEqual("--continue" in argv, persisted)
            self.assertEqual(instruction.encode() in prompt, not persisted)
            self.assertIn(target.encode(), prompt)

    async def test_recovery_retains_calculations_with_five_minutes_left(self):
        await self.exercise(persisted=True)

    async def test_early_cutoff_without_transcript_reissues_the_task(self):
        await self.exercise(persisted=False)

    async def test_complete_deliverable_skips_second_model_call(self):
        await self.exercise(persisted=True, finish_first=True)
