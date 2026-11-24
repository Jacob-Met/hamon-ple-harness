"""A pending local calculation can be followed by independent artifact authoring."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from zipfile import ZipFile

import archive_checkpoint
import harness

ROOT = Path(__file__).resolve().parents[1]


class InterruptedArchive(unittest.IsolatedAsyncioTestCase):
    async def run_capture(self, *, servers=(), state="missing", remaining=480):
        steps = json.loads((ROOT / "artifacts/foundation-strategy/trajectory.json").read_bytes())["steps"]
        self.assertNotIn("observation", steps[-1])
        pending = steps[-1]["tool_calls"][0]
        task = SimpleNamespace(instruction=steps[0]["message"],
                               workdir=steps[0]["message"].split("`", 2)[1],
                               mcp_servers=list(servers), log=mock.AsyncMock(), usage=mock.AsyncMock(),
                               time_left=lambda: remaining, final_message="")
        target = harness.delivery_guard.extract_targets(task.instruction, task.workdir)[0]["path"]
        calls = []

        async def process(argv, session, *, on_line, stdin, **kwargs):
            calls.append((argv, stdin.decode()))
            if len(calls) == 1:
                directory = Path(argv[argv.index("--session-dir") + 1])
                directory.mkdir()
                (directory / "unresolved.jsonl").write_text("{}\\n")
                message = {"role": "assistant", "stopReason": "toolUse", "content": [{
                    "type": "toolCall", "id": pending["tool_call_id"],
                    "name": pending["function_name"], "arguments": pending["arguments"],
                }]}
            else:
                message = {"role": "assistant", "stopReason": "stop", "content": [
                    {"type": "text", "text": "Observed the current task files."}]}
            await on_line(json.dumps({"type": "message_end", "message": message}).encode())
            return SimpleNamespace(code=0, stderr="")

        async def snapshot(_, targets, *, deadline=None):
            return {row["path"]: {"state": state if row["path"] == target else "missing"} for row in targets}

        relay = mock.MagicMock()
        relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
        with contextlib.ExitStack() as stack:
            def patch(module, name, value):
                stack.enter_context(mock.patch.object(module, name, value))
            patch(harness.taskbridge, "Relay", lambda _: relay)
            patch(harness.taskbridge, "run_process", process)
            patch(harness, "configure", lambda *args: ("provider", "model"))
            for name in ("intake_for_prompt", "provision_engineering_cli"):
                patch(harness.engineering_intake, name, mock.AsyncMock(return_value=""))
            patch(harness.archive_contract, "provision", mock.AsyncMock(return_value=("/tmp/stage", "publisher")))
            patch(harness.archive_contract, "run_command", mock.AsyncMock(return_value=False))
            patch(harness.delivery_guard, "snapshot", snapshot)
            result = await harness.solve(task)
        return calls, result, pending["arguments"]["command"]

    async def test_captured_unacknowledged_tool_receives_independent_archive_repair(self):
        calls, result, command = await self.run_capture()
        self.assertEqual(len(calls), 2)
        argv, repair = calls[1]
        self.assertNotIn("--continue", argv)
        self.assertNotIn("--session", argv)
        self.assertIn("publisher --first-only", repair)
        self.assertIn("Inspect the destination", repair)
        self.assertNotIn(command, repair)
        self.assertIn("missing", result)

    async def test_other_side_effects_unknown_state_and_exhausted_time_are_excluded(self):
        for kwargs in ({"servers": [{"name": "document-store"}]}, {"state": "unknown"},
                       {"state": "ready"}, {"state": "empty"}, {"remaining": 35}):
            with self.subTest(kwargs=kwargs):
                calls, _, _ = await self.run_capture(**kwargs)
                self.assertEqual(len(calls), 1)

    def test_fresh_publication_preserves_an_existing_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage"
            stage.mkdir()
            dest = root / "result.zip"
            (stage / "result.json").write_text('{"value": 7}')
            command = [sys.executable, archive_checkpoint.__file__,
                       json.dumps([str(stage), str(dest), ["result.json"]]), "--first-only"]
            subprocess.run(command, check=True, capture_output=True)
            first = dest.read_bytes()
            (stage / "result.json").write_text('{"value": 9}')
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(dest.read_bytes(), first)
            with ZipFile(dest) as exported:
                self.assertEqual(json.loads(exported.read("result.json")), {"value": 7})
