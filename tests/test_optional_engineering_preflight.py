"""Optional engineering setup must not prevent the participant from starting."""
import asyncio
import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import engineering_intake
import harness
import taskbridge


class Evaluator:
    """Serve task/result frames in memory using the real Session request IDs."""

    def __init__(self, rejected_operation):
        self.rejected_operation = rejected_operation
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait(json.dumps({"type": "task", "instruction": "fixture"}))
        self.sizes = {}
        self.writes = []
        self.terminal = []

    async def recv(self):
        return await self.incoming.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()

    async def send(self, frame):
        message = json.loads(frame)
        kind = message["type"]
        if kind == "exec":
            result = {"ok": self.rejected_operation != "exec", "exit_code": 1, "output": ""}
        elif kind == "write_file":
            path = message["path"]
            self.writes.append(path)
            previous = self.sizes.get(path, 0) if message["append"] else 0
            self.sizes[path] = previous + len(base64.b64decode(message["content_b64"]))
            result = {"ok": self.rejected_operation != "write_file" or len(self.writes) != 2,
                      "size": self.sizes[path]}
        elif kind in ("done", "error"):
            self.terminal.append(message)
            return
        else:
            return
        if not result["ok"]:
            result["error"] = "optional operation unavailable"
        await self.incoming.put(json.dumps(dict(result, type="result", id=message["id"])))


class OptionalEngineeringPreflight(unittest.IsolatedAsyncioTestCase):
    async def test_local_read_failure_and_short_acknowledgement_are_optional(self):
        original_read = Path.read_bytes
        for missing in ("hamon_geometry.py", "engineering_toolkit.py"):
            with self.subTest(missing=missing):
                def read(path):
                    if path.name == missing:
                        raise FileNotFoundError(missing)
                    return original_read(path)

                write = mock.AsyncMock(side_effect=lambda path, data: len(data))
                with mock.patch.object(Path, "read_bytes", new=read):
                    note = await engineering_intake.provision_engineering_cli(
                        SimpleNamespace(write_file=write))
                self.assertEqual(note, "")
                paths = [call.args[0] for call in write.await_args_list]
                self.assertNotIn("/tmp/hamon-engineering-toolkit.py", paths)
                self.assertEqual(bool(paths), missing == "engineering_toolkit.py")

        short = mock.AsyncMock(side_effect=lambda path, data: len(data) - 1)
        note = await engineering_intake.provision_engineering_cli(SimpleNamespace(write_file=short))
        self.assertEqual(note, "")
        short.assert_awaited_once()

    async def test_rejected_evaluator_requests_still_reach_the_participant(self):
        for rejected in ("exec", "write_file"):
            with self.subTest(rejected=rejected):
                evaluator = Evaluator(rejected)
                relay = mock.MagicMock()
                relay.__aenter__.return_value = SimpleNamespace(env=lambda: {})
                prompts = []

                async def participant(argv, session, *, on_line, **kwargs):
                    prompts.append(argv[argv.index("--append-system-prompt") + 1])
                    await on_line(json.dumps({
                        "type": "message_end",
                        "message": {"role": "assistant",
                                    "content": [{"type": "text", "text": "fixture answer"}]},
                    }).encode())
                    return SimpleNamespace(code=0, stderr="")

                with mock.patch.object(taskbridge, "Relay", return_value=relay), \
                     mock.patch.object(harness, "configure", return_value=("fixture", "model")), \
                     mock.patch.object(taskbridge, "run_process", new=participant):
                    await asyncio.wait_for(taskbridge._session(evaluator, harness.solve), 3)

                self.assertEqual(evaluator.terminal,
                                 [{"type": "done", "final_message": "fixture answer"}])
                self.assertEqual(len(prompts), 1)
                if rejected == "exec":
                    self.assertEqual(evaluator.writes[-1], "/tmp/hamon-engineering-toolkit.py")
                    self.assertIn("DETERMINISTIC ENGINEERING TOOLS", prompts[0])
                else:
                    self.assertEqual(len(evaluator.writes), 2)
                    self.assertNotIn("DETERMINISTIC ENGINEERING TOOLS", prompts[0])

    async def test_cancellation_propagates_from_both_preflights(self):
        for cancelled in (taskbridge.Cancelled, asyncio.CancelledError):
            for preflight in (engineering_intake.intake_for_prompt,
                              engineering_intake.provision_engineering_cli):
                with self.subTest(exception=cancelled.__name__, preflight=preflight.__name__):
                    operation = mock.AsyncMock(side_effect=cancelled())
                    session = SimpleNamespace(workdir="/", time_left=lambda: 60,
                                              exec=operation, write_file=operation)
                    with self.assertRaises(cancelled):
                        await preflight(session)
                    operation.assert_awaited_once()

    async def test_slow_successes_share_the_remaining_install_budget(self):
        # Model five seconds per write without consuming wall time. A write
        # crossing the deadline times out before its byte count is confirmed.
        for left, expected in ((None, [12, 7, 2]), (18, [6, 1]), (12, [])):
            with self.subTest(task_seconds_left=left):
                elapsed = 0
                timeouts = []
                uploaded = []

                async def write_file(path, data):
                    uploaded.append(path)
                    return len(data)

                async def timed_write(awaitable, timeout):
                    nonlocal elapsed
                    timeouts.append(timeout)
                    if timeout < 5:
                        elapsed += timeout
                        awaitable.close()
                        raise asyncio.TimeoutError
                    elapsed += 5
                    return await awaitable

                session = SimpleNamespace(write_file=write_file,
                                          time_left=lambda: None if left is None else left - elapsed)
                with mock.patch.object(engineering_intake, "monotonic", new=lambda: elapsed, create=True), \
                     mock.patch.object(engineering_intake.asyncio, "wait_for", new=timed_write):
                    note = await engineering_intake.provision_engineering_cli(session)
                self.assertEqual(note, "")
                self.assertEqual(timeouts, expected)
                self.assertNotIn("/tmp/hamon-engineering-toolkit.py", uploaded)
