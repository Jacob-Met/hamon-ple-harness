"""Real-process and Pi/read protocol controls for the soft delivery boundary.

The loopback provider is a deterministic protocol fixture, not an engineering
benchmark or a model/competition score.
"""
import asyncio
import base64
import contextlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import re
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import harness
import taskbridge
from test_cooperative_delivery_budget import FiniteTools

ROOT = Path(__file__).resolve().parents[1]


def batch_options(run):
    # The identical control also runs against the pinned pre-fix source.
    if "tool_batch_pending" not in inspect.signature(taskbridge.run_process).parameters:
        return {}
    return {"tool_batch_pending": run.pending_tool_batch, "tool_batch_reserve_sec": 1}


class ClockSession:
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds
        self.cancelled = asyncio.Event()
        self.logs = []

    def time_left(self):
        return max(0, self.deadline - time.monotonic())

    async def log(self, value):
        self.logs.append(value)


@unittest.skipUnless(os.name == "posix", "uses POSIX process groups")
class BatchProcessDeadline(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, delay, pending=True, cancel=False, reserve=.05):
        session = ClockSession(8 if cancel else 2)
        started = asyncio.Event()
        soft_cutoff = asyncio.Event()
        received = []
        active = False

        async def observe(line):
            nonlocal active
            received.append(line.decode())
            if line == b"started":
                active = True
                started.set()
            elif line == b"result":
                active = False

        code = ("import sys,time; print('started',flush=True); "
                "time.sleep(float(sys.argv[1])); print('result',flush=True); "
                "time.sleep(.05); print('flushed',flush=True)")
        def pending_at_boundary():
            soft_cutoff.set()
            return active and pending

        options = {}
        if "tool_batch_pending" in inspect.signature(taskbridge.run_process).parameters:
            options = {"tool_batch_pending": pending_at_boundary,
                       "tool_batch_reserve_sec": reserve}
        began = time.monotonic()
        task = asyncio.create_task(taskbridge.run_process(
            [sys.executable, "-c", code, str(delay)], session,
            on_line=observe, stop_before_sec=6 if cancel else 1, **options))
        if cancel:
            await asyncio.wait_for(started.wait(), 1)
            if options:
                await asyncio.wait_for(soft_cutoff.wait(), 7)
            began = time.monotonic()
            session.cancelled.set()
            with self.assertRaises(taskbridge.Cancelled):
                await asyncio.wait_for(task, 1)
            return None, received, time.monotonic() - began
        result = await asyncio.wait_for(task, 4)
        return result, received, time.monotonic() - began

    async def test_pending_result_and_flush_finish_past_soft_cutoff(self):
        result, received, elapsed = await self.exercise(1.65)
        print(json.dumps({"control": "pending-result", "received": received,
                          "stopped": result.stopped, "elapsed": elapsed}), flush=True)
        self.assertEqual(received, ["started", "result", "flushed"])
        self.assertEqual((result.code, result.stopped), (0, False))

    async def test_waiting_model_has_no_extension(self):
        result, received, _ = await self.exercise(1.65, pending=False)
        self.assertEqual(received, ["started"])
        self.assertTrue(result.stopped)

    async def test_stuck_tool_keeps_fixed_final_deadline(self):
        result, received, elapsed = await self.exercise(30)
        self.assertEqual(received, ["started"])
        self.assertTrue(result.stopped)
        self.assertLess(elapsed, 2.4)

    async def test_no_extra_headroom_keeps_original_cutoff(self):
        result, received, _ = await self.exercise(1.65, reserve=1)
        self.assertEqual(received, ["started"])
        self.assertTrue(result.stopped)

    async def test_cancel_during_wait_reaps_the_agent_immediately(self):
        _, received, elapsed = await self.exercise(30, cancel=True)
        self.assertEqual(received, ["started"])
        self.assertLess(elapsed, 1)


class BatchEligibility(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_result_cannot_extend_the_deadline(self):
        for bad in (["read"], {"id": "read"}, None, 1, "other"):
            with self.subTest(tool_call_id=bad):
                session = SimpleNamespace(final_message="", log=mock.AsyncMock(),
                                          usage=mock.AsyncMock())
                run = harness.Run(session, "fixture", budgeted=True)
                await run.on_event({"type": "session", "version": 3, "id": "fixture"})
                await run.on_event({"type": "message_end", "message": {
                    "role": "assistant", "stopReason": "toolUse",
                    "content": [{"type": "toolCall", "id": "read",
                                 "name": "read", "arguments": {"path": "input.json"}}]}})
                await run.on_event({"type": "message_end", "message": {
                    "role": "toolResult", "toolCallId": bad, "content": []}})
                self.assertTrue(run.invalid_tail)
                if hasattr(run, "pending_tool_batch"):
                    self.assertFalse(run.pending_tool_batch())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


class DelayedReadStream(FiniteTools):
    def __init__(self, task_dir):
        super().__init__()
        self.task_dir = task_dir

    def plan(self, request, index):
        delivery = "DELIVERY_PHASE" in json.dumps(request["messages"])
        if index == 1:
            return False, [("read", {"path": str(self.task_dir / "input.json")})], ""
        if not delivery:
            return True, [], "Unexpected further work-phase request."
        wrote = any(
            row.get("role") == "assistant" and "receipt.json" in json.dumps(row)
            for row in request["messages"]
        )
        if wrote:
            return True, [], "Delivered the observed input receipt."
        observed = None
        pattern = r'\{"nonce":\s*"([0-9a-f]+)",\s*"values":\s*(\[[^\]]+\])\}'
        for value in strings(request["messages"]):
            match = re.search(pattern, value)
            if match:
                observed = {"nonce": match.group(1), "total": sum(json.loads(match.group(2)))}
                break
        if observed is None:
            raise AssertionError("The genuine read result is absent from the resumed request")
        return False, [("write", {"path": str(self.task_dir / "receipt.json"),
                                  "content": json.dumps(observed)})], ""


@unittest.skipUnless(Path(harness.PI).is_file(), "install the submitted Pi via npm ci")
class RegisteredReadBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_read_is_acknowledged_then_delivered_without_reexecution(self):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        with tempfile.TemporaryDirectory(prefix="ple-read-boundary-") as tmp:
            root = Path(tmp)
            task_dir, agent_dir, session_dir, work = [
                root / name for name in ("task", "agent", "session", "work")
            ]
            for directory in (task_dir, agent_dir, session_dir, work):
                directory.mkdir()
            payload = {"nonce": secrets.token_hex(16),
                       "values": [((index * 37) % 101) - 50 for index in range(256)]}
            (task_dir / "input.json").write_text(json.dumps(payload))
            stream = DelayedReadStream(task_dir)
            observed = {}
            read_count = 0

            async def handler(socket):
                session = taskbridge.Session(socket, {
                    "instruction": "Read the input, calculate its total and save receipt.json.",
                    "workdir": str(task_dir), "limits": {"agent_timeout_sec": 30}})
                receiver = asyncio.create_task(session._read())
                try:
                    async with taskbridge.Relay(session) as relay:
                        env = relay.env()
                        env.update(PI_CODING_AGENT_DIR=str(agent_dir), PI_OFFLINE="1",
                                   PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0",
                                   NODE_NO_WARNINGS="1", TASK_WORKDIR=str(task_dir))
                        name, model = await asyncio.to_thread(
                            harness.configure, agent_dir, relay, session, env)
                        argv = [harness.PI, "--print", "--mode", "json",
                                "--session-dir", str(session_dir), "-nc", "-ns", "-np", "-na",
                                "-e", str(ROOT / "task-tools.ts"),
                                "-e", str(ROOT / "delivery-budget.ts"),
                                "--provider", name, "--model", model, "--thinking", "off"]
                        run = harness.Run(session, model, budgeted=True)
                        first = await taskbridge.run_process(
                            argv, session,
                            env=harness.delivery_process_env(
                                env, session, run.budget_steps, 10, budget_id=run.budget_id),
                            cwd=str(work), stdin=b"Read the supplied input.",
                            on_line=run.on_line, stop_before_sec=10, **batch_options(run))
                        observed.update(first_stopped=first.stopped,
                                        work_requests=len(stream.requests),
                                        results_recorded=len(run.observed_results))
                        matched = harness.acknowledged_tool_session(
                            session_dir, run.tool_turn, run.observed_results,
                            run.expected_calls, run.history, budget_id=run.budget_id)
                        observed["history_verified"] = matched is not None
                        records = [json.loads(line) for file in session_dir.glob("*.jsonl")
                                   for line in file.read_text().splitlines()]
                        markers = [entry for entry in records if entry.get("type") == "custom"
                                   and entry.get("customType") == "ple_delivery_boundary"]
                        observed["boundary_markers"] = len(markers)
                        observed["empty_abort_tail"] = bool(
                            records and harness.empty_budget_abort(records[-1].get("message", {})))
                        if matched is not None:
                            await taskbridge.run_process(
                                [*argv, "--session", str(matched)], session,
                                env=harness.delivery_process_env(
                                    env, session, run.budget_steps, 1,
                                    delivery=True, budget_id=run.budget_id),
                                cwd=str(work), stdin=b"DELIVERY_PHASE: save the observed total now.",
                                on_line=run.on_line, stop_before_sec=1)
                        observed["final_message"] = session.final_message
                except Exception as exc:
                    observed["handler_error"] = repr(exc)
                finally:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)

            async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as provider:
                url = f"http://127.0.0.1:{provider.sockets[0].getsockname()[1]}"
                env = {"MODEL_API": "google-cloud/claude-opus-5-5",
                       "MODEL_NAME": "claude-opus-5-5",
                       "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "local-fixture",
                       "MODEL_BASE_URL": url + "/v1", "MODEL_API_KEY": "local-fixture",
                       "NO_PROXY": "127.0.0.1,localhost"}
                with mock.patch.dict(os.environ, env):
                    async with serve(handler, "127.0.0.1", 0) as evaluator:
                        port = evaluator.sockets[0].getsockname()[1]
                        async with connect(f"ws://127.0.0.1:{port}") as socket:
                            async with asyncio.timeout(40):
                                async for frame in socket:
                                    message = json.loads(frame)
                                    if "id" not in message:
                                        continue
                                    result = {"type": "result", "id": message["id"], "ok": True}
                                    if message["type"] == "read_file":
                                        self.assertEqual(message["path"], str(task_dir / "input.json"))
                                        read_count += 1
                                        await asyncio.sleep(24)
                                        data = (task_dir / "input.json").read_bytes()
                                        result.update(content_b64=base64.b64encode(data).decode(),
                                                      size=len(data), truncated=False)
                                    elif message["type"] == "write_file":
                                        self.assertEqual(message["path"], str(task_dir / "receipt.json"))
                                        data = base64.b64decode(message["content_b64"])
                                        (task_dir / "receipt.json").write_bytes(data)
                                        result["size"] = len(data)
                                    else:
                                        result.update(ok=False, error="unsupported fixture action")
                                    try:
                                        await socket.send(json.dumps(result))
                                    except Exception:
                                        if observed.get("first_stopped"):
                                            break
                                        raise
            observed.update(read_requests=read_count,
                            total_model_requests=len(stream.requests),
                            delivered=(task_dir / "receipt.json").is_file())
            print("REGISTERED_READ_BOUNDARY " + json.dumps(observed, sort_keys=True), flush=True)
            self.assertEqual(stream.errors, [])
            self.assertNotIn("handler_error", observed, observed)
            self.assertEqual(read_count, 1)
            self.assertEqual(observed["work_requests"], 1, observed)
            self.assertTrue(observed["history_verified"], observed)
            self.assertEqual(observed["boundary_markers"], 1, observed)
            self.assertTrue(observed["empty_abort_tail"], observed)
            self.assertTrue(observed["delivered"], observed)
            self.assertEqual(json.loads((task_dir / "receipt.json").read_text()),
                             {"nonce": payload["nonce"], "total": sum(payload["values"])})
            self.assertEqual(len(stream.requests), 3)


if __name__ == "__main__":
    unittest.main()
