"""Actual evaluator/Pi protocols for direct verified JSON; no price oracle."""
import asyncio
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import engineering_intake
import hamon_pareto as router
import harness
from mock_sse_proxy import MockStream

EMPTY = {"schema": "hamon.ple.intake.v1", "scanned_files": 0,
         "engineering_file_count": 0, "by_extension": {}, "files": [], "warnings": []}
SINGLE = "Run the calculation below and return its JSON result."
BATCH = "Run the calculations below and return their JSON results in the same order."
PIPE = {"mode": "thermal_fluid", "spec": {"calculation": "pipe_network",
        "nodes": [{"id": "r", "fixed_head_m": 5}, {"id": "a", "demand_m3_s": 1e-6}],
        "pipes": [{"id": "p", "from": "r", "to": "a", "length_m": 5, "diameter_m": .1}]}}
CELL = {"mode": "electrochem_separations", "spec": {"subkind": "electrochemistry",
        "E0_V": 1.1, "n_electrons": 2, "T_K": 298.15,
        "activity_terms": [{"species": "product", "activity": .1, "stoich": 1},
                           {"species": "reactant", "activity": 1., "stoich": -1}]}}
CSTR = {"mode": "reactor_kinetics", "spec": {
        "reactor_type": "cstr", "order": 1, "C_A0": 1000., "v0": .1/60, "volume": .1,
        "A_preexp": 7.2e10/60, "E_a": 72750., "T0": 350.,
        "delta_T_adiabatic": 50000/239., "jacket_to_feed_heat_capacity_ratio": 50000/23900,
        "coolant_T_K": 300., "steady_state_branch": "all"}}

def instruction(value):
    return (BATCH if isinstance(value, list) else SINGLE) + "\n" + json.dumps(value)

def certificate(text, inventory=EMPTY, **kwargs):
    return router.route_from_intake_prompt(json.dumps(inventory), instruction=text, **kwargs)

class RecordedStream(MockStream):
    def __init__(self, answer):
        super().__init__(reply_text=answer)
        self.input_bytes = 0

    async def respond(self, reader, writer):
        headers = await reader.readuntil(b"\r\n\r\n")
        fields = dict(line.lower().split(b": ", 1) for line in headers.split(b"\r\n")[1:] if b": " in line)
        body = await reader.readexactly(int(fields[b"content-length"]))
        self.input_bytes += len(body)
        replay = asyncio.StreamReader()
        replay.feed_data(headers + body)
        replay.feed_eof()
        await super().respond(replay, writer)

async def task_course(text, inventory=EMPTY):
    """Only the exact read-only inventory probe gets an empty-world fact."""
    seen = []
    async def entry(socket):
        await harness.taskbridge._session(socket, harness.solve)
    async with serve(entry, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}") as socket:
            await socket.send(json.dumps({"type": "task", "instruction": text,
                                          "workdir": "/filesystem", "limits": {"agent_timeout_sec": 180}}))
            async with asyncio.timeout(90):
                async for raw in socket:
                    event = json.loads(raw)
                    seen.append(event)
                    if event["type"] == "done":
                        return event["final_message"], seen
                    if event["type"] == "error":
                        raise AssertionError(event)
                    if "id" not in event:
                        continue
                    reply = {"type": "result", "id": event["id"], "ok": False, "error": "empty fixture"}
                    if event["type"] == "exec":
                        argv = shlex.split(event["command"])
                        if (len(argv) == 6 and argv[:5] == ["cd", "/filesystem", "&&", "python3", "-c"]
                                and argv[5] == engineering_intake.REMOTE_PROBE):
                            reply = {"type": "result", "id": event["id"], "ok": True,
                                     "exit_code": 0, "output": json.dumps(inventory)}
                    await socket.send(json.dumps(reply))
    raise AssertionError("Task connection ended without done")

class ReplyContract(unittest.TestCase):
    def test_unseen_input_in_a_pruned_directory_prevents_direct_completion(self):
        text = instruction(PIPE)
        for relative in ("a/b/c/d/e/f/physical-input.csv", "build/physical-input.csv", ".hidden/physical-input.csv"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as folder:
                file = Path(folder)/relative
                file.parent.mkdir(parents=True)
                file.write_text("unread task input\n")
                run = subprocess.run([sys.executable, "-c", engineering_intake.REMOTE_PROBE],
                                     cwd=folder, capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
                record = json.loads(run.stdout)
                self.assertEqual(record["scanned_files"], 0)
                self.assertIn("directory_scan_incomplete", record["warnings"])
                self.assertIsNone(router.certified_json_reply(text, certificate(text, record)))

    def test_directory_read_failure_is_not_an_empty_context(self):
        def failed_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("unreadable task directory"))
            return iter(())
        output = io.StringIO()
        with mock.patch("os.walk", side_effect=failed_walk), contextlib.redirect_stdout(output):
            exec(engineering_intake.REMOTE_PROBE, {"__name__": "__main__"})
        record = json.loads(output.getvalue())
        self.assertIn("directory_scan_incomplete", record["warnings"])
        text = instruction(PIPE)
        self.assertIsNone(router.certified_json_reply(text, certificate(text, record)))

    def test_only_explicit_complete_instructions_authorize_return(self):
        text = instruction(PIPE)
        self.assertIsNotNone(router.certified_json_reply(text, certificate(text)))
        for changed in (json.dumps(PIPE), "```json\n"+json.dumps(PIPE)+"\n```",
                        text+"\nAlso explain the assumptions.", "Explain the calculation.\n"+text,
                        SINGLE+"\n"+json.dumps([PIPE]), BATCH+"\n"+json.dumps(PIPE)):
            with self.subTest(instruction=changed[:70]):
                self.assertIsNone(router.certified_json_reply(changed, certificate(changed)))

    def test_partial_batches_and_task_obligations_remain_model_work(self):
        bad = copy.deepcopy(PIPE)
        bad["spec"]["pipes"][0]["diameter_m"] = -1
        text = instruction([PIPE, bad])
        self.assertIsNone(router.certified_json_reply(text, certificate(text)))
        text = instruction(PIPE)
        for changes in ({"scanned_files": 1}, {"metadata": {"unread": True}},
                        {"files": [{"path": "drawing.pdf"}]}, {"warnings": ["scan truncated"]}):
            self.assertIsNone(router.certified_json_reply(text, certificate(text, {**EMPTY, **changes})))
        for counts in ({"mcp_server_count": 1}, {"deliverable_count": 1}):
            self.assertIsNone(router.certified_json_reply(text, certificate(text, **counts)))

    def test_stale_or_malformed_internal_plans_cannot_finish_a_task(self):
        text = instruction(PIPE)
        plan = certificate(text)
        for update in ({"presolve_verified": 1}, {"presolve_source_sha256": "stale"},
                       {"presolve_result_json": "null"}, {"presolve_result_json": "[]"},
                       {"presolve_result_json": '{"x":NaN}'}, {"presolve_result_json": '{"x":1,"x":2}'}):
            self.assertIsNone(router.certified_json_reply(text, {**plan, **update}))
        batch = instruction([PIPE, CELL])
        p = certificate(batch)
        self.assertIsNotNone(router.certified_json_reply(batch, p))
        p["presolve_result_json"] = json.dumps([json.loads(p["presolve_result_json"])[0]])
        self.assertIsNone(router.certified_json_reply(batch, p))

class InlineProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_real_task_protocol_finishes_without_model_configuration(self):
        env = {"PATH": os.environ["PATH"], "TMPDIR": tempfile.gettempdir()}
        with mock.patch.dict(os.environ, env, clear=True):
            for value in (PIPE, CELL, CSTR, [PIPE, CELL]):
                text = instruction(value)
                expected = json.loads(certificate(text)["presolve_result_json"])
                answer, events = await task_course(text)
                self.assertEqual(json.loads(answer), expected)
                self.assertEqual(sum(e["type"] == "done" for e in events), 1)
                self.assertEqual(sum(e["type"] == "exec" for e in events), 1)
                self.assertFalse(any(e["type"] in ("usage", "write_file") for e in events))

    async def test_cancel_or_deadline_before_calculation_does_no_fresh_work(self):
        for cancel in (False, True):
            session = mock.Mock(instruction=instruction(PIPE), mcp_servers=[], cancelled=asyncio.Event())
            session.time_left.return_value = 30 if cancel else 0
            if cancel:
                session.cancelled.set()
            with mock.patch.object(router, "route_from_intake_prompt") as solve:
                with self.assertRaises(harness.taskbridge.Cancelled):
                    await harness._certified_inline_reply(session, json.dumps(EMPTY), [])
                solve.assert_not_called()

    async def test_cancel_and_expiry_during_calculation_withhold_the_result(self):
        plan = certificate(instruction(PIPE))
        for cancel in (False, True):
            session = mock.Mock(instruction=instruction(PIPE), mcp_servers=[], cancelled=asyncio.Event())
            session.time_left.return_value = 30
            entered, release = asyncio.Event(), threading.Event()
            loop = asyncio.get_running_loop()
            def solve(*args, **kwargs):
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(5):
                    raise AssertionError("test failed to release bounded calculation")
                return plan
            with mock.patch.object(router, "route_from_intake_prompt", side_effect=solve):
                work = asyncio.create_task(harness._certified_inline_reply(session, json.dumps(EMPTY), []))
                try:
                    await asyncio.wait_for(entered.wait(), 3)
                    if cancel:
                        session.cancelled.set()
                    else:
                        session.time_left.return_value = 0
                    release.set()
                    with self.assertRaises(harness.taskbridge.Cancelled):
                        await work
                finally:
                    release.set()

    async def test_bare_presolve_yields_to_cancellation_before_model_start(self):
        text = json.dumps(PIPE)
        session = mock.Mock(instruction=text, workdir="/filesystem", mcp_servers=[],
                            cancelled=asyncio.Event(), log=mock.AsyncMock())
        session.time_left.return_value = 120
        loop = asyncio.get_running_loop()
        release = threading.Event()
        verify = router.route_from_intake_prompt

        def cancel():
            session.cancelled.set()
            release.set()

        def physical_verifier(*args, **kwargs):
            loop.call_soon_threadsafe(cancel)
            if not release.wait(1):
                raise AssertionError("bare presolve blocked the task event loop")
            result = verify(*args, **kwargs)
            self.assertTrue(result["presolve_verified"])
            return result

        with (mock.patch.object(engineering_intake, "intake_for_prompt", return_value=json.dumps(EMPTY)),
              mock.patch.object(engineering_intake, "provision_engineering_cli", return_value=""),
              mock.patch.object(harness, "configure", return_value=("anthropic", "claude-opus-5-5")),
              mock.patch.object(router, "route_from_intake_prompt", side_effect=physical_verifier) as route,
              mock.patch.object(harness.taskbridge, "run_process") as model):
            with self.assertRaises(harness.taskbridge.Cancelled):
                await harness.solve(session)
            route.assert_called_once()
            model.assert_not_called()

    @unittest.skipUnless(Path(harness.PI).is_file(), "requires the pinned Pi runtime")
    async def test_paired_actual_sse_requests_and_exact_replies(self):
        rows = []
        for value in (PIPE, CELL, CSTR, [PIPE, CELL]):
            text = instruction(value)
            answer = certificate(text)["presolve_result_json"]
            observed = []
            for baseline in (True, False):
                stream = RecordedStream(answer)
                async with await asyncio.start_server(stream.respond, "127.0.0.1", 0) as proxy:
                    url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
                    env = {"PATH": os.environ["PATH"], "TMPDIR": tempfile.gettempdir(),
                           "NO_PROXY": "127.0.0.1,localhost", "MODEL_NAME": "claude-opus-5-5",
                           "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "fixture",
                           "MODEL_BASE_URL": url+"/v1", "MODEL_API_KEY": "fixture"}
                    with mock.patch.dict(os.environ, env, clear=True):
                        if baseline:
                            with mock.patch.object(harness, "_certified_inline_reply", return_value=(None, None)):
                                final, events = await task_course(text)
                        else:
                            final, events = await task_course(text)
                self.assertEqual(json.loads(final), json.loads(answer))
                observed.append({"requests": stream.requests, "request_bytes": stream.input_bytes,
                                 "usage_events": sum(e["type"] == "usage" for e in events)})
            self.assertEqual([o["requests"] for o in observed], [1, 0])
            self.assertGreater(observed[0]["request_bytes"], 0)
            self.assertEqual(observed[1]["request_bytes"], 0)
            self.assertEqual(observed[1]["usage_events"], 0)
            rows.append(observed)
        print("DIRECT_JSON_PAIRED " + json.dumps({"courses": rows, "answers_equal": True}), flush=True)

if __name__ == "__main__":
    unittest.main()
