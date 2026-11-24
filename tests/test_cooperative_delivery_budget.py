"""Cooperative cutoff controls with the installed Pi, registered tools and localhost SSE."""
import asyncio
import base64
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import harness
import taskbridge
import test_final_materialize_stream as materialize_stream

ROOT = Path(__file__).resolve().parents[1]


class BudgetPolicy(unittest.TestCase):
    def test_explicit_steps_time_and_incomplete_batches(self):
        script = r"""
import {DeliveryBudget} from './delivery-budget.ts';
import assert from 'node:assert/strict';
let now=0;
const batch=(ids=['a','b'], results=ids) => ({
 message:{role:'assistant',stopReason:'toolUse',content:ids.map(id=>({type:'toolCall',id}))},
 toolResults:results.map(toolCallId=>({role:'toolResult',toolCallId}))
});
const turn=(budget,event=batch())=>{budget.assistant(event.message); const reason=budget.beforeTool(event.message.content[0]?.id); budget.finish(event); return reason;};
const cap=new DeliveryBudget({phase:'work',max_steps:7},()=>now);
for(let i=0;i<2;i++){now+=100;assert.equal(turn(cap),null)}
now+=100;assert.equal(turn(cap),'delivery step reserve');
assert.equal(cap.beforeTool('b'),'delivery step reserve');
const timed=new DeliveryBudget({phase:'work',phase_deadline_ms:900,task_deadline_ms:1200},()=>now);
now+=200;assert.equal(turn(timed),'delivery time reserve');
const final=new DeliveryBudget({phase:'delivery',max_steps:7,completed_steps:6},()=>now);
assert.equal(turn(final),'step limit');
const natural=new DeliveryBudget({phase:'work',max_steps:1},()=>now);
natural.assistant({role:'assistant',content:[{type:'text',text:'done'}]});
assert.equal(natural.beforeTool('a'),null);
for (const max_steps of [true,0,-1,1.5,'7',null]) {
 const invalid=new DeliveryBudget({phase:'work',max_steps},()=>now);
 assert.equal(turn(invalid),null);
}
const siblings=new DeliveryBudget({phase:'work',phase_deadline_ms:now+1000,task_deadline_ms:now+2000},()=>now);
siblings.assistant(batch().message); assert.equal(siblings.beforeTool('a'),null);
now+=2000;assert.equal(siblings.beforeTool('b'),null);
const malformed=new DeliveryBudget({phase:'work',max_steps:1},()=>now);
malformed.assistant(batch(['a','a']).message);assert.equal(malformed.beforeTool('a'),null);
console.log('budget policy controls passed');
"""
        result = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_abort_tail_requires_the_complete_zero_usage_record(self):
        import copy
        zero = {"role":"assistant","content":[],"stopReason":"error",
                "usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,
                         "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}}
        self.assertTrue(harness.empty_budget_abort(zero))
        for usage in (None, {}, {"totalTokens":17}, {"input":0,"output":0,"cacheRead":0,"cacheWrite":0}):
            self.assertFalse(harness.empty_budget_abort(dict(zero, usage=usage)))
        for key in ("input","output","cacheRead","cacheWrite","totalTokens"):
            changed = copy.deepcopy(zero)
            changed["usage"][key] = 1
            self.assertFalse(harness.empty_budget_abort(changed))
        for key in ("input","output","cacheRead","cacheWrite","total"):
            changed = copy.deepcopy(zero)
            changed["usage"]["cost"][key] = 1
            self.assertFalse(harness.empty_budget_abort(changed))
        changed = copy.deepcopy(zero)
        changed["usage"]["input"] = False
        self.assertFalse(harness.empty_budget_abort(changed))

    def test_environment_preserves_total_steps_and_shared_deadline(self):
        session = SimpleNamespace(limits={"max_steps": 9}, time_left=lambda: 1000)
        with mock.patch.object(harness, "time", return_value=10):
            env = harness.delivery_process_env({"PATH": "kept"}, session, 5, 90, delivery=True)
        value = json.loads(env["PLE_DELIVERY_BUDGET"])
        self.assertEqual(env["PATH"], "kept")
        self.assertEqual(value, {"phase":"delivery","completed_steps":5,"max_steps":9,
                                 "phase_deadline_ms":920000,"task_deadline_ms":1010000})
        self.assertTrue(harness.delivery_budget_exhausted(session, 9))
        self.assertFalse(harness.delivery_budget_exhausted(session, 8))


# The evaluator must stop this one-shot command before its second mutation.
SLOW_COMMAND = "printf 'started\\n' >> executions; sleep 16; touch late-write"


class FiniteTools:
    def __init__(self, mode="valid"):
        self.mode = mode
        self.requests = []
        self.errors = []
        self.tools = set()

    def plan(self, request, index):
        if self.mode in ("inflight", "cancel_inflight") and index == 1:
            return False, [("bash", {"command": SLOW_COMMAND, "timeout": 90})], ""
        delivery = "DELIVERY_PHASE" in json.dumps(request["messages"])
        already = any("delivery.json" in json.dumps(row) for row in request["messages"]
                      if row.get("role") == "assistant")
        final = (delivery and already) or index >= 10
        path = "delivery.json" if delivery else f"progress-{index}.json"
        calls = [("write", {"path":path,"content":json.dumps({"completed":index})})]
        if index == 3 and self.mode == "mixed":
            calls.append(("unknown_fixture_tool", {}))
        elif index == 3 and self.mode == "invalid":
            calls = [("unknown_fixture_tool", {}), ("write", {"path":path})]
        return final, calls, "Delivered."

    async def respond(self, reader, writer):
        try:
            raw = (await reader.readuntil(b"\r\n\r\n")).decode("ascii").split("\r\n")
            fields = dict(row.lower().split(": ", 1) for row in raw[1:] if ": " in row)
            request = json.loads(await reader.readexactly(int(fields["content-length"])))
            method, target, _ = raw[0].split()
            if [method, target.split("?", 1)[0]] != ["POST", "/v1/messages"]:
                raise AssertionError("unexpected provider route")
            self.requests.append(request)
            self.tools.update(tool["name"] for tool in request.get("tools", []))
            index = len(self.requests)
            final, calls, final_text = self.plan(request, index)
            events = [("message_start", {"message":{"id":f"local-{index}","type":"message",
                "role":"assistant","model":request["model"],"content":[],
                "usage":{"input_tokens":1,"output_tokens":0}}})]
            if final:
                events += [
                    ("content_block_start",{"index":0,"content_block":{"type":"text","text":""}}),
                    ("content_block_delta",{"index":0,"delta":{"type":"text_delta","text":final_text}}),
                ]
            else:
                for block, (name, arguments) in enumerate(calls):
                    events += [
                        ("content_block_start",{"index":block,"content_block":{
                            "type":"tool_use","id":f"call-{index}-{block}","name":name,"input":{}}}),
                        ("content_block_delta",{"index":block,"delta":{
                            "type":"input_json_delta","partial_json":json.dumps(arguments)}}),
                        ("content_block_stop",{"index":block}),
                    ]
            if final:
                events.append(("content_block_stop",{"index":0}))
            events += [
                ("message_delta",{"delta":{"stop_reason":"end_turn" if final else "tool_use",
                    "stop_sequence":None},"usage":{"output_tokens":1}}),
                ("message_stop",{}),
            ]
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            for kind, event in events:
                writer.write(f"event: {kind}\ndata: {json.dumps({'type':kind,**event})}\n\n".encode())
            await writer.drain()
        except Exception as exc:
            self.errors.append(repr(exc))
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


@unittest.skipUnless(Path(harness.PI).is_file(), "install the submitted Pi via npm ci")
class CooperativePi(unittest.IsolatedAsyncioTestCase):
    def test_evaluator_timeout_keeps_shorter_requests_and_whole_second_bounds(self):
        script = """
import {phaseExecLimit} from './task-tools.ts';
import assert from 'node:assert/strict';
assert.equal(phaseExecLimit(90, null, 10000), 90);
assert.equal(phaseExecLimit(2, 20000, 10000), 2);
assert.equal(phaseExecLimit(90, 20999, 10000), 8);
assert.equal(phaseExecLimit(90, 13000, 10000), 1);
assert.throws(() => phaseExecLimit(90, 12999, 10000), /not sent/);
"""
        subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                       cwd=ROOT, check=True, capture_output=True, timeout=20)

    async def test_complete_batch_stops_and_resumes_without_an_extra_request(self):
        await self.exercise("valid")

    async def test_malformed_sibling_does_not_consume_the_reserved_turns(self):
        await self.exercise("mixed")

    async def test_all_invalid_batch_still_reaches_delivery(self):
        await self.exercise("invalid")

    async def test_slow_inflight_bash_preserves_the_acknowledged_session(self):
        await self.exercise("inflight")

    async def test_cancel_during_admitted_bash_never_restarts_delivery(self):
        await self.exercise("cancel_inflight")

    async def exercise(self, mode):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        stream = FiniteTools(mode)
        with tempfile.TemporaryDirectory(prefix="ple-cooperative-") as tmp:
            root = Path(tmp)
            task_dir, agent_dir, session_dir, work = [root/name for name in ("task","agent","session","work")]
            for directory in (task_dir, agent_dir, session_dir, work):
                directory.mkdir()
            observed = {}
            slow = mode in ("inflight", "cancel_inflight")

            async def run_handler(session):
                async with taskbridge.Relay(session) as relay:
                    env = relay.env()
                    env.update(PI_CODING_AGENT_DIR=str(agent_dir),PI_OFFLINE="1",
                               PI_SKIP_VERSION_CHECK="1",PI_TELEMETRY="0",
                               NODE_NO_WARNINGS="1",TASK_WORKDIR=str(task_dir))
                    name, model = await asyncio.to_thread(harness.configure, agent_dir, relay, session, env)
                    argv = [harness.PI,"--print","--mode","json","--session-dir",str(session_dir),
                            "-nc","-ns","-np","-na","-e",str(ROOT/"task-tools.ts"),
                            "-e",str(ROOT/"delivery-budget.ts"),"--provider",name,
                            "--model",model,"--thinking","off"]
                    run = harness.Run(session,model,budgeted=True)
                    first = await taskbridge.run_process(argv,session,
                        env=harness.delivery_process_env(env,session,run.budget_steps,300,budget_id=run.budget_id),
                        cwd=str(work),stdin=b"Perform the next preparation step.",
                        on_line=run.on_line,stop_before_sec=300)
                    observed["requests_before_resume"] = len(stream.requests)
                    observed["first_status"] = first.code
                    observed["first_replies"] = run.replies
                    observed["budget_steps"] = run.budget_steps
                    matched = harness.acknowledged_tool_session(
                        session_dir,run.tool_turn,run.observed_results,run.expected_calls,run.history,budget_id=run.budget_id)
                    observed["matched"] = matched is not None
                    if mode in ("mixed", "invalid"):
                        observed["missing_marker_id_refused"] = harness.acknowledged_tool_session(
                            session_dir,run.tool_turn,run.observed_results,run.expected_calls,run.history) is None
                        observed["wrong_marker_id_refused"] = harness.acknowledged_tool_session(
                            session_dir,run.tool_turn,run.observed_results,run.expected_calls,run.history,
                            budget_id="not-this-run") is None
                    if matched is None:
                        observed["tail_roles"] = [[(r.get("message",{}).get("role"),r.get("message",{}).get("stopReason")) for r in list(map(json.loads,p.read_text().splitlines()))[-6:]] for p in session_dir.glob("*.jsonl")]
                    else:
                        await taskbridge.run_process([*argv,"--session",str(matched)],session,
                            env=harness.delivery_process_env(env,session,run.budget_steps,90,delivery=True,budget_id=run.budget_id),
                            cwd=str(work),stdin=b"DELIVERY_PHASE: write the calculated result now.",
                            on_line=run.on_line,stop_before_sec=90)
                    observed["last"] = session.final_message

            async def handler(socket):
                try:
                    async with asyncio.timeout(25):
                        await taskbridge._session(socket, run_handler)
                except Exception as exc:
                    observed["handler_error"] = repr(exc)

            async with await asyncio.start_server(stream.respond,"127.0.0.1",0) as provider:
                url = f"http://127.0.0.1:{provider.sockets[0].getsockname()[1]}"
                env = {"MODEL_API":"google-cloud/claude-opus-5-5","MODEL_NAME":"claude-opus-5-5",
                       "ANTHROPIC_BASE_URL":url,"ANTHROPIC_API_KEY":"local-fixture",
                       "MODEL_BASE_URL":url+"/v1","MODEL_API_KEY":"local-fixture",
                       "NO_PROXY":"127.0.0.1,localhost"}
                with mock.patch.dict(os.environ,env):
                    async with serve(handler,"127.0.0.1",0) as evaluator:
                        port = evaluator.sockets[0].getsockname()[1]
                        async with connect(f"ws://127.0.0.1:{port}") as socket:
                            await socket.send(json.dumps({
                                "type":"task", "instruction":"Calculate first, then write a delivery file.",
                                "workdir":str(task_dir),
                                "limits":{"max_steps":7, "agent_timeout_sec":20 if slow else 1200}}))
                            async with asyncio.timeout(40):
                                async for frame in socket:
                                    msg = json.loads(frame)
                                    if msg.get("type") == "exec" and slow:
                                        self.assertEqual(msg["command"].splitlines()[-1], SLOW_COMMAND)
                                        self.assertNotIn("exec_timeout", observed)
                                        limit = observed["exec_timeout"] = msg["timeout_sec"]
                                        child = await asyncio.create_subprocess_exec(
                                            "timeout", str(limit), "bash", "-c", msg["command"])
                                        if mode == "cancel_inflight":
                                            await socket.send(json.dumps({"type":"cancel"}))
                                        status = await child.wait()
                                        observed["exec_status"] = status
                                        if mode != "cancel_inflight":
                                            await socket.send(json.dumps({"type":"result","id":msg["id"],
                                                "ok":True,"exit_code":status,"output":"",
                                                "truncated":False,"timed_out":status == 124}))
                                    elif msg.get("type") == "write_file":
                                        path = Path(msg["path"])
                                        self.assertTrue(path.is_relative_to(task_dir))
                                        data = base64.b64decode(msg["content_b64"])
                                        path.parent.mkdir(parents=True,exist_ok=True)
                                        path.write_bytes(data)
                                        await socket.send(json.dumps({"type":"result","id":msg["id"],
                                                                      "ok":True,"size":len(data)}))
                                    elif msg.get("type") == "done":
                                        observed["done"] = msg["final_message"]
                                    elif "id" in msg:
                                        await socket.send(json.dumps({"type":"result","id":msg["id"],
                                                                      "ok":False,"error":"unsupported fixture action"}))
            self.assertEqual(stream.errors,[])
            self.assertNotIn("handler_error", observed, (observed,stream.errors,len(stream.requests)))
            if slow:
                self.assertEqual((task_dir/"executions").read_text(), "started\n")
                self.assertFalse((task_dir/"late-write").exists(), observed)
                self.assertGreaterEqual(observed["exec_timeout"], 1)
                self.assertLess(observed["exec_timeout"], 15)
                self.assertEqual(observed["exec_status"], 124)
                if mode == "cancel_inflight":
                    self.assertEqual(len(stream.requests), 1)
                    self.assertNotIn("requests_before_resume", observed)
                    self.assertFalse((task_dir/"delivery.json").exists())
                    self.assertIn("stopped", observed["done"])
                    return
                self.assertTrue(observed["matched"], observed)
                self.assertEqual(observed["requests_before_resume"], 1)
                self.assertEqual(len(stream.requests), 3)
                delivered_context = json.dumps(stream.requests[1]["messages"])
                self.assertIn(f'Command timed out after {observed["exec_timeout"]} seconds', delivered_context)
                self.assertEqual(json.loads((task_dir/"delivery.json").read_text()), {"completed":2})
                self.assertEqual(observed["last"], "Delivered.")
                return
            self.assertEqual(observed["requests_before_resume"],3,observed)
            self.assertEqual(observed["first_replies"],3 if mode == "valid" else 4,observed)
            self.assertEqual(observed["budget_steps"],3,observed)
            if mode in ("mixed", "invalid"):
                self.assertTrue(observed["missing_marker_id_refused"])
                self.assertTrue(observed["wrong_marker_id_refused"])
            self.assertTrue(observed["matched"],observed)
            self.assertEqual(len(stream.requests),5)
            self.assertFalse((task_dir/"progress-3.json").exists())
            self.assertEqual(json.loads((task_dir/"delivery.json").read_text()),{"completed":4})
            self.assertEqual(observed["last"],"Delivered.")


class BudgetRepairStream(FiniteTools):
    def __init__(self, script):
        super().__init__()
        self.script = str(script)
        self.input = str(Path(script).with_name("input.json"))

    def plan(self, request, index):
        users = [row for row in request["messages"] if row.get("role") == "user"]
        content = users[-1].get("content", []) if users else []
        delivery = index > 1 and (isinstance(content, str) or any(
            isinstance(block, dict) and block.get("type") == "text" for block in content))
        wrote = any(self.script in json.dumps(row) and '"write"' in json.dumps(row)
                    for row in request["messages"] if row.get("role") == "assistant")
        final = wrote or index >= 12
        calls = [("write", {"path":self.script,"content":materialize_stream.GENERATOR})] if delivery else [
            ("read", {"path":self.input})]
        return final, calls, "Calculation prepared."


@unittest.skipUnless(Path(harness.PI).is_file(), "install the submitted Pi via npm ci")
class CooperativeSolve(unittest.IsolatedAsyncioTestCase):
    course = materialize_stream.FinalMaterializationStream.course

    async def test_ordinary_solve_delivers_docx_after_step_reserve(self):
        await self.course(".docx", stream_factory=BudgetRepairStream, max_steps=7, expected_requests=5)

    async def test_ordinary_solve_delivers_six_archive_members_after_step_reserve(self):
        await self.course(".zip", stream_factory=BudgetRepairStream, max_steps=7, expected_requests=5)


if __name__ == "__main__":
    unittest.main()
