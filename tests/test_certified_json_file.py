"""Paired submitted Pi/Evaluator WebSocket course for a single explicit calculator JSON file."""
import asyncio
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import delivery_guard
import engineering_intake
import harness
from mock_sse_proxy import MockStream
from test_direct_json_delivery import case


class CertifiedFileCourse(unittest.IsolatedAsyncioTestCase):
    async def test_registered_file_and_fallbacks(self):
        stream = MockStream(reply_text="Participant fallback.")
        spec = case(50000)
        prompt = "Run the calculation below and write its JSON result to `result.json`.\n" + json.dumps(spec)

        async def entry(ws):
            await harness.taskbridge._session(ws, harness.solve)

        async def task(port, folder, *, corrupt_read=False, instruction=prompt):
            folder.mkdir(exist_ok=True)
            target = folder / "result.json"
            intro = "cd " + shlex.quote(str(folder)) + " && python3 -c " + shlex.quote(engineering_intake.REMOTE_PROBE)
            reads = writes = 0
            before_calls = stream.requests
            async with connect(f"ws://127.0.0.1:{port}") as ws, asyncio.timeout(90):
                await ws.send(json.dumps({"type":"task", "instruction":instruction, "workdir":str(folder),
                                          "limits":{"agent_timeout_sec":400}}))
                async for wire in ws:
                    msg = json.loads(wire)
                    if msg["type"] == "done":
                        return msg["final_message"], reads, writes, stream.requests - before_calls
                    if msg["type"] == "error":
                        self.fail(msg)
                    if "id" not in msg:
                        continue
                    reply = {"type":"result", "id":msg["id"], "ok":False, "error":"fixture operation unavailable"}
                    if msg["type"] == "exec":
                        cmd = msg["command"]
                        argv = shlex.split(cmd)
                        run = None
                        if cmd == intro:
                            run = [sys.executable, "-c", engineering_intake.REMOTE_PROBE]
                        elif argv[:3] == ["python3", "-c", harness._NEW_JSON_RESULT]:
                            self.assertEqual(argv[3:5], [str(folder),str(target)])
                            self.assertEqual(len(argv),6)
                            run = [sys.executable,*argv[1:]]
                            writes += 1
                        elif argv[:3] == ["python3","-c",delivery_guard.REMOTE_PROBE]:
                            requested = json.loads(argv[3])
                            # Participant fallback also fingerprints its three conventional
                            # materializer scripts. None of these reads publishes a file.
                            allowed = {target, folder/"solution.py", folder/"generate.py", folder/"build.py"}
                            self.assertTrue(requested)
                            self.assertTrue(all(row["kind"] == "file" and Path(row["path"]) in allowed
                                                for row in requested), requested)
                            run = [sys.executable,*argv[1:]]
                        if run is not None:
                            proc = await asyncio.create_subprocess_exec(*run, cwd=folder,
                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                            out, err = await proc.communicate()
                            self.assertEqual(proc.returncode,0,err.decode())
                            reply.update(ok=True,exit_code=0,output=out.decode(),timed_out=False)
                    elif msg["type"] == "read_file" and msg["path"] == str(target) and target.exists():
                        reads += 1
                        data = b'{"changed":true}' if corrupt_read else target.read_bytes()
                        reply.update(ok=True,size=len(data),truncated=False,
                                     content_b64=base64.b64encode(data).decode())
                    await ws.send(json.dumps(reply))
            self.fail("missing terminal answer")

        with tempfile.TemporaryDirectory(dir="/filesystem") as tmp:
            root=Path(tmp)
            async with await asyncio.start_server(stream.respond,"127.0.0.1",0) as proxy:
                url=f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
                env={"PATH":os.environ["PATH"],"HOME":str(root),"NO_PROXY":"localhost,127.0.0.1",
                     "MODEL_NAME":"claude-sonnet-4-5","ANTHROPIC_BASE_URL":url,"ANTHROPIC_API_KEY":"fixture"}
                with mock.patch.dict(os.environ,env,clear=True):
                    async with serve(entry,"127.0.0.1",0) as server:
                        port=server.sockets[0].getsockname()[1]
                        with mock.patch.object(harness,"_certified_json_file",mock.AsyncMock(return_value=None)):
                            base,_,_,base_calls = await task(port,root/"baseline")
                        self.assertGreater(base_calls,0)
                        self.assertFalse((root/"baseline/result.json").exists())
                        answer,reads,writes,calls = await task(port,root/"candidate")
                        self.assertEqual((reads,writes,calls),(1,1,0))
                        data=(root/"candidate/result.json").read_bytes()
                        self.assertIn(hashlib.sha256(data).hexdigest(),answer)
                        result=json.loads(data)
                        self.assertTrue(result["conservation_verified"])
                        # Independent normally consolidated double-drainage series.
                        tv=1e-7*31536000/(2.5**2)
                        u=1-sum(8/((2*k+1)*math.pi)**2*math.exp(-((2*k+1)*math.pi)**2*tv/4) for k in range(40))
                        analytical=u*5*0.25/1.8*math.log10(1.5)
                        self.assertAlmostEqual(result["settlement_at_time_m"],analytical,places=9)
                        changed,_,_,changed_calls=await task(port,root/"changed",corrupt_read=True)
                        self.assertGreater(changed_calls,0)
                        self.assertNotIn("Saved and verified",changed)
                        kept=root/"existing"
                        kept.mkdir()
                        (kept/"result.json").write_bytes(b"retained input")
                        _,_,_,existing_calls=await task(port,kept)
                        self.assertGreater(existing_calls,0)
                        self.assertEqual((kept/"result.json").read_bytes(),b"retained input")
                        print(json.dumps({"single_json_baseline_provider_calls":base_calls,
                                          "candidate_provider_calls":calls,"cold_reads":reads,
                                          "settlement_m":result["settlement_at_time_m"],
                                          "scope":"supplemental explicit file request"}))
