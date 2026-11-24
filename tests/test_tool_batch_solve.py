"""Normal solve() delivery of a late registered read through the task protocol."""
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import tempfile
import time
import unittest
from unittest import mock

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import delivery_guard
import harness
import taskbridge
from test_cooperative_delivery_budget import FiniteTools
from test_tool_batch_cutoff import strings


class StressReceiptProvider(FiniteTools):
    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target

    def plan(self, request, index):
        if index == 1:
            return False, [("read", {"path": str(self.source)})], ""
        messages = request["messages"]
        if "SECOND PASS — REQUIRED DELIVERY REPAIR" not in json.dumps(messages, ensure_ascii=False):
            raise AssertionError("Unexpected work turn before delivery repair")
        if index == 2:
            for message in messages:
                for part in message.get("content", []):
                    if not isinstance(part, dict) or part.get("type") != "tool_result":
                        continue
                    if part.get("tool_use_id") != "call-1-0":
                        continue
                    for text in strings(part.get("content")):
                        m = re.search(r"batch ([0-9a-f]{32}) force_N=(\d+) area_mm2=(\d+)", text)
                        if m:
                            batch, force, area = m.groups()
                            content = f"{batch} {int(force)/int(area):.6f} MPa"
                            return False, [("write", {"path": str(self.target), "content": content})], ""
            raise AssertionError("Original registered read absent from repair history")
        if index == 3:
            return True, [], "Saved the stress receipt."
        raise AssertionError("Unexpected extra model request")


@unittest.skipUnless(Path(harness.PI).is_file(), "install the submitted Pi with npm ci")
class ToolBatchSolve(unittest.IsolatedAsyncioTestCase):
    async def test_checked_delivery_resumes_read_and_writes_verified_output(self):
        # The protocol-test runner provisions /filesystem as a writable task root.
        with (tempfile.TemporaryDirectory(prefix="ple-solve-boundary-", dir="/filesystem") as work,
              tempfile.TemporaryDirectory(prefix="ple-solve-home-") as home):
            root = Path(work)
            source, target = root / "input.txt", root / "receipt.txt"
            batch = secrets.token_hex(16)
            force, area = 1000 + secrets.randbelow(9000), 10 + secrets.randbelow(90)
            source.write_text(f"batch {batch} force_N={force} area_mm2={area}")
            provider = StressReceiptProvider(source, target)
            phases = []
            phase_env = harness.delivery_process_env

            def record_phase(*args, **kwargs):
                env = phase_env(*args, **kwargs)
                phases.append(json.loads(env["PLE_DELIVERY_BUDGET"]))
                return env

            async def entry(socket):
                await taskbridge._session(socket, harness.solve)

            reads = writes = 0
            target_states = []
            async with await asyncio.start_server(provider.respond, "127.0.0.1", 0) as model_server:
                endpoint = f"http://127.0.0.1:{model_server.sockets[0].getsockname()[1]}"
                env = {"PATH": os.environ["PATH"], "HOME": home, "NO_PROXY": "127.0.0.1,localhost",
                       "MODEL_API": "google-cloud/claude-opus-5-5", "MODEL_NAME": "claude-opus-5-5",
                       "ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_API_KEY": "fixture",
                       "MODEL_BASE_URL": endpoint + "/v1", "MODEL_API_KEY": "fixture"}
                with (mock.patch.dict(os.environ, env, clear=True),
                      mock.patch.object(harness, "DELIVERY_RESERVE_SEC", 10),
                      mock.patch.object(harness, "STOP_BEFORE_SEC", 1),
                      mock.patch.object(harness, "MIN_DELIVERY_RUN_SEC", 1),
                      mock.patch.object(harness, "delivery_process_env", side_effect=record_phase),
                      mock.patch.object(harness.engineering_intake, "intake_for_prompt", mock.AsyncMock(return_value="")),
                      mock.patch.object(harness.engineering_intake, "provision_engineering_cli", mock.AsyncMock(return_value=""))):
                    async with serve(entry, "127.0.0.1", 0) as server:
                        port = server.sockets[0].getsockname()[1]
                        async with connect(f"ws://127.0.0.1:{port}") as socket:
                            await socket.send(json.dumps({
                                "type": "task", "workdir": work, "limits": {"agent_timeout_sec": 32},
                                "instruction": f"Read input.txt for the batch, force and area. Write the stress receipt to {target}."
                            }))
                            async with asyncio.timeout(42):
                                async for frame in socket:
                                    message = json.loads(frame)
                                    if message["type"] == "done":
                                        self.assertEqual(message["final_message"], "Saved the stress receipt.")
                                        break
                                    self.assertNotEqual(message["type"], "error", message)
                                    if "id" not in message:
                                        continue
                                    result = {"type": "result", "id": message["id"], "ok": True}
                                    if message["type"] == "read_file":
                                        self.assertEqual(message["path"], str(source))
                                        self.assertEqual([p["phase"] for p in phases], ["work"])
                                        reads += 1
                                        soft = phases[0]["phase_deadline_ms"] / 1000
                                        self.assertLess(time.time(), soft)
                                        # Delay the evaluator response beyond the real work-phase cutoff.
                                        await asyncio.sleep(max(0, soft - time.time()) + 0.8)
                                        data = source.read_bytes()
                                        result.update(content_b64=base64.b64encode(data).decode(), size=len(data), truncated=False)
                                    elif message["type"] == "write_file":
                                        self.assertEqual(message["path"], str(target))
                                        self.assertEqual([p["phase"] for p in phases], ["work", "delivery"])
                                        self.assertFalse(message.get("append"))
                                        writes += 1
                                        data = base64.b64decode(message["content_b64"], validate=True)
                                        target.write_bytes(data)
                                        result["size"] = len(data)
                                    else:
                                        self.assertEqual(message["type"], "exec")
                                        args = shlex.split(message["command"])
                                        self.assertEqual(args[:3], ["python3", "-c", delivery_guard.REMOTE_PROBE])
                                        self.assertEqual([row["path"] for row in json.loads(args[3])], [str(target)])
                                        probe = await asyncio.create_subprocess_exec(
                                            *args, cwd=root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                                        stdout, stderr = await probe.communicate()
                                        self.assertEqual(probe.returncode, 0, stderr.decode())
                                        target_states.append(json.loads(stdout)[str(target)]["state"])
                                        result.update(exit_code=0, output=stdout.decode(), timed_out=False, truncated=False)
                                    await socket.send(json.dumps(result))
                                else:
                                    self.fail("Evaluator received no done frame")
            self.assertEqual((reads, writes, len(provider.requests)), (1, 1, 3))
            self.assertEqual(provider.errors, [])
            self.assertEqual(target_states, ["missing", "missing", "ready"])
            self.assertEqual(target.read_text(), f"{batch} {force/area:.6f} MPa")
