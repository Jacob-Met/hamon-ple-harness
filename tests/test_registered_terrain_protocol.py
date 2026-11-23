"""Real Pi bash registration, Relay and Session on a synthetic terrain task."""

import asyncio
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import engineering_intake
import taskbridge
from test_terrain_entry import drawing, PROPOSED

ROOT = Path(__file__).resolve().parents[1]
NODE = r"""
import register from './task-tools.ts';
import { SessionManager } from '@earendil-works/pi-coding-agent';
const tools = new Map();
register({registerTool(t) {tools.set(t.name, t);}, on() {}});
const cwd = process.env.TASK_WORKDIR;
const ctx = {cwd, sessionManager: SessionManager.inMemory(cwd)};
const results = [];
for (const [i, command] of JSON.parse(process.env.PLE_TERRAIN_COMMANDS).entries()) {
  try {
    const result = await tools.get('bash').execute(String(i), {command, timeout: 20}, null, () => {}, ctx);
    results.push({completed: true, result});
  } catch (error) {
    results.push({completed: false, error: String(error)});
  }
}
console.log(JSON.stringify(results));
"""


async def exercise():
    record = {"writes": [], "executions": [], "done": False}
    with tempfile.TemporaryDirectory(prefix="ple-terrain-protocol-") as tmp:
        task = Path(tmp)
        (task / "site.dxf").write_bytes(drawing())
        (task / "partial.dxf").write_bytes(drawing(PROPOSED[:1]))
        commands = [shlex.join([
            sys.executable, "-E", "-s", "-S", "/tmp/hamon-engineering-toolkit.py",
            "dxf-takeoff", filename, "--existing-layer", "EG", "--proposed-layer", "FG",
        ]) for filename in ["site.dxf", "partial.dxf"]]

        async def run_tools(session):
            note = await engineering_intake.provision_engineering_cli(session)
            if "--existing-layer" not in note:
                raise AssertionError("terrain CLI was not provisioned and advertised")
            async with taskbridge.Relay(session) as relay:
                env = relay.env()
                env.update(TASK_WORKDIR="/ple-terrain-task",
                           PLE_TERRAIN_COMMANDS=json.dumps(commands))
                proc = await asyncio.create_subprocess_exec(
                    "node", "--experimental-strip-types", "--input-type=module", "-e", NODE,
                    cwd=ROOT, env=env, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), 60)
                except BaseException:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise
                if proc.returncode:
                    raise AssertionError(err.decode())
                record["registered_results"] = json.loads(out.decode().strip().splitlines()[-1])
            return "Synthetic terrain protocol checks complete"

        async def handler(ws):
            await taskbridge._session(ws, run_tools)

        async with serve(handler, "127.0.0.1", 0, max_size=taskbridge.MAX_FRAME,
                         ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", max_size=taskbridge.MAX_FRAME) as ws:
                await ws.send(json.dumps({
                    "type": "task", "protocol": 1, "session": "terrain-protocol",
                    "instruction": "Check the two synthetic terrain drawings.",
                    "workdir": "/ple-terrain-task", "limits": {"agent_timeout_sec": 100},
                }))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 80))
                    kind = msg["type"]
                    if kind == "done":
                        record["done"] = True
                        break
                    if kind == "error":
                        raise AssertionError(msg)
                    if kind in ("log", "usage"):
                        continue
                    reply = {"type": "result", "id": msg["id"]}
                    if kind == "write_file":
                        path = Path(msg["path"])
                        if path.parent != Path("/tmp") or path.suffix != ".py":
                            raise AssertionError(msg["path"])
                        data = base64.b64decode(msg["content_b64"], validate=True)
                        target = task / path.name
                        with target.open("ab" if msg.get("append", False) else "wb") as handle:
                            handle.write(data)
                        record["writes"].append(path.name)
                        reply.update(ok=True, size=target.stat().st_size)
                    elif kind == "exec":
                        # The local transport emulates the evaluator's task mount.
                        command = msg["command"].replace("/ple-terrain-task", str(task))
                        command = command.replace("/tmp/hamon-engineering-toolkit.py",
                                                  str(task / "hamon-engineering-toolkit.py"))
                        result = await asyncio.to_thread(
                            subprocess.run, ["bash", "-c", command], cwd=task,
                            capture_output=True, text=True, timeout=25,
                        )
                        record["executions"].append({
                            "exit_code": result.returncode, "stdout": result.stdout,
                            "stderr": result.stderr,
                        })
                        reply.update(exit_code=result.returncode, output=result.stdout+result.stderr,
                                     truncated=False, timed_out=False)
                    else:
                        raise AssertionError(msg)
                    await ws.send(json.dumps(reply))
    return record


class RegisteredTerrainProtocolTests(unittest.TestCase):
    def test_registered_bash_reaches_staged_terrain_and_preserves_refusal(self):
        record = asyncio.run(exercise())
        self.assertTrue(record["done"])
        self.assertIn("hamon_terrain_volume.py", record["writes"])
        self.assertEqual(len(record["registered_results"]), 2)
        pi_good, pi_partial = record["registered_results"]
        self.assertTrue(pi_good["completed"], pi_good)
        self.assertTrue(pi_partial["completed"], pi_partial)
        self.assertFalse(pi_good["result"].get("isError", False), pi_good)
        self.assertIs(pi_partial["result"]["isError"], True)
        self.assertEqual(pi_good["result"]["structuredContent"]["exit_code"], 0)
        self.assertEqual(pi_partial["result"]["structuredContent"]["exit_code"], 3)
        self.assertEqual(len(record["executions"]), 2)
        good, partial = record["executions"]
        self.assertEqual(good["exit_code"], 0, good)
        self.assertEqual(partial["exit_code"], 3, partial)
        good_reply, partial_reply = json.loads(good["stdout"]), json.loads(partial["stdout"])
        self.assertIs(good_reply["qualified"], True)
        self.assertEqual((good_reply["cut_m3"], good_reply["fill_m3"]), (0.125, 0.125))
        self.assertIs(partial_reply["qualified"], False)
        self.assertEqual(partial_reply["common_area_m2"], 0.5)
        for field in ("cut_m3", "fill_m3", "net_fill_m3"):
            self.assertIsNone(partial_reply[field])
        good_text = "\n".join(block["text"] for block in pi_good["result"]["content"]
                              if block["type"] == "text")
        self.assertEqual(json.loads(good_text), good_reply)
        self.assertEqual(json.loads(pi_good["result"]["structuredContent"]["output"]),
                         good_reply)
        self.assertEqual(json.loads(pi_partial["result"]["structuredContent"]["output"]),
                         partial_reply)
        partial_text = "\n".join(block["text"] for block in pi_partial["result"]["content"]
                                 if block["type"] == "text")
        self.assertIn("Command exited with code 3", partial_text)
        start = partial_text.index("{")
        caller_refusal, _ = json.JSONDecoder().raw_decode(partial_text[start:])
        self.assertEqual(caller_refusal, partial_reply)


if __name__ == "__main__":
    unittest.main()
