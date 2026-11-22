"""Registered file tools through the actual Relay, Session and WebSocket."""

import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import hamon_drawing_reader
import taskbridge

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "router_fixtures", ROOT / "tests/test_wired_router_e2e.py"
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
NODE = r"""
import register from './task-tools.ts';
const tools=new Map(); register({registerTool(t){tools.set(t.name,t)},on(){}});
const out={};
for (const c of JSON.parse(process.env.PLE_FILE_CASES)) {
  try {const r=await tools.get(c.tool).execute(c.id,c.params,null,()=>{},{});
    out[c.id]={completed:true,details:r.details};}
  catch(e){out[c.id]={completed:false,error:String(e)}}
}
console.log(JSON.stringify(out));
"""


async def exercise():
    with tempfile.TemporaryDirectory(prefix="ple-file-protocol-") as tmp:
        root = Path(tmp)
        task, local, scratch = (root / "task", root / "harness", root / "scratch")
        for p in (task, local, scratch):
            p.mkdir()
        data = {
            "model.ifc": fixtures.SAMPLE_IFC.encode(),
            "drawing ü.pdf": hamon_drawing_reader.build_minimal_vector_pdf(
                [
                    {
                        "width_pt": 612,
                        "height_pt": 792,
                        "texts": [(36, 36, "SCALE: 1:50")],
                        "segments": [(72, 144, 288, 144)],
                    }
                ]
            ),
            "circuit.sp": b"Bias and signal\nV1 in 0 DC 12 AC 4 90\nR1 in out 1k\nR2 out 0 1k\n.ac lin 1 1000 1000\n.end",
            "good.json": b'{"a":1,"b":2}',
            "bad.json": b'{"wrong":3}',
            "change.json": b'{"a":1}',
        }
        for name, raw in data.items():
            (task / name).write_bytes(raw)
        for name in ("good", "bad", "missing", "denied", "truncated", "change"):
            (local / (name + ".json")).write_text('{"a":99,"b":99}')
        (local / "drawing ü.pdf").write_bytes(b"wrong local PDF")
        pre = "check_deliverable_preflight"
        cases = [
            {
                "id": "ifc",
                "tool": "hamon_ifc_spatial_clash",
                "params": {"path": "model.ifc"},
            },
            {
                "id": "pdf",
                "tool": "hamon_drawing_reader",
                "params": {"path": str(local / "drawing ü.pdf")},
            },
            {
                "id": "circuit",
                "tool": "hamon_conservation_cad_verifier",
                "params": {
                    "path": "circuit.sp",
                    "mode": "circuit",
                    "spec": {"freq_hz": 0, "proposed_node_voltages": {"out": 6}},
                },
            },
        ]
        for name in ("good", "bad", "missing", "denied", "truncated"):
            cases.append(
                {
                    "id": name,
                    "tool": pre,
                    "params": {"path": name + ".json", "expected_task_ids": ["a"]},
                }
            )
        for suffix in ("first", "second"):
            cases.append(
                {
                    "id": suffix,
                    "tool": pre,
                    "params": {"path": "change.json", "expected_task_ids": ["a"]},
                }
            )
        record = {"reads": []}

        async def run_tools(session):
            async with taskbridge.Relay(session) as relay:
                env = relay.env()
                env.update(
                    TASK_WORKDIR=str(local),
                    TASK_HARNESS_SRC=str(ROOT),
                    TASK_HARNESS_PYTHON=sys.executable,
                    TMPDIR=str(scratch),
                    PLE_FILE_CASES=json.dumps(cases),
                )
                proc = await asyncio.create_subprocess_exec(
                    "node",
                    "--experimental-strip-types",
                    "--input-type=module",
                    "-e",
                    NODE,
                    cwd=ROOT,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), 90)
                except BaseException:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    raise
                if proc.returncode:
                    raise AssertionError(err.decode())
                record["results"] = json.loads(out.decode().strip().splitlines()[-1])
                record["temporary_files"] = [p.name for p in scratch.iterdir()]
            return "File protocol complete"

        async def handler(ws):
            await taskbridge._session(ws, run_tools)

        async with serve(
            handler, "127.0.0.1", 0, max_size=taskbridge.MAX_FRAME, ping_interval=None
        ) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(
                f"ws://127.0.0.1:{port}", max_size=taskbridge.MAX_FRAME
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "task",
                            "protocol": 1,
                            "session": "file-boundary",
                            "instruction": "Verify task files.",
                            "workdir": str(local),
                            "limits": {"agent_timeout_sec": 120},
                        }
                    )
                )
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 100))
                    if msg["type"] == "done":
                        break
                    if msg["type"] == "error":
                        raise AssertionError(msg)
                    if msg["type"] in ("usage", "log"):
                        continue
                    if msg["type"] != "read_file":
                        raise AssertionError(msg)
                    path = Path(msg["path"])
                    name = str(path.relative_to(local) if path.is_absolute() else path)
                    record["reads"].append(name)
                    reply = {"type": "result", "id": msg["id"]}
                    if name == "denied.json":
                        reply.update(ok=False, error="task read denied")
                    elif name == "truncated.json":
                        reply.update(
                            ok=True,
                            size=10_000_001,
                            truncated=True,
                            content_b64=base64.b64encode(b'{"a":1}').decode(),
                        )
                    elif not (task / name).is_file():
                        reply.update(ok=False, error="task file missing")
                    else:
                        raw = (task / name).read_bytes()
                        reply.update(
                            ok=True,
                            size=len(raw),
                            truncated=False,
                            content_b64=base64.b64encode(raw).decode(),
                        )
                        if name == "change.json":
                            (task / name).write_bytes(b'{"b":2}')
                    await ws.send(json.dumps(reply))
        return record


class RegisteredFileProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = asyncio.run(exercise())
        cls.results = cls.record["results"]

    def test_registered_files_reach_actual_task_bytes(self):
        for name in ("ifc", "pdf", "circuit", "good"):
            with self.subTest(tool=name):
                r = self.results[name]
                self.assertTrue(r["completed"], r)
                d = r["details"]
                self.assertTrue(d.get("ok", d.get("conservation_verified")), d)
        self.assertAlmostEqual(
            self.results["circuit"]["details"]["node_voltages"]["out"], 6
        )
        self.assertEqual(self.results["pdf"]["details"]["page_count"], 1)
        self.assertEqual(self.results["good"]["details"]["size_bytes"], 13)
        self.assertEqual(self.results["good"]["details"]["path"], "good.json")
        for name in ("model.ifc", "drawing ü.pdf", "circuit.sp", "good.json"):
            self.assertIn(name, self.record["reads"])

    def test_local_shadow_does_not_approve_bad_task_file(self):
        r = self.results["bad"]
        self.assertTrue(r["completed"], r)
        self.assertFalse(r["details"]["ok"], r)
        self.assertIn("bad.json", self.record["reads"])

    def test_unavailable_or_partial_bytes_never_fall_back_locally(self):
        for name, reason in (
            ("missing", "task file missing"),
            ("denied", "task read denied"),
            ("truncated", "truncat"),
        ):
            with self.subTest(case=name):
                r = self.results[name]
                self.assertFalse(r["completed"], r)
                self.assertIn(reason, r["error"].lower())
                self.assertIn(name + ".json", self.record["reads"])

    def test_repeated_verification_reads_changed_task_bytes(self):
        a, b = self.results["first"], self.results["second"]
        self.assertTrue(a["completed"] and a["details"]["ok"], a)
        self.assertTrue(b["completed"], b)
        self.assertFalse(b["details"]["ok"], b)
        self.assertEqual(self.record["reads"].count("change.json"), 2)
        self.assertEqual(self.record["temporary_files"], [])


if __name__ == "__main__":
    unittest.main()
