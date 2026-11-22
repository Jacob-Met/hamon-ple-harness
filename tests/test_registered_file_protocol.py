"""Registered file tools through the actual Relay, Session and WebSocket."""

import asyncio
import base64
import importlib.util
import json
import os
import stat
import subprocess
import zipfile
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
const {default: register} = await import(process.env.PLE_TEST_TOOL_MODULE);
const tools=new Map(); register({registerTool(t){tools.set(t.name,t)},on(){}});
const out={};
async function run(c) {
  try {const r=await tools.get(c.tool).execute(c.id,c.params,null,()=>{},{});
    out[c.id]={completed:true,details:r.details};}
  catch(e){out[c.id]={completed:false,error:String(e)}}
}
const cases=JSON.parse(process.env.PLE_FILE_CASES);
for(const c of cases.filter(c=>!c.parallel)) await run(c);
await Promise.all(cases.filter(c=>c.parallel).map(run));
console.log(JSON.stringify(out));
"""


async def exercise(tool_module="./task-tools.ts"):
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
        for name in ("good", "bad", "missing", "change"):
            (local / (name + ".json")).write_text('{"a":99,"b":99}')
        (local / "drawing ü.pdf").write_bytes(b"wrong local PDF")
        for name in ("denied", "truncated", "invalid64", "mismatch"):
            (local / (name + ".sp")).write_bytes(data["circuit.sp"])
        for name in ("good_dir", "bad_dir", "link_dir"):
            (task / name).mkdir()
            (local / name).mkdir()
            (task / name / "predictions.json").write_text(
                '{"a":1}' if name != "bad_dir" else '{"b":2}'
            )
            (local / name / "predictions.json").write_text('{"a":99}')
        (task / "link_dir" / "linked.json").symlink_to(task / "good.json")
        for name in ("good.zip", "other.zip", "linked.zip"):
            with zipfile.ZipFile(task / name, "w") as z:
                z.writestr("predictions.json", '{"a":1}')
                if name == "linked.zip":
                    info = zipfile.ZipInfo("linked.txt")
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, "predictions.json")
            with zipfile.ZipFile(local / name, "w") as z:
                z.writestr("predictions.json", '{"a":99}')
        (task / "owner's result.json").write_text('{"a":1}')
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
        for name in ("good", "bad", "missing"):
            cases.append(
                {
                    "id": name,
                    "tool": pre,
                    "params": {"path": name + ".json", "expected_task_ids": ["a"]},
                }
            )
        for name in (
            "good_dir",
            "bad_dir",
            "link_dir",
            "good.zip",
            "other.zip",
            "linked.zip",
            "owner's result.json",
        ):
            cases.append(
                {
                    "id": name,
                    "tool": pre,
                    "params": {"path": name, "expected_task_ids": ["a"]},
                    "parallel": name in ("good.zip", "other.zip"),
                }
            )
        for name in ("denied", "truncated", "invalid64", "mismatch"):
            cases.append(
                {
                    "id": name,
                    "tool": "hamon_conservation_cad_verifier",
                    "params": {"path": name + ".sp", "mode": "circuit"},
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
        record = {"reads": [], "exec_count": 0, "changed_runs": 0}

        async def run_tools(session):
            async with taskbridge.Relay(session) as relay:
                env = relay.env()
                env.update(
                    TASK_WORKDIR=str(local),
                    TASK_HARNESS_SRC=str(ROOT),
                    TASK_HARNESS_PYTHON=sys.executable,
                    TMPDIR=str(scratch),
                    PLE_FILE_CASES=json.dumps(cases),
                    PLE_TEST_TOOL_MODULE=tool_module,
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
                    if msg["type"] == "exec":
                        # Emulate the evaluator's separate task filesystem mount.
                        command = msg["command"].replace(str(local), str(task))
                        r = await asyncio.to_thread(
                            subprocess.run,
                            ["bash", "-c", command],
                            cwd=task,
                            capture_output=True,
                            text=True,
                            timeout=20,
                        )
                        record["exec_count"] += 1
                        if "change.json" in command:
                            record["changed_runs"] += 1
                            (task / "change.json").write_bytes(b'{"b":2}')
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "result",
                                    "id": msg["id"],
                                    "exit_code": r.returncode,
                                    "output": r.stdout + r.stderr,
                                    "timed_out": False,
                                    "truncated": False,
                                }
                            )
                        )
                        continue
                    if msg["type"] != "read_file":
                        raise AssertionError(msg)
                    path = Path(msg["path"])
                    name = str(path.relative_to(local) if path.is_absolute() else path)
                    record["reads"].append(name)
                    reply = {"type": "result", "id": msg["id"]}
                    if name == "denied.sp":
                        reply.update(ok=False, error="task read denied")
                    elif name == "truncated.sp":
                        reply.update(
                            ok=True,
                            size=10_000_001,
                            truncated=True,
                            content_b64=base64.b64encode(b'{"a":1}').decode(),
                        )
                    elif name == "invalid64.sp":
                        reply.update(
                            ok=True, size=3, truncated=False, content_b64="!!!!"
                        )
                    elif name == "mismatch.sp":
                        reply.update(
                            ok=True, size=999, truncated=False, content_b64="eA=="
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
        for name in ("model.ifc", "drawing ü.pdf", "circuit.sp"):
            self.assertIn(name, self.record["reads"])

    def test_local_shadow_does_not_approve_bad_task_file(self):
        r = self.results["bad"]
        self.assertTrue(r["completed"], r)
        self.assertFalse(r["details"]["ok"], r)
        self.assertGreater(self.record["exec_count"], 0)

    def test_unavailable_or_partial_bytes_never_fall_back_locally(self):
        for name in ("denied", "truncated", "invalid64", "mismatch"):
            with self.subTest(case=name):
                r = self.results[name]
                self.assertFalse(r["completed"], r)
                self.assertIn(name + ".sp", self.record["reads"])
        missing = self.results["missing"]
        self.assertTrue(missing["completed"], missing)
        self.assertFalse(missing["details"]["ok"], missing)

    def test_directories_archives_symlinks_and_quoted_paths_keep_native_semantics(self):
        for name in ("good_dir", "good.zip", "other.zip", "owner's result.json"):
            with self.subTest(path=name):
                r = self.results[name]
                self.assertTrue(r["completed"], r)
                self.assertTrue(r["details"]["ok"], r)
        for name in ("bad_dir", "link_dir", "linked.zip"):
            with self.subTest(path=name):
                r = self.results[name]
                self.assertTrue(r["completed"], r)
                self.assertFalse(r["details"]["ok"], r)
        for name in ("link_dir", "linked.zip"):
            self.assertIn(
                "symlink", str(self.results[name]["details"]["errors"]).lower()
            )
        r = self.results["good_dir"]["details"]
        self.assertEqual(r["file_count"], 1)
        self.assertEqual(r["total_uncompressed_bytes"], 7)
        self.assertEqual(r["scalar_payloads_checked"], 1)

    def test_repeated_verification_reads_changed_task_bytes(self):
        a, b = self.results["first"], self.results["second"]
        self.assertTrue(a["completed"] and a["details"]["ok"], a)
        self.assertTrue(b["completed"], b)
        self.assertFalse(b["details"]["ok"], b)
        self.assertEqual(self.record["changed_runs"], 2)
        self.assertEqual(self.record["temporary_files"], [])


if __name__ == "__main__":
    unittest.main()
