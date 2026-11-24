"""Check two authored-draft exports over the submitted Pi, SSE and evaluator socket."""
import asyncio
import base64
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

import archive_checkpoint
import archive_contract
import harness
import hamon_witness_package
import test_final_materialize_stream as existing
from office_package import check_office_package

ROOT = Path(harness.__file__).resolve().parent


@unittest.skipUnless(Path(harness.PI).is_file(), "actual installed Pi required")
class AuthoredDraftStream(unittest.IsolatedAsyncioTestCase):
    async def course(self, task):
        with tempfile.TemporaryDirectory(dir="/filesystem", prefix="ple-drafts-") as td:
            root = Path(td)
            tools = root / ".task-tmp"
            tools.mkdir()
            prompt = json.loads((ROOT / "artifacts" / task / "trajectory.json").read_bytes())["steps"][0]["message"]
            if task == "foundation-strategy":
                prompt = prompt.replace("/filesystem/dc_structural__foundation_strategy__blind", str(root))
                dest = root / "output/submission.zip"
                names = archive_contract.member_names(prompt)
                self.assertEqual(len(names), 6)
                source_zip = next((ROOT / "artifacts/eor-review-corrections/files").rglob("submission.zip"))
                template = root / "controlled-input.xlsx"
                with ZipFile(source_zip) as z:
                    template.write_bytes(z.read("constraint_register.xlsx"))
                from xml.etree import ElementTree as ET
                with ZipFile(template) as z:
                    sheet = next(e for e in ET.fromstring(z.read("xl/workbook.xml")).iter() if e.tag.endswith("}sheet")).get("name")
                rows = {}
                for name in names:
                    kind = Path(name).suffix
                    if kind == ".xlsx":
                        rows[name] = {"template": str(template), "cells": {sheet: {"A240": "Controlled input row"}}}
                    elif kind == ".json":
                        rows[name] = {"json": {"source": "Controlled input fixture", "value": 10}}
                    else:
                        self.assertEqual(kind, ".md")
                        rows[name] = {"text": "Controlled calculation from the local test fixture."}
                content = {"members": rows}
            else:
                dest = root / ".ple-delivery/Municipal_Witness_Test_Package.docx"
                names = []
                content = hamon_witness_package.example()
                content["case"] = "Controlled municipal case"
                content["scheduling_items"][0]["case"] = content["case"]
            dest.parent.mkdir()
            draft = dest.with_name(dest.name + ".draft.json")
            author = json.dumps(content)
            stream = existing.RepairStream(draft)
            observed = []
            proxy = await asyncio.start_server(stream.respond, "127.0.0.1", 0)
            async with proxy:
                url = "http://127.0.0.1:" + str(proxy.sockets[0].getsockname()[1])
                env = {"PATH": os.environ["PATH"], "HOME": str(root), "NO_PROXY": "127.0.0.1,localhost",
                       "MODEL_API": "google-cloud/claude-opus-5-5", "MODEL_NAME": "claude-opus-5-5",
                       "ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "fixture",
                       "MODEL_BASE_URL": url + "/v1", "MODEL_API_KEY": "fixture"}
                async def entry(ws):
                    await harness.taskbridge._session(ws, harness.solve)
                with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(existing, "GENERATOR", author):
                    async with serve(entry, "127.0.0.1", 0) as service:
                        async with connect("ws://127.0.0.1:" + str(service.sockets[0].getsockname()[1])) as ws:
                            await ws.send(json.dumps({"type": "task", "instruction": prompt, "workdir": str(root),
                                                      "limits": {"agent_timeout_sec": 210}}))
                            async with asyncio.timeout(100):
                                async for frame in ws:
                                    m = json.loads(frame)
                                    if m["type"] == "done":
                                        result = m.get("final_message", "")
                                        break
                                    self.assertNotEqual(m["type"], "error", m)
                                    if "id" not in m: continue
                                    reply = {"type": "result", "id": m["id"], "ok": True}
                                    if m["type"] == "write_file":
                                        raw = base64.b64decode(m["content_b64"])
                                        path = Path(m["path"])
                                        if path.parent == Path("/tmp"):
                                            path = tools / path.name
                                        else:
                                            self.assertEqual(path, draft)
                                            self.assertEqual(raw.decode(), author)
                                        with path.open("ab" if m.get("append") else "wb") as handle:
                                            handle.write(raw)
                                        reply["size"] = path.stat().st_size
                                    elif m["type"] == "exec":
                                        args = shlex.split(m["command"])
                                        if args[:3] == ["cd", str(root), "&&"]:
                                            args = args[3:]
                                            self.assertEqual(args, ["python3", "-c", harness.engineering_intake.REMOTE_PROBE])
                                        self.assertEqual(args[0], "python3")
                                        if args[1].startswith("/tmp/"):
                                            binfile = Path(args[1]).name
                                            args[1] = str(tools / binfile)
                                            if binfile.startswith("ple-archive-"):
                                                spec = json.loads(args[2])
                                                spec[0] = str(tools / Path(spec[0]).name)
                                                args[2] = json.dumps(spec)
                                            elif binfile == "hamon-engineering-toolkit.py":
                                                self.assertEqual(args[2], "materialize-draft")
                                                observed.append(args[3:5])
                                            else: self.fail("Unexpected task helper")
                                        else:
                                            self.assertEqual(args[1], "-c")
                                        p = await asyncio.create_subprocess_exec(*args, cwd=root,
                                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                                        output, error = await asyncio.wait_for(p.communicate(), m["timeout_sec"] + 2)
                                        reply.update(exit_code=p.returncode, output=output.decode(), timed_out=False, truncated=False)
                                        if len(args)>2 and args[2] == "materialize-draft":
                                            self.assertEqual(p.returncode, 0, output.decode() + error.decode())
                                    else: self.fail("Unexpected task action " + m["type"])
                                    await ws.send(json.dumps(reply))
                                else: self.fail("missing terminal delivery frame")
            self.assertEqual(observed, [[str(draft), str(dest)]])
            self.assertNotIn("Incomplete", result)
            self.assertIn("write", stream.tools)
            if names:
                archive_checkpoint.verify(dest, names)
                with ZipFile(dest) as z:
                    self.assertEqual(json.loads(z.read("foundation_geometry.json"))["value"], 10)
            else:
                with ZipFile(dest) as z: check_office_package(z, ".docx")
                self.assertIn("not independently verified", result)

    async def test_foundation_draft_to_six_member_archive(self):
        await self.course("foundation-strategy")

    async def test_solar_draft_to_named_word_copy(self):
        await self.course("solar-witness-test-package")
