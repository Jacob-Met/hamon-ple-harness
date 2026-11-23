"""Public task instructions and the live solve path, without a model or external writes."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import archive_contract
import harness
import hamon_witness_package


ROOT = Path(__file__).resolve().parents[1]
CLAIM = "All requested deliverables have been filed."


def public_instruction(name):
    return json.loads((ROOT / "artifacts" / name / "trajectory.json").read_bytes())["steps"][0]["message"]


class TerminalReceipt(unittest.IsolatedAsyncioTestCase):
    async def course(self, prompt, populate=None, *, unavailable=False, workdir="/filesystem/work", inspect=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = []
            session = SimpleNamespace(instruction=prompt, workdir=workdir,
                mcp_servers=[], final_message="", usage=mock.AsyncMock(), log=mock.AsyncMock(),
                time_left=lambda: 700)
            def real_path(value):
                path = Path(value)
                return path if path.is_relative_to(root) else root / str(path).lstrip("/")
            async def remote(command, timeout_sec):
                _, _, script, arguments = shlex.split(command)
                probes.extend(row["path"] for row in json.loads(arguments))
                if unavailable:
                    return {"exit_code": 1, "output": ""}
                stdout = io.StringIO()
                with (mock.patch("pathlib.Path", side_effect=real_path),
                      mock.patch.object(sys, "argv", ["-c", arguments]),
                      contextlib.redirect_stdout(stdout)):
                    exec(script, {})
                return {"exit_code": 0, "output": stdout.getvalue()}
            session.exec = remote
            async def participant(*args, on_line, **kwargs):
                if populate:
                    populate(root)
                await on_line(json.dumps({"type":"message_end", "message":{
                    "role":"assistant","content":[{"type":"text","text":CLAIM}]}}).encode())
                return SimpleNamespace(code=0, stderr="")
            relay = mock.MagicMock()
            relay.__aenter__.return_value = SimpleNamespace(env=lambda: dict(os.environ))
            with contextlib.ExitStack() as patches:
                for owner, name, value in [
                    (harness.taskbridge, "Relay", mock.Mock(return_value=relay)),
                    (harness.taskbridge, "run_process", mock.AsyncMock(side_effect=participant)),
                    (harness, "configure", mock.Mock(return_value=("model", "fixture"))),
                    (harness.engineering_intake, "intake_for_prompt", mock.AsyncMock(return_value="")),
                    (harness.engineering_intake, "provision_engineering_cli", mock.AsyncMock(return_value="")),
                    (harness.archive_contract, "provision", mock.AsyncMock(return_value=None)),
                ]:
                    patches.enter_context(mock.patch.object(owner, name, value))
                summary = await harness.solve(session)
            if inspect:
                inspect(root)
            return summary, probes

    async def test_missing_foundation_archive_and_partial_member_set(self):
        prompt = public_instruction("foundation-strategy")
        exact = "/filesystem/dc_structural__foundation_strategy__blind/output/submission.zip"
        missing, probes = await self.course(prompt, workdir=exact.split("/output/")[0])
        self.assertIn(exact + " (missing)", missing)
        self.assertNotIn(CLAIM, missing)
        self.assertIn(exact, probes)
        members = archive_contract.member_names(prompt)
        self.assertEqual(len(members), 6)
        def partial(root):
            dest = root / exact.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "w") as archive:
                for member in members[:-1]:
                    archive.writestr(member, "partial")
        answer, _ = await self.course(prompt, partial, workdir=exact.split("/output/")[0])
        self.assertIn("zip_root_members_mismatch", answer)
        self.assertNotIn(CLAIM, answer)

    async def test_structurally_complete_six_member_archive_preserves_report(self):
        prompt = public_instruction("foundation-strategy")
        taskroot = "/filesystem/dc_structural__foundation_strategy__blind"
        book = ROOT / "artifacts/fill-compaction-acceptance/files/filesystem/Fill_Placement_Records_BlockA_SlabOnGrade.xlsx"
        def complete(root):
            dest = root / taskroot.lstrip("/") / "output/submission.zip"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, "w") as archive:
                for member in archive_contract.member_names(prompt):
                    payload = book.read_bytes() if member.endswith(".xlsx") else (
                        b'{"result":"format-only control"}' if member.endswith(".json") else b"# format-only control")
                    archive.writestr(member, payload)
        answer, _ = await self.course(prompt, complete, workdir=taskroot)
        self.assertEqual(answer, CLAIM)

    async def test_solar_document_store_is_never_treated_as_a_local_file(self):
        answer, paths = await self.course(public_instruction("solar-witness-test-package"))
        self.assertIn("/filesystem/work/.ple-delivery/Municipal_Witness_Test_Package.docx", paths)
        self.assertTrue(all("commissioning folder" not in path for path in paths))
        self.assertIn("Municipal_Witness_Test_Package.docx", answer)
        self.assertIn("Program Office's municipal commissioning folder", answer)
        self.assertIn("not independently verified", answer)
        self.assertIn("Incomplete working copy:", answer)
        self.assertIn("(missing)", answer)

    async def test_named_return_without_destination_is_verified_as_a_working_copy(self):
        answer, paths = await self.course(public_instruction("solar-supplemental-review"))
        self.assertIn("/filesystem/work/.ple-delivery/Bramwick_Commons_Screen_Memo.docx", paths)
        self.assertIn("Incomplete working copy", answer)
        self.assertNotIn("External filing", answer)
        self.assertNotIn(CLAIM, answer)

    def test_root_workdir_places_external_copy_under_exported_output(self):
        prompt = public_instruction("solar-witness-test-package")
        terminal = harness.delivery_guard.TerminalDelivery(prompt, [])
        self.assertEqual(terminal.add_working_copy("/"),
                         "/app/output/.ple-delivery/Municipal_Witness_Test_Package.docx")

    async def recover_named_filing(self, inspect=None):
        prompt = public_instruction("solar-witness-test-package")
        name = "Municipal_Witness_Test_Package.docx"
        turns = []
        def recover(root):
            store = root / "fixture-document-store"
            copy = root / "filesystem/work/.ple-delivery" / name
            turns.append(len(turns) + 1)
            if len(turns) == 1:
                store.mkdir()
                _, content = hamon_witness_package.paragraphs(hamon_witness_package.example())
                hamon_witness_package.write_docx(store / name, content)
            else:
                copy.parent.mkdir(parents=True)
                copy.write_bytes((store / name).read_bytes())
        answer, paths = await self.course(prompt, recover, inspect=inspect)
        self.assertEqual(turns, [1, 2])
        self.assertNotIn("Incomplete", answer)
        self.assertIn("not independently verified", answer)
        self.assertIn(CLAIM, answer)
        self.assertGreaterEqual(paths.count("/filesystem/work/.ple-delivery/" + name), 3)

    async def test_named_filing_recovers_existing_document_without_reupload(self):
        await self.recover_named_filing()

    async def test_complete_local_named_destination_and_unavailable_observation(self):
        prompt = "Return one file named result.md, filed in /app/files/reports."
        def write(root):
            dest = root / "app/files/reports/result.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("Measured reaction: 3 kN.")
        answer, paths = await self.course(prompt, write)
        self.assertEqual(answer, CLAIM)
        self.assertIn("/app/files/reports/result.md", paths)
        unknown, _ = await self.course(prompt, unavailable=True)
        self.assertIn("could not be verified", unknown)
        self.assertNotIn("(missing)", unknown)

    async def test_empty_named_artifact_is_not_a_delivery(self):
        prompt = "Write the final deliverable to output/report.csv."
        def empty(root):
            dest = root / "filesystem/work/output/report.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
        answer, _ = await self.course(prompt, empty)
        self.assertIn("/filesystem/work/output/report.csv (empty)", answer)
        self.assertNotIn(CLAIM, answer)

    async def test_protocol_fallback_retains_receipt_on_failure_or_cancellation(self):
        for failure in (RuntimeError("participant stopped"), harness.taskbridge.Cancelled()):
            with self.subTest(kind=type(failure).__name__):
                ws = mock.AsyncMock()
                ws.recv.return_value = json.dumps({"type":"task","instruction":"Save to output/report.pdf."})
                async def idle_reader(session):
                    await asyncio.Event().wait()
                async def stopped(session):
                    path = "/app/output/report.pdf"
                    evidence = harness.delivery_guard.TerminalDelivery(session.instruction,
                                 [{"kind":"file","path":path}])
                    session.fallback_summary = evidence.summary
                    evidence.observe({path:{"state":"missing"}})
                    session.final_message = CLAIM
                    raise failure
                with mock.patch.object(harness.taskbridge.Session, "_read", idle_reader):
                    await harness.taskbridge._session(ws, stopped)
                final = json.loads(ws.send.await_args_list[-1].args[0])
                self.assertEqual(final["type"], "done")
                self.assertIn("report.pdf (missing)", final["final_message"])
                self.assertNotIn(CLAIM, final["final_message"])

    async def test_no_deliverable_contract_preserves_the_answer(self):
        answer, paths = await self.course("Explain the dimensions of a force.")
        self.assertEqual((answer, paths), (CLAIM, []))


if __name__ == "__main__":
    unittest.main()
