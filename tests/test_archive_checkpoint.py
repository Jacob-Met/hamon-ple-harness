import contextlib
import hashlib
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import archive_checkpoint
import archive_contract
import harness


ROOT = Path(__file__).resolve().parents[1]


def public_task(name):
    path = ROOT / "artifacts" / name / "trajectory.json"
    return json.loads(path.read_bytes())["steps"][0]["message"]


class ArchiveInterruption(unittest.TestCase):
    def test_bounded_office_validation_accounts_for_both_archives(self):
        memo = ROOT / "artifacts/permission-to-operate-memo/files/filesystem/Corwyn_PTO_Memo.docx"
        document = memo.read_bytes()
        with zipfile.ZipFile(io.BytesIO(document)) as office:
            expanded = sum(part.file_size for part in office.infolist())
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w") as output:
            output.writestr("memo.docx", document)
        with zipfile.ZipFile(package) as archive:
            needed = len(document) + expanded
            self.assertFalse(archive_checkpoint.verify_typed_members(archive, ["memo.docx"], needed - 1))
            self.assertTrue(archive_checkpoint.verify_typed_members(archive, ["memo.docx"], needed))

    def test_raw_foundation_contract_and_ambiguity(self):
        text = public_task("foundation-strategy")
        names = archive_contract.member_names(text)
        self.assertEqual(len(names), 6)
        self.assertIn("foundation_geometry.json", names)
        self.assertIn("foundation_schedule.xlsx", names)
        self.assertEqual(archive_contract.member_names(text.replace("exactly these six", "exactly these seven")), [])
        task = ROOT / "artifacts/foundation-strategy"
        self.assertFalse(any(task.rglob("submission.zip")))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "output" / "submission.zip"
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                archive_checkpoint.publish(Path(tmp)/"stage", dest, names)
            self.assertFalse(dest.exists())

    def test_eor_interruption_before_and_after_atomic_publication(self):
        names = archive_contract.member_names(public_task("eor-review-corrections"))
        official = next((ROOT/"artifacts/eor-review-corrections/files").rglob("submission.zip"))
        with tempfile.TemporaryDirectory() as tmp:
            stage, target = Path(tmp)/"stage", Path(tmp)/"output"/"submission.zip"
            with zipfile.ZipFile(official) as z:
                self.assertEqual(sorted(names), sorted(z.namelist()))
                z.extractall(stage)
            archive_checkpoint.publish(stage, target, names)
            archive_checkpoint.verify_checkpoint(stage, target, names)
            old_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            # Same source and member bytes, written directly to the previous
            # graded path. A kill before the ZIP central directory loses it.
            direct = Path(tmp)/"direct.zip"
            direct.write_bytes(target.read_bytes())
            rewrite = """
import os, signal, sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'w') as out:
    out.write(sys.argv[2], sys.argv[3])
    out.fp.flush()
    print('writing', flush=True)
    signal.pause()
"""
            control = subprocess.Popen([sys.executable, "-c", rewrite, str(direct),
                                        str(stage/names[0]), names[0]], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(control.stdout.readline().strip(), "writing")
            finally:
                control.kill()
                control.wait(timeout=5)
                control.stdout.close()
            with self.assertRaises(zipfile.BadZipFile):
                archive_checkpoint.verify(direct, names)
            note = stage / "design_basis.md"
            note.write_bytes(note.read_bytes() + b"\n")
            # Kill the real publisher after its validated tempfile is complete
            # and immediately before it can replace the last graded ZIP.
            pause = """
import json, os, signal, sys
from pathlib import Path
import archive_checkpoint
def stop(src, dst):
    print('validated', flush=True)
    signal.pause()
archive_checkpoint.os.replace = stop
stage, target, names = json.loads(sys.argv[1])
archive_checkpoint.publish(Path(stage), Path(target), names)
"""
            proc = subprocess.Popen([sys.executable, "-c", pause, json.dumps([str(stage),str(target),names])],
                                    cwd=ROOT, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(proc.stdout.readline().strip(), "validated")
            finally:
                proc.kill()
                proc.wait(timeout=5)
                proc.stdout.close()
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), old_hash)
            archive_checkpoint.verify(target, names)
            bad = stage/"constraint_register.xlsx"
            saved = bad.read_bytes()
            bad.write_bytes(b"unfinished spreadsheet")
            with self.assertRaises(zipfile.BadZipFile):
                archive_checkpoint.publish(stage, target, names)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), old_hash)
            bad.write_bytes(saved)
            archive_checkpoint.publish(stage, target, names)
            archive_checkpoint.verify_checkpoint(stage, target, names)
            self.assertNotEqual(hashlib.sha256(target.read_bytes()).hexdigest(), old_hash)



    def test_all_29_dev_worlds_cover_9_of_9_fixed_root_zip_contracts(self):
        expected_fixed = {
            "cooling-water-flush-plan": 7,
            "data-hall-device-takeoff": 5,
            "eor-review-corrections": 7,
            "foundation-strategy": 6,
            "frame-extension-method": 6,
            "lv-cable-sizing": 5,
            "roof-deck-design": 4,
            "seismic-diaphragm-shear": 5,
            "wind-diaphragm-shear": 5,
        }
        matched = {}
        for task_dir in sorted((ROOT / "artifacts").iterdir()):
            traj = task_dir / "trajectory.json"
            if not traj.is_file():
                continue
            names = archive_contract.member_names(public_task(task_dir.name))
            if names:
                matched[task_dir.name] = len(names)
                zips = list((task_dir / "files").rglob("*.zip")) if (task_dir / "files").exists() else []
                if zips:
                    with zipfile.ZipFile(zips[0]) as zf:
                        self.assertEqual(sorted(names), sorted(zf.namelist()))
        self.assertEqual(matched, expected_fixed)

    def test_cutoff_loose_output_hydration_and_pdf_dxf_md_validation(self):
        names = ["plan.dxf", "report.pdf", "basis.md"]
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            out_dir = Path(tmp) / "output"
            out_dir.mkdir(parents=True)
            target = out_dir / "submission.zip"
            (out_dir / "plan.dxf").write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n")
            (out_dir / "report.pdf").write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\n%%EOF\n")
            (out_dir / "basis.md").write_text("# Design Basis\nGoverning load case verified.\n")
            archive_checkpoint.publish(stage, target, names)
            archive_checkpoint.verify_checkpoint(stage, target, names)
            # Corrupt PDF header must be rejected without clobbering target
            good_sha = hashlib.sha256(target.read_bytes()).hexdigest()
            (stage / "report.pdf").write_bytes(b"NOT_A_PDF")
            with self.assertRaisesRegex(ValueError, "invalid PDF header"):
                archive_checkpoint.publish(stage, target, names)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), good_sha)

class ArchiveRoute(unittest.IsolatedAsyncioTestCase):
    async def test_provisioned_publisher_is_standalone(self):
        prompt = public_task("eor-review-corrections")
        original = next((ROOT / "artifacts/eor-review-corrections/files").rglob("submission.zip"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "submission.zip"

            async def write_helper(name, body):
                return (root / Path(name).name).write_bytes(body)

            session = SimpleNamespace(instruction=prompt, write_file=write_helper)
            plan = await archive_contract.provision(session, [{"kind": "file", "path": str(target)}])
            _, helper, argument = shlex.split(plan[1])
            stage, destination, names = json.loads(argument)
            stage = root / Path(stage).name
            with zipfile.ZipFile(original) as archive:
                archive.extractall(stage)
            command = [sys.executable, "-I", "-S", str(root / Path(helper).name),
                       json.dumps([str(stage), destination, names])]
            subprocess.run(command, check=True, cwd=root)
            subprocess.run(command + ["--verify"], check=True, cwd=root)
            archive_checkpoint.verify(target, names)

    async def exercise(self, complete):
        prompt = public_task("eor-review-corrections")
        original = next((ROOT/"artifacts/eor-review-corrections/files").rglob("submission.zip"))
        with tempfile.TemporaryDirectory() as tmp:
            workdir = "/filesystem/fixture"
            target = Path(tmp)/"graded.zip"
            stage = Path(tmp)/"members"
            members = archive_contract.member_names(prompt)
            session = SimpleNamespace(
                workdir=workdir, instruction=prompt, mcp_servers=[], final_message="",
                log=mock.AsyncMock(), usage=mock.AsyncMock(), time_left=lambda: 900,
            )
            plan = (str(stage), "registered-publisher")
            events = []
            async def process(argv, _, *, stdin, on_line, log_prefix, **kwargs):
                if log_prefix == "pi: ":
                    self.assertIn(b"FIRST COMPLETE VERSION", stdin)
                    self.assertFalse(target.exists())
                    with zipfile.ZipFile(original) as z:
                        z.extractall(stage)
                    if not complete:
                        (stage/members[-1]).unlink()
                    events.append("first pass")
                    await on_line(json.dumps({"type":"tool_execution_start","toolCallId":"staged",
                                              "toolName":"bash","args":{"command":f"write staged files in {stage}"}}).encode())
                    self.assertFalse(target.exists())
                    await on_line(b'{"type":"message_end","message":{"role":"toolResult","toolCallId":"staged","isError":false}}')
                    self.assertEqual(target.exists(), complete)
                    events.append("initial process still running")
                elif log_prefix == "pi-delivery-repair: ":
                    self.assertFalse(target.exists())
                    self.assertIn(b"REQUIRED DELIVERY REPAIR", stdin)
                    events.append("delivery repair")
                else:
                    archive_checkpoint.verify_checkpoint(stage, target, members)
                    self.assertIn(b"Work only on staged members", stdin)
                    events.append("refinement")
                await on_line(b'{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"phase complete"}]}}')
                return SimpleNamespace(code=0, stderr="")
            async def command(_, cmd):
                if cmd.endswith(" --verify"):
                    archive_checkpoint.verify_checkpoint(stage,target,members)
                    events.append("verified")
                else:
                    try:
                        archive_checkpoint.publish(stage,target,members)
                    except ValueError:
                        events.append("incomplete")
                        return False
                    events.append("published")
                return True
            targets = [dict(kind="file",path=str(target))]
            async def snapshot(_, ts, *, deadline=None):
                if ts != targets:
                    return {}
                return {str(target): {"state": "ready" if target.exists() else "missing",
                                      "fingerprint": "after" if target.exists() else ""}}
            relay = mock.MagicMock()
            relay.__aenter__.return_value = SimpleNamespace(env=lambda:dict(os.environ))
            with contextlib.ExitStack() as stack:
                for obj, attr, kw in [
                    (harness.taskbridge,"Relay", dict(return_value=relay)),
                    (harness.taskbridge,"run_process",dict(side_effect=process)),
                    (harness,"configure",dict(return_value=("model","fixture"))),
                    (harness.engineering_intake,"intake_for_prompt",dict(return_value="")),
                    (harness.engineering_intake,"provision_engineering_cli",dict(return_value="")),
                    (harness.delivery_guard,"snapshot",dict(side_effect=snapshot)),
                    (harness.delivery_guard,"extract_targets",dict(return_value=targets)),
                    (harness.archive_contract,"provision",dict(return_value=plan)),
                    (harness.archive_contract,"run_command",dict(side_effect=command)),
                ]:
                    stack.enter_context(mock.patch.object(obj,attr,**kw))
                await harness.solve(session)
            return events, target.exists()

    async def test_verified_first_package_precedes_optional_run(self):
        events, exists = await self.exercise(complete=True)
        self.assertTrue(exists)
        self.assertEqual(events, ["first pass", "published", "initial process still running", "verified", "refinement"])

    async def test_missing_member_allows_repair_and_prevents_refinement(self):
        events, exists = await self.exercise(complete=False)
        self.assertFalse(exists)
        self.assertEqual(events, ["first pass", "incomplete", "initial process still running", "incomplete", "delivery repair"])
