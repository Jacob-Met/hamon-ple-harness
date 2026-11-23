"""Locked output-contract cases from public PLE dev tasks plus negative controls.

These fixtures are intentionally independent of the parser implementation.
No hidden task files, model calls, or grading slots are used.
"""
import contextlib
import io
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import delivery_guard as guard


class ParseContracts(unittest.TestCase):
    def targets(self, prompt, cwd="/filesystem"):
        return [(t["path"], t["kind"]) for t in guard.extract_targets(prompt, cwd)]

    def test_named_docx(self):
        prompt = ("Prepare the signed engineering response and "
                  "save it as /filesystem/Halbray_Backfeed_Reply.docx.")
        self.assertEqual(self.targets(prompt),
                         [("/filesystem/Halbray_Backfeed_Reply.docx", "file")])

    def test_explicit_zip(self):
        prompt = ("Read task.md first; write your deliverable to "
                  "\`/filesystem/final_batch4__mechanical_task5_pcw_flush_mop__blind/output/submission.zip\`."
                  " Write a first COMPLETE version of \`output/submission.zip\` early.")
        self.assertEqual(self.targets(prompt,
                         "/filesystem/final_batch4__mechanical_task5_pcw_flush_mop__blind"),
                         [("/filesystem/final_batch4__mechanical_task5_pcw_flush_mop__blind/output/submission.zip",
                           "file")])

    def test_ifc_contract(self):
        prompt = ("Inputs: \`docs/example.ifc\`, \`agent_view.json\`. "
                  "Your deliverable is **\`submission/model.ifc\`** — an IFC4 BIM model.")
        self.assertEqual(self.targets(prompt, "/filesystem/DCBim"),
                         [("/filesystem/DCBim/submission/model.ifc", "file")])

    def test_submitted_generator_and_optional_runtime_outputs(self):
        prompt = ("Expected output One self-contained Python script at `output/build.py` "
                  "that builds the model and writes a file to sys.argv[1] "
                  "(default `output/model.ifc`). "
                  "Drawing notes (optional side file) Write a JSON file next to the model "
                  "(output/notes.json). Separately write the report to output/report.pdf.")
        self.assertEqual(self.targets(prompt), [
            ("/filesystem/output/build.py", "file"), ("/filesystem/output/report.pdf", "file")])

    def test_output_directory(self):
        prompt = ("Save the files that make up your answer under "
                  "\`output/\` in this workspace; do not edit the inputs.")
        self.assertEqual(self.targets(prompt), [("/filesystem/output", "dir")])

    def test_relative_memo(self):
        prompt = ("Put the corrected figures in a memo and save as "
                  "Correspondence/Client/Memo_to_Callaway_2026-10-31.docx.")
        self.assertEqual(self.targets(prompt),
                         [("/filesystem/Correspondence/Client/Memo_to_Callaway_2026-10-31.docx", "file")])

    def test_input_only_is_not_output(self):
        prompt = ("Read docs/example.ifc, input/agent_view.json and "
                  "check the dimensions in plans/base.pdf.")
        self.assertEqual(self.targets(prompt), [])

    def test_no_false_positive_on_discussion(self):
        prompt = ("Write a discussion about why docs/example.ifc "
                  "should not be treated as the deliverable.")
        self.assertEqual(self.targets(prompt), [])

    def test_escape_outside_task_is_refused(self):
        prompt = "Save the report as ../../secrets/report.docx."
        self.assertEqual(self.targets(prompt), [])

    def test_mixed_input_and_output(self):
        prompt = ("Read input/model.ifc; write final deliverable to "
                  "output/model.ifc.")
        self.assertEqual(self.targets(prompt),
                         [("/filesystem/output/model.ifc", "file")])


class DirectArchive(unittest.IsolatedAsyncioTestCase):
    async def test_direct_write_honors_declared_nonempty_root_files(self):
        path = "/filesystem/output/submission.zip"
        targets = [{"path": path, "kind": "file"}]
        prompt = ("Write a first complete version of output/submission.zip, then improve it. "
                  "Produce exactly these two files at the archive root:\n"
                  "1. `one.md`\n2. `two.md`")
        self.instruction = prompt
        with tempfile.TemporaryDirectory() as tmp:
            graded = Path(tmp).resolve() / "submission.zip"

            async def task_exec(command, timeout_sec):
                _, _, program, argument = shlex.split(command)
                # Map the task container's sole graded path to this real temporary file.
                def task_path(value):
                    self.assertEqual(value, path)
                    return graded
                output = io.StringIO()
                with (mock.patch("pathlib.Path", side_effect=task_path),
                      mock.patch.object(sys, "argv", ["-c", argument]),
                      contextlib.redirect_stdout(output)):
                    exec(program, {})
                return {"exit_code": 0, "output": output.getvalue()}

            self.exec = task_exec
            before = await guard.snapshot(self, targets)
            self.assertEqual(before, {path: {"state": "missing"}})
            complete = [("one.md", b"one"), ("two.md", b"two")]
            for entries, reason in [
                (complete[::-1], None),
                ([complete[0], ("two.md", b"   ")], "zip_invalid_member_format"),
                (complete[:1], "zip_root_members_mismatch"),
                (complete + [("extra.md", b"extra")], "zip_root_members_mismatch"),
                (complete + complete[:1], "zip_root_members_mismatch"),
                ([complete[0], ("sub/two.md", b"two")], "zip_root_members_mismatch"),
                ([complete[0], ("two.md", b"")], "zip_empty_root_member"),
            ]:
                with self.subTest(entries=entries):
                    with warnings.catch_warnings(), zipfile.ZipFile(graded, "w") as archive:
                        warnings.filterwarnings("ignore", message="Duplicate name")
                        for name, body in entries:
                            archive.writestr(name, body)
                    after = await guard.snapshot(self, targets)
                    expected = [{"path": path, "reason": reason}] if reason else []
                    self.assertEqual(guard.compare(before, after), expected)
            self.instruction = "Write output/submission.zip."
            self.assertEqual(guard.compare(before, await guard.snapshot(self, targets)), [])


class LargeDeliverable(unittest.IsolatedAsyncioTestCase):
    async def test_middle_edit_and_identical_rewrite(self):
        task_path = "/filesystem/output/site.ifc"
        targets = [{"path": task_path, "kind": "file"}]
        self.instruction = "Write your deliverable to " + task_path
        with tempfile.TemporaryDirectory() as tmp:
            graded = Path(tmp) / "site.ifc"
            data = b"ISO-10303-21;" + b" " * (1 << 20) + b"123" + b" " * (1 << 20)

            async def task_exec(command, timeout_sec):
                _, _, program, argument = shlex.split(command)
                output = io.StringIO()
                with (mock.patch("pathlib.Path", return_value=graded),
                      mock.patch.object(sys, "argv", ["-c", argument]),
                      contextlib.redirect_stdout(output)):
                    exec(program, {})
                return {"exit_code": 0, "output": output.getvalue()}

            self.exec = task_exec
            graded.write_bytes(data)
            before = await guard.snapshot(self, targets)
            graded.write_bytes(data.replace(b"123", b"124"))
            after = await guard.snapshot(self, targets)
            self.assertEqual(after[task_path]["state"], "ready")
            self.assertEqual(before[task_path]["size"], after[task_path]["size"])
            self.assertEqual(guard.compare(before, after), [])
            graded.write_bytes(data)
            self.assertEqual(guard.compare(before, await guard.snapshot(self, targets)),
                             [{"path": task_path, "reason": "unchanged"}])


class DirectoryDeliverable(unittest.IsolatedAsyncioTestCase):
    async def test_tree_uses_complete_contents_from_the_live_task_probe(self):
        task_path = "/filesystem/graded"
        targets = [{"path": task_path, "kind": "dir"}]
        self.instruction = "Save the files under graded/."
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve() / "graded"
            folder.mkdir()

            async def task_exec(command, timeout_sec):
                _, _, program, argument = shlex.split(command)
                output = io.StringIO()
                with (mock.patch("pathlib.Path", side_effect=lambda value: folder if value == task_path else Path(value)),
                      mock.patch.object(sys, "argv", ["-c", argument]),
                      contextlib.redirect_stdout(output)):
                    exec(program, {})
                return {"exit_code": 0, "output": output.getvalue()}

            self.exec = task_exec
            (folder / "placeholder").touch()
            self.assertEqual((await guard.snapshot(self, targets))[task_path]["state"], "empty")
            deep = folder.joinpath("a", "b", "c", "d", "e", "f")
            deep.mkdir(parents=True)
            payload = deep / "result.csv"
            payload.write_bytes(b"old,1")
            before = await guard.snapshot(self, targets)
            self.assertEqual(before[task_path]["files"], 1)
            old_times = (payload.stat().st_atime_ns, payload.stat().st_mtime_ns)
            os.utime(payload, ns=(old_times[0], old_times[1] + 1_000_000_000))
            self.assertEqual(guard.compare(before, await guard.snapshot(self, targets)),
                             [{"path": task_path, "reason": "unchanged"}])
            payload.write_bytes(b"new,1")
            os.utime(payload, ns=old_times)
            self.assertEqual(guard.compare(before, await guard.snapshot(self, targets)), [])
            (folder / "bad.pdf").write_bytes(b"not a PDF")
            self.assertEqual(guard.compare(before, await guard.snapshot(self, targets)),
                             [{"path": task_path, "reason": "pdf_header"}])


class CompareSnapshots(unittest.TestCase):
    def test_missing_and_invalid(self):
        before = {"/filesystem/output/submission.zip": {"state": "missing"}}
        after = {"/filesystem/output/submission.zip": {
            "state": "invalid", "reason": "zip_invalid"}}
        issues = guard.compare(before, after)
        self.assertEqual(issues[0]["reason"], "zip_invalid")

    def test_unchanged_prepopulated_stub(self):
        name = "/filesystem/submission/model.ifc"
        before = {name: {"state": "ready", "fingerprint": "a"}}
        after = {name: {"state": "ready", "fingerprint": "a"}}
        self.assertEqual(guard.compare(before, after),
                         [{"path": name, "reason": "unchanged"}])

    def test_new_valid_file(self):
        name = "/filesystem/Halbray_Backfeed_Reply.docx"
        self.assertEqual(guard.compare(
            {name: {"state": "missing"}},
            {name: {"state": "ready", "fingerprint": "new"}}), [])

    def test_unavailable_probe_does_not_fabricate_failure(self):
        name = "/filesystem/output/report.pdf"
        self.assertEqual(guard.compare({name: {"state": "unknown"}}, {}), [])


if __name__ == "__main__":
    unittest.main()
