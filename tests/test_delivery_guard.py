"""Locked output-contract cases from public PLE dev tasks plus negative controls.

These fixtures are intentionally independent of the parser implementation.
No hidden task files, model calls, or grading slots are used.
"""
import unittest

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
