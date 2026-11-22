"""Independent offline reconciliation contracts, no model or hidden test inputs."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from ship import audit_ple_delivery as audit


class CorpusAuditContracts(unittest.TestCase):
    def test_path_translation_matches_organizer_export(self):
        self.assertEqual(audit.exported_path(
            "filesystem/final_batch4__mechanical_task5_pcw_flush_mop__blind/output/submission.zip"),
            "/filesystem/final_batch4__mechanical_task5_pcw_flush_mop__blind/output/submission.zip")
        self.assertEqual(audit.exported_path(
            "app/files/output/report.docx"), "/app/files/output/report.docx")
        self.assertIsNone(audit.exported_path("other/source.txt"))
        self.assertIsNone(audit.exported_path("filesystem/../../secret.txt"))

    def test_archive_delivery_cold(self):
        root="/filesystem/dc_structural__foundation_strategy__blind"
        prompt=("You are working in '" + root + "'. "
                "Write the final deliverable to " + root + "/output/submission.zip.")
        result=audit.evaluate_task(
            "foundation-strategy",prompt,
            {"filesystem/dc_structural__foundation_strategy__blind/output/submission.zip":"hash"},
            reward=0.4)
        self.assertEqual(result["state"],"all_detected_targets_exported")
        self.assertEqual(len(result["matched"]),1)

    def test_no_output_is_not_success(self):
        root="/filesystem/dc_structural__foundation_strategy__blind"
        prompt="Write final deliverable to " + root + "/output/submission.zip."
        result=audit.evaluate_task("foundation-strategy",prompt,{},reward=0)
        self.assertEqual(result["state"],"no_detected_target_export")
        self.assertEqual(result["historical_template_reward"],0)

    def test_generic_output_folder_matches_real_files(self):
        prompt="Save the files that make up your answer under output/ in this workspace."
        result=audit.evaluate_task(
            "landscape-sheet-revision",prompt,
            {"app/files/output/Street_Tree_Memo.docx":"hash"})
        self.assertEqual(result["cwd_inferred"],"/app/files")
        self.assertEqual(result["state"],"all_detected_targets_exported")


    def test_nested_ifc_project_root_is_preserved(self):
        prompt=("Inputs: docs/example.ifc is a starter. "
                "## Expected outputSubmit submission/model.ifc. "
                "The final grade uses the final file only.")
        result=audit.evaluate_task("data-center-site-model",prompt,{
            "app/files/frontier__DCBim__p6_002/submission/model.ifc":"hash"})
        self.assertEqual(result["cwd_inferred"],"/app/files/frontier__DCBim__p6_002")
        self.assertEqual(result["state"],"all_detected_targets_exported")

    def test_mep_script_does_not_count_as_required_ifc(self):
        prompt=("Write your deliverable to output/model.ifc, "
                "and verify its geometry before final submission.")
        result=audit.evaluate_task("data-hall-mep-routing",prompt,{
            "app/files/dc_model__hall_seed000_d000__mep_routing/output/solution.py":"hash"})
        self.assertEqual(result["cwd_inferred"],
                         "/app/files/dc_model__hall_seed000_d000__mep_routing")
        self.assertEqual(result["state"],"no_detected_target_export")
        self.assertEqual(len(result["unmatched"]),1)

    def test_docstore_only_is_not_falsely_granted_local_file(self):
        prompt=("Return your work as a single Word document named "
                "Municipal_Witness_Test_Package.docx, filed in the "
                "Program Office's document store.")
        result=audit.evaluate_task("solar-witness-test-package",prompt,{})
        self.assertEqual(result["state"],"no_explicit_output_contract")


class ArtifactCustody(unittest.TestCase):
    def test_named_return_requires_one_matching_export(self):
        prompt = "Return a memo named Review.docx."
        item = {"state": "verified", "actual_blob": "example"}
        files = {"filesystem/Review.docx": item}
        row = {"targets": []}
        self.assertEqual(audit.custody(prompt, row, files)[0]["state"], "verified")
        files["app/files/Review.docx"] = item
        self.assertEqual(audit.custody(prompt, row, files)[0]["state"], "ambiguous_exports")

    def test_remote_filing_needs_destination_evidence(self):
        prompt = ("Return a memo named Review.docx, filed in the commissioning "
                  "folder in the document store, with a signed decision.")
        files = {"filesystem/Review.docx": {"state": "verified"}}
        result = audit.custody(prompt, {"targets": []}, files)[0]
        self.assertEqual(result["state"], "destination_receipt_unverified")
        self.assertEqual(result["destination"], "the commissioning folder in the document store")
        self.assertEqual(len(result["exports"]), 1)

    def test_declared_local_folder_uses_exact_path(self):
        prompt = "Return a memo named Review.docx, filed in /filesystem/reports."
        files = {"filesystem/Review.docx": {"state": "verified"}}
        self.assertEqual(audit.custody(prompt, {"targets": []}, files)[0]["state"], "missing_export")

    def test_verify_export_matches_git_blob_and_actual_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "filesystem/memo.txt"
            path.parent.mkdir()
            path.write_bytes(b"approved")
            expected = hashlib.sha1(b"blob 8\0approved").hexdigest()
            observed = audit.verify_export(root, "filesystem/memo.txt", expected)
            self.assertEqual(observed["state"], "verified")
            self.assertEqual(observed["sha256"], hashlib.sha256(b"approved").hexdigest())
            path.write_bytes(b"revised!")
            self.assertEqual(audit.verify_export(root, "filesystem/memo.txt", expected)["state"], "blob_mismatch")

    def test_status_needs_every_requested_destination(self):
        verified = {"state": "verified"}
        self.assertEqual(audit.delivery_status([verified]), "verified")
        self.assertEqual(audit.delivery_status([verified, {"state": "missing_export"}]), "incomplete")
        self.assertEqual(audit.delivery_status([verified, {"state": "destination_receipt_unverified"}]), "unverified")
        self.assertEqual(audit.delivery_status([]), "unverified")

    def test_terminal_observation_belongs_to_the_call(self):
        step = {"step_id": 11, "tool_calls": [{"tool_call_id": "last"}],
                "observation": {"results": [{"source_call_id": "earlier"}]}}
        self.assertEqual(audit.terminal_step([step])["without_observation"], ["last"])
        step["observation"]["results"].append({"source_call_id": "last"})
        self.assertEqual(audit.terminal_step([step])["without_observation"], [])


if __name__=="__main__":
    unittest.main()
