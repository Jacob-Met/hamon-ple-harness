"""Source provenance and invalid-inherited-artifacts gates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ship import check_artifact_provenance as p


class ProvenanceGateTests(unittest.TestCase):
    def make_project(self):
        tmp=tempfile.TemporaryDirectory()
        root=Path(tmp.name)
        for name in p.FIXED_RUNTIME:
            (root/name).write_text(f"fixture contents {name}")
        (root/"engineering_toolkit.py").write_text("def convert(): return 1\n")
        (root/"artifacts").mkdir()
        manifest={"harness":{"commit":"example","tree":"example",
                             "files":{name:"100644 "+p.blob_sha((root/name).read_bytes())
                                       for name in (p.FIXED_RUNTIME|{"engineering_toolkit.py"})}}}
        (root/"artifacts"/"manifest.json").write_text(json.dumps(manifest))
        return tmp,root

    def test_current_exact_source_is_necessary_for_fresh(self):
        tmp,root=self.make_project()
        with tmp:
            report=p.evaluate(root)
            self.assertEqual(report["status"],"fresh")
            self.assertEqual(report["source_files_checked"],len(p.FIXED_RUNTIME)+1)
            self.assertEqual(report["diverged"],[])

    def test_stale_source_and_unrepresented_new_module_fail(self):
        tmp,root=self.make_project()
        with tmp:
            (root/"harness.py").write_text("not the generator")
            (root/"engineering_intake.py").write_text("new runtime imported by harness")
            report=p.evaluate(root)
            self.assertEqual(report["status"],"stale")
            self.assertIn("engineering_intake.py",report["missing_in_manifest"])
            self.assertEqual([v["path"] for v in report["diverged"]],["harness.py"])
            self.assertEqual(p.main([str(root)]),3)

    def test_missing_old_executable_and_symlink_refused(self):
        tmp,root=self.make_project()
        with tmp:
            (root/"taskbridge.py").unlink()
            (root/"taskbridge.py").symlink_to(root/"harness.py")
            report=p.evaluate(root)
            self.assertEqual(report["status"],"stale")
            self.assertIn({"path":"taskbridge.py",
                           "reason":"missing_or_symlink_runtime_input"},report["unsafe"])

    def test_malformed_manifest_refused(self):
        tmp,root=self.make_project()
        with tmp:
            (root/"artifacts"/"manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"lacks a harness.files"):
                p.evaluate(root)

    def test_real_repository_requires_a_new_evaluation(self):
        root=Path(__file__).resolve().parents[1]
        manifest=root/"artifacts"/"manifest.json"
        if manifest.is_file():
            report=p.evaluate(root)
            # This can change when a genuine new dev run is checked in; the
            # correctness assertion is on the detector's witness structure,
            # not a frozen expectation that artifacts must remain stale.
            self.assertEqual(report["status"],"stale" if
                (report["diverged"] or report["missing_in_manifest"] or
                 report["unsafe"] or report["removed_executable"]) else "fresh")
            self.assertGreater(report["source_files_checked"],0)


if __name__=="__main__":
    unittest.main()
