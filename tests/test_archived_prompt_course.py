"""Qualification boundaries for the two-arm archived Pi/SSE replay."""
import json
from pathlib import Path
import tempfile
import unittest

from replay_empty_worlds import PROMPT_COURSE, ROOT, load_cases
from ship.audit_ple_delivery import task_workdir


class ArchivedPromptCourseTests(unittest.TestCase):
    def test_full_receipt_preserves_all_original_prompts_and_inferred_workdirs(self):
        artifacts = ROOT / "artifacts"
        cases = load_cases(artifacts)
        self.assertEqual(len(cases), 29)
        self.assertEqual([task for task, _, _ in cases], sorted(PROMPT_COURSE["prompts"]))
        for task, message, workdir in cases:
            files = artifacts / task / "files"
            exports = ["/" + str(path.relative_to(files)) for path in files.rglob("*") if path.is_file()]
            self.assertEqual(workdir, task_workdir(message, exports), task)

    def test_full_course_rejects_a_missing_world_and_an_unregistered_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task in PROMPT_COURSE["prompts"]:
                folder = root / task
                folder.mkdir()
                (folder / "trajectory.json").touch()
            (root / "foundation-strategy" / "trajectory.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing=.*foundation-strategy"):
                load_cases(root)
            (root / "foundation-strategy" / "trajectory.json").touch()
            extra = root / "unexpected-world"
            extra.mkdir()
            (extra / "trajectory.json").touch()
            with self.assertRaisesRegex(ValueError, "unknown=.*unexpected-world"):
                load_cases(root)

    def test_subset_rejects_unknown_duplicate_and_unavailable_task_ids(self):
        known = "foundation-strategy"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unknown=.*foundaton-strategy"):
                load_cases(ROOT / "artifacts", [known, "foundaton-strategy"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cases(ROOT / "artifacts", [known, known])
            with self.assertRaisesRegex(ValueError, "missing=.*foundation-strategy"):
                load_cases(root, [known])

    def test_only_the_original_first_user_prompt_enters_either_arm(self):
        task = "fire-test-validation-memo"
        original = load_cases(ROOT / "artifacts", [task])
        _, prompt, _ = original[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / task
            folder.mkdir()
            trace = folder / "trajectory.json"
            first = {"step_id": 1, "source": "user", "message": prompt}
            for replacement, problem in (
                (dict(first, message=prompt + " "), "digest"),
                (dict(first, source="agent"), "first trajectory step"),
                (dict(first, step_id=2), "first trajectory step"),
                (dict(first, step_id=True), "first trajectory step"),
            ):
                trace.write_text(json.dumps({"steps": [replacement]}))
                with self.assertRaisesRegex(ValueError, problem):
                    load_cases(folder.parent, [task])
            trace.write_text(json.dumps({"steps": [first, {"message": "later held out of replay"}]}))
            self.assertEqual(load_cases(folder.parent, [task]), original)
