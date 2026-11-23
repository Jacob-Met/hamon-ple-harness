"""Compile dated documentary evidence through the task-container CLI."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree
from zipfile import ZipFile

import engineering_intake
from hamon_witness_package import W, example, paragraphs


class ScopeAndSchedule(unittest.TestCase):
    def test_asof_scope_release_and_witness_decisions_are_separate(self):
        spec = example()
        base = spec["arrays"][0]
        cleared = {**base, "id": "Array B", "carriage": "carried", "release_request": "supported"}
        outside = {"id": "Array C", "scope": "out", "scope_basis": base["scope_basis"]}
        spec["arrays"].extend([cleared, outside])
        result, _ = paragraphs(spec)
        self.assertEqual(result["witness_required"], ["Example array"])
        self.assertEqual(result["supported_release_requests"], ["Array B"])
        self.assertTrue(result["schedule_matches"])
        for key, target in (("scope_basis", outside), ("installation_basis", cleared),
                            ("carriage_basis", cleared), ("release_basis", cleared)):
            with self.subTest(key=key):
                original = target[key]
                target[key] = dict(original, date="2025-01-03")
                with self.assertRaisesRegex(ValueError, "postdates"):
                    paragraphs(spec)
                target[key] = original

    def test_standing_items_form_an_exact_multiset(self):
        spec = example()
        item = spec["scheduling_items"][0]
        spec["scheduling_items"] = [dict(item, standing="no"), dict(item, id="2", array="Other")]
        self.assertEqual(paragraphs(spec)[0]["schedule_issues"],
                         {"missing": ["Example array"], "unexpected": ["Other"], "duplicate": []})
        spec["scheduling_items"] = [item, dict(item, id="3")]
        self.assertEqual(paragraphs(spec)[0]["schedule_issues"]["duplicate"], ["Example array"])
        spec["scheduling_items"] = [dict(item, case="Another case")]
        with self.assertRaisesRegex(ValueError, "different case"):
            paragraphs(spec)

    def test_incomplete_evidence_cannot_become_a_final_package(self):
        for key in ("installed_equipment", "carriage_basis", "release_basis", "tests"):
            with self.subTest(key=key):
                spec = example()
                del spec["arrays"][0][key]
                with self.assertRaises(ValueError):
                    paragraphs(spec)
        spec = example()
        spec["case_snapshot"]["date"] = "2025-01-01"
        with self.assertRaisesRegex(ValueError, "as-of"):
            paragraphs(spec)

    def test_provisioned_cli_generates_and_withholds_a_new_word_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Receiver:
                async def write_file(self, path, data):
                    return (root / Path(path).name).write_bytes(data)
            note = asyncio.run(engineering_intake.provision_engineering_cli(Receiver()))
            self.assertIn("witness-package", note)
            command = [sys.executable, "-E", "-s", "-S", str(root / "hamon-engineering-toolkit.py")]
            spec = json.loads(subprocess.check_output(command + ["witness-package", "--example"], cwd=root))
            inp, out = root / "case.json", root / "package.docx"
            def execute(data):
                inp.write_text(json.dumps(data))
                return subprocess.run(command + ["witness-package", str(inp), "--output", str(out)],
                                      cwd=root, capture_output=True, text=True)
            missing = deepcopy(spec)
            missing["scheduling_items"] = []
            self.assertEqual(execute(missing).returncode, 3)
            self.assertFalse(out.exists())
            self.assertEqual(execute(spec).returncode, 0)
            with ZipFile(out) as z:
                body = ElementTree.fromstring(z.read("word/document.xml"))
                words = " ".join(n.text or "" for n in body.iter("{"+W+"}t"))
                for text in ("Example array", "EX-1", "Designated witness", "Approved procedure", "Standards Engineering"):
                    self.assertIn(text, words)
            self.assertEqual(subprocess.run(command + ["format-guard", str(out)], capture_output=True).returncode, 0)
            saved = out.read_bytes()
            self.assertEqual(execute(spec).returncode, 2)
            self.assertEqual(out.read_bytes(), saved)
