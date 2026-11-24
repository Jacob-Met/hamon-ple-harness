"""Exercise real draft publication and the ordinary freshness/selection contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import archive_checkpoint
import delivery_materialize
import hamon_delivery_draft as draft
import hamon_witness_package as witness
from office_package import check_office_package


class DraftPublication(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        roots = patch.object(draft, "TASK_ROOTS", (self.root,))
        roots.start()
        self.addCleanup(roots.stop)

    def publish(self, target, names, data, *, fingerprint=None):
        source = target.with_name(target.name + ".draft.json")
        content = json.dumps(data).encode()
        source.write_bytes(content)
        fingerprint = fingerprint or hashlib.sha256(str(len(content)).encode() + content).hexdigest()
        return draft.render(source, target, names, fingerprint)

    def test_signed_off_schedule_creates_a_working_copy(self):
        data = witness.example()
        data["case"] = "Controlled municipal commissioning case"
        data["scheduling_items"][0]["case"] = data["case"]
        dest = self.root / "Municipal_Witness_Test_Package.docx"
        self.assertEqual(self.publish(dest, [], data), {"state": "executed"})
        with ZipFile(dest) as z:
            check_office_package(z, ".docx")
            self.assertIn(data["case"], z.read("word/document.xml").decode())
        old = dest.read_bytes()
        with self.assertRaises(ValueError):
            self.publish(dest, [], data)
        self.assertEqual(dest.read_bytes(), old)

    def test_example_unresolved_case_and_changed_draft_stay_missing(self):
        target = self.root / "witness.docx"
        data = witness.example()
        for source, fingerprint in [(data, None), (data, "stale fingerprint")]:
            with self.assertRaises(ValueError):
                self.publish(target, [], source, fingerprint=fingerprint)
        data["scheduling_items"] = []
        with self.assertRaises(ValueError):
            self.publish(target, [], data)
        self.assertFalse(target.exists())

    def test_six_authored_members_use_original_workbook_templates(self):
        original = next((Path(__file__).resolve().parents[1] / "artifacts/eor-review-corrections/files").rglob("submission.zip"))
        template = self.root / "register.xlsx"
        with ZipFile(original) as z:
            template.write_bytes(z.read("constraint_register.xlsx"))
        with ZipFile(template) as z:
            book = ET.fromstring(z.read("xl/workbook.xml"))
            name = next(e for e in book.iter() if e.tag.endswith("}sheet")).attrib["name"]
            files = {p: z.read(p) for p in z.namelist()}
        members = {
            "foundation_geometry.json": {"json": {"source": "controlled geometry record"}},
            "ground_improvement_plan.json": {"json": {"source": "controlled plan record"}},
            "foundation_schedule.xlsx": {"template": str(template), "cells": {name: {"A240": 12.5}}},
            "constraint_register.xlsx": {"template": str(template), "cells": {name: {"A240": "Source requirement"}}},
            "foundation_feasibility_matrix.md": {"text": "Controlled comparison from the project fixture."},
            "design_basis.md": {"text": "Controlled design evidence from the fixture."},
        }
        output = self.root / "submission.zip"
        self.publish(output, list(members), {"members": members})
        archive_checkpoint.verify(output, list(members))
        import io
        with ZipFile(output) as result, ZipFile(io.BytesIO(result.read("constraint_register.xlsx"))) as revised:
            differs = [p for p, data in files.items() if revised.read(p) != data]
            self.assertEqual(len(differs), 1)
            cell = next(e for e in ET.fromstring(revised.read(differs[0])).iter() if e.get("r") == "A240")
            self.assertEqual("".join(cell.itertext()), "Source requirement")
        output.unlink()
        missing = dict(members)
        missing.pop("design_basis.md")
        with self.assertRaises(ValueError):
            self.publish(output, list(members), {"members": missing})
        self.assertFalse(output.exists())
        members["constraint_register.xlsx"]["template"] = "/etc/passwd"
        with self.assertRaises(ValueError):
            self.publish(output, list(members), {"members": members})
        self.assertFalse(output.exists())

    def test_owner_formulas_are_preserved_and_their_old_values_are_recalculated(self):
        xml = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>2</v></c><c r="B1"><f>A1*2</f><v>4</v></c></row></sheetData></worksheet>'
        changed = draft.patch_sheet(xml, {"A1": 5})
        refreshed, formulas = draft.invalidate_formula_cache(changed)
        self.assertTrue(formulas)
        root = ET.fromstring(refreshed)
        cell = next(n for n in root.iter() if n.get("r") == "B1")
        self.assertEqual([n.tag.rsplit("}", 1)[-1] for n in cell], ["f"])
        self.assertEqual(cell[0].text, "A1*2")
        with self.assertRaises(ValueError):
            draft.patch_sheet(xml, {"B1": 10})

    def test_draft_needs_a_qualified_task_and_fresh_observation(self):
        dest = "/filesystem/project/output/return.zip"
        targets = [{"path": dest, "kind": "file"}]
        self.assertFalse(any("draft_members" in x for x in delivery_materialize.candidates(targets)))
        rows = delivery_materialize.candidates(targets, archive_members=["body.md"])
        self.assertEqual(rows[-1]["script"], dest + ".draft.json")
        source = rows[-1]["script"]
        ready = {source: {"state": "ready", "fingerprint": "observed bytes"}}
        self.assertEqual(delivery_materialize.plan(rows, [{"path": dest}], ready, ready), [])
        chosen = delivery_materialize.plan(rows, [{"path": dest}], {source: {"state":"missing"}}, ready)
        self.assertEqual(chosen, [{**rows[-1], "fingerprint":"observed bytes"}])
