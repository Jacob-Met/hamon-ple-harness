"""Workbook receipts for conversions whose stated source contradicts reconstructed units."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import package, worksheet


class ConversionSourceTests(unittest.TestCase):
    def evaluate(self, source, units, cache):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source-frame.xlsx"
            package(path)
            with ZipFile(path) as archive:
                parts = {name: archive.read(name) for name in archive.namelist()}
            parts["xl/worksheets/1.xml"] = worksheet([[
                ("A", 'CONVERT(2,"m","cm")', 200),
                ("B", f'CONVERT({source},"{units[0]}","{units[1]}")', cache),
                ("C", "B1*2", None if cache is None else cache*2),
            ]])
            with ZipFile(path, "w") as archive:
                for name, data in parts.items():
                    archive.writestr(name, data)
            audit = xlsx_audit(path)
            guard = audit_local_deliverable(path)
        rows = {row["cell"]: row for kind in ("conversion", "arithmetic")
                for row in audit[kind + "_audit"]["examples"] if row["sheet"] == "Results"}
        return rows, guard

    def test_proven_source_contradiction_rejects_with_or_without_its_numeric_cache(self):
        for units, kind, cached in [(("m","mm"), "unit_scale_mismatch", 400000),
                                    (("kg","g"), "dimension_mismatch", 400000)]:
            for cache in (cached, None):
                with self.subTest(units=units, cache=cache):
                    rows, guard = self.evaluate("A1*2", units, cache)
                    self.assertFalse(guard["ok"])
                    self.assertEqual(rows["B1"]["status"], kind)
                    self.assertEqual(rows["B1"]["from_unit"], units[0])
                    self.assertEqual(rows["C1"]["reason"], "upstream_" + kind)
                    self.assertNotIn("expected", rows["B1"])

    def test_corrected_and_unresolved_sources_stay_distinct(self):
        rows, guard = self.evaluate("A1*2", ("cm","mm"), 4000)
        self.assertTrue(guard["ok"])
        self.assertEqual(rows["B1"]["expected"], 4000)
        self.assertEqual(rows["C1"]["status"], "matched")
        rows, guard = self.evaluate("A1*Z1", ("kg","g"), 400000)
        self.assertTrue(guard["ok"])
        self.assertEqual(rows["B1"]["status"], "unverified")

    def test_required_frame_survives_an_unlabelled_addend(self):
        rows, guard = self.evaluate("A1+1", ("m","mm"), 201000)
        self.assertFalse(guard["ok"])
        self.assertEqual(rows["B1"]["status"], "unit_scale_mismatch")
        self.assertEqual(rows["B1"]["required_scale_si"], "1/100")
        self.assertEqual(rows["B1"]["declared_scale_si"], "1")
