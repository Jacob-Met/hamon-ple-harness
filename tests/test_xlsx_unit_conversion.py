"""Audit numeric unit-conversion caches in a saved two-sheet package."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import MAIN, package, worksheet


def workbook(path, formulas):
    package(path)
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/workbook.xml"] = parts["xl/workbook.xml"].replace(b"Ratings 2026", b"Field's Data")
    parts["xl/worksheets/1.xml"] = worksheet([[("A", f, cached)] for f, cached in formulas])
    parts["xl/worksheets/2.xml"] = worksheet([
        [("A", None, 25.4), ("B", "12.7*2", 25.4)],
        [("A", "12.7*2", 25.4)], [("A", None, 25.4)]])
    # The A3 cached value is an array follower, not a literal dependency.
    root = ET.fromstring(parts["xl/worksheets/2.xml"])
    master = root.find(f".//{{{MAIN}}}c[@r='A2']/{{{MAIN}}}f")
    master.attrib.update(t="array", ref="A2:A3")
    parts["xl/worksheets/2.xml"] = ET.tostring(root)
    with ZipFile(path, "w") as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


class UnitConversionTests(TestCase):
    def test_saved_cross_sheet_value_and_dimension(self):
        source = "'field''s Data'!$A$1"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "field.xlsx"
            workbook(path, [
                (f'CONVERT({source},"mm","in")', 1),
                (f'CONVERT({source},"mm","in")', 25.4),
                (f'CONVERT({source},"mm","kg")', 0)])
            audit = xlsx_audit(str(path))["conversion_audit"]
            self.assertEqual(audit["counts"], dict(matched=1, mismatch=1, dimension_mismatch=1, unverified=0))
            bad = audit["examples"][1]
            self.assertEqual((bad["source_sheet"], bad["source_cell"], bad["expected"], bad["cached"]),
                             ("Field's Data", "A1", 1, 25.4))
            self.assertFalse(audit_local_deliverable(str(path))["ok"])

    def test_arithmetic_is_reconstructed_while_other_dependencies_stay_unverified(self):
        sheet = "'Field''s Data'!"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.xlsx"
            workbook(path, [
                (f'CONVERT({sheet}B1,"mm","in")', 999),
                (f'CONVERT({sheet}A3,"mm","in")', 999),
                (f'CONVERT({sheet}A1,"MM","in")', 999),
                (f'CONVERT({sheet}A1,"mm","in")', None),
                ('CONVERT(\'[other.xlsx]Sheet\'!A1,"mm","in")', 999),
                (f'CONVERT({sheet}XFE1,"mm","in")', 999),
                (f'CONVERT({sheet}A1+1,"mm","in")', 999)])
            counts = xlsx_audit(str(path))["conversion_audit"]["counts"]
            self.assertEqual(counts, dict(matched=0, mismatch=2, dimension_mismatch=0, unverified=5))
            self.assertFalse(audit_local_deliverable(str(path))["ok"])
