"""Read genuine shared master and follower cells from a saved OOXML workbook."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from engineering_toolkit import xlsx_audit
from test_xlsx_exact_lookup import MAIN, package


def audit_shared(rows, group_range="B1:B3", key="A1", reverse=False):
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "catalog.xlsx"
        package(path)
        with ZipFile(path) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        part = "xl/worksheets/1.xml"
        root = ET.fromstring(parts[part])
        ns = "{" + MAIN + "}"
        data = root.find(ns + "sheetData")
        for index, (kind, group, literal, cached) in enumerate(rows):
            row = data[index]
            key_cell, result_cell = row.findall(ns + "c")
            key_cell.find(ns + "v").text = str(literal)
            f = result_cell.find(ns + "f")
            f.attrib = {"t": "shared", "si": group}
            if kind == "master":
                f.set("ref", group_range)
                f.text = f"VLOOKUP({key},'Ratings 2026'!$A$1:$C$4,2,FALSE)"
            else:
                f.text = None
            result_cell.find(ns + "v").text = str(cached)
        if reverse:
            data[:] = list(reversed(data))
        parts[part] = ET.tostring(root)
        with ZipFile(path, "w") as archive:
            for name, raw in parts.items():
                archive.writestr(name, raw)
        return xlsx_audit(str(path))


class SharedExactLookupTests(TestCase):
    def test_relative_rows_and_pinned_catalog(self):
        rows = [("master", "0", 1, 25), ("follower", "00", 2, 999), ("follower", "0", 3, 42)]
        for reverse in (False, True):
            audit = audit_shared(rows, reverse=reverse)
            cells = {r["cell"]: r for r in audit["lookup_audit"]["examples"]}
            self.assertEqual([cells[f"B{i}"]["status"] for i in range(1, 4)], ["matched","mismatch","matched"])
            self.assertEqual((cells["B2"]["cached"], cells["B2"]["expected"]), (999, 15))

    def test_absolute_row_and_range_membership(self):
        rows = [("master", "0", 1, 25), ("follower", "0", 2, 25), ("follower", "0", 3, 42)]
        audit = audit_shared(rows, group_range="B1:B2", key="A$1")
        cells = {r["cell"]: r for r in audit["lookup_audit"]["examples"]}
        self.assertEqual([cells[f"B{i}"]["status"] for i in range(1, 4)], ["matched","matched","unverified"])

    def test_duplicate_master_or_orphan_has_no_certified_follower(self):
        for group in ("12", "19"):
            rows = [("master", "12", 1, 25), ("follower", group, 2, 999), ("master", "12", 3, 42)]
            audit = audit_shared(rows)
            self.assertEqual(audit["sheets"][0]["unresolved_shared_formula_followers"], 1)
            cells = {r["cell"]: r for r in audit["lookup_audit"]["examples"]}
            self.assertNotIn("B2", cells)
            self.assertEqual(cells["B1"]["status"], "unverified")
