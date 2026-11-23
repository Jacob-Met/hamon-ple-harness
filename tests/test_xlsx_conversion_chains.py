"""Recompute saved conversion chains from numeric literals."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import MAIN, package, worksheet


class ConversionChains(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "engineering.xlsx"

    def evaluate(self, first, second=(), edit=None):
        package(self.path)
        with ZipFile(self.path) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        key = "xl/worksheets/1.xml"
        parts[key] = worksheet([first])
        parts["xl/worksheets/2.xml"] = worksheet([second])
        if edit:
            root = ET.fromstring(parts[key])
            edit(root)
            parts[key] = ET.tostring(root)
        with ZipFile(self.path, "w") as archive:
            for name, contents in parts.items():
                archive.writestr(name, contents)
        result = xlsx_audit(self.path)["conversion_audit"]
        return {e["cell"]: e for e in result["examples"]}, result["counts"]

    def test_stale_and_clean_intermediate_are_recomputed(self):
        for cm, metres, expected in (
            (999, 9.99, ("mismatch", "mismatch")),
            (999, 1, ("mismatch", "matched")),
            (100, 9.99, ("matched", "mismatch")),
            (100, 1, ("matched", "matched")),
            (None, 1, ("unverified", "matched")),
        ):
            with self.subTest(cm=cm, metres=metres):
                result, _ = self.evaluate([
                    ("C", 'CONVERT(B1,"cm","m")', metres),
                    ("B", 'CONVERT(A1,"m","cm")', cm), ("A", None, 1)])
                self.assertEqual((result["B1"]["status"], result["C1"]["status"]), expected)
                self.assertEqual((result["B1"]["expected"], result["C1"]["expected"]), (100, 1))
                self.assertEqual(result["C1"]["source_value"], 100)
                self.assertEqual((result["C1"]["root_cell"], result["C1"]["conversion_depth"]), ("A1", 2))
                self.assertEqual(audit_local_deliverable(self.path)["ok"], "mismatch" not in expected)

    def test_forward_cross_sheet_and_unit_continuity(self):
        ref = "'Ratings 2026'!$B$1"
        result, counts = self.evaluate([
            ("A", f'CONVERT({ref},"cm","mm")', 254), ("C", 'CONVERT(A1,"mm","in")', 10),
            ("D", 'CONVERT(A1,"mm","m")', .254), ("E", 'CONVERT(A1,"m","in")', 10000)],
            [("A", None, 10), ("B", 'CONVERT(A1,"in","cm")', 25.4)])
        self.assertEqual((counts["matched"], counts["unverified"]), (4, 1))
        self.assertEqual((result["C1"]["root_sheet"], result["C1"]["root_cell"]), ("Ratings 2026", "A1"))
        self.assertEqual(result["D1"]["conversion_depth"], 3)
        self.assertEqual(result["E1"]["reason"], "upstream_unit_disagrees")

    def test_refuses_cycles_and_unsupported_dependencies(self):
        result, counts = self.evaluate([
            ("A", 'CONVERT(B1,"m","m")', 1), ("B", 'CONVERT(A1,"m","m")', 1),
            ("C", 'CONVERT(B1,"m","cm")', 100), ("D", "2*2", 4),
            ("E", 'CONVERT(D1,"kg","g")', 4000), ("F", 'CONVERT(F1,"m","m")', 1),
            ("G", None, 7), ("H", 'CONVERT(G1,"m","kg")', 7), ("I", 'CONVERT(H1,"kg","g")', 7000)])
        for address in ("A1", "B1", "C1", "F1"):
            self.assertEqual(result[address]["reason"], "cyclic_reference")
        self.assertEqual((counts["dimension_mismatch"], counts["unverified"]), (1, 6))
        self.assertEqual(result["E1"]["reason"], "unsupported_or_missing_dependency")
        self.assertEqual(result["I1"]["status"], "unverified")

    def test_array_result_region_cannot_supply_formula(self):
        def array_region(root):
            root.find(f".//{{{MAIN}}}c[@r='B1']/{{{MAIN}}}f").attrib.update(t="array", ref="B1:C1")
        result, counts = self.evaluate([
            ("A", None, 1), ("B", "A1:A2", 1),
            ("C", 'CONVERT(A1,"m","cm")', 100), ("D", 'CONVERT(C1,"cm","m")', 1)], edit=array_region)
        self.assertEqual(counts["unverified"], 2)
        self.assertEqual(result["D1"]["status"], "unverified")

    def test_forward_chains_at_formula_index_limit(self):
        def column(index):
            chars = []
            while index:
                index, letter = divmod(index - 1, 26)
                chars.append(chr(ord("A") + letter))
            return "".join(reversed(chars))

        for length in (256, 257):
            with self.subTest(length=length):
                cells = [("A", None, 3)]
                for index in range(2, length + 2):
                    source = column(index + 1) if index < length + 1 else "A"
                    cells.append((column(index), f'CONVERT({source}1,"m","m")', 3))
                rows, counts = self.evaluate(cells)
                self.assertEqual(counts["matched"], 256 if length == 256 else 0)
                self.assertEqual(counts["unverified"], 0 if length == 256 else 257)
                if length == 256:
                    self.assertEqual(rows["B1"]["conversion_depth"], 256)
