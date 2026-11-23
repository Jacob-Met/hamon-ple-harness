"""Reconcile arithmetic formula results with independently reconstructed input values."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import package, worksheet


class ArithmeticWorkbookTests(TestCase):
    def workbook(self, rows, *, options=""):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "calculation.xlsx"
        package(path)
        with ZipFile(path) as original:
            parts = {p: original.read(p) for p in original.namelist()}
        parts["xl/worksheets/1.xml"] = worksheet(rows)
        parts["xl/worksheets/2.xml"] = worksheet([[("A", None, 2), ("B", None, 3)]])
        if options:
            parts["xl/workbook.xml"] = parts["xl/workbook.xml"].replace(
                b"</workbook>", options.encode() + b"</workbook>")
        with ZipFile(path, "w") as archive:
            for name, data in parts.items():
                archive.writestr(name, data)
        return path

    def test_cross_sheet_graph_uses_literal_roots_even_with_stale_intermediate(self):
        source = "'ratings 2026'!$A$1+'Ratings 2026'!B1"
        for middle, end, wrong in ((5, 20, 0), (5, 33, 1), (7, 20, 1), (7, 28, 2)):
            with self.subTest(middle=middle, end=end):
                path = self.workbook([[("A", source, middle), ("B", "A1 * 4", end)]])
                audit = xlsx_audit(str(path))["arithmetic_audit"]
                self.assertEqual(audit["counts"], {"matched": 2-wrong, "mismatch": wrong, "unverified": 0})
                self.assertEqual(audit["examples"][-1]["expected"], 20)
                self.assertEqual(audit_local_deliverable(path)["ok"], wrong == 0)

    def test_ordinary_cli_distinguishes_stale_correct_and_unsupported(self):
        cli = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
        for formula, cached, status in (("2*10", 20, 0), ("2*10", 33, 3), ("SUM(2,3)", 5, 0)):
            with self.subTest(formula=formula, cached=cached):
                path = self.workbook([[("A", formula, cached)]])
                proc = subprocess.run([sys.executable, str(cli), "format-guard", str(path)],
                                      capture_output=True, text=True, timeout=15)
                self.assertEqual(proc.returncode, status, proc.stdout + proc.stderr)
                result = json.loads(proc.stdout)
                if formula.startswith("SUM"):
                    self.assertEqual(result["unverified_arithmetic_formulas"], 1)
                elif status:
                    self.assertIn("arithmetic_mismatches=1", result["reason"])

    def test_numeric_precedence_scientific_notation_and_shared_dependency(self):
        path = self.workbook([[("A", ".5E+1", 5), ("B", "-(A1-1)/2+A1*3", 13),
                               ("C", "B1+A1+B1", 31)]])
        self.assertEqual(xlsx_audit(str(path))["arithmetic_audit"]["counts"]["matched"], 3)
        self.assertTrue(audit_local_deliverable(path)["ok"])

    def test_unknown_functions_and_cycles_do_not_supply_literals(self):
        path = self.workbook([[("A", "SUM(2,3)", 5), ("B", "A1*4", 20),
                               ("C", "D1+1", 12), ("D", "C1-1", 11),
                               ("E", "1/0", 1), ("F", "Missing!A1", 2)]])
        audit = xlsx_audit(str(path))["arithmetic_audit"]
        self.assertEqual(audit["counts"], {"matched": 0, "mismatch": 0, "unverified": 6})
        self.assertEqual(audit_local_deliverable(path)["unverified_arithmetic_formulas"], 6)

    def test_precision_sensitive_cancellation_does_not_reject_a_saved_zero(self):
        path = self.workbook([[("A", "0.5-0.4-0.1", 0), ("B", "A1*1e12", 0)]])
        counts = xlsx_audit(str(path))["arithmetic_audit"]["counts"]
        self.assertEqual(counts, {"matched": 0, "mismatch": 0, "unverified": 2})
        self.assertTrue(audit_local_deliverable(path)["ok"])

    def test_precision_as_displayed_is_outside_reconstruction(self):
        path = self.workbook([[("A", "1/3", .33)]], options='<calcPr fullPrecision="0"/>')
        counts = xlsx_audit(str(path))["arithmetic_audit"]["counts"]
        self.assertEqual(counts, {"matched": 0, "mismatch": 0, "unverified": 1})
