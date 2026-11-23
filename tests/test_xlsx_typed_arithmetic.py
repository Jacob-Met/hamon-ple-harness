"""Cross-sheet numeric caches and explicit SI dimensions on the saved XLSX package."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from test_xlsx_exact_lookup import package, worksheet


def workbook_parts(source):
    """Include the package manifest and root relationship used by spreadsheet applications."""
    with ZipFile(source) as original:
        parts = {name: original.read(name) for name in original.namelist()}
    parts["[Content_Types].xml"] = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.sheet.main+xml"/>'
        + ''.join(f'<Override PartName="/xl/worksheets/{i}.xml" ContentType="application/vnd.'
                  'openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in (1,2))
        + '</Types>').encode()
    parts["_rels/.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rWorkbook" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>').encode()
    return parts


class TypedWorkbookTests(unittest.TestCase):
    def guards(self, results, ratings):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "support-load.xlsx"
            package(path)
            parts = workbook_parts(path)
            parts["xl/worksheets/1.xml"] = worksheet([results])
            parts["xl/worksheets/2.xml"] = worksheet([ratings])
            with ZipFile(path, "w") as book:
                for name, data in parts.items():
                    book.writestr(name, data)
            cli = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
            audit = subprocess.run([sys.executable, str(cli), "xlsx-audit", str(path)],
                                   check=True, capture_output=True, text=True)
            guard = subprocess.run([sys.executable, str(cli), "format-guard", str(path)],
                                   capture_output=True, text=True)
            self.assertIn(guard.returncode, (0, 3), guard.stderr)
            return json.loads(audit.stdout), guard.returncode, json.loads(guard.stdout)

    def test_independently_reconstructs_both_original_wrong_caches(self):
        for middle, total, exit_code in [(899,902,3),(40,43,0)]:
            with self.subTest(middle=middle):
                audit, code, guard = self.guards(
                    [("A",None,3),("B","'Ratings 2026'!B1*2",middle),("C","B1+A1",total)],
                    [("A",None,5),("B","A1*4",20)])
                examples = {(r["sheet"],r["cell"]):r for r in audit["arithmetic_audit"]["examples"]}
                self.assertEqual([examples["Results",c]["expected"] for c in ("B1","C1")],[40,43])
                self.assertEqual(code, exit_code, guard)

    def test_explicit_units_reject_numerically_correct_incompatible_sum(self):
        results = [("A","'Ratings 2026'!B1*2",2000),
                   ("B","'Ratings 2026'!D1",3600),("C","A1+B1",5600),("D","C1/2",2800)]
        ratings = [("A",None,1),("B",'CONVERT(A1,"m","mm")',1000),("C",None,1),
                   ("D",'CONVERT(C1,"hr","sec")',3600)]
        audit, code, guard = self.guards(results, ratings)
        self.assertEqual(audit["conversion_audit"]["counts"]["matched"],2)
        records = {r["cell"]:r for r in audit["arithmetic_audit"]["examples"]}
        self.assertEqual(records["C1"]["status"],"dimension_mismatch")
        self.assertEqual(records["C1"]["left_dimension"],[0,1,0,0,0])
        self.assertEqual(records["C1"]["right_dimension"],[0,0,1,0,0])
        self.assertEqual(records["D1"]["reason"],"upstream_dimension_mismatch")
        self.assertEqual(code,3,guard)
        self.assertIn("arithmetic_dimension_mismatches=1",guard["reason"])

    def test_product_ratio_cancellation_and_unlabelled_cells(self):
        ratings = [("A",None,1),("B",'CONVERT(A1,"m","cm")',100),
                   ("C",None,2),("D",'CONVERT(C1,"hr","sec")',7200)]
        a, b = "'Ratings 2026'!B1", "'Ratings 2026'!D1"
        audit, code, guard = self.guards(
            [("A",a+"*"+b,720000),("B","A1/"+b,100),("C","B1+"+a,200),
             ("D","B1+'Ratings 2026'!A1",101),("E","B1+1",101)],ratings)
        records = {r["cell"]:r for r in audit["arithmetic_audit"]["examples"]}
        self.assertEqual(code,0,guard)
        self.assertEqual(records["A1"]["si_dimension"],[0,1,1,0,0])
        self.assertEqual(records["C1"]["si_dimension"],[0,1,0,0,0])
        self.assertNotIn("si_dimension",records["D1"])
        self.assertNotIn("si_dimension",records["E1"])

    def test_stale_conversion_is_never_an_input_and_cycle_remains_unknown(self):
        ratings = [("A",None,1),("B",'CONVERT(A1,"m","cm")',8),
                   ("C",'CONVERT(B1,"cm","mm")',80)]
        audit, code, guard = self.guards(
            [("A","'Ratings 2026'!C1*2",160),("B","C1+1",5),("C","B1*2",10),
             ("D","SUM(1,2)",3),("E","D1+1",4)],ratings)
        records = {r["cell"]:r for r in audit["arithmetic_audit"]["examples"]}
        self.assertEqual(records["A1"]["expected"],2000)
        self.assertEqual(records["A1"]["status"],"mismatch")
        self.assertEqual(records["B1"]["reason"],"cyclic_or_deep_dependency")
        self.assertEqual(records["D1"]["status"],"unverified")
        self.assertEqual(records["E1"]["status"],"unverified")
        self.assertEqual(code,3,guard)
