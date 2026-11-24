"""Coverage and isolation of bounded workbook calculation graphs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import package, worksheet


class WorkbookComponents(unittest.TestCase):
    def audit(self, first, second):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "components.xlsx"
            package(path)
            with ZipFile(path) as source:
                parts = {name: source.read(name) for name in source.namelist()}
            parts["xl/worksheets/1.xml"] = worksheet(first)
            parts["xl/worksheets/2.xml"] = worksheet(second)
            with ZipFile(path, "w") as target:
                for name, data in parts.items():
                    target.writestr(name, data)
            return xlsx_audit(path), audit_local_deliverable(path)

    def test_cross_sheet_forward_producers_and_untrusted_caches(self):
        # Each of the 140 rows is independent despite one shared literal input.
        first = [[("A", f"'Ratings 2026'!B{i}*2", 12 if i < 140 else 28)]
                 for i in range(1, 141)]
        second = [[("A", None, 3), ("B", "$A$1*2", 6 if i < 140 else 14)]
                  for i in range(1, 141)]
        audit, guard = self.audit(first, second)
        report = audit["arithmetic_audit"]
        self.assertEqual(report["counts"], {"matched": 278, "mismatch": 2, "unverified": 0})
        wrong = {(r["sheet"], r["cell"]): r["expected"] for r in report["examples"]
                 if r["status"] == "mismatch"}
        self.assertEqual(wrong, {("Results", "A140"): 12, ("Ratings 2026", "B140"): 6})
        self.assertFalse(guard["ok"])

    def test_oversize_component_does_not_consume_the_next_sheets_checks(self):
        # One producer feeds 259 siblings. The independent second sheet is still checked.
        first = [[("A", "2*3" if i == 1 else "$A$1*2", 6 if i == 1 else 12)]
                 for i in range(1, 261)]
        audit, guard = self.audit(first, [[("A", "2*3", 99)]])
        rows = audit["arithmetic_audit"]
        self.assertEqual(rows["counts"], {"matched": 256, "mismatch": 1, "unverified": 4})
        self.assertEqual(rows["limits"], {"component_omitted": 4, "index_omitted": 0})
        self.assertTrue(all(r["reason"] == "arithmetic_component_limit"
                            for r in rows["examples"] if r["status"] == "unverified"))
        self.assertFalse(guard["ok"])
