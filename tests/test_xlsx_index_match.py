"""True cross-sheet INDEX/MATCH returns resolved independently of stored caches."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import package, worksheet


class IndexMatchTests(TestCase):
    def make_sheet(self, formula, cached, table=None):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "allowables.xlsx"
        package(path)
        with ZipFile(path) as archive:
            parts = {n: archive.read(n) for n in archive.namelist()}
        parts["xl/worksheets/1.xml"] = worksheet([[("A", None, 2), ("B", formula, cached)]])
        if table is None:
            table = [[("A", None, 1)], [("A", None, 2), ("C", None, 110)],
                     [("A", None, 2), ("C", None, 175)], [("C", None, 250)]]
        parts["xl/worksheets/2.xml"] = worksheet(table)
        with ZipFile(path, "w") as archive:
            for n, value in parts.items():
                archive.writestr(n, value)
        return path

    formula = "INDEX('Ratings 2026'!$C$2:$C$4,MATCH(Results!A1,'Ratings 2026'!A1:A3,0))"

    def test_offset_and_first_exact_duplicate(self):
        for cache in (110, 175):
            with self.subTest(cache=cache):
                path = self.make_sheet(self.formula, cache)
                result = xlsx_audit(path)["lookup_audit"]["examples"][0]
                self.assertEqual(result["source_cell"], "C3")
                self.assertEqual(result["expected"], 175)
                self.assertEqual(result["status"], "matched" if cache == 175 else "mismatch")
                self.assertEqual(audit_local_deliverable(str(path))["ok"], cache == 175)

    def test_unsupported_forms_never_validate_cached_results(self):
        for change, into in ((",0)", ",1)"), (",0)", ")"), ("A1:A3","A1:B3"),
                             ("A1:A3","A1:A4"), ("A1:A3","A1:A2049"),
                             ("'Ratings 2026'","'[other.xlsx]Ratings 2026'"),
                             ("Results!A1","Unknown!A1")):
            with self.subTest(change=change, replacement=into):
                result = xlsx_audit(self.make_sheet(self.formula.replace(change, into), 175))["lookup_audit"]
                self.assertEqual(result["counts"]["unverified"], 1)
                self.assertTrue(result["examples"][0]["reason"])

    def test_unknown_preceding_key_and_cached_formula_source(self):
        for table in (
            [[("A","1+0",1)], [("A",None,2)], [("C",None,175)]],
            [[("A",None,1)], [("A",None,2)], [("C","7*25",175)]],
        ):
            result = xlsx_audit(self.make_sheet(self.formula, 175, table))["lookup_audit"]
            self.assertEqual(result["counts"]["unverified"], 1)
