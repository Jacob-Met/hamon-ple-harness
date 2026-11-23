"""Saved workbook checks across arithmetic and exact unit-scale dependencies."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from engineering_toolkit import audit_local_deliverable, xlsx_audit
from test_xlsx_exact_lookup import MAIN, package, worksheet
import test_xlsx_typed_arithmetic as fixtures


class UnitGraphTests(unittest.TestCase):
    def evaluate(self, results, ratings=(), types=None, full_precision=True):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unit-graph.xlsx"
            package(path)
            parts = fixtures.workbook_parts(path)
            parts["xl/worksheets/1.xml"] = worksheet([results])
            parts["xl/worksheets/2.xml"] = worksheet([ratings])
            if types:
                root = ET.fromstring(parts["xl/worksheets/1.xml"])
                for cell, typ in types.items():
                    root.find(f".//{{{MAIN}}}c[@r='{cell}']").set("t", typ)
                parts["xl/worksheets/1.xml"] = ET.tostring(root)
            if not full_precision:
                root = ET.fromstring(parts["xl/workbook.xml"])
                ET.SubElement(root, f"{{{MAIN}}}calcPr", {"fullPrecision": "0"})
                parts["xl/workbook.xml"] = ET.tostring(root)
            with ZipFile(path, "w") as book:
                for name, contents in parts.items():
                    book.writestr(name, contents)
            audit = xlsx_audit(path)
            return audit, audit_local_deliverable(path)

    @staticmethod
    def records(audit, kind):
        return {(row["sheet"], row["cell"]): row for row in audit[kind+"_audit"]["examples"]}

    def test_literal_unary_arithmetic_and_nested_convert(self):
        audit, guard = self.evaluate([
            ("A", 'CONVERT(-1,"m","cm")', -100),
            ("B", 'CONVERT(1+1,"m","cm")', 200),
            ("C", 'CONVERT(2*3,"m","cm")', 600),
            ("D", 'CONVERT(CONVERT(3,"m","cm")/3,"cm","m")', 1),
            ("E", 'CONVERT(2*3,"m","cm")', 999)])
        self.assertEqual(audit["conversion_audit"]["counts"],
                         dict(matched=4, mismatch=1, dimension_mismatch=0, unverified=0))
        self.assertFalse(guard["ok"])
        self.assertEqual(self.records(audit, "conversion")["Results", "E1"]["expected"], 600)

    def test_wrong_arithmetic_cache_never_feeds_conversion_or_dependents(self):
        audit, guard = self.evaluate([
            ("A", 'CONVERT(\'Ratings 2026\'!B1,"m","cm")', 600),
            ("B", "A1*2", 1200)], [("A", None, 3), ("B", "A1*2", 999)])
        arithmetic = self.records(audit, "arithmetic")
        conversion = self.records(audit, "conversion")["Results", "A1"]
        self.assertEqual((arithmetic["Ratings 2026", "B1"]["status"],
                          arithmetic["Ratings 2026", "B1"]["expected"]), ("mismatch", 6))
        self.assertEqual((conversion["status"], conversion["expected"]), ("matched", 600))
        self.assertEqual((conversion["root_sheet"], conversion["root_cell"]), ("Ratings 2026", "A1"))
        self.assertEqual(arithmetic["Results", "B1"]["status"], "matched")
        self.assertFalse(guard["ok"])

    def test_exact_scale_cancellation_through_products_and_ratios(self):
        cm, metres, inches = ("'Ratings 2026'!"+cell for cell in ("B1", "C1", "D1"))
        audit, guard = self.evaluate([
            ("A", f"{cm}*{metres}/{cm}", 2),
            ("B", 'CONVERT(A1,"m","cm")', 200),
            ("C", f"{inches}*{cm}/{inches}", 200),
            ("D", 'CONVERT(C1,"cm","m")', 2),
            ("E", f"{cm}/{cm}", 1),
            ("F", "A1/C1", .01)], [
                ("A", None, 2), ("B", 'CONVERT(A1,"m","cm")', 200),
                ("C", 'CONVERT(A1,"m","m")', 2),
                ("D", 'CONVERT(A1,"m","in")', float(Fraction(2)/Fraction(127,5000)))])
        self.assertTrue(guard["ok"], guard)
        rows = self.records(audit, "arithmetic")
        self.assertEqual([rows["Results", c]["unit_scale_si"] for c in ("A1", "C1", "E1", "F1")],
                         ["1", "1/100", "1", "100"])
        self.assertEqual(rows["Results", "F1"]["si_dimension"], [0,0,0,0,0])

    def test_equal_dimensions_with_different_scales_fail_and_taint_dependents(self):
        audit, guard = self.evaluate([
            ("A", "'Ratings 2026'!B1", 200), ("B", "'Ratings 2026'!C1", 2),
            ("C", "A1+B1", 202), ("D", "C1*2", 404)], [
                ("A", None, 2), ("B", 'CONVERT(A1,"m","cm")', 200),
                ("C", 'CONVERT(A1,"m","m")', 2)])
        rows = self.records(audit, "arithmetic")
        self.assertEqual(rows["Results", "C1"]["status"], "unit_scale_mismatch")
        self.assertEqual((rows["Results", "C1"]["left_scale_si"],
                          rows["Results", "C1"]["right_scale_si"]), ("1/100", "1"))
        self.assertEqual(rows["Results", "D1"]["reason"], "upstream_unit_scale_mismatch")
        self.assertNotIn("expected", rows["Results", "D1"])
        self.assertFalse(guard["ok"])
        self.assertIn("arithmetic_unit_scale_mismatches=1", guard["reason"])

    def test_explicit_rescaling_makes_the_same_sum_well_typed(self):
        audit, guard = self.evaluate([
            ("A", '\'Ratings 2026\'!C1+CONVERT(\'Ratings 2026\'!B1,"cm","m")', 4),
            ("B", 'CONVERT(A1,"m","cm")', 400)], [
                ("A", None, 2), ("B", 'CONVERT(A1,"m","cm")', 200),
                ("C", 'CONVERT(A1,"m","m")', 2)])
        self.assertTrue(guard["ok"], guard)
        self.assertEqual(self.records(audit, "arithmetic")["Results", "A1"]["unit_scale_si"], "1")
        self.assertEqual(self.records(audit, "conversion")["Results", "B1"]["expected"], 400)

    def test_conversion_cache_types_and_missing_cache_are_distinct(self):
        audit, guard = self.evaluate([
            ("A", 'CONVERT(2*3,"m","cm")', 600),
            ("B", 'CONVERT(2*3,"m","cm")', 1),
            ("C", 'CONVERT(2*3,"m","cm")', None)], types={"A1":"str", "B1":"b"})
        rows = self.records(audit, "conversion")
        self.assertEqual([rows["Results", c]["status"] for c in ("A1", "B1", "C1")],
                         ["mismatch", "mismatch", "unverified"])
        self.assertEqual(rows["Results", "A1"]["cached_type"], "str")
        self.assertEqual(rows["Results", "B1"]["cached_type"], "b")
        self.assertEqual(rows["Results", "C1"]["expected"], 600)
        self.assertFalse(guard["ok"])

    def test_precision_as_displayed_does_not_certify_conversion_caches(self):
        audit, guard = self.evaluate([("A", 'CONVERT(2*3,"m","cm")', 999)], full_precision=False)
        self.assertEqual(audit["conversion_audit"]["counts"]["unverified"], 1)
        self.assertEqual(audit["conversion_audit"]["counts"]["mismatch"], 0)

    def test_shared_producer_is_immutable_across_many_independent_aliases(self):
        def column(index):
            chars = []
            while index:
                index, digit = divmod(index-1, 26)
                chars.append(chr(65+digit))
            return "".join(reversed(chars))
        cells = [("A", 'CONVERT(1,"m","cm")', 100)]
        cells.extend((column(index), "A1", 100) for index in range(2, 48))
        audit, guard = self.evaluate(cells)
        self.assertTrue(guard["ok"], guard)
        self.assertEqual(audit["arithmetic_audit"]["counts"]["matched"], 46)

    def test_unsupported_and_inconsistent_unit_dependencies_remain_unverified(self):
        audit, guard = self.evaluate([
            ("A", 'CONVERT(\'Ratings 2026\'!B1,"m","cm")', 300),
            ("B", 'CONVERT(\'Ratings 2026\'!C1,"cm","m")', .03),
            ("C", 'CONVERT(2*3,"C","F")', 42.8)], [
                ("A", None, 3), ("B", "SUM(1,2)", 3),
                ("C", 'CONVERT(A1,"m","m")', 3)])
        rows = self.records(audit, "conversion")
        self.assertTrue(all(rows["Results", cell]["status"] == "unverified" for cell in ("A1","B1","C1")))
        self.assertEqual(rows["Results", "B1"]["reason"], "upstream_unit_disagrees")
        self.assertTrue(guard["ok"], guard)
