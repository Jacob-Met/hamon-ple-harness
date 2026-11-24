"""Physical contradictions survive unlabelled terms and absent numeric caches."""
import unittest

import test_xlsx_unit_graph as fixtures


class UnitConstraintTests(unittest.TestCase):
    def evaluate(self, cells, ratings=(), types=None):
        return fixtures.UnitGraphTests().evaluate(cells, ratings, types)

    @staticmethod
    def records(audit):
        return {(row['sheet'], row['cell']): row
                for kind in ('arithmetic', 'conversion')
                for row in audit[kind + '_audit']['examples']}

    def test_unlabelled_addends_cannot_erase_a_later_dimensional_conflict(self):
        length = 'CONVERT(1,"m","cm")'
        time = 'CONVERT(1,"hr","sec")'
        variants = [
            (length + '+0', 100, 'A1+' + time, 3700),
            (length + '+1', 101, 'A1+' + time, 3701),
            ('1+' + length, 101, 'A1+' + time, 3701),
            (length + "+'Ratings 2026'!A1", 101, 'A1+' + time, 3701),
            ('(' + length + '+1)*2', 202, 'A1+' + time, 3802),
            ('(' + length + '+1)/2', 50.5, 'A1+' + time, 3650.5),
            (length + '+1', 101, '-A1+' + time, 3499),
        ]
        for expression, first, downstream, second in variants:
            with self.subTest(expression=expression, downstream=downstream):
                audit, guard = self.evaluate([
                    ('A', expression, first), ('B', downstream, second),
                    ('C', 'B1*2', second*2)], [('A', None, 1)])
                rows = self.records(audit)
                self.assertEqual(rows['Results', 'B1']['status'], 'dimension_mismatch')
                self.assertEqual(rows['Results', 'C1']['reason'], 'upstream_dimension_mismatch')
                self.assertFalse(guard['ok'], guard)

    def test_unlabelled_addends_cannot_erase_a_later_scale_conflict(self):
        audit, guard = self.evaluate([
            ('A', 'CONVERT(1,"m","cm")+1', 101),
            ('B', 'A1+CONVERT(1,"m","m")', 102)])
        self.assertEqual(self.records(audit)['Results', 'B1']['status'], 'unit_scale_mismatch')
        self.assertFalse(guard['ok'], guard)

    def test_valid_additions_report_a_condition_without_inventing_input_units(self):
        audit, guard = self.evaluate([
            ('A', 'CONVERT(1,"m","cm")+1', 101),
            ('B', 'A1+CONVERT(2,"m","cm")', 301),
            ('C', 'CONVERT(B1,"cm","m")', 3.01)])
        rows = self.records(audit)
        self.assertTrue(guard['ok'], guard)
        self.assertEqual(rows['Results', 'A1']['status'], 'matched')
        self.assertNotIn('si_dimension', rows['Results', 'A1'])
        self.assertEqual(rows['Results', 'A1']['conditional_unit_scale_si'], '1/100')
        self.assertEqual(rows['Results', 'C1']['expected'], 3.01)

    def test_unknown_multiplicative_unit_does_not_become_dimensionless(self):
        audit, guard = self.evaluate([
            ('A', 'CONVERT(1,"m","cm")+1', 101),
            ('B', "A1*'Ratings 2026'!A1", 202),
            ('C', 'B1+CONVERT(1,"hr","sec")', 3802)], [('A', None, 2)])
        rows = self.records(audit)
        self.assertTrue(guard['ok'], guard)
        self.assertNotIn('si_dimension', rows['Results', 'B1'])
        self.assertNotIn('conditional_si_dimension', rows['Results', 'B1'])

    def test_explicit_conversion_conflict_is_independent_of_cache_type_and_presence(self):
        for value, types in [(None, None), (1, {'A1':'str'}), (1, {'A1':'b'})]:
            with self.subTest(value=value, types=types):
                audit, guard = self.evaluate([('A', 'CONVERT(1,"m","kg")', value)], types=types)
                self.assertEqual(audit['conversion_audit']['counts']['dimension_mismatch'], 1)
                self.assertFalse(guard['ok'], guard)

    def test_explicit_conversion_conflict_precedes_missing_or_cyclic_numeric_input(self):
        cases = [
            [('A', 'CONVERT(\'Ratings 2026\'!X1,"m","kg")', None)],
            [('A', 'CONVERT(B1,"m","kg")', None), ('B', 'A1*2', None)],
            [('A', 'CONVERT(\'Absent sheet\'!A1,"m","kg")', None)],
            [('A', 'CONVERT(1E999,"m","kg")', None)],
        ]
        for cells in cases:
            with self.subTest(cells=cells):
                audit, guard = self.evaluate(cells)
                self.assertEqual(self.records(audit)['Results', 'A1']['status'], 'dimension_mismatch')
                self.assertFalse(guard['ok'], guard)
