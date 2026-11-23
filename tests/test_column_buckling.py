"""Requested-member checks for the Johnson/Euler flexural column model."""

import math
import unittest

from hamon_column_buckling import column_buckling


class ColumnBucklingTests(unittest.TestCase):
    def setUp(self):
        self.member = dict(E=200e9, sigma_y=250e6, L=8, A=.005)
        self.strong = .05 * .1**3 / 12
        self.weak = .1 * .05**3 / 12

    def test_different_end_restraints(self):
        axes = {"x": {"I": self.strong, "K": 2}, "y": {"I": self.weak, "K": .5}}
        result = column_buckling(dict(self.member, principal_axes=axes))
        self.assertTrue(result["verified"])
        self.assertEqual(result["governing_axes"], ["x"])
        for name, axis in axes.items():
            expected = math.pi**2 * self.member["E"] * axis["I"] / (axis["K"] * self.member["L"])**2
            actual = result["principal_axis_results"][name]
            self.assertAlmostEqual(actual["P_cr_N"] / expected, 1)
            self.assertEqual(actual["regime"], "euler_elastic")

    def test_equal_and_near_equal_axes(self):
        axes = {"x": {"I": 4e-6, "K": 2}, "y": {"I": 1e-6, "K": 1}}
        self.assertEqual(column_buckling(dict(self.member, principal_axes=axes))["governing_axes"], ["x", "y"])
        axes["x"]["I"] = math.nextafter(axes["x"]["I"], math.inf)
        self.assertEqual(column_buckling(dict(self.member, principal_axes=axes))["governing_axes"], ["y"])

    def test_johnson_force_and_geometric_scaling(self):
        r = dict(self.member, L=.5, I=self.strong)
        original = column_buckling(r)
        yield_load = r["sigma_y"] * r["A"]
        euler_load = math.pi**2 * r["E"] * r["I"] / r["L"]**2
        self.assertEqual(original["regime"], "johnson_inelastic")
        self.assertAlmostEqual(original["P_cr_N"] / (yield_load - yield_load**2 / (4 * euler_load)), 1)
        enlarged = column_buckling(dict(r, L=10*r["L"], A=100*r["A"], I=10000*r["I"]))
        self.assertAlmostEqual(enlarged["P_cr_N"] / original["P_cr_N"], 100)
        self.assertAlmostEqual(enlarged["slenderness_ratio"], original["slenderness_ratio"])

    def test_invalid_member_withholds_load(self):
        baseline = dict(self.member, I=self.strong)
        for name in ("E", "sigma_y", "A", "I", "L", "K"):
            for value in (-1, 0, True, math.inf, math.nan):
                with self.subTest(name=name, value=value):
                    result = column_buckling(dict(baseline, **{name: value}))
                    self.assertFalse(result["verified"])
                    self.assertNotIn("P_cr_N", result)
                    self.assertIn("error", result)

    def test_large_radius_ratio_has_nonzero_slenderness(self):
        result = column_buckling(dict(self.member, A=1e-157, I=1e159, L=1e156))
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["slenderness_ratio"], .01)

    def test_incomplete_or_ambiguous_axes(self):
        axis = {"I": self.weak, "K": 1}
        for spec in (dict(principal_axes={"x": axis}), dict(principal_axes={"x": axis, "y": {}}),
                     dict(I=self.strong, principal_axes={"x": axis, "y": axis})):
            with self.subTest(spec=spec):
                result = column_buckling(dict(self.member, **spec))
                self.assertFalse(result["verified"])
                self.assertNotIn("P_cr_N", result)
