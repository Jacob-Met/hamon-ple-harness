"""Physical section cuts and reactions of a horizontal pin-and-roller beam."""
import unittest
from unittest.mock import patch

import hamon_dual_verifier as dual


def solve(pin, roller, points=(), **options):
    spec = {"supports": [{"type": "roller", "x": roller}, {"type": "pin", "x": pin}],
            "point_loads": [{"x": x, "Fx": fx, "Fy": fy} for x, fx, fy in points], **options}
    return dual.solve_domain_problem("static_equilibrium_2d", spec)


class RollerSection(unittest.TestCase):
    def test_bending_moment_vanishes_at_either_simple_end(self):
        for pin, roller in ((0, 10), (10, 0)):
            with self.subTest(pin=pin):
                result = solve(pin, roller, [(5, 0, -100)], total_length_m=10)
                self.assertTrue(result["dual_verified"])
                self.assertEqual((result["R_Ay_N"], result["R_By_N"]), (50, 50))
                self.assertEqual(result["M_support_B_Nm"], 0)

    def test_signed_overhang_with_left_or_right_roller(self):
        # Cut the overhang outboard of B. Its 4 m of UDL and 20 N at 3 m
        # give 4*4*2 + 20*3 = 92 N m of hogging; the point on B has no arm.
        original = [(0, 2, -8), (4, 0, 6), (11, -7, -20), (8, 0, -9)]
        for reflected in (False, True):
            x = (lambda value: 12 - value) if reflected else (lambda value: value)
            for sign in (-1, 1):
                with self.subTest(reflected=reflected, sign=sign):
                    loads = [(x(u), fx * sign, fy * sign) for u, fx, fy in original]
                    r = solve(x(3), x(8), loads, total_length_m=12, udl_N_per_m=4 * sign)
                    self.assertTrue(r["dual_verified"])
                    self.assertAlmostEqual(r["R_Ay_N"], sign * 15.2)
                    self.assertAlmostEqual(r["R_By_N"], sign * 63.8)
                    self.assertEqual(r["R_Ax_N"], sign * 5)
                    self.assertEqual(r["M_support_B_Nm"], sign * 92)

    def test_implicit_span_translates_with_its_supports(self):
        # No overhang length is specified. A 6 m UDL at 10 N/m acts at
        # the midpoint between the supports even at survey coordinates.
        for origin in (-250000.0, 17.0, 500000.0):
            for reverse in (False, True):
                pin, roller = (origin + 6, origin) if reverse else (origin, origin + 6)
                with self.subTest(origin=origin, reverse=reverse):
                    r = solve(pin, roller, udl_N_per_m=10)
                    self.assertEqual((r["R_Ay_N"], r["R_By_N"], r["M_support_B_Nm"]), (30, 30, 0))

    def test_lightly_loaded_pin_and_fallback(self):
        for sympy in (dual.HAS_SYMPY, False):
            with self.subTest(sympy=sympy), patch.object(dual, "HAS_SYMPY", sympy):
                r = solve(10, 0, [(0, 0, -1e16), (10, 0, -1)], total_length_m=10)
                self.assertEqual((r["R_Ay_N"], r["R_By_N"], r["M_support_B_Nm"]), (1, 1e16, 0))

    def test_loads_respect_explicit_extent_and_implicit_overhangs(self):
        for outside in (-2, 12):
            with self.subTest(outside=outside), self.assertRaises(ValueError):
                solve(0, 10, [(outside, 0, -100)], total_length_m=10)
        r = solve(0, 10, [(0, 0, -10), (10, 0, -25)], total_length_m=10)
        self.assertEqual((r["R_Ay_N"], r["R_By_N"], r["M_support_B_Nm"]), (10, 25, 0))
        for pin, roller, overhang in ((0, 10, 12), (10, 0, -2)):
            with self.subTest(pin=pin):
                r = solve(pin, roller, [(overhang, 0, -100)])
                self.assertEqual((r["R_Ay_N"], r["R_By_N"], r["M_support_B_Nm"]), (-20, 120, 200))

    def test_unresisted_or_inconsistent_support_layout_is_refused(self):
        for pin, roller in ((2, 2), (-1, 4), (1, 7)):
            with self.subTest(pin=pin, roller=roller), self.assertRaises(ValueError):
                solve(pin, roller, total_length_m=6)
