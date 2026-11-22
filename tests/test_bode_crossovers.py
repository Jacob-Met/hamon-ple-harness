"""Analytic frequency-margin controls for continuous real rational transfer functions."""
from decimal import Decimal, localcontext
import math
import json
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def margins(num, den):
    result = run_router_conservation_cad_verifier({
        "mode": "control", "spec": {"mode": "transfer_function", "num": num, "den": den}})
    return json.loads(json.dumps(result, allow_nan=False))


class BodeCrossoverTests(unittest.TestCase):
    def test_unit_scaling_and_exact_tangent(self):
        for k in (1e-9, 1, 1e9):
            result = margins([k], [1, 0])
            self.assertAlmostEqual(result["w_gc_rad_s"]/k, 1)
            self.assertEqual(result["phase_margin_deg"], 90)
        for a in (Decimal("0.001"), Decimal("1"), Decimal("100000")):
            # |D|²-|N|² = (w²-7a²)² touches zero without changing sign.
            result = margins([str(24*a*a)], [1, str(6*a), str(25*a*a)])
            self.assertEqual(len(result["gain_crossovers"]), 1)
            self.assertAlmostEqual(result["w_gc_rad_s"]/float(a), math.sqrt(7))
            self.assertAlmostEqual(result["phase_margin_deg"], 180-math.degrees(math.atan2(math.sqrt(7),3)))

    def test_narrow_resonance_and_smallest_absolute_margin(self):
        r = margins([0.0005], [1, 0.0001, 1])
        crossings = r["gain_crossovers"]
        self.assertEqual(len(crossings), 2)
        self.assertLess(crossings[0]["w_rad_s"], 1)
        self.assertGreater(crossings[1]["w_rad_s"], 1)
        for c in crossings:
            w = c["w_rad_s"]
            self.assertAlmostEqual(math.hypot(1-w*w, 0.0001*w), 0.0005, places=12)
        self.assertEqual(r["w_gc_rad_s"], crossings[1]["w_rad_s"])
        r = margins([8], [1, 0, 10, 1, 9, 0])
        self.assertEqual([c["w_rad_s"] for c in r["phase_crossovers"]], [1, 3])
        self.assertEqual(r["w_pc_rad_s"], 3)
        self.assertAlmostEqual(r["gain_margin_db"], 20*math.log10(9/8))

    def test_wrapped_phase_and_undefined_frequencies(self):
        r = margins([1], [math.comb(8,k) for k in range(9)])
        roots = [c["w_rad_s"] for c in r["phase_crossovers"]]
        self.assertEqual(len(roots), 2)
        for w, degrees in zip(roots, (180, 540)):
            self.assertAlmostEqual(8*math.degrees(math.atan(w)), degrees)
        r = margins([1], [1,0,1])
        self.assertEqual(r["negative_real_intervals_rad_s"], [[1,None]])
        self.assertEqual(r["imaginary_axis_poles_rad_s"], [1])
        self.assertEqual(r["phase_crossovers"], [])
        self.assertAlmostEqual(r["w_gc_rad_s"], math.sqrt(2))
        r = margins([1,0,1], [1,0,1])
        self.assertEqual(r["unity_gain_intervals_rad_s"], [[0,1],[1,None]])
        self.assertEqual(r["removable_holes_rad_s"], [1])
        self.assertEqual(r["imaginary_axis_poles_rad_s"], [])
        self.assertIsNone(r["w_gc_rad_s"])
        r = margins([1,0,1], [1,0,2,0,1])
        self.assertEqual(r["imaginary_axis_poles_rad_s"], [1])
        self.assertEqual(r["removable_holes_rad_s"], [])

    def test_dc_pole_and_removable_origin_use_finite_json(self):
        for num, den, expected, status in (([2], [1,0], None, "pole_at_zero"),
                                           ([2,0], [1,0], 2, "finite_limit"),
                                           ([2,0,0], [1,0], 0, "finite_limit"),
                                           ([1e-16], [1,1e-16], 1, "finite"),
                                           ([1], [1,1e-310], None, "outside_float_range")):
            r = margins(num, den)
            self.assertEqual((r["dc_gain"], r["dc_gain_status"]), (expected, status))

    def test_exact_positive_axis_unit_gain_uses_consistent_phase_branch(self):
        # L = 1 + (s^4-2)/(s+1) equals +1 exactly at an irrational frequency.
        r = margins([1,0,0,1,-1], [1,1])
        crossing = min(r["gain_crossovers"], key=lambda v: abs(v["w_rad_s"] - 2**0.25))
        self.assertAlmostEqual(crossing["w_rad_s"], 2**0.25)
        self.assertEqual(crossing["phase_margin_deg"], -180)

    def test_exact_negative_axis_classification_near_a_zero(self):
        with localcontext() as ctx:
            ctx.prec = 65
            for sign in (-1, 1):
                c = Decimal(2).sqrt() + sign*Decimal("1e-25")
                r = margins([-1,0,0,1,2,str(c)], [1,0,3])
                if sign > 0:
                    self.assertEqual(r["phase_crossovers"], [])
                else:
                    self.assertAlmostEqual(r["w_pc_rad_s"], 2**0.25)
                    self.assertAlmostEqual(r["gain_margin_db"], 500 + 20*math.log10(3-math.sqrt(2)), places=9)
