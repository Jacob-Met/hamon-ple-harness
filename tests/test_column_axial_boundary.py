"""Source-accurate AISC H1 axial branch decisions on the registered route."""
from fractions import Fraction
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def interaction(pu, pc):
    return run_dual_verify_spec({
        "mode": "domain",
        "kind": "column_interaction_aisc",
        "spec": {"Pu": pu, "Pc": pc, "Mux": "4*kip_ft", "Mcx": "10*kip_ft",
                 "Muy": 3, "Mcy": 12},
    })


class ColumnAxialBoundary(unittest.TestCase):
    def check_ratio(self, pu, pc, ratio):
        result = interaction(pu, pc)
        bending = Fraction(4, 10) + Fraction(3, 12)
        if ratio >= Fraction(1, 5):
            equation = "H1-1a"
            utilization = ratio + Fraction(8, 9) * bending
        else:
            equation = "H1-1b"
            utilization = ratio / 2 + bending
        self.assertTrue(result["dual_verified"])
        self.assertEqual(result["governing_equation"], equation)
        self.assertEqual(result["axial_ratio"], float(ratio))
        self.assertAlmostEqual(result["interaction_ratio"], float(utilization), places=15)

    def test_equal_ratio_is_independent_of_force_unit_and_expression_order(self):
        for pu, pc in (
            ("3*kip", "15*kip"), ("1*lbf", "5*lbf"), ("3*kip", "15000*lbf"),
            ("1*kip/3", "5*kip/3"), ("sqrt(9)*kip", "max(10,15)*kip"),
            ("(1e18+3)-1e18", 15), (3, 15), ("3*ksi*in2", "15*kip"),
        ):
            with self.subTest(pu=pu, pc=pc):
                self.check_ratio(pu, pc, Fraction(1, 5))

    def test_each_side_retains_source_precision_without_an_equality_tolerance(self):
        for amount in ("2.99999999999999999999", "3.00000000000000000001",
                       str(math.nextafter(3., 0.)), str(math.nextafter(3., 4.))):
            with self.subTest(amount=amount):
                self.check_ratio(f"{amount}*kip", "15*kip", Fraction(amount) / 15)

    def test_ordinary_low_and_high_utilizations_stay_on_their_existing_branches(self):
        for pu, ratio in (("1*kip", Fraction(1, 10)), ("4*kip", Fraction(2, 5))):
            with self.subTest(pu=pu):
                self.check_ratio(pu, "10*kip", ratio)

    def test_accepts_scalar_functions_and_preserves_field_dimensions(self):
        self.check_ratio("sin(pi/2)*kN", "10*kN", Fraction(1, 10))
        with self.assertRaisesRegex(ValueError, "Pu"):
            interaction("3*kPa", "15*kN")
