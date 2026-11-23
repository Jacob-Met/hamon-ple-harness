"""Unit contracts for the registered AISC column checks."""
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


BUCKLING = "column_buckling_aisc_e3"
INTERACTION = "column_interaction_aisc"
E3 = {"L": 3, "r": 0.05, "E": 200e9, "Fy": 250e6, "Ag": 0.01}
H1 = {"Pu": 100000, "Pc": 500000, "Mux": 10000, "Mcx": 50000,
      "Muy": 5000, "Mcy": 25000}


def column(kind, spec):
    return run_dual_verify_spec({"mode": "domain", "kind": kind, "spec": spec})


class ColumnDimensions(unittest.TestCase):
    def test_incompatible_units_are_rejected_even_if_the_si_magnitude_is_unchanged(self):
        cases = ((BUCKLING, E3, {"L": "3*s", "r": "0.05*kg", "E": "200e9*N",
                                 "Fy": "250e6*N/m", "Ag": "0.01*m"}),
                 (INTERACTION, H1, {"Pu": "100000*Pa", "Pc": "500000*J", "Mux": "10000*N",
                                    "Mcx": "50000*N", "Muy": "5000*N", "Mcy": "25000*N"}))
        for kind, base, replacements in cases:
            for field, invalid in replacements.items():
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    column(kind, {**base, field: invalid})

    def test_cancelled_dimensions_and_zero_wrong_units_cannot_supply_a_force(self):
        for invalid in ("100000*m/m", "100000*rad", "0*m"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Pu"):
                column(INTERACTION, {**H1, "Pu": invalid})

    def test_si_numbers_unit_expressions_and_bare_numeric_expressions_agree(self):
        mixed = {"L": "3000*mm", "r": "5*cm", "E": "200*GPa",
                 "Fy": "250*MPa", "Ag": "100*cm2"}
        for length, explicit in ((3, "3000*mm"), (12, "1200*cm")):
            expected = column(BUCKLING, {**E3, "L": length})
            self.assertTrue(expected["dual_verified"])
            self.assertEqual(expected, column(BUCKLING, {**mixed, "L": explicit}))
            self.assertEqual(expected, column(BUCKLING, {**E3, "L": str(length), "E": "2 * 100e9"}))
            slenderness = length / E3["r"]
            self.assertAlmostEqual(expected["Fe_Pa"] / E3["E"], math.pi**2 / slenderness**2)
        for pu, equation in ((50, "H1-1b"), (100, "H1-1a")):
            expected = column(INTERACTION, {**H1, "Pu": pu * 1000})
            explicit = {"Pu": f"{pu}*kN", "Pc": "500*kN", "Mux": "10*kN*m",
                        "Mcx": "50*kN*m", "Muy": "5*kN*m", "Mcy": "25*kN*m"}
            self.assertEqual(expected, column(INTERACTION, explicit))
            self.assertEqual(expected["governing_equation"], equation)
            self.assertTrue(expected["passes"])
