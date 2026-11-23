"""End-force checks for the prismatic two-span continuous-beam solver."""
import unittest
from fractions import Fraction

import hamon_dual_verifier as dv


def solve(q1, q2, span1=4, span2=4):
    return dv.run_dual_verify_spec({
        "mode": "domain",
        "kind": "continuous_beam_2span",
        "spec": {
            "span1": f"{span1}*m", "span2": f"{span2}*m",
            "udl1": f"{q1}*N/m", "udl2": f"{q2}*N/m",
        },
    })


class TwoSpanSupportReactions(unittest.TestCase):
    def test_cancelling_end_reactions_at_three_load_scales(self):
        for q in (1, 1000, 1000000):
            for reverse in (False, True):
                with self.subTest(scale=q, reverse=reverse):
                    loads = (7*q, q) if reverse else (q, 7*q)
                    result = solve(*loads)
                    expected = [12*q, 20*q, 0] if reverse else [0, 20*q, 12*q]
                    self.assertEqual([result[k] for k in ("R_A_N", "R_B_N", "R_C_N")], expected)
                    self.assertTrue(result["dual_verified"])

    def test_small_signed_reaction_is_preserved(self):
        for delta in (2.0**-24, -2.0**-24):
            with self.subTest(delta=delta):
                result = solve(1, 7 + delta)
                self.assertEqual(result["R_A_N"], -delta / 4)

    def test_opposing_loads_with_zero_middle_reaction_and_support_moment(self):
        result = solve(2, -2)
        self.assertEqual([result[k] for k in ("R_A_N", "R_B_N", "R_C_N")], [4, 0, -4])
        self.assertEqual(result["M_B_hogging_Nm"], 0.0)
        self.assertEqual(result["M_max_Nm"], 4.0)

    def test_unequal_spans_and_opposing_loads(self):
        result = solve(2, -4, span1=3, span2=5)
        for field, expected in (
            ("R_A_N", Fraction(511, 96)),
            ("R_B_N", Fraction(-643, 60)),
            ("R_C_N", Fraction(-1377, 160)),
            ("M_B_abs_Nm", Fraction(223, 32)),
            ("M_B_sagging_Nm", Fraction(223, 32)),
            ("M_B_hogging_Nm", 0),
        ):
            self.assertEqual(result[field], float(expected))

    def test_same_numbers_with_incompatible_units(self):
        fields = {"span1": "4*m", "span2": "4*m", "udl1": "1*N/m",
                  "udl2": "7*N/m", "E": "200*GPa", "I": "1e-4*m**4"}
        for name, value in fields.items():
            with self.subTest(field=name):
                spec = dict(fields, **{name: f"({value})*s/m"})
                with self.assertRaisesRegex(ValueError, name + " must have units"):
                    dv.solve_continuous_beam_2span_dual(spec)
        for name in ("udl1", "udl2"):
            with self.subTest(zero=name):
                with self.assertRaisesRegex(ValueError, name + " must have units"):
                    dv.solve_continuous_beam_2span_dual(dict(fields, **{name: "0*kg"}))

    def test_rescaled_units_and_coherent_si_numbers(self):
        numeric = {"span1": 4, "span2": 5, "udl1": -2, "udl2": 7,
                   "E": 200e9, "I": 1e-4}
        rescaled = {"span1": "4000*mm", "span2": "500*cm", "udl1": "-0.002*kN/m",
                    "udl2": "7*kg/s**2", "E": "200000*MPa", "I": "10000*cm**4"}
        baseline = dv.solve_continuous_beam_2span_dual(numeric)
        for name, value in rescaled.items():
            with self.subTest(field=name):
                result = dv.solve_continuous_beam_2span_dual(dict(numeric, **{name: value}))
                self.assertEqual(result, baseline)
        self.assertEqual(dv.solve_continuous_beam_2span_dual(rescaled), baseline)
