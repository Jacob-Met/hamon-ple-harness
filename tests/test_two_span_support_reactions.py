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
            ("M_B_hogging_Nm", Fraction(223, 32)),
        ):
            self.assertEqual(result[field], float(expected))
