"""AISC H1 treats required axial and flexural strengths as magnitudes."""
import unittest
from fractions import Fraction

from hamon_dual_verifier import run_dual_verify_spec


def solve(pu, moment_x, moment_y, **changes):
    spec = dict(Pu=pu, Pc=100, Mux=moment_x, Mcx=100, Muy=moment_y, Mcy=100)
    return run_dual_verify_spec(dict(mode="domain", kind="column_interaction_aisc",
                                    spec=dict(spec, **changes)))


class ColumnSignedBending(unittest.TestCase):
    def test_signed_axes_cannot_cancel_the_governing_corner(self):
        for pu in (10, 30):
            axial = Fraction(pu, 100)
            for sx in (1,-1):
                for sy in (1,-1):
                    mx, my = sx*55, sy*55
                    # Orthogonal plane bending of a symmetric section: choosing the
                    # opposite extreme corner reverses either contribution independently.
                    governing = max(x*mx+y*my for x in (-1,1) for y in (-1,1))
                    bending = Fraction(governing,100)
                    ratio = axial+Fraction(8,9)*bending if pu==30 else axial/2+bending
                    r=solve(pu,mx,my)
                    self.assertTrue(r["dual_verified"])
                    self.assertAlmostEqual(r["interaction_ratio"], float(ratio))
                    self.assertEqual(r["passes"], ratio<=1)

    def test_axial_action_reversal_uses_the_capacity_supplied_for_that_action(self):
        for pu in (0, 10, 20, 30):
            forward, reverse = solve(pu,24,-18), solve(-pu,24,-18)
            for field in ("axial_ratio","governing_equation","interaction_ratio","passes"):
                self.assertEqual(forward[field],reverse[field])

    def test_available_strengths_are_positive(self):
        for updates in (dict(Pc=-100),dict(Mcx=-100),dict(Mcy=-100),
                        dict(Pc=0),dict(Mcx=0),dict(Mcy=0)):
            with self.assertRaises(ValueError):
                solve(30,55,-55,**updates)
