"""Signed support moments and unilateral contact on a constant-EI two-span beam."""
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def beam(q1, q2, supports=None):
    spec = dict(span1=4, span2=4, udl1=q1, udl2=q2, E=200e9, I=.001)
    if supports is not None:
        spec["support_conditions"] = supports
    return run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec=spec))


class TwoSpanContact(unittest.TestCase):
    def test_upward_load_does_not_report_hogging_at_middle_support(self):
        # Full uplift simply reverses the moment diagram of the downward load.
        r = beam(-1000, -1000)
        self.assertTrue(r["dual_verified"])
        self.assertEqual(r["M_B_hogging_Nm"], 0.)

    def test_right_end_bearing_lifts_under_left_span_live_load(self):
        # Bilateral pin constraints require a downward 250 N hold-down at C.
        self.assertEqual(beam(1000, 0)["R_C_N"], -250.)
        r = beam(1000, 0, {"C": "compression_only"})
        self.assertTrue(r["dual_verified"])
        self.assertEqual(r["R_C_N"], 0.)
        self.assertEqual((r["R_A_N"], r["R_B_N"]), (2000., 2000.))
