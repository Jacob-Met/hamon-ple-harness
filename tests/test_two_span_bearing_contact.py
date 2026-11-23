"""Signed support moments and unilateral contact on a constant-EI two-span beam."""
import math
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

    def test_signed_moment_fields_and_bilateral_hold_down(self):
        downward = beam(1000, 1000)
        upward = beam(-1000, -1000)
        self.assertEqual(downward["M_B_Nm"], -2000.)
        self.assertEqual(upward["M_B_Nm"], 2000.)
        self.assertEqual(upward["M_B_sagging_Nm"], 2000.)
        self.assertEqual(downward["M_max_sagging_Nm"], 1125.)
        self.assertEqual(upward["M_max_hogging_Nm"], 1125.)
        self.assertEqual(downward["M_max_Nm"], upward["M_max_Nm"])
        loaded = beam(1000, 0, {"C": "bilateral"})
        self.assertEqual(loaded["support_states"]["C"], "hold_down")
        self.assertEqual(loaded["support_displacements_m"]["C"], 0.)

    def test_end_gap_from_simple_span_rotation_and_mirrored_load(self):
        # The overhang is straight and rotates with the unloaded end of the
        # simply supported loaded span: wL^3/(24EI).
        expected_gap = 1000. * 4**4 / (24 * 200e9 * .001)
        for q, end, other in [((1000, 0), "C", "A"), ((0, 1000), "A", "C")]:
            r = beam(*q, {key: "compression_only" for key in "ABC"})
            self.assertEqual(r["support_states"][end], "lifted")
            self.assertEqual(r["support_states"][other], "bearing")
            self.assertAlmostEqual(r["support_displacements_m"][end], expected_gap, places=16)
            self.assertEqual(r["M_B_Nm"], 0.)
            self.assertEqual(r["M_max_sagging_Nm"], 2000.)

    def test_middle_bearing_releases_under_uniform_upward_load(self):
        r = beam(-1000, -1000, {"B": "compression_only"})
        self.assertEqual([r[f"R_{name}_N"] for name in "ABC"], [-4000., 0., -4000.])
        self.assertEqual(r["support_states"], dict(A="hold_down", B="lifted", C="hold_down"))
        self.assertEqual(r["M_B_Nm"], -8000.)
        self.assertAlmostEqual(r["support_displacements_m"]["B"], 5*1000*8**4/(384*200e9*.001))

    def test_contact_boundary_and_two_immediate_load_neighbors(self):
        for q, expected in [(math.nextafter(7., 0.), "bearing"), (7., "touching"),
                            (math.nextafter(7., math.inf), "lifted")]:
            r = beam(q, 1., {"C": "compression_only"})
            self.assertEqual(r["support_states"]["C"], expected)
            self.assertEqual(r["R_C_N"] * r["support_displacements_m"]["C"], 0.)
            self.assertGreaterEqual(r["R_C_N"], 0.)
            self.assertGreaterEqual(r["support_displacements_m"]["C"], 0.)

    def test_unrestrained_rigid_modes_and_impossible_loading(self):
        bearings = {name: "compression_only" for name in "ABC"}
        with self.assertRaisesRegex(ValueError, "no restrained contact equilibrium"):
            beam(-1000, -1000, bearings)
        with self.assertRaisesRegex(ValueError, "nonunique"):
            beam(0, 0, bearings)
        with self.assertRaisesRegex(ValueError, "nonunique"):
            beam(0, 0, dict(B="compression_only", C="compression_only"))
        # A central bilateral support and two just-touching bearings admit no tilt.
        r = beam(0, 0, dict(A="compression_only", C="compression_only"))
        self.assertEqual(r["support_states"], dict(A="touching", B="bilateral", C="touching"))

    def test_support_contract_is_explicit(self):
        for invalid in ({"D": "compression_only"}, {"C": "pinned_maybe"}, ["A"]):
            with self.assertRaisesRegex(ValueError, "support_conditions"):
                beam(1000, 1000, invalid)

    def test_nonzero_contact_values_below_binary64_range(self):
        # A released overhang rotates by qL^3/(24EI) at B. Its C gap is
        # positive, but qL^4/(24EI) cannot be published as a nonzero float.
        base = dict(span1=2.**-15, span2=2.**-15, udl1=2.**-1000, udl2=0.,
                    E=2.**30, I=1., support_conditions={"C": "compression_only"})
        # R_A = 3qL/8 is also nonzero and rounds to zero for this tiny load.
        other = dict(span1=1., span2=1., udl1=math.ulp(0.), udl2=math.ulp(0.),
                     E=1., I=1., support_conditions={"A": "compression_only"})
        for spec in (base, other):
            with self.assertRaisesRegex(ValueError, "reporting range"):
                run_dual_verify_spec(dict(mode="domain", kind="continuous_beam_2span", spec=spec))
