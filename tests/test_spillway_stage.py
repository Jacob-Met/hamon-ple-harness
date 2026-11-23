"""Public free-overflow discharge, stage and freeboard contact regressions."""
import math
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def spillway(q, crest=1000, top=1002, minimum=1, **extra):
    data = dict(Q_design_m3_s=q, Cd_si=1, weir_length_m=1, crest_el_m=crest,
                dam_top_el_m=top, min_freeboard_m=minimum)
    data.update(extra)
    return run_dual_verify_spec(dict(mode="domain", kind="spillway_weir", spec=data))


class SpillwayStage(unittest.TestCase):
    def test_one_metre_overtopping_is_invariant_under_elevation_datum(self):
        for crest in (0, -(2**54), 2**54, 2**54+1):
            with self.subTest(crest=crest):
                result = spillway(1, crest=crest, top=crest, minimum=0)
                self.assertTrue(result["dual_verified"], result)
                self.assertFalse(result["passes_freeboard"], result)
                self.assertEqual(result["freeboard_m"], -1)

    def test_adjacent_design_discharges_at_survey_freeboard_contact(self):
        for flow, ok in ((math.nextafter(1., 0.), True), (1., True),
                         (math.nextafter(1., math.inf), False)):
            with self.subTest(flow=flow):
                result = spillway(flow)
                self.assertEqual(result["passes_freeboard"], ok, result)
                self.assertEqual(result["Q_capacity_with_freeboard_m3_s"], 1)

    def test_no_overflow_gives_a_stage_bound(self):
        for top, minimum, guarantee in ((1002, 1, True), (1002, 2, True),
                                        (1002, 3, None)):
            with self.subTest(minimum=minimum):
                result = spillway(0, top=top, minimum=minimum)
                self.assertTrue(result["dual_verified"], result)
                self.assertIsNone(result["water_surface_el_m"])
                self.assertIsNone(result["freeboard_m"])
                self.assertEqual(result["head_required_m"], 0)
                self.assertEqual(result["water_surface_el_upper_bound_m"], 1000)
                self.assertEqual(result["freeboard_lower_bound_m"], top-1000)
                self.assertIs(result["passes_freeboard"], guarantee)

    def test_positive_heads_outside_legacy_bracket_are_unique(self):
        for head, flow in ((2.**-16, 2.**-24), (64, 512)):
            with self.subTest(head=head):
                result = spillway(flow, crest=-5, top=100, minimum=2)
                self.assertEqual(result["head_required_m"], head)
                self.assertEqual(result["freeboard_m"], 105-head)
                self.assertTrue(result["passes_freeboard"], result)

    def test_native_integer_elevations_keep_sub_float_clearance(self):
        datum = 2**54
        for top, ok in ((datum, False), (datum+1, True), (datum+2, True)):
            with self.subTest(top=top):
                result = spillway(1, crest=datum, top=top, minimum=0)
                self.assertEqual(result["passes_freeboard"], ok, result)
                self.assertEqual(result["freeboard_m"], top-datum-1)

    def test_zero_flow_stage_observation_selects_one_family_member(self):
        for stage, expected in ((1000, False), (999, True), (998, True)):
            result = spillway(0, top=1001, minimum=2, water_surface_el_m=stage)
            self.assertTrue(result["dual_verified"])
            self.assertEqual(result["water_surface_el_m"], stage)
            self.assertEqual(result["freeboard_m"], 1001-stage)
            self.assertIs(result["passes_freeboard"], expected)
        with self.assertRaises(ValueError):
            spillway(0, water_surface_el_m=1001)

    def test_positive_stage_constraints_and_signed_elevation(self):
        flow = math.nextafter(1., math.inf)
        self.assertFalse(spillway(flow, water_surface_el_m=1001.)["passes_freeboard"])
        with self.assertRaises(ValueError):
            spillway(flow, water_surface_el_m=1001)
        result = spillway(1, crest=-1, top=0, minimum=0, water_surface_el_m=0)
        self.assertEqual(result["water_surface_el_m"], 0)
        self.assertTrue(result["passes_freeboard"])

    def test_resolved_scalar_ranges_and_constitutive_domain(self):
        for changes in (dict(Q_design_m3_s=-1), dict(Cd_si=0), dict(weir_length_m=-1),
                        dict(min_freeboard_m=-1), dict(crest_el_m=math.inf),
                        dict(tailwater_el_m=1001)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                spillway(1, **changes)
        # A finite positive head is preserved even if the coefficient and width
        # have a product outside the finite binary64 range.
        result = spillway(2.**900, crest=0, top=2.**-200, minimum=0,
                          Cd_si=2.**600, weir_length_m=2.**600)
        self.assertEqual(result["head_required_m"], 2.**-200)
        with self.assertRaises(ValueError):
            spillway(2.**-1074, crest=0, top=1, minimum=0,
                     Cd_si=2.**1023, weir_length_m=2.**1023)
