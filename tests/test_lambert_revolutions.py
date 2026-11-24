"""Published long-flight transfers and explicit revolution-branch controls."""
import math
import unittest

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


# Der, The Superior Lambert Algorithm, AMOS 2011, example 1.
# Source-pinned independent transcription:
# https://github.com/jorgepiloto/lamberthub/blob/5010709b000bb6b29831ad628fa4b1ce1f519036/tests/test_multirev_solvers.py
REFERENCE = {
    ("short", "lower_x"): ([.50335770, .61869408, -1.57176904],
                            [-4.18334626, -1.13262727, 6.13307091]),
    ("short", "upper_x"): ([-2.45759553, 1.16945801, .43161258],
                            [-5.53841370, .01822220, 5.49641054]),
    ("long", "lower_x"): ([1.33645655, -.94654565, .30211211],
                           [4.93628678, .39863416, -5.61593092]),
    ("long", "upper_x"): ([-1.38861608, -.47836611, 2.21280154],
                           [3.92901545, 1.50871943, -6.52926969]),
}


def solve(**kwargs):
    params = dict(subkind="lambert_transfer", mu=1., r1_m=[1.,0.,0.],
                  r2_m=[0.,1.,0.], tof_s=5*math.pi/2, revolutions=1,
                  revolution_branch="upper_x")
    params.update(kwargs)
    return dispatch_wave7_verifier("lagrangian_orbital", params)


class RevolvingLambertTests(unittest.TestCase):
    def test_four_published_der_transfers(self):
        inputs = dict(mu=3.986004418e14,
                      r1_m=[22592145.603, -1599915.239, -19783950.506],
                      r2_m=[1922067.697, 4054157.051, -8925727.465],
                      tof_s=36000)
        for (path, branch), (v1, v2) in REFERENCE.items():
            with self.subTest(path=path, branch=branch):
                result = solve(**inputs, transfer_path=path, revolution_branch=branch)
                self.assertTrue(result["verified"], result)
                self.assertTrue(result["endpoint_verified"], result)
                self.assertEqual(result["propagated_revolutions"], 1)
                for field, expected in (("v1_m_s", v1), ("v2_m_s", v2)):
                    for actual, reference_km_s in zip(result[field], expected):
                        self.assertAlmostEqual(actual, 1000*reference_km_s, delta=.0001)

    def test_circular_orbits_after_one_to_32_complete_revolutions(self):
        for count in (1, 2, 8, 32):
            for path, angle, branch, direction in (
                    ("short", math.pi/2, "upper_x", 1),
                    ("long", 3*math.pi/2, "lower_x", -1)):
                with self.subTest(count=count, path=path):
                    result = solve(revolutions=count, transfer_path=path,
                                   revolution_branch=branch, tof_s=2*math.pi*count+angle)
                    self.assertTrue(result["verified"], result)
                    self.assertEqual(result["propagated_revolutions"], count)
                    self.assertTrue(result["revolution_count_verified"])
                    for actual, expected in zip(result["v1_m_s"], (0.,direction,0.)):
                        self.assertAlmostEqual(actual, expected, delta=3e-13)
                    self.assertAlmostEqual(result["swept_angle_rad"],
                                           2*math.pi*count+angle, delta=3e-13)

    def test_branches_are_distinct_and_claims_do_not_swap_them(self):
        lower = solve(revolution_branch="lower_x")
        upper = solve(revolution_branch="upper_x")
        self.assertTrue(lower["verified"] and upper["verified"], (lower, upper))
        self.assertLess(lower["root_parameter_x"], upper["root_parameter_x"])
        self.assertGreater(math.dist(lower["v1_m_s"], upper["v1_m_s"]), .1)
        rejected = solve(revolution_branch="lower_x", proposed_v1_m_s=upper["v1_m_s"])
        self.assertFalse(rejected["verified"], rejected)
        self.assertTrue(rejected["endpoint_verified"], rejected)

    def test_bifurcation_and_infeasible_flight_times(self):
        reference = solve()
        self.assertTrue(reference["verified"], reference)
        minimum = reference["minimum_tof_s"]
        self.assertFalse(solve(tof_s=.999*minimum)["verified"])
        roots = [solve(tof_s=1.00000001*minimum, revolution_branch=branch)
                 for branch in ("lower_x", "upper_x")]
        self.assertTrue(all(r["verified"] for r in roots), roots)
        self.assertLess(roots[0]["root_parameter_x"], roots[1]["root_parameter_x"])
        for result in roots:
            self.assertLess(result["endpoint_relative_error"], 1e-10)

    def test_branch_and_count_are_not_silently_defaulted(self):
        for changes in ({"revolutions":False}, {"revolutions":1.0}, {"revolutions":-1},
                        {"revolutions":33}, {"revolution_branch":None},
                        {"revolution_branch":"fastest"}, {"revolutions":0},
                        {"revolutions":True}, {"tof_s":1}):
            with self.subTest(changes=changes):
                self.assertFalse(solve(**changes)["verified"])
        baseline = dict(subkind="lambert_transfer", mu=1, r1_m=[1,0,0],
                        r2_m=[0,1,0], tof_s=math.pi/2)
        result = dispatch_wave7_verifier("lagrangian_orbital", baseline)
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["propagated_revolutions"], 0)


if __name__ == "__main__":
    unittest.main()
