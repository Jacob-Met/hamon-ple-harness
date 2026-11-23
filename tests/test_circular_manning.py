import math
import unittest

from mpmath import mp

from hamon_dual_verifier import solve_domain_problem


def run(q, **changes):
    spec = {"cross_section": "circular", "diameter_m": 1., "manning_n": .013,
            "slope_S": .001, "Q_m3_s": q}
    return solve_domain_problem("manning_channel", dict(spec, **changes))


class CircularManningTests(unittest.TestCase):
    def setUp(self):
        c = self.c = mp.clone()
        c.dps = 90
        self.n, self.slope = c.mpf(.013), c.mpf(.001)

        def section(depth):
            # Intersect the circle with a horizontal chord at physical depth.
            half_chord = c.sqrt(depth*(1-depth))
            arc = c.acos(1-2*depth)
            area = arc/4 - (c.mpf(".5")-depth)*half_chord
            return area, arc
        self.section = section

        def discharge(y):
            a, p = section(y)
            return a*(a/p)**(c.mpf(2)/3)*c.sqrt(self.slope)/self.n
        self.discharge = discharge
        # dA/dy is chord width, dP/dy is its reciprocal times 2D.
        self.peak_y = c.findroot(lambda y: 5*y*(1-y)*section(y)[1]-section(y)[0], (.90, .95))
        self.full_q, self.peak_q = float(discharge(1)), float(discharge(self.peak_y))

    def test_two_distinct_free_surface_states_replay_conveyance(self):
        q = .7885087931997831
        r = run(q)
        self.assertTrue(r["dual_verified"])
        self.assertNotIn("normal_depth_m", r)
        self.assertEqual(r["full_section_gravity_flow_m3_s"], self.full_q)
        self.assertEqual(r["maximum_gravity_flow_m3_s"], self.peak_q)
        states = r["normal_depth_solutions"]
        self.assertEqual([s["discharge_branch"] for s in states], ["increasing","decreasing"])
        for state, guess in zip(states, ((.84,.88),(.98,.995))):
            depth = self.c.findroot(lambda y: self.discharge(y)-q, guess)
            self.assertEqual(state["normal_depth_m"], float(depth))
            self.assertEqual(state["crown_clearance_m"], float(1-depth))
            area, perimeter = self.section(depth)
            self.assertEqual(state["velocity_m_s"], float(q/area))
            width = 2*self.c.sqrt(depth*(1-depth))
            self.assertEqual(state["froude_number"], float(q/area*self.c.sqrt(width/(self.c.mpf(9.80665)*area))))

    def test_crown_and_peak_have_different_one_ulp_neighborhoods(self):
        for q, count, full, feasible in (
            (math.nextafter(self.full_q, 0.),1,False,True), (self.full_q,1,True,True),
            (math.nextafter(self.full_q, math.inf),2,False,True),
            (math.nextafter(self.peak_q, 0.),2,False,True), (self.peak_q,1,False,True),
            (math.nextafter(self.peak_q, math.inf),0,False,False)):
            r = run(q)
            with self.subTest(q=q):
                self.assertIs(r["requested_state_feasible"], feasible)
                self.assertIs(r["dual_verified"], feasible)
                self.assertEqual(len(r["normal_depth_solutions"]),count)
                self.assertEqual("full_section_boundary" in r,full)
                for state in r["normal_depth_solutions"]:
                    c = self.c.mpf(state["crown_clearance_m"])
                    self.assertGreater(c,0)
                    self.assertTrue(math.isclose(float(self.discharge(1-c)), q, rel_tol=2e-15))
        peak = run(self.peak_q)["normal_depth_solutions"][0]
        self.assertEqual(peak["discharge_branch"], "maximum")
        self.assertEqual(peak["normal_depth_m"], float(self.peak_y))
        high = run(math.nextafter(self.full_q, math.inf))["normal_depth_solutions"][1]
        self.assertEqual(high["normal_depth_m"], 1.)
        self.assertGreater(high["free_surface_width_m"], 0.)
        self.assertLess(high["crown_clearance_m"], 1e-28)

    def test_geometrically_similar_flows_and_nearly_dry_state(self):
        primary = run(.79)["normal_depth_solutions"]
        # For D multiplied by 8, Q at fixed relative depth scales as 8^(8/3)=256.
        large = run(.79*256, diameter_m=8.)["normal_depth_solutions"]
        for first, other in zip(primary, large):
            self.assertEqual(other["normal_depth_m"], 8*first["normal_depth_m"])
            self.assertEqual(other["crown_clearance_m"], 8*first["crown_clearance_m"])
            self.assertEqual(other["velocity_m_s"], 4*first["velocity_m_s"])
        thin = run(1e-180)
        self.assertTrue(thin["dual_verified"])
        self.assertGreater(thin["normal_depth_m"],0.)
        self.assertEqual(len(thin["normal_depth_solutions"]),1)

    def test_legacy_trapezoid_and_mixed_shape_input(self):
        legacy = {"Q_m3_s":1., "bottom_width_m":2., "side_slope_z":1.,
                  "manning_n":.013, "slope_S":.001}
        self.assertEqual(solve_domain_problem("manning_channel",legacy),
                         solve_domain_problem("manning_channel",dict(legacy,cross_section="trapezoidal")))
        with self.assertRaises(ValueError):
            run(.75,bottom_width_m=1.)
