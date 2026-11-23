import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(curve, static, resistance, **options):
    spec = {"calculation": "pump_operating_point",
            "curve": [{"flow_m3_s": q, "head_m": h} for q, h in curve],
            "static_head_m": static, "system_resistance_s2_m5": resistance, **options}
    return run_router_conservation_cad_verifier({"spec": spec})


class PumpOperatingPointTests(unittest.TestCase):
    def test_hydraulic_institute_worked_example(self):
        # Hydraulic Institute Data Tool, table 1.C.1 and equation 1.A.3(c).
        # https://datatool.pumps.org/pump-fundamentals/combined.html
        ft = .3048
        gpm = .003785411784 / 60
        table = [(0, 380), (150, 330.5), (200, 296), (300, 200)]
        curve = [(q*gpm, h*ft) for q, h in table]
        result = solve(curve, 265*ft, .000775*ft/gpm**2)
        self.assertTrue(result["has_rated_intersection"])
        point, = result["operating_points"]
        self.assertAlmostEqual(point["total_flow_m3_s"]/gpm, 200)
        self.assertAlmostEqual(point["head_m"]/ft, 296)

    def test_parallel_bank_matches_registered_duty_curve(self):
        result = solve([(0,60), (.02,40), (.03,15)], 10, 7000, identical_pumps=2)
        point, = result["operating_points"]
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(point["flow_per_pump_m3_s"], .025)
        self.assertAlmostEqual(point["total_flow_m3_s"], .05)
        self.assertAlmostEqual(point["head_m"], 27.5)

    def test_rising_segment_two_crossings_and_classified_tangency(self):
        # H_p = 1+q; residual = q²-q+3/16. Endpoint signs alone miss both.
        curve = [(0,1), (1,2)]
        out = solve(curve, 19/16, 1)
        self.assertEqual([p["total_flow_m3_s"] for p in out["operating_points"]], [.25,.75])
        self.assertEqual(len(solve(curve, 1.25, 1)["operating_points"]), 1)
        self.assertEqual(len(solve(curve, math.nextafter(1.25, 0), 1)["operating_points"]), 2)
        self.assertEqual(solve(curve, math.nextafter(1.25, 2), 1)["operating_points"], [])

    def test_knots_intervals_endpoints_and_no_extrapolation(self):
        out = solve([(0,1),(1,3),(2,3),(3,3),(4,2),(5,3)], 3, 0, identical_pumps=2)
        self.assertEqual(out["continuous_total_flow_intervals_m3_s"], [[2,6]])
        self.assertEqual([p["total_flow_m3_s"] for p in out["operating_points"]], [10])
        out = solve([(0,3),(1,2),(2,1)], 2, 0)
        self.assertEqual(out["operating_points"], [{"flow_per_pump_m3_s": 1,
                                                    "total_flow_m3_s": 1, "head_m": 2}])
        self.assertEqual(solve([(1,4),(2,2)], 5, 0)["operating_points"], [])
        self.assertEqual(solve([(1,4),(2,2)], -1, 0)["operating_points"], [])

    def test_nonbinary_root_and_arbitrary_flow_scale(self):
        for scale in (2.0**-450, 1., 2.0**450):
            out = solve([(0,2), (3*scale,2)], -1, 1/scale**2)
            q = out["operating_points"][0]["total_flow_m3_s"]
            self.assertAlmostEqual(q/scale, math.sqrt(3))
            self.assertEqual(out["operating_points"][0]["head_m"], 2)

    def test_invalid_physics_and_mixed_duty_inputs(self):
        for options in ({"identical_pumps": True}, {"per_pump_input_W": 100}, {"total_flow_m3_s": .1}):
            self.assertFalse(solve([(0,2),(1,1)], 1, 1, **options)["ok"])
        self.assertFalse(solve([(0,2),(0,1)], 1, 0)["ok"])
        self.assertFalse(solve([(0,2),(1,1)], 1, -1)["ok"])
