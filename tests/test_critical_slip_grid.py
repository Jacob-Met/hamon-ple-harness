import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route


def acads():
    # ACADS verification problem 1; all coordinates are metres and stresses are Pa.
    return {"calculation": "critical_slip_circle_grid", "method": "bishop", "slide_direction": "left",
            "ground_profile_m": [[20, 25], [30, 25], [50, 35], [70, 35]], "base_elevation_m": 20,
            "unit_weight_N_m3": 20000, "cohesion_Pa": 3000, "friction_angle_deg": 19.6,
            "slices_per_surface": 32, "circle_grid": {"center_x_m": list(range(26, 39, 2)),
            "center_y_m": list(range(43, 57, 2)), "radius_m": list(range(18, 33))}}


def single():
    return dict(acads(), circle_grid={"center_x_m": [30], "center_y_m": [53], "radius_m": [28]})


class CriticalCircleGrid(unittest.TestCase):
    def test_published_acads_bishop_and_slice_refinement(self):
        spec = acads()
        previous = None
        for slices in (16, 32, 64):
            answer = route({"spec": dict(spec, slices_per_surface=slices)})
            self.assertTrue(answer["conservation_verified"], answer)
            self.assertEqual(sum(answer["counts"].values()), answer["trials"])
            self.assertGreater(answer["counts"]["qualified"], 1)
            self.assertGreater(answer["counts"]["excluded_geometry"], 0)
            factor = answer["minimum"]["factor_of_safety"]
            # Slide2 publishes 0.987 for a different search/discretization.
            self.assertLess(abs(factor - 0.987), 0.004)
            if previous is not None:
                self.assertLess(abs(factor - previous), 0.001)
            previous = factor
            self.assertFalse(answer["continuous_global_minimum_verified"])

    def test_published_slide2_center_radius_and_surface_endpoints(self):
        grid = {"center_x_m": [29.104], "center_y_m": [55.527], "radius_m": [30.527]}
        actual = route({"spec": dict(single(), circle_grid=grid)})["minimum"]
        self.assertAlmostEqual(actual["sliding_span_m"][0], 30.027, delta=0.002)
        self.assertAlmostEqual(actual["sliding_span_m"][1], 51.700, delta=0.002)
        self.assertAlmostEqual(actual["factor_of_safety"], 0.987, delta=0.001)

    def test_dry_crack_and_reflection(self):
        for method in ("bishop", "janbu_uncorrected"):
            spec = dict(single(), method=method)
            intact = route({"mode": "civil_geotech", "spec": spec})
            self.assertTrue(intact["conservation_verified"], intact)
            cracked = route({"spec": dict(spec, dry_tension_crack_depth_m=0.5)})
            self.assertTrue(cracked["conservation_verified"], cracked)
            span = cracked["minimum"]["sliding_span_m"]
            self.assertLess(span[1], intact["minimum"]["sliding_span_m"][1])
            self.assertAlmostEqual(35 - (53 - math.sqrt(28**2 - (span[1]-30)**2)), 0.5)
            reflected = dict(spec, slide_direction="right", dry_tension_crack_depth_m=0.5,
                             ground_profile_m=[[-x, y] for x, y in reversed(spec["ground_profile_m"])],
                             circle_grid={**spec["circle_grid"], "center_x_m": [-30]})
            other = route({"spec": reflected})
            self.assertAlmostEqual(other["minimum"]["factor_of_safety"], cracked["minimum"]["factor_of_safety"])
            self.assertAlmostEqual(other["minimum"]["sliding_span_m"][0], -span[1])
            self.assertNotAlmostEqual(cracked["minimum"]["factor_of_safety"], intact["minimum"]["factor_of_safety"])

    def test_finite_grid_and_admissibility_controls(self):
        sample = single()
        weak = route({"spec": dict(sample, cohesion_Pa=1500)})
        self.assertLess(weak["minimum"]["factor_of_safety"], route({"spec": sample})["minimum"]["factor_of_safety"])
        for change in ({"slide_direction": "right"}, {"dry_tension_crack_depth_m": 20},
                       {"water_table_m": 4}, {"slices_per_surface": True},
                       {"circle_grid": {**sample["circle_grid"], "radius_m": [40]}},
                       {"circle_grid": {**sample["circle_grid"], "radius_m": [28, 28]}}):
            with self.subTest(change=change):
                self.assertFalse(route({"spec": {**sample, **change}})["conservation_verified"])
