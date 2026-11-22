"""Independent influence limits and ordinary consolidation-router checks."""
import math
import unittest
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route


def load(cx=0, cy=0, width=2, length=2, pressure=1000):
    return dict(center_x_m=cx, center_y_m=cy, width_m=width, length_m=length, net_pressure_Pa=pressure)


def check(loads, point=(0, 0, 1), **changes):
    spec = dict(calculation="consolidation_1d", H_clay_m=2, e0=.9, Cc=.25,
                sigma_v0_eff_Pa=50000, cv_m2_s=.0001, time_s=0, drainage="double",
                rectangular_loads=loads, stress_point=dict(zip(("x_m", "y_m", "depth_m"), point)))
    spec.update(changes)
    return route({"mode": "civil_geotech", "spec": spec})


class RectangularStressTest(unittest.TestCase):
    def test_square_corner_and_center(self):
        # A square of side z has Fadum corner influence 0.17522 (USACE Fig. 6-8).
        corner = check([load(width=1, length=1)], point=(.5, .5, 1))
        self.assertTrue(corner["conservation_verified"])
        self.assertAlmostEqual(corner["derived_delta_sigma_v_Pa"] / 1000, .17522, places=5)
        center = check([load()])
        self.assertAlmostEqual(center["derived_delta_sigma_v_Pa"], 4 * corner["derived_delta_sigma_v_Pa"])
        self.assertAlmostEqual(center["derived_delta_sigma_v_Pa"], center["corner_delta_sigma_v_Pa"])

    def test_translation_subdivision_and_proposed_value(self):
        whole = check([load(cx=3, cy=4, width=4, length=2)], point=(9, 8, 2))
        stress = whole["derived_delta_sigma_v_Pa"]
        halves = check([load(cx=-1), load(cx=1)], point=(6, 4, 2), delta_sigma_v_Pa=stress)
        self.assertTrue(halves["conservation_verified"])
        self.assertAlmostEqual(halves["derived_delta_sigma_v_Pa"], stress)
        false_claim = check([load(cx=-1), load(cx=1)], point=(6, 4, 2), delta_sigma_v_Pa=1000)
        self.assertTrue(false_claim["ok"])
        self.assertFalse(false_claim["conservation_verified"])
        self.assertEqual(halves["primary_consolidation_settlement_m"], false_claim["primary_consolidation_settlement_m"])

    def test_point_load_limit_and_surface_limits(self):
        # A small area with total force P tends to 3 P z^3 / (2 pi R^5).
        outside = check([load(width=.001, length=.001, pressure=1e9)], point=(3, 4, 2))
        self.assertTrue(outside["conservation_verified"])
        limit = 3 * 1000 * 2**3 / (2 * math.pi * math.sqrt(29)**5)
        self.assertAlmostEqual(outside["derived_delta_sigma_v_Pa"] / limit, 1, places=6)
        for point, fraction in [((0, 0, 1e-7), 1), ((1, 0, 1e-7), .5),
                                ((1, 1, 1e-7), .25), ((2, 2, 1e-7), 0)]:
            with self.subTest(point=point):
                out = check([load()], point=point)
                self.assertTrue(out["conservation_verified"], out)
                self.assertAlmostEqual(out["derived_delta_sigma_v_Pa"] / 1000, fraction, places=6)

    def test_far_field_and_legacy_scalar(self):
        out = check([load(width=1, length=1, pressure=1.e45)], point=(1.e9, 0, 1))
        self.assertTrue(out["conservation_verified"])
        self.assertAlmostEqual(out["derived_delta_sigma_v_Pa"], 3 / (2 * math.pi), places=10)
        scalar = dict(calculation="consolidation_1d", H_clay_m=2, e0=.9, Cc=.25,
                      sigma_v0_eff_Pa=50000, cv_m2_s=.0001, time_s=0, drainage="double",
                      delta_sigma_v_Pa=45000)
        legacy = route({"mode": "civil_geotech", "spec": scalar})
        self.assertTrue(legacy["conservation_verified"])
        self.assertNotIn("derived_delta_sigma_v_Pa", legacy)

    def test_invalid_geometries(self):
        for loads, pt in [([], (0, 0, 1)), ([load(pressure=-1)], (0, 0, 1)),
                          ([load(width=0)], (0, 0, 1)), ([load()], (0, 0, 0))]:
            with self.subTest(loads=loads, point=pt):
                self.assertFalse(check(loads, pt)["ok"])
        out = check([load()], stress_point=None, delta_sigma_v_Pa=45000)
        self.assertFalse(out["ok"])

    def test_bearing_and_algebraic_entry_points_check_geometry(self):
        from hamon_dual_verifier import solve_domain_problem
        geometry = dict(rectangular_loads=[load()], stress_point={"x_m": 0, "y_m": 0, "depth_m": 1})
        bearing = dict(calculation="bearing_capacity", bearing_method="vesic", footing_B_m=2,
                       depth_Df_m=1, gamma_N_m3=18000, c_prime_Pa=10000, phi_deg=30, FS=3,
                       H_clay_m=2, e0=.9, Cc=.25, sigma_v0_eff_Pa=50000, **geometry)
        algebraic = dict(H_m=2, e0=.9, Cc=.25, Cr=.035, sigma_v0_Pa=50000,
                         sigma_p_Pa=50000, max_settlement_m=1, **geometry)
        for explicit in (False, True):
            if explicit:
                bearing["delta_sigma_v_Pa"] = 999
                algebraic["delta_sigma_Pa"] = 999
            a = route({"mode": "civil_geotech", "spec": bearing})
            b = solve_domain_problem("consolidation_settlement", algebraic)
            self.assertTrue(a["ok"], a)
            self.assertEqual(a["conservation_verified"], not explicit)
            self.assertEqual(b["dual_verified"], not explicit)
            self.assertAlmostEqual(a["primary_consolidation_settlement_m"], b["settlement_m"])


