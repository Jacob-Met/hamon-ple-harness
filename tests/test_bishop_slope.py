"""Circular-slice equilibrium, independent oracle and real registered tool checks."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from hamon_bishop import verify_bishop_simplified
from hamon_bishop_circle import verify_bishop_circle
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def specimen():
    return {"calculation": "bishop_simplified", "slip_surface": "circular", "circle_radius_m": 10.,
            "slices": [dict(width_m=1., weight_N_m=w, base_angle_deg=a, cohesion_Pa=c, phi_deg=p)
                       for a, w, c, p in zip([-20., 0., 20., 40., 60.],
                                            [20000., 80000., 150000., 180000., 110000.],
                                            [2000., 3000., 4000., 5000., 6000.],
                                            [15., 20., 25., 30., 35.])]}


def tension_specimen():
    value = specimen()
    value["slices"] = [dict(width_m=1., weight_N_m=1000., base_angle_deg=60., cohesion_Pa=10000., phi_deg=0.),
                       dict(width_m=1., weight_N_m=200000., base_angle_deg=20., cohesion_Pa=1000., phi_deg=0.)]
    return value


class OracleColumn(np.ndarray):
    """The unchanged reference function uses only these two column operations."""
    def to_numpy(self, dtype):
        return np.asarray(self, dtype=dtype)

    @property
    def iloc(self):
        return self


def external_reference(spec):
    path = Path(__file__).parent / "oracles" / "xslope_bishop_source.py"
    loader = importlib.util.spec_from_file_location("independent_xslope_bishop", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    rows, radius = spec["slices"], spec["circle_radius_m"]
    angles = np.radians([r["base_angle_deg"] for r in rows])
    columns = {
        "alpha": [r["base_angle_deg"] for r in rows], "phi": [r["phi_deg"] for r in rows],
        "c": [r["cohesion_Pa"] for r in rows], "w": [r["weight_N_m"] for r in rows],
        "u": [r.get("pore_pressure_Pa", 0.) for r in rows],
        "dl": np.asarray([r["width_m"] for r in rows]) / np.cos(angles),
        "x_c": radius * np.sin(angles), "y_cb": -radius * np.cos(angles),
        "xo": [0.] * len(rows), "yo": [0.] * len(rows),
    }
    table = {key: np.asarray(value, dtype=float).view(OracleColumn) for key, value in columns.items()}
    return module.bishop_independent(table, fs_init=2., tol=1e-13, max_iter=2000)


def physical_profile(slices=20, *, original_cohesion=False):
    """Derive actual slice weights from the pinned XSLOPE simple1 geometry.

    Workbook main!D8=Imperial; profile!A9:B11=(0,0),(20,20),(100,20);
    circles!B3:E3=(10,40,Depth,0); mat!C11,F11,G11=(125,500,0).
    Our tangent bases and integrated weights differ from XSLOPE's arc/chord
    discretization. Compare the independent function on this same table, not
    the rounded README output for its different table. The frictional control
    explicitly changes c=0 and phi=35 while retaining the published geometry.
    """
    from scipy.integrate import quad
    ft, lbf = .3048, 4.4482216152605
    left, right = (50-math.sqrt(2300))/2, 10+math.sqrt(1200)
    edges = [left+(right-left)*i/slices for i in range(slices+1)]
    circle = dict(calculation="bishop_slip_circle", slide_direction="left",
                  circle=dict(center_x_m=10*ft, center_y_m=40*ft, radius_m=40*ft), slices=[])
    table = dict(calculation="bishop_simplified", slip_surface="circular", circle_radius_m=40*ft, slices=[])
    for a, b in zip(edges, edges[1:]):
        area = quad(lambda x: min(x,20)-40+math.sqrt(1600-(x-10)**2), a, b,
                    points=[20] if a < 20 < b else None)[0]
        c, phi = (500*lbf/ft**2, 0.) if original_cohesion else (0., 35.)
        weight = 125*area*lbf/ft
        circle["slices"].append(dict(left_x_m=a*ft, right_x_m=b*ft, weight_N_per_m=weight,
                                    cohesion_Pa=c, friction_angle_deg=phi, pore_pressure_Pa=0.))
        table["slices"].append(dict(width_m=(b-a)*ft, weight_N_m=weight, cohesion_Pa=c, phi_deg=phi,
                                  base_angle_deg=math.degrees(math.asin(((a+b)/2-10)/40))))
    return circle, table


class BishopSlices(unittest.TestCase):
    def assert_closed(self, spec, result):
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["equilibrium_verified"])
        factor = result["factor_of_safety"]
        for row, reaction in zip(spec["slices"], result["slice_reactions"]):
            angle, phi = map(math.radians, (row["base_angle_deg"], row["phi_deg"]))
            length = row["width_m"] / math.cos(angle)
            effective = reaction["effective_normal_N_m"]
            shear = reaction["mobilized_shear_N_m"]
            pressure = row.get("pore_pressure_Pa", 0.)
            self.assertGreaterEqual(effective, 0.)
            self.assertGreater(reaction["m_alpha"], 0.)
            vertical = (effective + pressure * length) * math.cos(angle) + shear * math.sin(angle)
            strength = row["cohesion_Pa"] * length + effective * math.tan(phi)
            self.assertAlmostEqual(vertical / row["weight_N_m"], 1., places=11)
            self.assertAlmostEqual(factor * shear / strength, 1., places=11)
        driving = math.fsum(r["weight_N_m"] * math.sin(math.radians(r["base_angle_deg"])) for r in spec["slices"])
        self.assertAlmostEqual(math.fsum(r["mobilized_shear_N_m"] for r in result["slice_reactions"]) / driving,
                               1., places=11)

    def test_tiny_tensile_contact_cannot_disappear_during_normalization(self):
        spec = specimen()
        spec["circle_radius_m"] = 1.
        spec["slices"] = [
            dict(width_m=1., weight_N_m=1e300, base_angle_deg=30., cohesion_Pa=0., phi_deg=35.),
            dict(width_m=1., weight_N_m=1e-200, base_angle_deg=0., cohesion_Pa=0., phi_deg=0.,
                 pore_pressure_Pa=1e-100),
        ]
        result = verify_bishop_simplified(spec)
        self.assertFalse(result["equilibrium_verified"])
        self.assertIn("normalization", result["reason"])

    def test_nonzero_pressure_or_cohesion_width_product_must_survive(self):
        for key in ("cohesion_Pa", "pore_pressure_Pa"):
            with self.subTest(key=key):
                spec = specimen()
                spec["slices"][0].update(width_m=1e-200, **{key: 1e-200})
                result = verify_bishop_simplified(spec)
                self.assertFalse(result["equilibrium_verified"])
                self.assertIn("width multiplication", result["reason"])

    def test_published_geometry_with_frictional_control_matches_independent_reference(self):
        for count in (20, 80):
            with self.subTest(count=count):
                circle, table = physical_profile(count)
                result = verify_bishop_circle(circle)
                self.assertTrue(result["conservation_verified"], result)
                self.assertAlmostEqual(result["factor_of_safety"], external_reference(table), places=11)
                self.assertTrue(all(r["effective_normal_N_per_m"] > 0 for r in result["slices"]))

    def test_positive_cohesive_equilibrium_factor_does_not_override_tensile_contact(self):
        circle, table = physical_profile(original_cohesion=True)
        factor = external_reference(table)
        self.assertGreater(factor, 1.)
        last = table["slices"][-1]
        alpha = math.radians(last["base_angle_deg"])
        shear = last["cohesion_Pa"]*last["width_m"]/math.cos(alpha)/factor
        normal = (last["weight_N_m"]-shear*math.sin(alpha))/math.cos(alpha)
        self.assertLess(normal, 0.)
        result = verify_bishop_circle(circle)
        self.assertFalse(result["conservation_verified"])
        self.assertFalse(result["compressive_contact_verified"])
        self.assertIn("tensile", result["reason"])

    def test_scaled_moments_survive_overflow_of_an_unreported_force_sum(self):
        edges = [-.08, -.075, -.07, -.065, -.06]
        spec = dict(calculation="bishop_slip_circle", slide_direction="right",
                    circle=dict(center_x_m=0., center_y_m=0., radius_m=.1),
                    slices=[dict(left_x_m=a, right_x_m=b, weight_N_per_m=1., cohesion_Pa=0.,
                                 friction_angle_deg=35., pore_pressure_Pa=0.)
                            for a,b in zip(edges,edges[1:])])
        ordinary = verify_bishop_circle(spec)
        for row in spec["slices"]: row["weight_N_per_m"] *= 1e308
        large = verify_bishop_circle(spec)
        self.assertTrue(large["conservation_verified"], large)
        self.assertAlmostEqual(large["factor_of_safety"], ordinary["factor_of_safety"], places=13)
        self.assertAlmostEqual(large["driving_moment_Nm_per_m"]/1e308,
                               ordinary["driving_moment_Nm_per_m"], places=14)

    def test_single_slice_matches_independent_wedge_force_balance(self):
        spec = specimen()
        spec["slices"] = [dict(width_m=1., weight_N_m=100000., base_angle_deg=30., cohesion_Pa=0., phi_deg=35.)]
        result = verify_bishop_simplified(spec)
        self.assert_closed(spec, result)
        self.assertAlmostEqual(result["factor_of_safety"], math.tan(math.radians(35)) / math.tan(math.radians(30)), places=13)

    def test_dry_and_wet_mixed_slices_match_pinned_independent_reference(self):
        values = []
        for water_fraction in (0., .1, .2, .3):
            for cohesion_factor in (.5, 1., 2.):
                with self.subTest(water=water_fraction, cohesion=cohesion_factor):
                    spec = specimen()
                    for row in spec["slices"]:
                        row["pore_pressure_Pa"] = water_fraction * row["weight_N_m"] / row["width_m"]
                        row["cohesion_Pa"] *= cohesion_factor
                    result = verify_bishop_simplified(spec)
                    self.assert_closed(spec, result)
                    self.assertAlmostEqual(result["factor_of_safety"], external_reference(spec), places=11)
                    if cohesion_factor == 1.:
                        values.append(result["factor_of_safety"])
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_undrained_strength_matches_explicit_moment_equation(self):
        spec = specimen()
        for row in spec["slices"]:
            row.update(cohesion_Pa=5000., phi_deg=0.)
        resisting = math.fsum(r["cohesion_Pa"] * r["width_m"] / math.cos(math.radians(r["base_angle_deg"])) for r in spec["slices"])
        driving = math.fsum(r["weight_N_m"] * math.sin(math.radians(r["base_angle_deg"])) for r in spec["slices"])
        result = verify_bishop_simplified(spec)
        self.assert_closed(spec, result)
        self.assertAlmostEqual(result["factor_of_safety"], resisting / driving, places=13)

    def test_negative_base_angle_selects_the_physical_quadratic_root(self):
        spec = specimen()
        spec["slices"] = [dict(width_m=1., weight_N_m=10000., base_angle_deg=-60., cohesion_Pa=0., phi_deg=45.),
                          dict(width_m=1., weight_N_m=100000., base_angle_deg=30., cohesion_Pa=0., phi_deg=10.)]
        triples = [(r["weight_N_m"] * math.tan(math.radians(r["phi_deg"])),
                    math.sin(math.radians(r["base_angle_deg"])) * math.tan(math.radians(r["phi_deg"])),
                    math.cos(math.radians(r["base_angle_deg"]))) for r in spec["slices"]]
        (a, b, c), (A, B, C) = triples
        driving = math.fsum(r["weight_N_m"] * math.sin(math.radians(r["base_angle_deg"])) for r in spec["slices"])
        roots = np.roots([driving * c * C, driving * (c * B + C * b) - a * C - A * c,
                          driving * b * B - a * B - A * b])
        pole = max(0., -b / c, -B / C)
        physical, = [float(root) for root in roots if np.isreal(root) and root > pole]
        result = verify_bishop_simplified(spec)
        self.assert_closed(spec, result)
        self.assertAlmostEqual(result["factor_of_safety"], physical, places=12)
        self.assertGreater(result["factor_bracket"][0], pole)

    def test_tensile_base_is_not_certified_by_a_scalar_root(self):
        result = verify_bishop_simplified(tension_specimen())
        self.assertFalse(result["ok"])
        self.assertFalse(result["conservation_verified"])
        self.assertIsNone(result["factor_of_safety"])
        self.assertIn("tensile", result["reason"])

    def test_force_scaling_and_radius_preserve_factor(self):
        baseline = verify_bishop_simplified(specimen())
        for scale in (1e-120, 1e-6, 1e6, 1e120):
            with self.subTest(scale=scale):
                spec = specimen()
                for row in spec["slices"]:
                    row["weight_N_m"] *= scale
                    row["cohesion_Pa"] *= scale
                spec["circle_radius_m"] *= 17
                result = verify_bishop_simplified(spec)
                self.assert_closed(spec, result)
                self.assertAlmostEqual(result["factor_of_safety"], baseline["factor_of_safety"], places=12)
                self.assertAlmostEqual(result["driving_moment_Nm_per_m"] / scale / 17,
                                       baseline["driving_moment_Nm_per_m"], places=8)

    def test_slice_table_subdivision_preserves_summed_equilibrium(self):
        spec = specimen()
        baseline = verify_bishop_simplified(spec)
        spec["slices"] = [dict(row, width_m=row["width_m"] / 2, weight_N_m=row["weight_N_m"] / 2)
                          for row in spec["slices"] for _ in range(2)]
        result = verify_bishop_simplified(spec)
        self.assert_closed(spec, result)
        self.assertAlmostEqual(result["factor_of_safety"], baseline["factor_of_safety"], places=12)

    def test_model_scope_and_incomplete_inputs_are_refused(self):
        cases = []
        for changes in ({"slip_surface": "noncircular"}, {"horizontal_seismic_coefficient": .1},
                        {"reinforcement_N_m": 10000.}, {"surface_water_depth_m": 2.},
                        {"circle_radius_m": 0.}, {"slices": []}, {"calculation": "janbu"},
                        {"domain": "structural"}):
            cases.append(dict(specimen(), **changes))
        for key, value in [("width_m", 0.), ("weight_N_m", -1.), ("base_angle_deg", 90.),
                           ("base_angle_deg", math.nan), ("cohesion_Pa", True), ("phi_deg", 90.),
                           ("pore_pressure_Pa", -1.), ("surface_load_N_m", 1.)]:
            spec = specimen(); spec["slices"][0][key] = value; cases.append(spec)
        spec = specimen(); del spec["slices"][0]["cohesion_Pa"]; cases.append(spec)
        spec = specimen(); spec["slices"][0]["id"] = spec["slices"][1]["id"] = "same"; cases.append(spec)
        spec = specimen()
        for row in spec["slices"]: row["base_angle_deg"] = -10.
        cases.append(spec)
        for index, spec in enumerate(cases):
            with self.subTest(index=index):
                result = verify_bishop_simplified(spec)
                self.assertFalse(result["ok"], result)
                self.assertIsNone(result["factor_of_safety"])

    def test_proposal_and_required_factor_are_distinct_from_equilibrium(self):
        spec = dict(specimen(), required_factor_of_safety=1.5, proposed_factor_of_safety=3.)
        result = verify_bishop_simplified(spec)
        self.assertTrue(result["equilibrium_verified"])
        self.assertFalse(result["proposed_factor_matches"])
        self.assertFalse(result["conservation_verified"])
        self.assertFalse(result["selected_surface_meets_required_factor"])
        self.assertFalse(result["slice_geometry_verified"])
        self.assertFalse(result["horizontal_force_equilibrium_checked"])
        self.assertFalse(result["critical_surface_search_performed"])

    def test_public_router_keeps_the_existing_circle_geometry_api(self):
        spec = dict(calculation="bishop_slip_circle", circle=dict(center_x_m=0., center_y_m=13., radius_m=10.),
                    slide_direction="right", slices=[dict(left_x_m=-6., right_x_m=-4., weight_N_per_m=1000.,
                        cohesion_Pa=120., friction_angle_deg=30., pore_pressure_Pa=95.)])
        for mode in ("auto", "civil_geotech"):
            with self.subTest(mode=mode):
                result = run_router_conservation_cad_verifier({"mode": mode, "spec": spec})
                self.assertTrue(result["conservation_verified"], result)
                self.assertTrue(result["compressive_contact_verified"])
                self.assertTrue(result["slice_geometry_verified"])
                self.assertFalse(result["horizontal_force_equilibrium_checked"])
                self.assertFalse(result["critical_surface_search_performed"])
                self.assertEqual(result["calculation"], "bishop_slip_circle")


if __name__ == "__main__":
    unittest.main()
