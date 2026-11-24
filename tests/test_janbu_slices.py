"""Independent reference surfaces and validity checks for simplified Janbu."""

import copy
import json
import math
from pathlib import Path
import unittest

from hamon_janbu_slices import verify_janbu_slices


def seequent():
    path = Path(__file__).parent / "fixtures" / "slope_seequent_hand_calculation.json"
    table = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {"id": str(number), "width_m": b, "weight_N_m": w * 1000.0,
         "base_angle_deg": -angle, "pore_pressure_Pa": u * 1000.0}
        for number, _, b, w, angle, _, u in table["rows"]
    ]
    return {"calculation": "janbu_slices", "c_prime_Pa": 5000.0, "phi_deg": 30.0,
            "slices": rows}


def ibaraki(count, water_elevation=None):
    # Published midpoint construction, independent of the solver.
    cx, cz, x0 = 6.0, 18.0, -1.0
    radius2 = (x0 - cx)**2 + cz**2
    x1 = cx + math.sqrt(radius2 - (cz - 10.0)**2)
    width = (x1 - x0) / count
    rows = []
    for i in range(count):
        x = x0 + (i + 0.5) * width
        dz = math.sqrt(radius2 - (x - cx)**2)
        base = cz - dz
        ground = min(10.0, max(0.0, x / 1.5))
        head = 0.0 if water_elevation is None else max(0.0, min(ground, water_elevation) - base)
        rows.append({"id": str(i + 1), "width_m": width,
                     "weight_N_m": 18000.0 * width * (ground - base),
                     "base_angle_deg": math.degrees(math.atan2(x - cx, dz)),
                     "pore_pressure_Pa": 9810.0 * head})
    return {"calculation": "janbu_slices", "c_prime_Pa": 10000.0, "phi_deg": 30.0,
            "slices": rows}


def horizontal_residual(spec, factor):
    # Recover normal and mobilized shear from the dimensional slice free body.
    terms = []
    t = math.tan(math.radians(spec["phi_deg"]))
    for r in spec["slices"]:
        alpha = math.radians(r["base_angle_deg"])
        co, si = math.cos(alpha), math.sin(alpha)
        length = r["width_m"] / co
        constant = (spec["c_prime_Pa"] - r["pore_pressure_Pa"] * t) * length
        normal = (r["weight_N_m"] - constant * si / factor) / (co + t * si / factor)
        shear = (constant + normal * t) / factor
        terms.append(shear * co - normal * si)
    return math.fsum(terms)


class JanbuSlicesTests(unittest.TestCase):
    def test_seequent_inputs_with_independent_horizontal_force_reference(self):
        spec = seequent()
        out = verify_janbu_slices(spec)
        self.assertTrue(out["verified"], out)
        self.assertTrue(out["base_contact_verified"])
        self.assertEqual(out["method"], "janbu_simplified_uncorrected")
        self.assertAlmostEqual(out["factor_of_safety"], 1.1247783031418455, places=11)
        self.assertLess(abs(horizontal_residual(spec, out["factor_of_safety"])), 1e-7)
        self.assertFalse(out["slice_geometry_verified"])
        self.assertFalse(out["critical_surface_search_performed"])

    def test_published_ibaraki_roots_and_tensile_slice(self):
        for n, water, reference, delta, tensile in (
            (6, None, 1.8293, 0.00005, []),
            (50, None, 1.8677, 0.00005, ["50"]),
            (50, 4.0, 1.427, 0.0005, ["50"]),
        ):
            with self.subTest(n=n, water=water):
                out = verify_janbu_slices(ibaraki(n, water))
                self.assertTrue(out["solution_found"], out)
                self.assertTrue(out["equilibrium_verified"])
                self.assertAlmostEqual(out["factor_of_safety"], reference, delta=delta)
                self.assertEqual(out["tensile_slice_ids"], tensile)
                self.assertEqual(out["base_contact_verified"], not tensile)
                self.assertEqual(out["verified"], not tensile)

    def test_small_analytical_root(self):
        spec = {"calculation": "janbu_slices", "c_prime_Pa": 1e-100, "phi_deg": 0.0,
                "slices": [{"width_m": 1.0, "weight_N_m": 1.0, "base_angle_deg": 45.0}]}
        out = verify_janbu_slices(spec)
        self.assertTrue(out["verified"], out)
        self.assertTrue(math.isclose(out["factor_of_safety"], 2e-100, rel_tol=1e-12))

    def test_signed_angles_and_admissible_denominator_branch(self):
        spec = seequent()
        for r in spec["slices"]:
            r["base_angle_deg"] *= -1
        self.assertFalse(verify_janbu_slices(spec)["solution_found"])
        for r in spec["slices"]:
            r["base_angle_deg"] = abs(r["base_angle_deg"])
        unsigned = verify_janbu_slices(spec)
        self.assertGreater(abs(unsigned["factor_of_safety"] - 1.1247783031418455), 0.01)
        spec = {"calculation": "janbu_slices", "c_prime_Pa": 0.0, "phi_deg": 60.0,
                "slices": [{"width_m": 1.0, "weight_N_m": 1.0, "base_angle_deg": -70.0},
                           {"width_m": 1.0, "weight_N_m": 10.0, "base_angle_deg": 30.0}]}
        out = verify_janbu_slices(spec)
        self.assertTrue(out["verified"], out)
        self.assertGreater(out["factor_of_safety"], math.tan(math.radians(70.0)) *
                           math.tan(math.radians(60.0)))

    def test_force_scaling_and_subdivision(self):
        spec = seequent()
        for scale in (1e-120, 1e120):
            changed = copy.deepcopy(spec)
            changed["c_prime_Pa"] *= scale
            for r in changed["slices"]:
                r["weight_N_m"] *= scale
                r["pore_pressure_Pa"] *= scale
            out = verify_janbu_slices(changed)
            self.assertTrue(out["verified"], out)
            self.assertAlmostEqual(out["factor_of_safety"], 1.1247783031418455, places=11)
        split = [{**r, "id": r["id"] + suffix, "width_m": r["width_m"] / 2,
                  "weight_N_m": r["weight_N_m"] / 2}
                 for r in spec["slices"] for suffix in ("a", "b")]
        out = verify_janbu_slices({**spec, "slices": split})
        self.assertTrue(out["verified"], out)
        self.assertAlmostEqual(out["factor_of_safety"], 1.1247783031418455, places=11)

    def test_threshold_and_proposed_factor(self):
        spec = seequent()
        for threshold, meets in ((1.0, True), (1.15, False)):
            out = verify_janbu_slices({**spec, "required_factor_of_safety": threshold})
            self.assertTrue(out["verified"], out)
            self.assertEqual(out["meets_required_factor"], meets)
        out = verify_janbu_slices({**spec, "proposed_factor_of_safety": 2.0})
        self.assertFalse(out["proposed_factor_consistent"])
        self.assertFalse(out["verified"])

    def test_unsupported_and_malformed_inputs(self):
        spec = seequent()
        for extra in ({"seismic_kh": 0.1}, {"method": "corrected_janbu"}, {"slices": []}):
            with self.subTest(extra=extra):
                self.assertFalse(verify_janbu_slices({**spec, **extra})["verified"])
        for key, value in (("width_m", 0.0), ("weight_N_m", True), ("phi_deg", 90.0),
                           ("base_angle_deg", 90.0), ("pore_pressure_Pa", -1.0),
                           ("weight_N_m", float("nan"))):
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(spec)
                changed["slices"][0][key] = value
                out = verify_janbu_slices(changed)
                self.assertFalse(out["verified"])
                json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
