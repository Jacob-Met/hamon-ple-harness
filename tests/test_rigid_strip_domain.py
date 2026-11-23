"""Mechanical scope controls for signed strip actions and an area-pressure witness."""
from fractions import Fraction
from itertools import permutations
import unittest

from hamon_bearing_capacity import verify_rigid_strip_contact
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec


class StripDomainTest(unittest.TestCase):
    def test_load_system_is_invariant_to_origin_translation_and_action_order(self):
        # A balanced horizontal pair and a vertical force reproduce the
        # accumulated moment about each translated base center.
        for sign in (-1, 1):
            for cx, cz in [(0, 0), (-3.5, 8), (2**20, -(2**20))]:
                actions = [
                    {"horizontal_N_m": sign * 16, "height_m": cz + 3},
                    {"horizontal_N_m": sign * -16, "height_m": cz},
                    {"vertical_N_m": 64, "x_m": cx + sign / 4},
                ]
                # M = ±(16*3 + 64/4) = ±64. On a 3 m-wide strip this gives
                # an edge-attached triangle of width 1.5 with area 64.
                expected_moment = sign * 64
                for ordered in permutations(actions):
                    spec = {"calculation": "rigid_strip_contact", "base_width_m": 3,
                            "base_center": {"x_m": cx, "height_m": cz},
                            "applied_loads": list(ordered)}
                    out = run_router_conservation_cad_verifier({"spec": spec})
                    self.assertTrue(out["conservation_verified"], out)
                    self.assertEqual(out["horizontal_resultant_N_m"], 0)
                    self.assertEqual(out["vertical_resultant_N_m"], 64)
                    self.assertEqual(out["moment_Nm_m"], expected_moment)
                    left, right = map(Fraction, out["contact_interval_m"])
                    width = right - left
                    peak = Fraction(2 * 64, 1) / width
                    centroid = (left + 2 * right) / 3 if sign > 0 else (2 * left + right) / 3
                    self.assertEqual(width * peak / 2, 64)
                    self.assertEqual(width * peak * centroid / 2, expected_moment)
                    self.assertEqual(float(peak), out["q_max_Pa"])

    def test_two_orthogonal_strips_do_not_certify_a_positive_affine_area_field(self):
        # Exact triangular integrals on [-1,1]^2 for p=max(0,x+y).
        # At fixed x, integrate y from -x to 1: int p dy = (1+x)^2/2.
        def moment(power):
            return Fraction(2, power + 1) if power % 2 == 0 else Fraction(0)
        normal = (moment(0) + 2 * moment(1) + moment(2)) / 2
        x_moment = (moment(1) + 2 * moment(2) + moment(3)) / 2
        self.assertEqual(normal, Fraction(4, 3))
        self.assertEqual(x_moment, Fraction(2, 3))  # y moment follows by symmetry

        # The product of the two equilibrium strip marginals has the same
        # three resultants. Its square footprint and peak differ from the
        # diagonal affine patch, whose area and peak are both exactly 2.
        strip = verify_rigid_strip_contact({"calculation": "rigid_strip_contact",
                                           "base_width_m": 2, "vertical_resultant_N_m": 4,
                                           "moment_Nm_m": 2})
        self.assertTrue(strip["conservation_verified"], strip)
        self.assertEqual(strip["contact_interval_m"], [-0.5, 1.0])
        self.assertEqual(strip["q_max_Pa"], float(Fraction(16, 3)))
        begin = Fraction(-1, 2)
        width = 1 - begin
        strip_area = width ** 2 / 2
        scale = normal / strip_area ** 2
        self.assertEqual(scale * strip_area ** 2 * (begin + 2) / 3, x_moment)
        self.assertEqual(width ** 2, Fraction(9, 4))
        self.assertEqual(scale * width ** 2, Fraction(64, 27))
        self.assertGreater(scale * (-begin) ** 2, 0)  # true pressure at origin is zero

        # Scale the field by three so all public input resultants are integers.
        for sx in (-1, 1):
            for sy in (-1, 1):
                spec = {"calculation": "rigid_strip_contact", "base_width_m": 2,
                        "base_length_m": 2, "vertical_resultant_N": 4,
                        "moment_x_Nm": sx * 2, "moment_y_Nm": sy * 2}
                routes = [verify_rigid_strip_contact(spec),
                          run_router_conservation_cad_verifier({"mode": "civil_geotech", "spec": spec}),
                          run_router_conservation_cad_verifier({"spec": spec})]
                for out in routes:
                    self.assertFalse(out["conservation_verified"], out)
                    self.assertFalse(out["ok"], out)
                dual = run_dual_verify_spec({"mode": "domain", "kind": "rigid_strip_contact",
                                             "spec": spec})
                self.assertFalse(dual["dual_verified"], dual)


if __name__ == "__main__":
    unittest.main()
