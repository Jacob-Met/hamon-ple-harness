"""Signed stiffness and inertia of a rigid patch on an eccentric spring."""
import math
import unittest
from fractions import Fraction

from hamon_wave7_engineering_verifiers import solve_lagrangian_orbital


def solve(mass, stiffness):
    return solve_lagrangian_orbital({"subkind": "lagrangian_modes", "M": mass, "K": stiffness})


class RigidPatchModes(unittest.TestCase):
    def test_free_rotation_weak_spring_and_negative_tangent_stiffness(self):
        # Spring elongation is x + arm*theta. The neighboring rotational
        # stiffness values add or remove one binary64 increment from K22.
        for arm in (3/4096, 7/65536):
            for mass in ([[3, 0], [0, 5]], [[2, .5], [.5, 3]]):
                for direction in (-1, 0, 1):
                    c = math.nextafter(arm**2, math.copysign(math.inf, direction)) if direction else arm**2
                    k = [[1, arm], [arm, c]]
                    out = solve(mass, k)
                    # The roots obey det(K-lambda*M)=A*lambda²-B*lambda+C.
                    a = Fraction(mass[0][0])*mass[1][1] - Fraction(mass[0][1])**2
                    b = Fraction(mass[1][1])+Fraction(c)*mass[0][0]-2*Fraction(arm)*mass[0][1]
                    det = Fraction(c)-Fraction(arm)**2
                    with self.subTest(arm=arm, mass=mass, direction=direction):
                        if det < 0:
                            self.assertFalse(out["verified"], out)
                            self.assertIn("negative_stiffness", out["error"])
                            continue
                        self.assertTrue(out["verified"], out)
                        self.assertEqual(out["rigid_mode_count"], int(not det))
                        high = (float(b)+math.sqrt(float(b*b-4*a*det)))/(2*float(a))
                        low = float(det/a)/high
                        for actual, expected in zip(out["eigenvalues_omega2"], (low, high)):
                            self.assertTrue(math.isclose(actual, expected, rel_tol=2e-15), out)
                        self.assertTrue(math.isclose(out["natural_frequencies_rad_s"][0],
                                                     math.sqrt(low), rel_tol=2e-15))

    def test_multiple_rigid_modes_and_independent_soft_translation(self):
        # Two free patch translations, one independent torsion spring, and
        # one unit-strength spring. Both physical zero modes must survive.
        out = solve([[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,5]],
                    [[0,0,0,0],[0,0,0,0],[0,0,2**-80,0],[0,0,0,10]])
        self.assertTrue(out["verified"], out)
        self.assertEqual(out["rigid_mode_count"], 2)
        for value, interval, expected in zip(out["eigenvalues_omega2"], out["eigenvalue_intervals_omega2"],
                                             (0., 0., 2**-82, 2.)):
            self.assertTrue(math.isclose(value, expected, rel_tol=2e-15))
            self.assertLessEqual(interval[0], expected)
            self.assertGreaterEqual(interval[1], expected)
