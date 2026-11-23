"""Exact characteristic-polynomial and energy checks through the Wave 7 dispatcher."""
import math
import unittest
from fractions import Fraction
from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def modes(m, k):
    return dispatch_wave7_verifier("lagrangian_orbital", {"subkind": "lagrangian_modes", "M": m, "K": k})


def root_counts(mass, stiffness, point):
    """Count roots below and equal to a point on the 2-DOF characteristic parabola."""
    a, b, d = (Fraction(x) for x in (mass[0][0], mass[0][1], mass[1][1]))
    e, f, h = (Fraction(x) for x in (stiffness[0][0], stiffness[0][1], stiffness[1][1]))
    quadratic, trace_term, constant = a*d-b*b, e*d+h*a-2*b*f, e*h-f*f
    value = quadratic*point*point - trace_term*point + constant
    slope = 2*quadratic*point-trace_term
    if value < 0:
        return 1, 0
    if value == 0:
        return (0, 2) if slope == 0 else (int(slope > 0), 1)
    return (2 if slope > 0 else 0), 0


class ModalInertia(unittest.TestCase):
    def test_scalar_energy_and_units(self):
        for m in (2.0**-100, 1.0, 2.0**100):
            for k in (2.0**-80, 1.0, 2.0**80):
                with self.subTest(m=m, k=k):
                    self.assertEqual(modes([[m]], [[-k]])["error"], "unstable_negative_stiffness_eigenvalue")
                    r = modes([[m]], [[k]])
                    self.assertTrue(r["verified"], r)
                    lo, hi = map(Fraction, r["eigenvalue_intervals_omega2"][0])
                    root = Fraction(k)/Fraction(m)
                    self.assertLessEqual(lo, root)
                    self.assertLessEqual(root, hi)
                    self.assertAlmostEqual(r["natural_frequencies_Hz"][0]/(math.sqrt(float(root))/math.tau), 1.0)
            self.assertEqual(modes([[m]], [[0]])["eigenvalue_intervals_omega2"], [[0.0, 0.0]])

    def test_downdate_and_nullity(self):
        for scale in (2.0**-40, 1.0, 2.0**40):
            for delta in (-2.0**-45, 0.0, 2.0**-45):
                with self.subTest(scale=scale, delta=delta):
                    m = [[1., 0.], [0., 1.]]
                    k = [[scale, scale], [scale, scale*(1+delta)]]
                    r = modes(m, k)
                    if delta < 0:
                        self.assertEqual(r["error"], "unstable_negative_stiffness_eigenvalue")
                        self.assertNotIn("natural_frequencies_Hz", r)
                        continue
                    self.assertTrue(r["verified"], r)
                    self.assertEqual(r["rigid_mode_count"], int(delta == 0))
                    for i, ((lo, hi), val) in enumerate(zip(r["eigenvalue_intervals_omega2"], r["eigenvalues_omega2"])):
                        self.assertLessEqual(lo, val)
                        self.assertLessEqual(val, hi)
                        n_lo, _ = root_counts(m, k, Fraction(lo))
                        n_hi, at_hi = root_counts(m, k, Fraction(hi))
                        self.assertLessEqual(n_lo, i)
                        self.assertGreater(n_hi+at_hi, i)

    def test_shear_and_mixed_diagonal(self):
        for s in (2.0**-15, 1., 2.0**15):
            # Congruent forms preserve the eigenvalues 1 and -2^-18.
            m = [[1., s], [s, 1+s*s]]
            k = [[1., s], [s, s*s-2.0**-18]]
            self.assertEqual(modes(m, k)["error"], "unstable_negative_stiffness_eigenvalue")
        self.assertEqual(modes([[1., 0.], [0., 1.]], [[0., 2.0**-40], [2.0**-40, 0.]])["error"],
                         "unstable_negative_stiffness_eigenvalue")

    def test_repeated_modes(self):
        m = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        for k, count in (([[0.]*3 for _ in range(3)], 3),
                         ([[2., 0., 0.], [0., 2., 0.], [0., 0., 0.]], 1),
                         ([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], 2)):
            r = modes(m, k)
            self.assertTrue(r["verified"], r)
            self.assertEqual(r["rigid_mode_count"], count)
            self.assertEqual(r["eigenvalue_intervals_omega2"][:count], [[0.0, 0.0]]*count)

    def test_unresolved_and_invalid_inputs(self):
        r = modes([[2., .25], [.25, 1.]], [[1., 1.], [1., 1.+2.0**-47]])
        # The determinant is positive; different LAPACK builds may resolve the
        # small root. A verified zero or unstable response is always wrong.
        if r["verified"]:
            self.assertGreater(r["eigenvalue_intervals_omega2"][0][0], 0.)
            self.assertEqual(r["rigid_mode_count"], 0)
        else:
            self.assertIn(r["error"], ("unresolved_positive_modal_frequency", "unresolved_dual_modal_frequency",
                                       "unresolved_modal_solver_residual"))
            self.assertNotIn("natural_frequencies_Hz", r)
        r = modes([[2.0**1023]], [[2.0**-1074]])
        self.assertFalse(r["verified"])
        self.assertNotIn("natural_frequencies_Hz", r)
        for m, k, error in (([[0.]], [[1.]], "non_spd_mass_matrix"),
                            ([[1., .1], [0., 1.]], [[1., 0.], [0., 1.]], "non_symmetric_mass_or_stiffness"),
                            ([], [], "matrix_shape_mismatch")):
            self.assertEqual(modes(m, k)["error"], error)


if __name__ == "__main__":
    unittest.main()
