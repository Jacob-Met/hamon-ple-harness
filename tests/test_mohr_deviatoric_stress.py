"""Hydrostatic pressure shifts principal stresses without changing shear yield."""
import math
from decimal import Decimal, localcontext
import unittest

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def stress(**components):
    return dispatch_wave6_verifier("solid_mohr_vessel_buckling",
                                   {"subkind": "mohr_stress", **components})


class MohrDeviatoricStressTests(unittest.TestCase):
    def test_hydrostatic_shift_preserves_a_shear_yield_state(self):
        for pressure in (0., 2.**56, -2.**56):
            for field in ("tau_xy", "tau_xz", "tau_yz"):
                with self.subTest(pressure=pressure, field=field):
                    r = stress(sigma_x=pressure, sigma_y=pressure, sigma_z=pressure,
                               sigma_yield=2., **{field: -1.})
                    self.assertTrue(r["verified"])
                    self.assertEqual(r["tau_max_abs"], 1.)
                    self.assertAlmostEqual(r["sigma_von_mises"], math.sqrt(3.), places=14)
                    self.assertAlmostEqual(r["safety_factor_vm"], 2/math.sqrt(3.), places=14)
                    self.assertEqual(r["safety_factor_tresca"], 1.)

    def test_rotated_repeated_principal_state_under_pressure(self):
        # The pure-shear tensor with every off-diagonal equal to tau has
        # eigenvalues 2*tau, -tau, -tau. Their spread is 3*abs(tau).
        for pressure in (2.**56, -2.**56):
            for tau in (-1., 1.):
                with self.subTest(pressure=pressure, tau=tau):
                    r = stress(sigma_x=pressure, sigma_y=pressure, sigma_z=pressure,
                               tau_xy=tau, tau_xz=tau, tau_yz=tau, sigma_yield=6.)
                    self.assertTrue(r["verified"])
                    self.assertEqual(r["tau_max_abs"], 1.5)
                    self.assertEqual(r["sigma_von_mises"], 3.)
                    self.assertEqual(r["safety_factor_tresca"], 2.)
                    self.assertEqual(r["safety_factor_vm"], 2.)

    def test_absolute_principal_roots_preserve_small_and_zero_stresses(self):
        # Positive 2-by-2 block with determinant 1; recover its small
        # eigenvalue by reciprocal product instead of subtracting close terms.
        n = 2**20
        r = stress(sigma_x=n*n+1, sigma_y=1, sigma_z=-2, tau_xy=n)
        with localcontext() as ctx:
            ctx.prec = 110
            trace = Decimal(n*n+2)
            big = (trace+(trace*trace-4).sqrt())/2
            self.assertEqual((r["sigma_1"], r["sigma_2"], r["sigma_3"]),
                             (float(big), float(1/big), -2.))
        self.assertTrue(r["verified"])
        singular = stress(sigma_x=n*n, sigma_y=1, sigma_z=-2, tau_xy=n)
        self.assertTrue(singular["verified"])
        self.assertEqual((singular["sigma_1"], singular["sigma_2"], singular["sigma_3"]),
                         (n*n+1, 0., -2.))

    def test_three_dimensional_shear_rescaling_and_pure_pressure(self):
        # Two perpendicular shears 3s and 4s give the slant shear 5s.
        for scale in (2.**-550, 1., 2.**505):
            r = stress(tau_xy=3*scale, tau_xz=-4*scale, sigma_yield=10*scale)
            with self.subTest(scale=scale):
                self.assertTrue(r["verified"])
                self.assertEqual((r["sigma_1"], r["sigma_2"], r["sigma_3"]),
                                 (5*scale, 0., -5*scale))
                self.assertEqual(r["tau_max_inplane"], 3*scale)
                self.assertEqual(r["tau_max_abs"], 5*scale)
                self.assertEqual(r["safety_factor_tresca"], 1.)
                self.assertTrue(math.isclose(r["sigma_von_mises"]/scale, math.sqrt(75.), rel_tol=2e-16))
        r = stress(sigma_x=12., sigma_y=12., sigma_z=12., sigma_yield=10.)
        self.assertTrue(r["verified"])
        self.assertEqual((r["sigma_von_mises"], r["tau_max_abs"]), (0., 0.))
        self.assertIsNone(r["safety_factor_vm"])
        self.assertIsNone(r["safety_factor_tresca"])
