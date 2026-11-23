"""Hydrostatic pressure shifts principal stresses without changing shear yield."""
import math
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
