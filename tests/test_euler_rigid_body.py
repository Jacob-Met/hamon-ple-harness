"""Independent geometry, work and direction controls on the Wave 7 rigid-body path."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import hamon_wave7_engineering_verifiers as wave


def solve(i, w=(1, 0, 0), tau=(0, 0, 0), duration=1, steps=96):
    return wave.dispatch_wave7_verifier("lagrangian_orbital", {
        "subkind": "euler_rigid_body", "I": i, "omega0": w, "torque": tau,
        "t_final": duration, "n_steps": steps,
    })


class EulerRigidBody(unittest.TestCase):
    def test_mass_second_moments(self):
        for i in range(3):
            for scale in (2.0**-65, 1., 2.0**65):
                moments = [scale]*3
                moments[i] = math.nextafter(2*scale, math.inf)
                self.assertEqual(solve(moments)["error"], "unrealizable_principal_moments")
                moments[i] = 2*scale
                self.assertTrue(solve(moments)["verified"])
        # A discrete point mass at each +/- coordinate axis realizes this triple.
        second = (2., 3., 5.)
        self.assertTrue(solve([sum(second) - x for x in second])["verified"])

    def test_spherical_torque_work_and_momentum(self):
        for inertia in (1e-200, 1e-18, 1., 1e80, 1e200):
            with self.subTest(inertia=inertia):
                r = solve([inertia]*3, (1, 0, 0), (0, inertia, 0))
                self.assertTrue(r["verified"], r)
                self.assertFalse(r["torque_free"])
                self.assertIsNone(r["precession_rate_rad_s"])
                np.testing.assert_allclose(r["omega_final_rad_s"], [1, 1, 0], atol=1e-13)
                self.assertAlmostEqual(r["kinetic_energy_J"]/inertia, 1.)
                self.assertAlmostEqual(r["work_done_J"]/inertia, .5)
                self.assertAlmostEqual(r["angular_momentum_norm_Nms"]/inertia, math.sqrt(2))
                self.assertAlmostEqual(r["initial_body_omega_azimuth_rate_rad_s"], 1.)
                self.assertAlmostEqual(r["initial_inertial_H_direction_speed_rad_s"], 1.)
                self.assertAlmostEqual(r["torque_half_H2_change_scaled"], .5)
                if inertia in (1e-200, 1e200):
                    self.assertIsNone(r["torque_half_H2_change_Nms2"])
                else:
                    self.assertAlmostEqual(r["torque_half_H2_change_Nms2"]/inertia/inertia, .5)
                self.assertLess(max(r["work_balance_residual"], r["angular_momentum_balance_residual"]), 1e-12)

    def test_free_top_body_azimuth_and_undefined_directions(self):
        r = solve([2, 2, 3], (.6, .8, .7))
        rate = .35
        expected = [.6*math.cos(rate) - .8*math.sin(rate),
                    .6*math.sin(rate) + .8*math.cos(rate), .7]
        self.assertTrue(r["verified"], r)
        np.testing.assert_allclose(r["omega_final_rad_s"], expected, atol=1e-10)
        self.assertAlmostEqual(r["precession_rate_rad_s"], rate)
        self.assertAlmostEqual(r["initial_body_omega_azimuth_rate_rad_s"], rate)
        self.assertEqual(r["initial_inertial_H_direction_speed_rad_s"], 0.)
        self.assertIsNone(solve([2, 2, 3], (0, 0, 1))["precession_rate_rad_s"])
        no_h = solve([2, 2, 3], (0, 0, 0), (0, 1, 0))
        self.assertIsNone(no_h["initial_inertial_H_direction_speed_rad_s"])
        self.assertIsNone(no_h["initial_body_omega_azimuth_rate_rad_s"])

    def test_constant_body_velocity_with_inertial_torque(self):
        # tau = omega cross H holds body omega constant while inertial H turns.
        r = solve([2, 3, 4], (1, 2, 3), (6, -6, 2))
        self.assertTrue(r["verified"], r)
        np.testing.assert_allclose(r["omega_final_rad_s"], [1, 2, 3], atol=1e-12)
        self.assertAlmostEqual(r["work_done_J"], 0., places=11)
        self.assertAlmostEqual(r["initial_body_omega_azimuth_rate_rad_s"], 0.)
        self.assertAlmostEqual(r["initial_inertial_H_direction_speed_rad_s"], math.sqrt(76/184))

    def test_extreme_norm_and_exact_no_time(self):
        for inertia in (1e-200, 1e200):
            r = solve([inertia]*3, duration=0)
            self.assertTrue(r["verified"], r)
            self.assertEqual(r["angular_momentum_norm_Nms"], inertia)
            self.assertEqual(r["kinetic_energy_J"], .5*inertia)
        w = [.13, .57, -.24]
        self.assertEqual(solve([2, 3, 5], w, duration=0)["omega_final_rad_s"], w)

    def test_incomplete_and_inconsistent_integration(self):
        short = SimpleNamespace(success=True, t=[0., .9], y=np.ones((3, 2)))
        with patch.object(wave.integrate, "solve_ivp", return_value=short):
            self.assertEqual(solve([1, 1, 1])["error"], "incomplete_rigid_body_integrator")
        wrong = SimpleNamespace(success=True, t=[0., 1.], y=np.zeros((3, 2)))
        with patch.object(wave.integrate, "solve_ivp", return_value=wrong):
            self.assertFalse(solve([1, 1, 1])["verified"])
        coarse = solve([2, 3, 4], (2, 3, 4), (.4, 0, 0), 1, steps=1)
        self.assertFalse(coarse["verified"])
        self.assertGreater(coarse["work_balance_residual"], 1e-8)
        self.assertGreater(coarse["angular_momentum_balance_residual"], 1e-8)


if __name__ == "__main__":
    unittest.main()
