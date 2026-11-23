"""Passive per-unit-length lines through the registered engineering receiver."""
import math
import unittest

import numpy as np
from scipy.linalg import expm

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


LINE = dict(R_ohm_per_m=0.23, L_H_per_m=8e-7, G_S_per_m=0.00002,
            C_F_per_m=9e-11, frequency_Hz=13.25e6, length_m=8)
LOSSLESS = dict(R_ohm_per_m=0, L_H_per_m=2**-10, G_S_per_m=0,
                C_F_per_m=2**-16, frequency_Hz=256, length_m=8)


def solve(spec):
    return run_router_conservation_cad_verifier(
        {"mode": "rf_transmission_rlc", "spec": {"subkind": "transmission_line", **spec}})


class DistributedLineTests(unittest.TestCase):
    def test_port_against_nodal_generator_exponential(self):
        for frequency in (0, 0.125, 13.25e6):
            for length in (0, 0.02, 8):
                for load, vector in (({"R_L": 33, "X_L": -17}, (33-17j, 1)),
                                     ({"load_kind": "open"}, (1, 0)),
                                     ({"load_kind": "short"}, (0, 1))):
                    spec = dict(LINE, frequency_Hz=frequency, length_m=length, **load)
                    w = 2j * math.pi * frequency
                    z = spec["R_ohm_per_m"] + w * spec["L_H_per_m"]
                    y = spec["G_S_per_m"] + w * spec["C_F_per_m"]
                    v, i = expm(length * np.array([[0, z], [y, 0]])) @ vector
                    with self.subTest(frequency=frequency, length=length, load=load):
                        out = solve(spec)
                        self.assertTrue(out["verified"], out)
                        gamma = complex(out["alpha_np_m"], out["beta_rad_m"])
                        z0 = complex(out["Z0_real"], out["Z0_imag"])
                        self.assertGreaterEqual(gamma.real, 0)
                        self.assertGreaterEqual(gamma.imag, 0)
                        np.testing.assert_allclose([gamma * z0, gamma / z0], [z, y], rtol=1e-13)
                        if i:
                            np.testing.assert_allclose(complex(out["Z_in_real"], out["Z_in_imag"]), v/i, rtol=2e-12, atol=1e-12)
                        else:
                            self.assertEqual(out["Z_in_kind"], "open")
                        if v:
                            np.testing.assert_allclose(complex(out["Y_in_real"], out["Y_in_imag"]), i/v, rtol=2e-12, atol=1e-12)
                        else:
                            self.assertEqual(out["Z_in_kind"], "short")

    def test_series_and_shunt_dc_limits(self):
        dc = dict(LINE, frequency_Hz=0, length_m=8)
        for r, g, load, expected in ((2, 0, {"R_L": 100}, 116),
                                     (0, 0.01, {"R_L": 100}, 100/9),
                                     (0, 0.01, {"load_kind": "open"}, 12.5),
                                     (2, 0, {"load_kind": "short"}, 16)):
            out = solve(dict(dc, R_ohm_per_m=r, G_S_per_m=g, **load))
            self.assertTrue(out["verified"], out)
            self.assertEqual(out["Z_in_real"], expected)
            self.assertEqual(out["Z_in_imag"], 0)
            self.assertIsNone(out["gamma_load_mag"])
            self.assertEqual(out["VSWR_status"], "no_characteristic_impedance")

    def test_exact_nodes_and_adjacent_frequencies(self):
        for load, kind in (({"load_kind": "open"}, "short"), ({"load_kind": "short"}, "open")):
            out = solve(dict(LOSSLESS, **load))
            self.assertTrue(out["verified"], out)
            self.assertEqual(out["Z_in_kind"], kind)
            below, above = (solve(dict(LOSSLESS, frequency_Hz=math.nextafter(256., value), **load))
                            for value in (0., math.inf))
            self.assertTrue(below["verified"] and above["verified"])
            self.assertLess(below["Z_in_imag"] * above["Z_in_imag"], 0)
        out = solve(dict(LOSSLESS, R_L=100))
        self.assertEqual(out["Z_in_real"], .64)
        self.assertEqual(out["Z_in_imag"], 0)
        dc = solve(dict(LOSSLESS, frequency_Hz=0, R_L=100))
        self.assertEqual((dc["Z0_real"], dc["Z_in_real"]), (8., 100.))

    def test_incomplete_overdetermined_and_nonpassive_inputs_abstain(self):
        for key in LINE:
            spec = dict(LINE, R_L=100)
            del spec[key]
            self.assertFalse(solve(spec)["verified"], key)
        for extra in ({"Z0":50}, {"length_wavelengths":0}, {"beta_rad_m":0}, {"alpha_np_m":0},
                      {"unknown_length":2}, {"frequency_Hz":-1}, {"R_ohm_per_m":-0.1}, {"G_S_per_m":float("nan")},
                      {"C_F_per_m":True}, {"L_H_per_m":"1e-7"}, {"length_m":-1}):
            self.assertFalse(solve(dict(LINE, **extra))["verified"], extra)
