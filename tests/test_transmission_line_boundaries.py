import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


class TransmissionLineBoundaryTests(unittest.TestCase):
    def solve(self, **spec):
        result = run_router_conservation_cad_verifier({
            "mode": "rf_transmission_rlc",
            "spec": {"subkind": "transmission_line", "Z0": 50.0, **spec},
        })
        self.assertTrue(result["verified"], result)
        return result

    def test_zero_length_preserves_load_and_finite_vswr(self):
        for load in (1e-14, 1e-6, 2e16, 1e20):
            with self.subTest(load=load):
                result = self.solve(R_L=load, length_wavelengths=0.0)
                self.assertAlmostEqual(result["Z_in_real"] / load, 1.0)
                self.assertEqual(result["Z_in_imag"], 0.0)
                self.assertAlmostEqual(result["VSWR"] / max(load/50.0, 50.0/load), 1.0)

    def test_lossy_quarter_wave_at_high_mismatch(self):
        load = 1e-6
        loss = 1e-8
        result = self.solve(R_L=load, length_wavelengths=0.25, alpha_np_m=loss, length_m=1.0)
        expected = 50 * (50 + load * math.tanh(loss)) / (load + 50 * math.tanh(loss))
        self.assertLess(abs(result["Z_in_real"] / expected - 1.0), 2e-14)
        self.assertEqual(result["Z_in_imag"], 0.0)

    def test_quarter_wave_neighbours_keep_opposite_reactance(self):
        low = self.solve(R_L=1e-8, length_wavelengths=math.nextafter(0.25, 0.0))
        high = self.solve(R_L=1e-8, length_wavelengths=math.nextafter(0.25, 1.0))
        self.assertGreater(low["Z_in_imag"], 0.0)
        self.assertLess(high["Z_in_imag"], 0.0)

