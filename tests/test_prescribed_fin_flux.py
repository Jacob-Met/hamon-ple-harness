"""Independent boundary controls for the registered steady, uniform fin verifier."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def fin(z, base, tip, x=0.375, **claim):
    spec = dict(mode="convective_fin_1d", tip_bc="prescribed_temp", length_m=1.0,
                k_w_mk=2.0, area_m2=0.02, perimeter_m=0.4, h_w_m2k=z*z*0.1,
                t_inf_c=0.0, t_base_c=base, t_tip_c=tip, x_eval_m=x, **claim)
    return run_router_conservation_cad_verifier({"mode": "pde", "spec": spec})


class PrescribedFinFluxTests(unittest.TestCase):
    def test_either_endpoint_can_drive_heat_across_an_ambient_base(self):
        for base, tip in ((0.0, 50.0), (0.0, -50.0), (5e-13, 50.0), (-5e-13, -50.0)):
            with self.subTest(base=base, tip=tip):
                expected = 0.04 * (base * math.cosh(1.0) - tip) / math.sinh(1.0)
                result = fin(1., base, tip, claimed_field="q_fin_w", claimed_value=expected)
                self.assertTrue(result["passed"], result)
                self.assertAlmostEqual(result["q_fin_w"], expected, places=12)
                self.assertAlmostEqual(result["q_fvm_w"], expected, delta=2e-5)
                self.assertFalse(fin(1., base, tip, claimed_field="q_fin_w", claimed_value=0.)["passed"])

    def test_opposed_boundary_contributions_can_cancel(self):
        # cosh(log(2)) = 1.25; the exact base heat flux vanishes.
        result = fin(math.log(2), 20., 25.)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["q_fin_w"], 0., places=12)

    def test_weak_cooling_common_mode_and_long_fin_penetration(self):
        z, temperature = 1e-3, 50.
        result = fin(z, temperature, temperature)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["q_fin_w"], .04*z*temperature*math.tanh(z/2), places=14)
        result = fin(800., 50., -20., x=1.)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["q_fin_w"], .04*800*50, places=9)
        self.assertEqual(result["t_eval_c"], -20.)

    def test_zero_normalization_denominator_cannot_verify_a_numeric_claim(self):
        for field in ("fin_efficiency", "fin_effectiveness"):
            with self.subTest(field=field):
                self.assertIsNone(fin(1., 0., 50.)[field])
                self.assertFalse(fin(1., 0., 50., claimed_field=field, claimed_value=1.)["passed"])

    def test_frozen_two_end_flux_and_temperature_matrix(self):
        from pathlib import Path
        import json
        import subprocess
        import sys
        import tempfile

        root = Path(__file__).resolve().parents[1]
        course = root / "tests" / "fin_flux" / "course.py"
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "fin.json"
            subprocess.run([sys.executable, str(course), str(root), str(results)], check=True)
            rows = json.loads(results.read_text())
        physical = sum(row["qualified"] for row in rows)
        rejected = sum(row.get("wrong_rejected", False) for row in rows)
        exceptions = sum("error" in row for row in rows)
        print(f"Prescribed-tip flux and temperature {physical}/{len(rows)}, "
              f"wrong-claim refusals {rejected}/{len(rows)}, exceptions {exceptions}")
        self.assertEqual((len(rows), physical, rejected, exceptions), (189, 189, 189, 0))
