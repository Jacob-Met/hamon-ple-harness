"""Exercise pile-group compression ratings through the deployed JSON interface."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hamon_dual_verifier import _pile_decimal_capacities, run_dual_verify_spec

BASE = {
    "n_rows": 3, "n_cols": 4, "pile_D_m": 0.45, "spacing_m": 1.35,
    "length_m": 18.0, "cu_Pa": 60000.0, "adhesion_alpha": 0.75, "Nc_tip": 9.0,
    "FS": 2.5, "applied_load_N": 1000000,
}


def request(**changes):
    return {"mode": "domain", "kind": "pile_group_capacity", "spec": BASE | changes}


def solve(**changes):
    return run_dual_verify_spec(request(**changes))


class PileCompressionTests(unittest.TestCase):
    def test_individual_and_block_limits(self):
        for rows, columns, length, spacing, deduction, expected_mode in (
            (2, 2, 5, 1, .5, "individual_efficiency"),
            (10, 10, 100, 1 + math.sqrt(2), .25, "block_failure"),
        ):
            with self.subTest(rows=rows):
                # The chosen layouts have exact angle fractions of pi/4 and pi/8.
                result = solve(n_rows=rows, n_cols=columns, pile_D_m=1, spacing_m=spacing,
                               length_m=length, cu_Pa=1, Nc_tip=9, adhesion_alpha=1, FS=2)
                single = math.pi * (9 / 4 + length)
                eff_count = rows * columns - deduction * ((rows-1)*columns + (columns-1)*rows)
                width, depth = 1 + (columns-1)*spacing, 1 + (rows-1)*spacing
                block = 9 * width * depth + 2 * (width + depth) * length
                self.assertAlmostEqual(result["Q_group_individual_N"], eff_count * single)
                self.assertAlmostEqual(result["Q_group_block_N"], block)
                self.assertAlmostEqual(result["Q_allowable_N"], min(eff_count * single, block) / 2)
                self.assertEqual(result["governing_mode"], expected_mode)

    def test_noncontact_angle_and_boundaries(self):
        # diameter/spacing = tan(pi/8) and each adjacent pair removes a quarter
        # of a pile from the effective count.
        for rows, cols in ((1, 1), (1, 7), (3, 4), (4, 3)):
            with self.subTest(rows=rows, cols=cols):
                r = solve(n_rows=rows, n_cols=cols, pile_D_m=1, spacing_m=1 + math.sqrt(2))
                neighbors = (rows - 1) * cols + (cols - 1) * rows
                self.assertAlmostEqual(r["group_efficiency_eta"], 1 - neighbors / (4 * rows * cols))
        self.assertEqual(solve(n_rows=1, n_cols=1, spacing_m=.1)["group_efficiency_eta"], 1)
        self.assertAlmostEqual(solve(n_rows=10**50, n_cols=10**50, spacing_m=.45)["group_efficiency_eta"]/1e-50, 1)

    def test_physical_input_domain(self):
        for field, values in {
            "n_rows": (False, -2, 0, 2.5), "n_cols": (True, 1.99),
            "applied_load_N": (-1., -1e9, math.nan), "FS": (0, -1, math.inf),
            "spacing_m": (-1, .1), "pile_D_m": (0, -1), "length_m": (0, -1),
            "adhesion_alpha": (-.2,), "Nc_tip": (-1,), "cu_Pa": (-1,),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    solve(**{field: value})
        for changes in ({"n_rows": 3.0}, {"n_rows": "3"}, {"Nc_tip": 0},
                        {"adhesion_alpha": 0}, {"cu_Pa": 0}, {"applied_load_N": 0}):
            self.assertTrue(solve(**changes)["dual_verified"])
        self.assertFalse(solve(applied_load_N=1e9)["passes_pile_group"])

    def test_nongoverning_error_is_caught(self):
        for index in (0, 1, 2, 3, 4):
            with self.subTest(index=index):
                def changed(*args):
                    values = list(_pile_decimal_capacities(*args))
                    values[index] *= 2
                    return values
                with patch("hamon_dual_verifier._pile_decimal_capacities", side_effect=changed):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        solve()

    def test_deployed_cli(self):
        toolkit = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "job.json"
            for load, expected_status in ((1e9, 0), (-1e9, 2)):
                filename.write_text(json.dumps(request(applied_load_N=load)))
                run = subprocess.run([sys.executable, toolkit, "dual-verify", filename],
                                     capture_output=True, text=True)
                reply = json.loads(run.stdout)
                self.assertEqual(run.returncode, expected_status)
                if load > 0:
                    self.assertFalse(reply["passes_pile_group"])
                    self.assertTrue(reply["dual_verified"])
                else:
                    self.assertFalse(reply["ok"])
                    self.assertNotIn("passes_pile_group", reply)
                    self.assertIn("uplift", reply["message"])


if __name__ == "__main__":
    unittest.main()
