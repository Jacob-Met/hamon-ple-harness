"""Passive-load operating points and extrema through the task's dual-verify entry point."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from hamon_dual_verifier import run_dual_verify_spec


BASE = {
    "S_rated_VA": 1000,
    "V_sec_rated_V": 100,
    "R_eq_ohm": 0.2,
    "X_eq_ohm": 0.4,
    "P_core_W": 5,
    "load_fraction": 0.5,
    "pf": 0.8,
    "pf_mode": "lagging",
}


def payload(**changes):
    return {
        "mode": "domain",
        "kind": "transformer_efficiency_regulation",
        "spec": {**BASE, **changes},
    }


def solve(**changes):
    return run_dual_verify_spec(payload(**changes))


class TestTransformerPassiveEfficiency(unittest.TestCase):
    def test_explicit_phase_and_power_balance(self):
        with localcontext() as ctx:
            ctx.prec = 60
            for phase in ("leading", "lagging"):
                sign = -1 if phase == "leading" else 1
                for pf in (0.0, 0.6, 0.8, 1.0):
                    for load in (0, 0.25, 0.5, 1, 1.5):
                        with self.subTest(phase=phase, pf=pf, load=load):
                            result = solve(pf_mode=phase, load_fraction=load, pf=pf)
                            current = Decimal(str(load)) * 10
                            p = Decimal.from_float(pf)
                            sine = (1 - p * p).sqrt()
                            r, x = Decimal.from_float(0.2), Decimal.from_float(0.4)
                            # |V + IZ|² = V² + |IZ|² + 2 V Re(IZ).
                            norm2 = 10000 + current**2 * (r * r + x * x)
                            norm2 += 200 * current * (r * p + sign * x * sine)
                            self.assertAlmostEqual(
                                result["E2_no_load_V"], float(norm2.sqrt()), places=12
                            )
                            self.assertAlmostEqual(
                                result["P_copper_W"], float(current * current * r)
                            )
                            power = 100 * current * p
                            self.assertAlmostEqual(
                                result["efficiency"],
                                float(power / (power + 5 + current * current * r)),
                            )
                            self.assertTrue(result["dual_verified"])
        self.assertLess(solve(pf_mode="LEADING")["voltage_regulation_pct"], 0)

    def test_unrestricted_peak_and_all_defined_tie(self):
        point = solve(P_core_W=80)
        self.assertEqual(point["optimal_load_fraction"], 2.0)
        self.assertEqual(point["optimal_load_fraction_status"], "unique")
        at = solve(P_core_W=80, load_fraction=2)["efficiency"]
        for other in (0.5, 1, 1.99, 2.01, 4):
            self.assertGreater(
                at, solve(P_core_W=80, load_fraction=other)["efficiency"]
            )
        for load in (0, 0.5, 2):
            result = solve(pf=0, load_fraction=load)
            self.assertEqual(result["efficiency"], 0.0)
            self.assertIsNone(result["optimal_load_fraction"])
            self.assertEqual(
                result["optimal_load_fraction_status"],
                "all_loads_with_defined_efficiency",
            )

    def test_vanishing_losses_and_undefined_operating_efficiency(self):
        copper_only = solve(P_core_W=0)
        self.assertEqual(copper_only["efficiency"], float(Fraction(400, 405)))
        self.assertIsNone(copper_only["optimal_load_fraction"])
        self.assertEqual(
            copper_only["optimal_load_fraction_status"],
            "supremum_as_load_approaches_zero",
        )
        self.assertGreater(
            solve(P_core_W=0, load_fraction=0.25)["efficiency"],
            copper_only["efficiency"],
        )
        core_only = solve(R_eq_ohm=0)
        self.assertEqual(core_only["efficiency"], float(Fraction(400, 405)))
        self.assertIsNone(core_only["optimal_load_fraction"])
        self.assertEqual(
            core_only["optimal_load_fraction_status"], "supremum_as_load_increases"
        )
        self.assertGreater(
            solve(R_eq_ohm=0, load_fraction=1)["efficiency"], core_only["efficiency"]
        )
        lossless = solve(R_eq_ohm=0, P_core_W=0)
        self.assertEqual(lossless["efficiency"], 1.0)
        self.assertEqual(
            lossless["optimal_load_fraction_status"],
            "all_loads_with_defined_efficiency",
        )
        for spec, status in [
            ({"load_fraction": 0, "P_core_W": 0}, "supremum_as_load_approaches_zero"),
            (
                {"load_fraction": 0, "R_eq_ohm": 0, "P_core_W": 0},
                "all_loads_with_defined_efficiency",
            ),
            ({"pf": 0, "R_eq_ohm": 0, "P_core_W": 0}, "undefined_at_all_loads"),
        ]:
            result = solve(**spec)
            self.assertTrue(math.isfinite(result["E2_no_load_V"]))
            self.assertIsNone(result["efficiency"])
            self.assertEqual(
                result["efficiency_status"], "undefined_zero_input_real_power"
            )
            self.assertEqual(result["optimal_load_fraction_status"], status)

    def test_invalid_load_domain(self):
        for key, value in [
            ("S_rated_VA", 0),
            ("V_sec_rated_V", -2),
            ("load_fraction", -0.5),
            ("P_core_W", -5),
            ("R_eq_ohm", -0.2),
            ("pf", -0.01),
            ("pf", 1.2),
            ("pf_mode", "typo"),
            ("X_eq_ohm", float("nan")),
            ("pf", True),
            ("P_core_W", None),
            ("load_fraction", float("inf")),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                solve(**{key: value})

    def test_extreme_rated_current_avoids_spurious_zero_optimum(self):
        # Irated² overflows binary64, but copper loss, voltage drop, eta and
        # the unique positive optimum are all representable on this operating point.
        point = solve(
            S_rated_VA=1e200,
            V_sec_rated_V=1,
            R_eq_ohm=1e-300,
            X_eq_ohm=0,
            P_core_W=1,
            load_fraction=1e-100,
            pf=1,
        )
        self.assertAlmostEqual(point["optimal_load_fraction"] / 1e-50, 1.0)
        self.assertAlmostEqual(point["P_copper_W"] / 1e-100, 1.0)
        point = solve(
            S_rated_VA=1e300,
            V_sec_rated_V=1e-100,
            R_eq_ohm=1e100,
            X_eq_ohm=0,
            P_core_W=1e-100,
            load_fraction=0,
        )
        self.assertEqual(point["efficiency"], 0.0)
        self.assertIsNone(point["optimal_load_fraction"])
        self.assertEqual(
            point["optimal_load_fraction_status"], "unique_outside_float_range"
        )

    def test_task_cli_reports_partial_or_rejects_invalid_input(self):
        toolkit = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "transformer.json"
            for changes, code in [({"R_eq_ohm": 0}, 0), ({"pf": 1.2}, 2)]:
                file.write_text(json.dumps(payload(**changes)))
                run = subprocess.run(
                    [sys.executable, str(toolkit), "dual-verify", str(file)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(run.returncode, code)
                result = json.loads(run.stdout)
                self.assertIs(result["ok"], code == 0)
                if code == 0:
                    self.assertTrue(result["dual_verified"])
                    self.assertIsNone(result["optimal_load_fraction"])
                else:
                    self.assertNotIn("dual_verified", result)
