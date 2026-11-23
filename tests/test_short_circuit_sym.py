import math
import unittest
from decimal import Decimal, localcontext

from hamon_dual_verifier import run_dual_verify_spec


def solve(**changes):
    spec = dict(S_base_kVA=1000, V_ll_V=480, Z_tx_pct=5,
                S_utility_sc_MVA=20, breaker_aic_kA=12.5)
    return run_dual_verify_spec({"mode": "domain", "kind": "short_circuit_sym",
                                 "spec": dict(spec, **changes)})


def fault_current(tx_xr, utility_xr):
    """Direct sum of per-phase R and X in secondary ohms, from nameplate data."""
    with localcontext() as ctx:
        ctx.prec = 100
        re = im = Decimal(0)
        for ratio in (tx_xr, utility_xr):
            # Both rated component impedances equal 0.01152 ohm at 480 V.
            z = Decimal("0.01152")
            if ratio is None:
                im += z
            else:
                xr = Decimal.from_float(float(ratio))
                r = z/(1+xr*xr).sqrt()
                re += r
                im += r*xr
        return Decimal(480)/(3*(re*re+im*im)).sqrt()


class ShortCircuitTests(unittest.TestCase):
    def test_same_magnitudes_both_sides_of_breaker(self):
        for ratios, adequate in [((6, 6), True), ((6, 1), False), ((1, 6), False)]:
            out = solve(transformer_X_over_R=ratios[0], utility_X_over_R=ratios[1])
            actual = fault_current(*ratios)
            self.assertEqual(out["I_sc_sym_A"], float(actual))
            self.assertEqual(out["aic_ok"], adequate)
            self.assertLessEqual(Decimal(out["I_sc_sym_min_A"]), actual)
            self.assertGreaterEqual(Decimal(out["I_sc_sym_max_A"]), actual)
        unknown = solve()
        self.assertIsNone(unknown["aic_ok"])
        self.assertEqual(unknown["aic_status"], "indeterminate")
        self.assertIsNone(unknown["I_sc_sym_A"])
        self.assertIsNone(unknown["Z_total_pu"])
        self.assertEqual(unknown["co_phasal_I_sc_sym_A"], float(fault_current(6, 6)))

    def test_missing_angles_and_partial_information(self):
        for key in ("utility_X_over_R", "transformer_X_over_R"):
            for xr in (0, .5, 1, 6):
                out = solve(**{key: xr})
                endpoints = [fault_current(xr, other) for other in (0, xr, None)]
                lo, hi = Decimal(out["I_sc_sym_min_A"]), Decimal(out["I_sc_sym_max_A"])
                self.assertLessEqual(lo, min(endpoints))
                self.assertGreaterEqual(hi, max(endpoints))
                self.assertGreater(Decimal(math.nextafter(float(lo), math.inf)), min(endpoints))
                self.assertLess(Decimal(math.nextafter(float(hi), 0)), max(endpoints))
        for rating, status in [(12, "fail"), (15, "indeterminate"), (18, "pass")]:
            self.assertEqual(solve(breaker_aic_kA=rating)["aic_status"], status)

    def test_breaker_neighbors_and_exact_magnitude_without_angles(self):
        for ratios in [(1, 3), (1e-180, 1e180), (6, 6)]:
            with localcontext() as ctx:
                ctx.prec = 100
                actual_ka = fault_current(*ratios)/1000
            center = float(actual_ka)
            for rating in (math.nextafter(center, 0), center, math.nextafter(center, math.inf)):
                out = solve(breaker_aic_kA=rating, transformer_X_over_R=ratios[0],
                            utility_X_over_R=ratios[1])
                self.assertEqual(out["aic_ok"], Decimal(rating) >= actual_ka)
        out = solve(Z_tx_pct=0, breaker_aic_kA=25)
        self.assertTrue(out["aic_ok"])
        self.assertAlmostEqual(out["I_sc_sym_A"], 20e6/(math.sqrt(3)*480))
        self.assertEqual(solve(breaker_aic_kA=0)["aic_status"], "fail")

    def test_impossible_source_data(self):
        for key, value in [("V_ll_V", 0), ("S_utility_sc_MVA", -1),
                           ("Z_tx_pct", -0.01), ("breaker_aic_kA", -1),
                           ("utility_X_over_R", float("inf")), ("transformer_X_over_R", -.1)]:
            with self.assertRaises(ValueError):
                solve(**{key: value})
