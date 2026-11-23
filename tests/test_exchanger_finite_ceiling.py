"""Regression tests for representable readings alongside finite-NTU ceilings."""
import math
import unittest

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def sizing(kind, field, value, hot=1000.0, cold=2000.0):
    return dispatch_wave7_verifier("exchangers_radiation", {
        "subkind": "heat_exchanger", "flow_arrangement": kind,
        "C_h": hot, "C_c": cold, "T_h_in": 400.0, "T_c_in": 300.0,
        field: value,
    })


class FiniteCeilingTests(unittest.TestCase):
    def test_finite_inverse_near_rounded_ceiling_in_original_units(self):
        # Reference NTUs from 120-digit bisection of the forward rating equation.
        cases = [
            ("parallel", "effectiveness", 0.6666666666666666, 24.95329850015803),
            ("shell_tube_1_2", "T_c_out", 338.1966011250105, 31.070024730705104),
            ("crossflow_cmax_mixed", "Q", 78693.86805747332, 44.17397512805165),
            ("crossflow_cmin_mixed", "effectiveness", 0.8646647167633873, 75.59117167177169),
        ]
        for kind, field, reading, expected in cases:
            result = sizing(kind, field, reading)
            self.assertTrue(result["verified"], result)
            self.assertTrue(math.isclose(result["NTU"], expected, rel_tol=5e-12))
            impossible = math.nextafter(reading, math.inf)
            self.assertFalse(sizing(kind, field, impossible)["verified"])


    def test_true_parallel_equilibrium_with_nonterminating_capacity_ratio(self):
        # Cmin/Cmax is 1/3. Arbitrary decimal precision alone leaves a false gap.
        for field, value in [("effectiveness", 0.75), ("Q", 75.0),
                             ("T_h_out", 325.0), ("T_c_out", 325.0)]:
            self.assertFalse(sizing("parallel", field, value, hot=1.0, cold=3.0)["verified"])

    def test_true_shell_equilibrium_from_duty_and_outlet(self):
        # For Cmin/Cmax=3/4, sqrt(1+r^2)=5/4 and the limiting duty is 200 W.
        for field, value in [("Q", 200.0), ("T_c_out", 350.0)]:
            self.assertFalse(sizing("shell_tube_1_2", field, value, hot=3.0, cold=4.0)["verified"])
