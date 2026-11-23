"""Check published counterflow terminal approaches through the registered router."""
import math
from fractions import Fraction
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def rate(**changes):
    spec = dict(subkind="heat_exchanger", flow_arrangement="counterflow",
                C_h=1., C_c=1., T_h_in=1e18, T_c_in=10., UA=1e18)
    spec.update(changes)
    return run_router_conservation_cad_verifier({"mode":"exchangers_radiation", "spec":spec})


class CounterflowTerminal(unittest.TestCase):
    def test_equal_capacity_retains_the_final_kelvin(self):
        expected = float(10 + (Fraction(10**18)-10)/(10**18+1))
        result = rate()
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["T_h_out_K"], expected)
        self.assertTrue(rate(T_h_out=expected)["verified"])
        self.assertFalse(rate(T_h_out=10.)["verified"])

    def test_hot_minimum_capacity_uses_exponential_approach(self):
        # For Cr=1/2, 1-epsilon = exp(-NTU/2)/(2-exp(-NTU/2)).
        k = math.exp(-40.)
        expected = 10. + (1e18-10.)*k/(2.-k)
        result = rate(C_c=2., UA=80.)
        self.assertTrue(result["verified"], result)
        self.assertAlmostEqual(result["T_h_out_K"], expected, places=12)
        self.assertFalse(rate(C_c=2., UA=80., T_h_out=10.)["verified"])

    def test_resolved_approach_on_ordinary_kelvin_scale(self):
        # At NTU=40 and Cr=1/2 the terminal gap is about 0.206 microkelvin.
        # A 0.050 microkelvin reading differs by <1e-9 of the full stream drop.
        params = dict(C_c=2., UA=40., T_h_in=500., T_c_in=300.)
        decay = math.exp(-20.)
        expected = 300. + 200. * decay / (2. - decay)
        self.assertTrue(rate(**params, T_h_out=expected)["verified"])
        adverse = rate(**params, T_h_out=300.000000050)
        self.assertFalse(adverse["verified"])
        self.assertGreater(adverse["observation_residuals"]["T_h_out"], 0.7)

    def test_hot_maximum_capacity_and_zero_conductance(self):
        ch=math.nextafter(1., 2.)
        # NTU*(1-Cr)>200, so the exponential changes the result by <1e-75 K.
        expected=float(10 + (Fraction(10**18)-10)*(Fraction(ch)-1)/Fraction(ch))
        result=rate(C_h=ch)
        self.assertTrue(result["verified"],result)
        self.assertEqual(result["T_h_out_K"],expected)
        zero=rate(UA=0.)
        self.assertTrue(zero["verified"],zero)
        self.assertEqual(zero["T_h_out_K"],1e18)

if __name__ == "__main__":
    unittest.main()
