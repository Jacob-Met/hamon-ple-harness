import unittest

from hamon_pareto import dispatch_ple_tool


class FractionalUnitTests(unittest.TestCase):
    def test_rational_powers_on_both_sides(self):
        out = dispatch_ple_tool("hamon_conservation_cad_verifier", {
            "lhs": {"value": 1, "unit": "cm^1/2/s^3/2"},
            "rhs": {"value": 0.1, "unit": "m^1/2/s^3/2"},
        })
        self.assertTrue(out["conservation_verified"])
        self.assertAlmostEqual(out["lhs_si"], 0.1)
        self.assertEqual(out["lhs_dim_7"], [0, 0.5, -1.5, 0, 0, 0, 0])

    def test_multiple_quotients_remain_ambiguous(self):
        with self.assertRaises(ValueError):
            dispatch_ple_tool("hamon_conservation_cad_verifier", {
                "lhs": {"value": 1, "unit": "kg/m^1/2/s"},
                "rhs": {"value": 1, "unit": "1"},
            })
