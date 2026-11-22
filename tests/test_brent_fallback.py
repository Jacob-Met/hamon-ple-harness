import math
import unittest
from _scipy_compat import _gk15_panel, brentq_fallback


class BrentBracketTest(unittest.TestCase):
    def test_previous_iterate_updates_the_bracket(self):
        for fn, expected in (
            (lambda x: x * x - 2, math.sqrt(2)),
            (lambda x: math.exp(-x) - 0.1, math.log(10)),
        ):
            with self.subTest(root=expected):
                actual = brentq_fallback(fn, 0, 10)
                self.assertAlmostEqual(actual, expected, places=10)
                self.assertAlmostEqual(fn(actual), 0, places=10)

    def test_mach_counterflow_and_parallel_ntu_brackets(self):
        gamma = 1.4

        def area_mach(m, ratio):
            base = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m)
            exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))
            return (base ** exp) / m - ratio

        r_half = area_mach(0.5, 0.0)
        m_half = brentq_fallback(area_mach, 1e-6, 1.0 - 1e-9, args=(r_half,))
        self.assertAlmostEqual(m_half, 0.5, places=10)

        m_sub3 = brentq_fallback(area_mach, 1e-6, 1.0 - 1e-9, args=(3.0,))
        self.assertAlmostEqual(area_mach(m_sub3, 3.0), 0.0, places=10)

        ua = brentq_fallback(lambda u: (u / 1200.0) / (1.0 + u / 1200.0) - (2.0 / 3.0), 50.0, 10000.0)
        self.assertAlmostEqual(ua, 2400.0, places=7)

        ntu = brentq_fallback(lambda n: 0.5 * (1.0 - math.exp(-2.0 * n)) - 0.25, 0.0, 10.0)
        self.assertAlmostEqual(ntu, 0.5 * math.log(2.0), places=10)

    def test_embedded_gauss_weights_match_their_nodes(self):
        for degree in (0, 2, 10):
            value, error = _gk15_panel(lambda x: x ** degree, -1.0, 1.0, ())
            self.assertAlmostEqual(value, 2 / (degree + 1), places=13)
            self.assertLess(error, 1e-12)


    def test_end_to_end_wave6_wave7_solvers_without_scipy(self):
        from types import SimpleNamespace
        import _scipy_compat as sc
        import hamon_wave6_physics_verifiers as w6
        import hamon_wave7_engineering_verifiers as w7

        orig_w6 = (w6.integrate, w6.optimize)
        orig_w7 = (w7.integrate, w7.linalg, w7.optimize)
        try:
            w6.integrate = SimpleNamespace(quad=sc.quad_fallback, solve_ivp=sc.solve_ivp_fallback)
            w6.optimize = SimpleNamespace(brentq=sc.brentq_fallback)
            w7.integrate = SimpleNamespace(quad=sc.quad_fallback, solve_ivp=sc.solve_ivp_fallback)
            w7.optimize = SimpleNamespace(brentq=sc.brentq_fallback)
            w7.linalg = SimpleNamespace(
                lu_factor=sc.lu_factor_fallback,
                lu_solve=sc.lu_solve_fallback,
                eigh=sc.eigh_fallback,
            )
            gamma = 1.4
            for mach_expected, regime in ((0.5, "subsonic"), (3.0, "supersonic")):
                ar = (1.0 / mach_expected) * (
                    (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach_expected * mach_expected)
                ) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
                out = w6.solve_compressible_gas_acoustics(
                    {"subkind": "isentropic", "area_ratio": ar, "gamma": gamma, "regime": regime}
                )
                self.assertTrue(out["verified"])
                self.assertAlmostEqual(out["mach"], mach_expected, places=11)
            hx_eq = w7.solve_exchanger_or_radiation(
                {"subkind": "heat_exchanger", "flow_arrangement": "counterflow",
                 "C_h": 2400.0, "C_c": 2400.0, "T_h_in": 400.0, "T_c_in": 300.0, "UA": 2400.0}
            )
            hx_inv = w7.solve_exchanger_or_radiation(
                {"subkind": "heat_exchanger", "flow_arrangement": "counterflow",
                 "C_h": 2400.0, "C_c": 2400.0, "T_h_in": 400.0, "T_c_in": 300.0,
                 "effectiveness": hx_eq["effectiveness"]}
            )
            self.assertTrue(hx_inv["verified"])
            self.assertAlmostEqual(hx_inv["UA_W_K"], 2400.0, places=9)
            hx_par = w7.solve_exchanger_or_radiation(
                {"subkind": "heat_exchanger", "flow_arrangement": "parallel",
                 "C_h": 2000.0, "C_c": 2000.0, "T_h_in": 400.0, "T_c_in": 300.0, "effectiveness": 0.25}
            )
            self.assertTrue(hx_par["verified"])
            self.assertAlmostEqual(hx_par["UA_W_K"], 2000.0 * (0.5 * math.log(2.0)), places=9)
        finally:
            w6.integrate, w6.optimize = orig_w6
            w7.integrate, w7.linalg, w7.optimize = orig_w7


if __name__ == "__main__":
    unittest.main()
