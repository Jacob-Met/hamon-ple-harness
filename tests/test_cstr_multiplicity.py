"""Single-CSTR multiplicity and actual mass-balance regressions."""
import math
import unittest
from hamon_wave6_physics_verifiers import solve_reactor_kinetics as solve

FIRST = dict(reactor_type="cstr", order=1., C_A0=1., v0=1., T0=300.,
             delta_T_adiabatic=200., k_ref=1., T_ref=350., E_a=80000.,
             gas_phase=False, volume=1.)
BUFFALO = dict(reactor_type="cstr", order=2., C_A0=15000., v0=1e-6,
               T0=323.15, delta_T_adiabatic=20000*.015/(.93*.35*4.184),
               A_preexp=3.24e6, E_a=25000*4.184, gas_phase=False, volume=500e-6)

class CstrMultiplicity(unittest.TestCase):
    def test_first_order_has_three_distinct_balanced_roots(self):
        result = solve({**FIRST, "steady_state_branch": "all"})
        expected = (.01342170533580371, .14297312550405095, .99973756033601)
        self.assertEqual(result["steady_state_count"], 3)
        for row, ref in zip(result["steady_states"], expected):
            self.assertAlmostEqual(row["conversion_X"], ref, delta=2e-14)
            self.assertLess(row["residual"], 1e-9)
            self.assertTrue(row["verified"])

    def test_published_second_order_liquid_example(self):
        # Carl R. F. Lund, AFCoKaRE Example 24.1; rates converted to SI.
        # Independent source gives 0.03%, 38.8%, 97.6% at 323/409/538 K.
        result = solve({**BUFFALO, "steady_state_branch": "all"})
        self.assertEqual(result["steady_state_count"], 3)
        for row, conversion, temperature in zip(result["steady_states"],
                (.000302971760963978, .38814825217155074, .9761184274704138),
                (323.2167391601457, 408.6519896228667, 538.1711090461829)):
            self.assertAlmostEqual(row["conversion_X"], conversion, delta=2e-13)
            self.assertAlmostEqual(row["exit_T_K"], temperature, delta=2e-10)
        self.assertTrue(result["verified"])

    def test_implicit_branch_is_refused_and_each_branch_is_selectable(self):
        with self.assertRaisesRegex(ValueError, "multiple physical"):
            solve(FIRST)
        for i in range(3):
            selected = solve({**FIRST, "steady_state_branch": i})
            self.assertEqual(selected["steady_state_branch"], i)
            self.assertEqual(selected["conversion_X"], selected["steady_states"][i]["conversion_X"])

    def test_invalid_branch_selector_is_not_coerced(self):
        for bad in (True, -1, 3, 1.0, "hot"):
            with self.subTest(selector=bad), self.assertRaises(ValueError):
                solve({**FIRST, "steady_state_branch": bad})

    def test_isothermal_unique_and_zero_order_endpoints(self):
        for volume, expected in ((0., 0.), (.1, .1), (1., 1.)):
            result = solve(dict(reactor_type="cstr", order=0., k=1., C_A0=1.,
                                v0=1., volume=volume))
            self.assertEqual(result["conversion_X"], expected)
            self.assertTrue(result["verified"])
        with self.assertRaisesRegex(ValueError, "no physical"):
            solve(dict(reactor_type="cstr", order=0., k=1., C_A0=1., v0=1., volume=1.1))
        result = solve(dict(reactor_type="cstr", order=1., k=.2, C_A0=1., v0=1., volume=5.))
        self.assertEqual(result["conversion_X"], .5)

    def test_constant_rate_ignores_unused_activation_field(self):
        p=dict(reactor_type="cstr", order=1., k=.2, C_A0=1., v0=1.,
               T0=300., delta_T_adiabatic=200., volume=5.)
        self.assertEqual(solve(p), solve({**p, "E_a": 80000.}))

    def test_prefactor_takes_precedence_over_reference_rate(self):
        self.assertEqual(solve({**BUFFALO, "steady_state_branch":"all"}),
                         solve({**BUFFALO, "k_ref":1e100, "T_ref":1.,
                                "steady_state_branch":"all"}))

    def test_prescribed_conversion_sizing_remains_single_valued(self):
        p={k:v for k,v in FIRST.items() if k!="volume"}
        x=.14297312550405095
        result=solve({**p, "conversion":x})
        self.assertAlmostEqual(result["volume_m3"],1.,delta=1e-12)
        self.assertEqual(result["conversion_X"],x)
        self.assertTrue(result["verified"])

    def test_exact_fold_and_adjacent_float_volumes(self):
        p=dict(reactor_type="cstr", order=-1., k=1., C_A0=1., F_A0=4.,
               v0=1., volume=1., steady_state_branch="all")
        result=solve(p)
        self.assertEqual([r["conversion_X"] for r in result["steady_states"]],[.5])
        below=solve({**p,"volume":math.nextafter(1.,0.)})
        self.assertEqual(below["steady_state_count"],2)
        self.assertLess(below["steady_states"][0]["conversion_X"],.5)
        self.assertGreater(below["steady_states"][1]["conversion_X"],.5)
        with self.assertRaisesRegex(ValueError,"no physical"):
            solve({**p,"volume":math.nextafter(1.,math.inf)})
        normalized=solve({**p,"F_A0":1.,"volume":.25})
        self.assertEqual(normalized["steady_states"],result["steady_states"])

    def test_zero_order_rescaling_and_adjacent_endpoint_volumes(self):
        p=dict(reactor_type="cstr",order=0.,k=6.,C_A0=1.,F_A0=3.,
               v0=1.,volume=.5,steady_state_branch="all")
        self.assertEqual(solve(p)["steady_states"][0]["conversion_X"],1.)
        self.assertLess(solve({**p,"volume":math.nextafter(.5,0.)})["steady_states"][0]["conversion_X"],1.)
        with self.assertRaisesRegex(ValueError,"no physical"):
            solve({**p,"volume":math.nextafter(.5,math.inf)})

    def test_temperature_and_expansion_invalid_intervals_are_rejected(self):
        for extra in ({"delta_T_adiabatic":-400.},{"epsilon_A":-1.},
                      {"v0":0.},{"volume":-1.},{"volume":math.inf}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                solve({**FIRST,**extra,"steady_state_branch":"all"})

if __name__=="__main__":
    unittest.main()

