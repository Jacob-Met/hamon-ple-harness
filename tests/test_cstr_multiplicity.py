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

    def test_non_dyadic_and_rational_order_folds(self):
        for order,c0,feed,root in ((-2.,1.,6.75,1/3),(-.5,3.,1.5,2/3)):
            p=dict(reactor_type="cstr",order=order,k=1.,C_A0=c0,F_A0=feed,
                   volume=1.,steady_state_branch="all")
            with self.subTest(order=order):
                at=solve(p)
                self.assertEqual(at["steady_state_count"],1)
                self.assertAlmostEqual(at["steady_states"][0]["conversion_X"],root,delta=1e-16)
                self.assertTrue(at["verified"])
                self.assertEqual(solve({**p,"volume":math.nextafter(1.,0.)})["steady_state_count"],2)
                with self.assertRaisesRegex(ValueError,"no physical"):
                    solve({**p,"volume":math.nextafter(1.,math.inf)})

    def test_zero_activation_reference_fold_is_not_duplicated(self):
        p=dict(reactor_type="cstr",order=-1.,C_A0=1.,F_A0=4.,volume=1.,
               k_ref=1.,T_ref=600.,E_a=0.,T0=500.,delta_T_adiabatic=200.,
               steady_state_branch="all")
        result=solve(p)
        self.assertEqual(result["steady_state_count"],1)
        self.assertEqual(result["steady_states"][0]["conversion_X"],.5)
        self.assertTrue(result["verified"])

    def test_exact_reference_temperature_fold_and_adjacent_volumes(self):
        p=dict(reactor_type="cstr",order=1.,k_ref=1.,T_ref=512.,
               E_a=2048*8.31446261815324,T0=256.,delta_T_adiabatic=512.,
               C_A0=1.,F_A0=1.,volume=1.,steady_state_branch="all")
        at=solve(p)
        self.assertEqual(at["steady_state_count"],2)
        self.assertEqual(at["steady_states"][-1]["conversion_X"],.5)
        self.assertAlmostEqual(at["steady_states"][0]["conversion_X"],.02676548297,delta=1e-11)
        self.assertTrue(at["verified"])
        self.assertEqual(solve({**p,"volume":math.nextafter(1.,0.)})["steady_state_count"],1)
        self.assertEqual(solve({**p,"volume":math.nextafter(1.,math.inf)})["steady_state_count"],3)

    def test_large_conversion_and_subnormal_residual_are_not_erased(self):
        p=dict(reactor_type="cstr",order=1.,k=1.,C_A0=1.,F_A0=1.,volume=1e100)
        result=solve(p)
        self.assertTrue(result["verified"])
        self.assertEqual(result["conversion_X"],1.)
        self.assertEqual(result["unreacted_fraction"],1e-100)
        subnormal=solve({**p,"F_A0":math.ulp(0.),"volume":1.5})
        self.assertFalse(subnormal["verified"])
        self.assertAlmostEqual(subnormal["residual"],1/3,delta=1e-15)
        self.assertTrue(solve({**p,"F_A0":math.ulp(0.),"volume":1.})["verified"])

    def test_near_complete_state_retains_unreacted_coordinate(self):
        for volume in (1.7815094986951783,10.,1e8):
            with self.subTest(volume=volume):
                p={**FIRST,"order":.5,"volume":volume,"steady_state_branch":"all"}
                result=solve(p)
                self.assertTrue(result["verified"])
                state=result["steady_states"][-1]
                x,u=state["conversion_X"],state["unreacted_fraction"]
                self.assertGreater(u,0.)
                temperature=300.+200.*x
                k=math.exp(-80000./8.31446261815324*(1/temperature-1/350.))
                self.assertLess(abs(x-volume*k*math.sqrt(u)),1e-12)
                if volume==1e8:
                    self.assertEqual(x,1.)
                    self.assertLess(u,1e-20)
                    selected=solve({**p,"steady_state_branch":0})
                    self.assertTrue(selected["verified"])
                    self.assertEqual(selected["unreacted_fraction"],u)

    def test_temperature_and_expansion_invalid_intervals_are_rejected(self):
        for extra in ({"delta_T_adiabatic":-400.},{"epsilon_A":-1.},
                      {"v0":0.},{"volume":-1.},{"volume":math.inf}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                solve({**FIRST,**extra,"steady_state_branch":"all"})

if __name__=="__main__":
    unittest.main()

