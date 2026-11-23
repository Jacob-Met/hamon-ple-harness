"""Independent current and terminal-voltage controls through the registered tool."""
import json
import math
import unittest

from hamon_pareto import dispatch_ple_tool as dispatch, select_domain_tools as select

R = 8.314462618
F = 96485.33212
TOOL = "hamon_conservation_cad_verifier"


def solve(**overrides):
    spec = dict(subkind="electrochemistry", E0_V=1.1, n_electrons=1, T_K=298.15,
                Q_reaction=1, current_A=1, R_internal_ohm=0, exchange_current_A=.1,
                transfer_coefficient=.5)
    spec.update(overrides)
    return json.loads(json.dumps(dispatch(TOOL, {"mode":"electrochem_separations", "spec":spec}), allow_nan=False))


class ButlerVolmer(unittest.TestCase):
    def test_symmetric_reference_and_routing(self):
        self.assertIn(TOOL, select("Calculate Butler-Volmer activation", []))
        eta = 2*R*298.15/F * math.asinh(5)
        for mode, sign in (("galvanic",-1), ("electrolytic",1)):
            r = solve(mode=mode, R_internal_ohm=.025)
            self.assertTrue(r["verified"], r)
            self.assertAlmostEqual(r["kinetic_overpotential_V"], sign*eta, places=13)
            self.assertAlmostEqual(r["V_cell_V"], 1.1 + sign*(eta+.025), places=13)
            zero = solve(mode=mode, current_A=0, R_internal_ohm=12, transfer_coefficient=.2)
            self.assertTrue(zero["verified"])
            self.assertEqual((zero["kinetic_overpotential_V"], zero["kinetic_current_A"], zero["V_cell_V"]),
                             (0,0,1.1))

    def test_forward_current_constructs_independent_asymmetric_controls(self):
        for alpha in (.05,.2,.5,.85,.97):
            for u in (-20.,-2.,-.01,1e-9,.01,2.,20.):
                for temp,n in ((265.,1),(335.,2)):
                    with self.subTest(alpha=alpha, u=u, temp=temp, n=n):
                        i0 = .125
                        current = i0 * (math.expm1(alpha*u) - math.expm1(-(1-alpha)*u))
                        mode = "galvanic" if current < 0 else "electrolytic"
                        out = solve(current_A=abs(current), exchange_current_A=i0, transfer_coefficient=alpha,
                                    mode=mode, kinetic_n_electrons=n, T_K=temp)
                        self.assertTrue(out["verified"],out)
                        self.assertTrue(math.isclose(out["kinetic_overpotential_V"],u*R*temp/(n*F),
                                                     rel_tol=3e-12,abs_tol=1e-20))
                        self.assertEqual(out["kinetic_current_A"],current)
                        self.assertLess(out["kinetic_current_relative_residual"],1e-11)

    def test_partial_conflicting_and_invalid_inputs(self):
        base = {"subkind":"electrochemistry","E0_V":1.1,"n_electrons":1,"current_A":1}
        for extra in ({"exchange_current_A":0.1},{"transfer_coefficient":.5},{"kinetic_n_electrons":1},
                      {"exchange_current_A":0,"transfer_coefficient":.5},
                      {"exchange_current_A":1,"transfer_coefficient":1},
                      {"exchange_current_A":True,"transfer_coefficient":.5},
                      {"exchange_current_A":1,"transfer_coefficient":.5,"eta_activation_V":.02}):
            with self.subTest(extra=extra):
                out=dispatch(TOOL,{"mode":"electrochem_separations","spec":dict(base,**extra)})
                self.assertFalse(out["verified"])
        for extra, voltage in (({},1.1),({"eta_activation_V":.125},.975)):
            out=dispatch(TOOL,{"mode":"electrochem_separations","spec":dict(base,**extra)})
            self.assertTrue(out["verified"])
            self.assertNotIn("activation_model",out)
            self.assertAlmostEqual(out["V_cell_V"],voltage)

    def test_extreme_current_ratio_and_small_signal(self):
        for i,i0 in ((1e200,1e-200),(1e-120,1e120)):
            out=solve(current_A=i,exchange_current_A=i0,mode="electrolytic")
            self.assertTrue(out["verified"],out)
            reference=(2*R*298.15/F*(math.log(i)-math.log(i0)) if i>i0 else R*298.15/F*(i/i0))
            self.assertTrue(math.isclose(out["kinetic_overpotential_V"],reference,rel_tol=3e-14))
