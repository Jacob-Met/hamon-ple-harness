"""Physical open/short load boundaries through the registered RF solver."""
import json
import math
import unittest

from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(**params):
    result = run_router_conservation_cad_verifier({
        "mode": "rf_transmission_rlc", "spec": {"subkind": "transmission_line", **params}})
    json.dumps(result, allow_nan=False)
    return result


class LoadTerminationTests(unittest.TestCase):
    def test_lossless_nodes_and_eighth_waves(self):
        for kind, zero, quarter, sign, angle in (("open","open","short",-1,0),
                                                 ("short","short","open",1,180)):
            for offset in (0, .5, 8, -1):
                for turns, expected, reactance in ((0,zero,0),(.125,"finite",sign*50),
                                                    (.25,quarter,0),(.375,"finite",-sign*50)):
                    with self.subTest(kind=kind, turns=turns, offset=offset):
                        out = solve(load_kind=kind, Z0=50, length_wavelengths=offset+turns)
                        self.assertTrue(out["verified"],out)
                        self.assertEqual(out["load_kind"],kind)
                        self.assertEqual(out["Z_in_kind"],expected)
                        self.assertEqual((out["Z_in_real"],out["Z_in_imag"]),
                                         (None,None) if expected=="open" else (0.,reactance))
                        self.assertEqual((out["Y_in_real"],out["Y_in_imag"]),
                                         (None,None) if expected=="short" else
                                         (0., -1/reactance if reactance else 0.))
                        self.assertEqual((out["gamma_load_mag"],out["gamma_load_deg"]),(1.,angle))
                        self.assertEqual((out["VSWR"],out["VSWR_status"]),(None,"unbounded"))
                        self.assertEqual((out["return_loss_dB"],out["return_loss_status"]),(0.,"finite"))
                        if kind=="open":
                            self.assertIsNone(out["quarter_wave_Z_T"])

    def test_abcd_boundary_columns_and_nearby_finite_states(self):
        ctx = mp.clone()
        ctx.dps = 150
        for kind in ("open","short"):
            for z0 in (50., 37+4j):
                for turns, loss in ((.173,.08),(.392,20),(.25,1e-55),
                                    (math.nextafter(.25,0.),0),(math.nextafter(.25,1.),0),
                                    (math.nextafter(.5,1.),1e-12)):
                    with self.subTest(kind=kind,z0=z0,turns=turns,loss=loss):
                        out = solve(load_kind=kind,Z0=str(z0),length_wavelengths=turns,
                                    alpha_np_m=loss,length_m=1.)
                        self.assertTrue(out["verified"],out)
                        self.assertEqual(out["Z_in_kind"],"finite")
                        # ABCD columns for I_L=0 and V_L=0. At an exact quarter
                        # turn use cos(pi/2)=0 to keep spurious residuals out of the oracle.
                        angle = 2*ctx.pi*ctx.mpf(turns)
                        co, si = (0,1) if turns==.25 else (ctx.cos(angle),ctx.sin(angle))
                        c = ctx.mpc(ctx.cosh(loss)*co, ctx.sinh(loss)*si)
                        s = ctx.mpc(ctx.sinh(loss)*co, ctx.cosh(loss)*si)
                        z = ctx.mpc(z0)*c/s if kind=="open" else ctx.mpc(z0)*s/c
                        for prefix, value in (("Z_in",z),("Y_in",1/z)):
                            for field, expected in (("real",value.real),("imag",value.imag)):
                                self.assertTrue(math.isclose(out[prefix+"_"+field],float(expected),
                                                             rel_tol=1e-14),out)

    def test_ordinary_finite_load_default_and_no_contradictory_boundaries(self):
        self.assertEqual(solve(load_kind="open")["Z_in_kind"],"open")
        self.assertEqual(solve(load_kind="short")["Z_in_kind"],"short")
        for kind in ("short","impedance"):
            explicit = solve(load_kind=kind,R_L=0,X_L=0,length_wavelengths=.183,Z0=37)
            self.assertEqual(explicit,solve(R_L=0,X_L=0,length_wavelengths=.183,Z0=37))
        self.assertEqual(solve(Z0=50)["Z_in_real"],100)
        for extra in ({"load_kind":"open","R_L":100}, {"load_kind":"open","X_L":0},
                      {"load_kind":"short","R_L":1}, {"load_kind":"short","X_L":1},
                      {"load_kind":"unknown"}, {"load_kind":None}, {"load_kind":"open","R_L":math.inf}):
            with self.subTest(extra=extra):
                self.assertFalse(solve(Z0=50,**extra)["verified"])
