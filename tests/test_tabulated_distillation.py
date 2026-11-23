"""Independent graphical construction using line intersections and bisection."""
import math
import unittest

import numpy as np
from scipy.optimize import brentq

TABLE = [[0, 0], [.25, .46], [.5, .63], [.7, .72], [.85, .94], [1, 1]]
BARRIER = [[0, 0], [.25, .38], [.5, .5], [.75, .8], [1, 1]]
PARAMS = dict(subkind="distillation", z_F=.5, x_D=.95, x_B=.05, F_mol_s=100.)


def graphical(table, q, reflux, z=.5, xd=.95, xb=.05):
    x, y = np.array(table).T
    d = (z-xb)/(xd-xb)
    b = 1-d
    liquid, vapor = reflux*d, (reflux+1)*d
    strip_v = vapor-(1-q)
    assert min(liquid, strip_v) > 0
    def rect(value):
        return (liquid*value+d*xd)/vapor
    def strip(value):
        return ((strip_v+b)*value-b*xb)/strip_v
    switch = brentq(lambda value: rect(value)-strip(value), xb, xd)
    knots = np.unique([xb, xd, switch, *[k for k in x if xb<k<xd]])
    gaps = np.interp(knots,x,y)-np.minimum(rect(knots),strip(knots))
    limit = int(np.argmin(gaps))
    if gaps[limit] <= 1e-13:
        return None, knots[limit], gaps[limit]
    current = xd
    target_y = xd
    liquids = []
    for i in range(2000):
        nxt = brentq(lambda t: float(np.interp(t,x,y))-target_y, 0, 1, xtol=1e-15)
        liquids.append(nxt)
        if nxt <= xb:
            return liquids, i+(current-xb)/(current-nxt), switch
        current = nxt
        target_y = min(rect(current),strip(current))
    raise AssertionError("oracle contact budget exhausted")


def solve(**kw):
    from hamon_wave7_engineering_verifiers import solve_electrochem_or_separation
    return solve_electrochem_or_separation({**PARAMS, **kw})


class MeasuredEquilibrium(unittest.TestCase):
    def test_interior_pinch_before_any_contact(self):
        for r in (4., 11.5, 11.49):
            self.assertIsNone(graphical(TABLE,.5,r)[0])
            result = solve(equilibrium_table=TABLE,q=.5,R=r)
            self.assertFalse(result["verified"])
            self.assertAlmostEqual(result["R_min"],11.5)
            self.assertAlmostEqual(result["limiting_composition"],.7)
            self.assertNotIn("N_stages_integer",result)
        self.assertFalse(solve(equilibrium_table=BARRIER,q=.5,R=1e10)["verified"])

    def test_graphical_stages_nonunity_feed_and_minimum_reflux(self):
        for q in (-.5,0.,.5,1.,1.6):
            for r in (12.,14.,23.):
                with self.subTest(q=q,R=r):
                    liquids, stages, switch = graphical(TABLE,q,r)
                    self.assertIsNotNone(liquids)
                    result = solve(equilibrium_table=TABLE,q=q,R=r)
                    self.assertTrue(result["verified"],result)
                    self.assertEqual(result["N_stages_integer"],len(liquids))
                    self.assertEqual(result["feed_stage"],next(i for i,x in enumerate(liquids,1) if x<=switch))
                    self.assertAlmostEqual(result["N_theoretical"],stages,places=10)
                    self.assertAlmostEqual(result["feed_intersection_x"],switch)
                    self.assertAlmostEqual(result["R_min"],11.5)

    def test_dense_constant_volatility_control(self):
        alpha=2.5
        coords=np.linspace(0,1,201)
        table=[[float(x),float(alpha*x/(1+(alpha-1)*x))] for x in coords]
        for q in (0,.5,1):
            ideal=solve(alpha=alpha,q=q,R=4)
            observed=solve(equilibrium_table=table,q=q,R=4)
            self.assertTrue(ideal["verified"] and observed["verified"])
            self.assertEqual(ideal["N_stages_integer"],observed["N_stages_integer"])
            self.assertLess(abs(ideal["N_theoretical"]-observed["N_theoretical"]),.003)
            self.assertLess(abs(ideal["R_min"]-observed["R_min"]),.001)

    def test_reject_ambiguous_or_invalid_relation(self):
        for table in ([[0,0],[.5,.4],[.5,.6],[1,1]],[[0,0],[.3,.6],[.5,.6],[1,1]],
                      [[0,0],[.5,math.nan],[1,1]],[[False,0],[1,1]],[[.1,0],[1,1]]):
            self.assertFalse(solve(equilibrium_table=table,q=.5,R=4)["verified"])
        self.assertFalse(solve(alpha=2.5,equilibrium_table=BARRIER,q=.5,R=4)["verified"])

    def test_three_feed_equilibrium_crossings(self):
        xs = [i/10 for i in range(11)]
        ys = [0,.24,.27,.37,.50,.77,.79,.83,.88,.92,1]
        table = list(map(list, zip(xs,ys)))
        q = -3
        def feed_balance(x):
            return q*x+(1-q)*np.interp(x,xs,ys)-.5
        roots = [brentq(feed_balance,a,b) for a,b in zip(xs,xs[1:])
                 if feed_balance(a)*feed_balance(b)<0]
        np.testing.assert_allclose(roots,[5/66,17/90,.22])
        denied = solve(equilibrium_table=table,q=q,R=9.4)
        self.assertFalse(denied["verified"])
        self.assertAlmostEqual(denied["R_min"],66/7)
        self.assertAlmostEqual(denied["limiting_composition"],roots[-1])
        approved = solve(equilibrium_table=table,q=q,R=10)
        profile, count, _ = graphical(table,q,10)
        self.assertTrue(approved["verified"])
        self.assertEqual(approved["N_stages_integer"],len(profile))
        self.assertAlmostEqual(approved["N_theoretical"],count)

    def test_zero_circulation_and_section_limits(self):
        table = [[0,0],[.25,.75],[1,1]]
        args = dict(equilibrium_table=table,x_B=.25,z_F=.5,x_D=.75)
        for q,r,section in [(1,0,"stripping_only"),(0,1,"rectifying_only"),
                            (.5,0,"single_equilibrium_flash")]:
            result = solve(**args,q=q,R=r)
            self.assertTrue(result["verified"],result)
            self.assertEqual(result["section_mode"],section)
            self.assertEqual(result["N_stages_integer"],1)
        inconsistent = solve(**{**args,"equilibrium_table":[[0,0],[.25,.76],[1,1]]},q=.5,R=0)
        self.assertFalse(inconsistent["verified"])
        self.assertIn("flash",inconsistent["error"])
