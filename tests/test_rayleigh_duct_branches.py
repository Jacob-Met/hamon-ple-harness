"""Fixed-inlet Rayleigh states against the original constant-area balances."""
import math
import unittest
from decimal import Decimal as D, localcontext

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def solve(mach, heat, **kwargs):
    spec = dict(subkind="rayleigh", M1=mach, q_J_kg=heat, T01=300., cp=1000., gamma=1.4)
    return dispatch_wave6_verifier("compressible_acoustics", dict(spec, **kwargs))


def velocity_oracle(mach, heat, gamma, inlet_t, cp, branch):
    # With inlet mass flux fixed, use v=u2/u1 as the unknown. Momentum gives
    # p=P2/P1 and continuity gives T2/T1=p*v. Bisect the original energy
    # equation independently on either side of its stagnation-enthalpy maximum.
    with localcontext() as ctx:
        ctx.prec = 110
        m, q, g, t, c = map(D, (mach, heat, gamma, inlet_t, cp))
        gm2 = g*m*m
        vstar = (1+gm2)/((g+1)*m*m)
        lo, hi = ((vstar, 1+1/gm2) if branch == "supersonic" else (D(0), vstar))
        for _ in range(320):
            v = (lo+hi)/2
            p = 1+gm2*(1-v)
            energy = c*t*((2*p*v+(g-1)*m*m*v*v)/(2+(g-1)*m*m)-1)
            if (energy > q) == (branch == "supersonic"):
                lo = v
            else:
                hi = v
        v = (lo+hi)/2
        p = 1+gm2*(1-v)
        return float(m*(v/p).sqrt()), float(p)


class RayleighBranchTests(unittest.TestCase):
    def test_heat_above_sonic_budget_cannot_certify(self):
        r = solve(2., 156250.)
        self.assertFalse(r["verified"])
        self.assertFalse(r["requested_state_feasible"])
        self.assertTrue(r["choked"])
        self.assertNotIn("M2", r)
        self.assertAlmostEqual(r["reachable_sonic_state"]["heat_J_kg"], 78125., places=8)

    def test_finite_supersonic_cooling_beyond_previous_bracket(self):
        for q in [-114729.78444543749, -90000., 0., 39062.5]:
            with self.subTest(q=q):
                r = solve(2., q)
                expected = velocity_oracle(2., q, 1.4, 300., 1000., "supersonic")
                self.assertTrue(r["verified"] and r["wire_rounding_verified"], r)
                self.assertEqual((r["M2"], r["P2_over_P1"]), expected)
        self.assertGreater(solve(2., -114729.78444543749)["M2"], 99.9)

    def test_both_limits_and_adjacent_finite_cooling(self):
        # These two limiting heats are exactly representable for this inlet.
        p = dict(gamma=2., T01=1., cp=1.)
        sonic, floor = .125, -.15625
        r = solve(2., sonic, **p)
        self.assertTrue(r["verified"] and r["choked"])
        self.assertEqual(r["M2"],1.)
        q = math.nextafter(floor, math.inf)
        r = solve(2., q, **p)
        self.assertTrue(r["verified"], r)
        self.assertEqual((r["M2"],r["P2_over_P1"]),velocity_oracle(2.,q,2.,1.,1.,"supersonic"))
        for q in (floor, math.nextafter(floor, -math.inf)):
            r = solve(2., q, **p)
            self.assertFalse(r["verified"] or r["requested_state_feasible"])
            self.assertNotIn("M2", r)
            self.assertEqual(r["limiting_heat_J_kg"], floor)

    def test_cooling_from_sonic_identifies_all_finite_branches(self):
        params = dict(gamma=2., T01=1., cp=1.)
        unresolved = solve(1., -.0625, **params)
        self.assertFalse(unresolved["verified"])
        self.assertEqual(set(unresolved["available_branches"]), {"subsonic", "supersonic"})
        for branch in ("subsonic", "supersonic"):
            r = solve(1., -.0625, branch=branch, **params)
            self.assertTrue(r["verified"], r)
            self.assertEqual((r["M2"],r["P2_over_P1"]),velocity_oracle(1.,-.0625,2.,1.,1.,branch))
        only_subsonic = solve(1., -.25, **params)
        self.assertTrue(only_subsonic["verified"] and only_subsonic["M2"] < 1)
        self.assertFalse(solve(1., -.25, branch="supersonic", **params)["verified"])
        self.assertFalse(solve(1., .01, **params)["verified"])
        self.assertTrue(solve(1., 0., **params)["verified"])
