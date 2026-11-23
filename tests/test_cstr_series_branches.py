"""Physical trajectories of equal-volume CSTR cascades."""
import math
import unittest
from fractions import Fraction

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


HOT = dict(reactor_type="cstr_series", order=1, C_A0=1., v0=1., T0=300.,
           delta_T_adiabatic=200., k_ref=1., T_ref=350., E_a=80000.,
           gas_phase=False, volume=1.)


def solve(params):
    return dispatch_wave6_verifier("reactor_kinetics", params)


def physical_root(inlet, left, right):
    def residual(x):
        temperature = 300 + 200*x
        rate = math.exp((80000/8.31446261815324)*(1/350-1/temperature))*(1-x)
        return x-inlet-.5*rate

    low_positive = residual(left) > 0
    assert low_positive != (residual(right) > 0)
    for _ in range(54):
        mid = (left+right)/2
        if (residual(mid) > 0) == low_positive:
            left = mid
        else:
            right = mid
    return (left+right)/2


class SeriesBranches(unittest.TestCase):
    def test_one_tank_matches_single_and_needs_explicit_selection(self):
        single = solve(dict(HOT, reactor_type="cstr", steady_state_branch="all"))
        series = solve(dict(HOT, n_stages=1, steady_state_branch="all"))
        self.assertEqual(series["steady_state_count"], 3)
        self.assertEqual([s["stages"][0] for s in series["steady_states"]],
                         single["steady_states"])
        for n in (1, 2):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "multiple physical"):
                solve(dict(HOT, n_stages=n))

    def test_independent_balance_and_temperature_on_all_five_paths(self):
        # Independent sign brackets in the original dimensional mole equation.
        inputs = [physical_root(0., a, b) for a, b in ((0,.03), (.15,.3), (.99,1.))]
        paths = [(inputs[0], physical_root(inputs[0], a, b))
                 for a, b in ((.006,.04), (.1,.3), (.99,1.))]
        paths += [(x, physical_root(x, max(.99,x), 1.)) for x in inputs[1:]]
        states = solve(dict(HOT, n_stages=2, steady_state_branch="all"))
        self.assertTrue(states["verified"])
        self.assertEqual(states["steady_state_count"], len(paths))
        for state, expected in zip(states["steady_states"], sorted(paths, key=lambda pair:pair[1])):
            previous, previous_t = 0., 300.
            for stage, x in zip(state["stages"], expected):
                with self.subTest(path=expected, x=x):
                    self.assertAlmostEqual(stage["conversion_X"], x, delta=6e-15)
                    t, u = stage["exit_T_K"], stage["unreacted_fraction"]
                    self.assertAlmostEqual(t-previous_t, 200*(x-previous), delta=2e-12)
                    consumed = .5 * math.exp(80000/8.31446261815324*(1/350-1/t))*u
                    self.assertAlmostEqual(consumed, x-previous, delta=3e-14)
                    previous, previous_t = x, t
        selected = solve(dict(HOT, n_stages=2, steady_state_branch=3))
        self.assertEqual(selected["stages"], states["steady_states"][3]["stages"])

    def test_complementary_cascade_and_ordinary_isothermal_stages(self):
        for v in (2., 20., 2e100):
            with self.subTest(volume=v):
                result = solve(dict(reactor_type="cstr_series", k=1., volume=v, n_stages=2.))
                self.assertTrue(result["verified"])
                u = 1/(1+Fraction(v)/2)
                self.assertEqual([s["unreacted_fraction"] for s in result["stages"]],
                                 [float(u), float(u*u)])

    def test_depleted_zero_order_and_no_downstream_root(self):
        p = dict(reactor_type="cstr_series", k=1., order=0., volume=1., n_stages=3)
        self.assertTrue(solve(p)["verified"])
        self.assertEqual(solve(p)["conversion_X"], 1.)
        with self.assertRaisesRegex(ValueError, "no physical"):
            solve(dict(p, volume=1.5))
        self.assertEqual([s["conversion_X"] for s in solve(dict(p,volume=0))["stages"]],
                         [0.,0.,0.])
        for n in (0, -1, 1.5, True):
            with self.subTest(stages=n), self.assertRaises(ValueError):
                solve(dict(p,n_stages=n))

    def test_exact_upstream_rational_at_a_downstream_fold(self):
        # Per tank (X-Xin)(1-X)=4/25. Stage one can end at 1/5 or 4/5.
        # Only 1/5 has a continuation: a repeated root at X=3/5.
        p = dict(reactor_type="cstr_series", n_stages=2, volume=2., k=4.,
                 F_A0=25., order=-1., steady_state_branch="all")
        result = solve(p)
        self.assertTrue(result["verified"])
        self.assertEqual(result["steady_state_count"], 1)
        self.assertEqual([r["conversion_X"] for r in result["steady_states"][0]["stages"]],
                         [.2, .6])
        self.assertEqual(solve(dict(p, volume=math.nextafter(2.,0.)))["steady_state_count"], 2)
        with self.assertRaisesRegex(ValueError, "no physical"):
            solve(dict(p, volume=math.nextafter(2.,math.inf)))
