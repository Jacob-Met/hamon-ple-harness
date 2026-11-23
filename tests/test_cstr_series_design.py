"""Inverse design controls independent of the backward interval recurrence."""
import math
import unittest
import numpy as np
from scipy.optimize import brentq
from hamon_wave6_physics_verifiers import dispatch_wave6_verifier, solve_reactor_kinetics

HOT = dict(reactor_type="cstr_series", n_stages=2, order=1, C_A0=1.,
           v0=1., T0=300., delta_T_adiabatic=200., k_ref=1., T_ref=350.,
           E_a=80000., gas_phase=False)


def independent_balances(case, design):
    """Check the public, rounded outputs with the defining physical equations."""
    count = case.get("n_stages", 2)
    c0, flow = case.get("C_A0", 1.), case.get("v0", 1.)
    feed = case.get("F_A0", c0*flow)
    t0 = case.get("T0", 300.)
    rise = case.get("delta_T_adiabatic", 0.)
    rise = rise if abs(rise) > 1e-15 else 0.
    eps = case.get("epsilon_A", 0.)
    gas = case.get("gas_phase", abs(eps) > 1e-15)
    order = case.get("order", 1.)
    previous, worst = 0., 0.
    for stage in design["stages"]:
        x, u = stage["conversion_X"], stage["unreacted_fraction"]
        temperature = t0+rise*x if rise else case.get("T", t0)
        if "A_preexp" in case and "E_a" in case:
            rate = case["A_preexp"]*math.exp(-case["E_a"]/(8.31446261815324*temperature))
        elif all(key in case for key in ("k_ref", "T_ref", "E_a")):
            rate = case["k_ref"]*math.exp(-case["E_a"]/8.31446261815324*
                                         (1/temperature-1/case["T_ref"]))
        else:
            rate = case["k"]
        concentration = c0*u/(1+eps*x)*(t0/temperature if gas else 1.)
        consumed = feed*(x-previous)
        reacted = design["volume_m3"]/count*rate*concentration**order
        scale = max(abs(consumed), abs(reacted))
        worst = max(worst, abs(consumed-reacted)/scale if scale else 0.)
        if not (0 <= previous <= x <= case["conversion"]):
            raise AssertionError("nonphysical stage order")
        if not math.isclose(stage["exit_T_K"], temperature, rel_tol=2e-15):
            raise AssertionError("energy balance")
        previous = x
    if previous != case["conversion"]:
        raise AssertionError("wrong final conversion")
    return worst


class CstrSeriesDesign(unittest.TestCase):
    def all_designs(self, **overrides):
        case = dict(HOT, **overrides)
        result = solve_reactor_kinetics(dict(case, design_branch="all"))
        self.assertTrue(result["verified"])
        for design in result["designs"]:
            self.assertLess(independent_balances(case, design), 2e-11)
        return result

    def test_requested_conversions_do_not_collapse_to_branch_jump(self):
        volumes = []
        for target in (.01, .05, .1, .2, .4, .6, .8, .99):
            with self.subTest(target=target):
                result = self.all_designs(conversion=target)
                volumes.append(tuple(row["volume_m3"] for row in result["designs"]))
        self.assertEqual(len(set(volumes)), 8)

    def test_independent_forward_all_paths_reach_each_design(self):
        for target in (.2, .99):
            inverse = self.all_designs(conversion=target)
            for design in inverse["designs"]:
                forward = solve_reactor_kinetics(dict(HOT, volume=design["volume_m3"],
                                                      steady_state_branch="all"))
                error = min(max(abs(a["conversion_X"]-b["conversion_X"])
                                for a, b in zip(path["stages"], design["stages"]))
                            for path in forward["steady_states"])
                self.assertLess(error, 1e-10)

    def test_two_tank_stationary_polynomial_finds_same_three_designs(self):
        # Eliminate volume: x*r(t)=(t-x)*r(x). Stationary cuts of this
        # logarithmic equation bound every monotonic branch independently.
        t = .99
        P = np.polynomial.Polynomial
        x, temperature = P([0., 1.]), P([300., 200.])
        numerator = (t*(1-x)*temperature**2 + x*(t-x)*temperature**2
                     - (80000./8.31446261815324)*200.*x*(t-x)*(1-x))
        stationary = sorted(float(z.real) for z in numerator.roots()
                            if abs(z.imag) < 1e-10 and 0 < z.real < t)
        self.assertEqual(len(stationary), 2)
        def log_rate(z):
            return -(80000./8.31446261815324)*(1/(300.+200.*z)-1/350.)+math.log1p(-z)
        def equation(z):
            return math.log(z)-math.log(t-z)+log_rate(t)-log_rate(z)
        cuts = [math.nextafter(0., 1.), *stationary, math.nextafter(t, 0.)]
        roots = [brentq(equation, a, b, xtol=1e-14)
                 for a, b in zip(cuts, cuts[1:]) if equation(a)*equation(b) < 0]
        expected = sorted(2*(t-z)/math.exp(log_rate(t)) for z in roots)
        actual = [row["volume_m3"] for row in self.all_designs(conversion=t)["designs"]]
        self.assertEqual(len(actual), 3)
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a/b, 1., places=11)

    def test_ambiguous_design_requires_an_explicit_selector(self):
        with self.assertRaisesRegex(ValueError, "multiple physical"):
            solve_reactor_kinetics(dict(HOT, conversion=.99))
        all_rows = self.all_designs(conversion=.99)["designs"]
        for index, expected in enumerate(all_rows):
            selected = solve_reactor_kinetics(dict(HOT, conversion=.99, design_branch=index))
            self.assertEqual(selected["volume_m3"], expected["volume_m3"])
        for selector in (True, -1, 3, 1., "0"):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                solve_reactor_kinetics(dict(HOT, conversion=.99, design_branch=selector))
        with self.assertRaisesRegex(ValueError, "use design_branch"):
            solve_reactor_kinetics(dict(HOT, conversion=.2, steady_state_branch="all"))

    def test_isothermal_first_order_closed_form_including_dyadic_root(self):
        for stages in (1, 2, 3, 4, 8):
            case = dict(reactor_type="cstr_series", n_stages=stages, order=1,
                        conversion=.875, C_A0=2., v0=.1, k=.4)
            result = solve_reactor_kinetics(case)
            expected = stages*.1/.4*((1-.875)**(-1/stages)-1)
            self.assertAlmostEqual(result["volume_m3"]/expected, 1., places=13)
            self.assertEqual(result["design_count"], 1)
            self.assertLess(independent_balances(case, result), 2e-12)

    def test_zero_order_and_single_stage_endpoint(self):
        for stages in (1, 2, 3, 8):
            case = dict(reactor_type="cstr_series", n_stages=stages, order=0,
                        conversion=.5, C_A0=1., v0=1., k=3.)
            result = solve_reactor_kinetics(case)
            self.assertAlmostEqual(result["volume_m3"], 1/6, places=15)
            self.assertLess(independent_balances(case, result), 2e-12)
        for target in (0., 1.):
            result = solve_reactor_kinetics(dict(case, conversion=target))
            self.assertEqual(result["conversion_X"], target)
            self.assertTrue(result["verified"])

    def test_gas_phase_fractional_order_and_rate_precedence(self):
        case = dict(reactor_type="cstr_series", n_stages=3, order=.7, conversion=.72,
                    C_A0=2.4, v0=.3, F_A0=.81, T0=290., T=320.,
                    delta_T_adiabatic=35., epsilon_A=.6, gas_phase=True,
                    A_preexp=200., E_a=12000., k_ref=999., T_ref=800., k=999.)
        result = solve_reactor_kinetics(dict(case, design_branch="all"))
        self.assertEqual(result["design_count"], 1)
        for design in result["designs"]:
            self.assertLess(independent_balances(case, design), 2e-11)

    def test_unrepresentable_space_time_refuses_instead_of_verifying_infinity(self):
        case = dict(reactor_type="cstr_series", n_stages=2, k=1., order=0,
                    C_A0=1., F_A0=1e300, v0=1e-100, conversion=.5)
        with self.assertRaisesRegex(ValueError, "space time is not representable"):
            solve_reactor_kinetics(case)

    def test_dispatch_reaches_inverse_design_and_rejects_invalid_inputs(self):
        result = dispatch_wave6_verifier({"domain":"reactor_kinetics",
                                          "params":dict(HOT, conversion=.2)})
        self.assertEqual(result["design_count"], 1)
        for change in ({"n_stages":True}, {"n_stages":0}, {"n_stages":1.5},
                       {"conversion":-1.}, {"conversion":1.}, {"conversion":float("nan")},
                       {"k_ref":0.}, {"F_A0":0.}, {"epsilon_A":-1.},
                       {"delta_T_adiabatic":-400.}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_reactor_kinetics({**HOT, "conversion":.2, **change})


if __name__ == "__main__":
    unittest.main()
