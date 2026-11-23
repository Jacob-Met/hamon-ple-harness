"""Independent mole/energy checks for the constant-coefficient PFR design equation."""
import math
import unittest
from decimal import Decimal, localcontext

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def solve(**params):
    return dispatch_wave6_verifier("reactor_kinetics", {"reactor_type": "pfr", "k": 1., **params})


class PowerLawPfr(unittest.TestCase):
    def test_tiny_dose_and_trace_remaining(self):
        for volume in (1e-300, 1e-26, .25, 30., 40., 690.):
            with self.subTest(volume=volume), localcontext() as c:
                c.prec = 420
                u = (-Decimal(volume)).exp()
                result = solve(volume=volume, delta_T_adiabatic=72.5, T0=285.)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["conversion_X"], float(1-u))
                self.assertEqual(result["unreacted_fraction"], float(u))
                self.assertEqual(result["exit_T_K"], float(Decimal(285)+Decimal("72.5")*(1-u)))

    def test_forward_and_inverse_near_first_order_and_gas_feed(self):
        for n in (0., .5, math.nextafter(1., 0.), 1., math.nextafter(1., 2.), 2., 3.):
            with self.subTest(order=n), localcontext() as c:
                c.prec = 120
                # Inlet concentration at the specified gas operating temperature
                # is C_A0*T0/T. Feed flow is given independently of that state.
                params = dict(order=n, C_A0=3., k=.75, F_A0=5., v0=2.,
                              gas_phase=True, T0=280., T=420.)
                a, x = Decimal(n)-1, Decimal("0.625")
                coeff = Decimal(".75")*(Decimal(2)**Decimal(n))/5
                integral = ((1-x)**(-a)-1)/a if a else -(1-x).ln()
                sized = solve(**params, conversion=float(x), volume=-1.)
                self.assertTrue(sized["verified"], sized)
                self.assertEqual(sized["volume_m3"], float(integral/coeff))
                inverse = solve(**params, volume=sized["volume_m3"])
                self.assertTrue(inverse["verified"], inverse)
                self.assertAlmostEqual(inverse["conversion_X"], float(x), delta=2e-16)
                self.assertEqual(inverse["exit_T_K"],420.)

    def test_finite_depletion_and_its_neighbors(self):
        for n, limit in ((-1., .5), (0.,1.), (.5,2.)):
            with self.subTest(order=n):
                at = solve(order=n, volume=limit)
                self.assertTrue(at["verified"], at)
                self.assertEqual(at["unreacted_fraction"], 0.)
                self.assertEqual(solve(order=n, conversion=1.)["volume_m3"], limit)
                below = solve(order=n, volume=math.nextafter(limit,0.))
                self.assertGreater(below["unreacted_fraction"], 0.)
                self.assertTrue(below["verified"], below)
                if n > 0:
                    self.assertEqual(solve(order=n, volume=3.)["unreacted_fraction"], 0.)
                else:
                    with self.assertRaisesRegex(ValueError, "depletion"):
                        solve(order=n, volume=math.nextafter(limit,math.inf))

    def test_near_zero_order_retains_which_side_of_depletion_applies(self):
        # At dimensionless exposure one, the original integrated equation is
        # U**(1-n)=n. A tiny positive n therefore leaves a nonzero fraction.
        for n in (2.**-500, math.ulp(0.)):
            with self.subTest(order=n), localcontext() as c:
                c.prec = 440
                d = Decimal(n)
                expected = (d.ln()/(1-d)).exp()
                result = solve(volume=1., order=n)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["unreacted_fraction"], float(expected))
                with self.assertRaisesRegex(ValueError, "depletion"):
                    solve(volume=1., order=-n)

    def test_invalid_or_unrepresentable_states_do_not_certify(self):
        for params in (dict(order=0.,conversion=1.2), dict(conversion=-.1),
                       dict(volume=-1.), dict(conversion=.5,epsilon_A=-2.),
                       dict(conversion=.8,delta_T_adiabatic=-400.),
                       dict(conversion=1.,order=1.), dict(conversion=1.,order=2.),
                       dict(volume=750.), dict(volume=2.,k=0.)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                solve(**params)
        # The nearest subnormal remaining fraction here loses more than
        # 0.1 percent even though the reported conversion rounds to one.
        self.assertFalse(solve(volume=740.)["verified"])
        self.assertEqual(solve(volume=0.)["unreacted_fraction"], 1.)
        smallest = math.ulp(0.)
        # Both the minimum design volume and residence time are public values.
        self.assertFalse(solve(order=.5, conversion=1., F_A0=3*smallest, k=4.)["verified"])
        with self.assertRaisesRegex(ValueError, "space time"):
            solve(volume=1e-250, v0=1e150, F_A0=1.)

    def test_cooling_and_contraction_only_through_the_reached_outlet(self):
        for rise, x in ((-400.,.2), (-300.,.5)):
            sized = solve(conversion=x, T0=300., delta_T_adiabatic=rise)
            inverse = solve(volume=sized["volume_m3"], T0=300., delta_T_adiabatic=rise)
            for r in (sized, inverse):
                self.assertTrue(r["verified"],r)
                self.assertAlmostEqual(r["exit_T_K"],300.+rise*x)
        tail = solve(volume=400., T0=300., delta_T_adiabatic=-300.)
        self.assertTrue(tail["verified"],tail)
        self.assertAlmostEqual(tail["exit_T_K"]/(300*math.exp(-400)),1.,delta=2e-15)
        with self.assertRaisesRegex(ValueError,"temperature"):
            solve(order=0.,volume=.75,T0=300.,delta_T_adiabatic=-400.)
        # Contracting gas stops having positive bulk volume at X=.5.
        # Its first-quarter design is valid and reversible on the numeric route.
        mixed = dict(k_ref=1., T_ref=300., E_a=5000., delta_T_adiabatic=-350., epsilon_A=-2.)
        r = solve(conversion=.25, **mixed)
        inv = solve(volume=r["volume_m3"], **mixed)
        self.assertTrue(r["verified"] and inv["verified"], (r,inv))
        self.assertAlmostEqual(inv["conversion_X"],.25)

    def test_numerical_variable_rate_route_cannot_certify_zero_consumption(self):
        # Heating changes this Arrhenius rate, retaining the quadrature route.
        p = dict(k_ref=1., T_ref=300., E_a=5000., T0=300., delta_T_adiabatic=35.)
        result = solve(volume=1e-26, **p)
        if result["conversion_X"] == 0:
            self.assertFalse(result["verified"])
        ordinary = solve(conversion=.3, **p)
        self.assertTrue(ordinary["verified"], ordinary)
        self.assertAlmostEqual(ordinary["exit_T_K"],310.5)
