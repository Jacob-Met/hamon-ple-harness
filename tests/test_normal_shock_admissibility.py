import math
import unittest
from decimal import Decimal, localcontext

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def shock(mach=2., **inputs):
    return run_router_conservation_cad_verifier({
        "mode": "compressible_acoustics",
        "spec": {"subkind": "normal_shock", "M1": mach, **inputs},
    })


class NormalShockAdmissibilityTests(unittest.TestCase):
    def test_rejects_negative_absolute_pressure_and_heat_capacity(self):
        for values in ({"P1": -101325.}, {"gamma": .8}):
            with self.subTest(values=values):
                self.assertFalse(shock(1.1, **values)["verified"])

    def test_weak_shock_retains_positive_stagnation_loss(self):
        mach = math.nextafter(1., math.inf)
        r = shock(mach, gamma=1.4, R=287.05)
        self.assertTrue(r["verified"], r)
        self.assertLessEqual(r["P02_over_P01"], 1.)
        with localcontext() as ctx:
            ctx.prec = 170
            g, gas, m = map(Decimal.from_float, (1.4, 287.05, mach))
            # Static downstream pressure follows from momentum after the
            # energy/mass compression ratio fixes the downstream velocity.
            rho = (g+1)*m*m/(2+(g-1)*m*m)
            p = 1+g*m*m*(1-1/rho)
            s = g/(g-1)*(p/rho).ln()-p.ln()
            self.assertEqual(r["entropy_increase_J_kgK"], float(gas*s))
            self.assertEqual(r["stagnation_pressure_loss_fraction"], float(1-(-s).exp()))
            self.assertEqual(r["P02_over_P01"], float((-s).exp()))

    def test_sonic_limit_and_subsonic_refusal(self):
        r = shock(1., P1=83200., T1=270.)
        self.assertTrue(r["verified"], r)
        self.assertFalse(r["shock_present"])
        self.assertEqual(r["shock_state"], "sonic_limit")
        for key in ("M2", "P2_over_P1", "rho2_over_rho1", "T2_over_T1", "P02_over_P01"):
            self.assertEqual(r[key], 1.)
        self.assertEqual(r["entropy_increase_J_kgK"], 0.)
        self.assertEqual(r["stagnation_pressure_loss_fraction"], 0.)
        self.assertFalse(shock(math.nextafter(1., 0.))["verified"])

    def test_static_states_replay_mass_momentum_and_total_energy(self):
        cases = [(1.2, 1.3, 250., 90e3, 275.), (3., 5/3, 190., 1e240, 1e-220),
                 (1e50, 1.4, 287., 1e-200, 1e-200),
                 (1.4, math.nextafter(1., math.inf), 80., 18000., 340.)]
        for mach, gamma, gas, p1, t1 in cases:
            with self.subTest(mach=mach, gamma=gamma):
                r = shock(mach, gamma=gamma, R=gas, P1=p1, T1=t1)
                self.assertTrue(r["verified"], r)
                self.assertTrue(r["shock_present"])
                self.assertGreater(r["entropy_increase_J_kgK"], 0.)
                self.assertGreater(r["stagnation_pressure_loss_fraction"], 0.)
                with localcontext() as ctx:
                    ctx.prec = 180
                    m, g, gas_d, p, t = map(Decimal.from_float, (mach,gamma,gas,p1,t1))
                    p2,t2,m2 = map(Decimal.from_float, (r["P2_Pa"],r["T2_K"],r["M2"]))
                    density1, density2 = p/(gas_d*t), p2/(gas_d*t2)
                    v1, v2 = m*(g*gas_d*t).sqrt(), m2*(g*gas_d*t2).sqrt()
                    cp = g*gas_d/(g-1)
                    pairs = ((density1*v1, density2*v2),
                             (p+density1*v1*v1, p2+density2*v2*v2),
                             (cp*t+v1*v1/2, cp*t2+v2*v2/2))
                    for upstream, downstream in pairs:
                        self.assertLess(abs(upstream-downstream)/upstream, Decimal("8e-16"))
                    # Independent compression and entropy from normal mass,
                    # momentum and a calorically perfect equilibrium equation.
                    compressed = (g+1)*m*m/(2+(g-1)*m*m)
                    rise = 1+g*m*m*(1-1/compressed)
                    entropy = (rise.ln()-g*compressed.ln())/(g-1)
                    self.assertEqual(r["entropy_increase_J_kgK"], float(gas_d*entropy))
                    self.assertEqual(r["P02_over_P01"], float((-entropy).exp()))

    def test_explicit_domain_and_reporting_limits(self):
        for args in ({"R":0.}, {"T1":-1.}, {"P1":0.}, {"gamma":1.},
                     {"P1":math.inf}, {"T1":math.nan}):
            with self.subTest(args=args):
                self.assertFalse(shock(**args)["verified"])
        self.assertFalse(shock(1e100, gamma=1.4)["verified"])

    def test_all_energy_and_entropy_use_the_same_caloric_gas(self):
        for kwargs in ({"cp": 100., "R": 287.05, "gamma": 1.4},
                       {"cp": 500., "R": 287.05, "gamma": 1.4},
                       {"cp": -1.}):
            self.assertFalse(shock(**kwargs)["verified"])
        no_r = shock(gamma=1.4, cp=900.)
        self.assertTrue(no_r["verified"], no_r)
        with localcontext() as ctx:
            ctx.prec = 120
            g = Decimal.from_float(1.4)
            self.assertEqual(no_r["R_effective_J_kgK"], float(900*(g-1)/g))
        explicit = shock(2., cp=1004.5, R=287., gamma=1.4)
        inferred = shock(2., cp=1004.5, R=287.)
        self.assertTrue(explicit["verified"], explicit)
        self.assertEqual(explicit, inferred)
        self.assertEqual(explicit["cp_J_kgK"], 1004.5)
        self.assertEqual(explicit["R_effective_J_kgK"], 287.)
        self.assertEqual(explicit["gamma_effective"], 1.4)
