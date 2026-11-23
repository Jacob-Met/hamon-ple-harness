"""Independent integration of the positive entropy production along the Hugoniot."""
import math
import unittest
from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


class ShockEntropyIntegral(unittest.TestCase):
    def run_shock(self, mach, **gas):
        return run_router_conservation_cad_verifier(dict(mode="compressible_acoustics",
            spec=dict(subkind="normal_shock", M1=mach, P1=73123., T1=227., **gas)))

    def test_entropy_and_original_fluxes_from_one_gas(self):
        ctx = mp.clone()
        ctx.dps = 125
        gases = [
            (dict(gamma=1.5, R=40.), ctx.mpf("1.5"), ctx.mpf(40)),
            (dict(gamma=1.25, cp=1000.), ctx.mpf("1.25"), ctx.mpf(200)),
            (dict(gamma=1.4, cp=700., R=200.), ctx.mpf(7)/5, ctx.mpf(200)),
        ]
        for spec,g,gas in gases:
            for mach in (math.nextafter(1., math.inf), 1.015625, 3.75, 1e24):
                with self.subTest(gas=spec, mach=mach):
                    result = self.run_shock(mach, **spec)
                    self.assertTrue(result["verified"], result)
                    # Differentiate the Rankine-Hugoniot entropy along y=log(M1²).
                    # This nonnegative integrand is regular at y=0 and for large y.
                    def production(y):
                        q = ctx.expm1(y)
                        return 2*g*q*q/((g+1+2*g*q)*(g+1+(g-1)*q))
                    s = ctx.quad(production, [0, 2*ctx.log(mach)])
                    for key,ref in [("entropy_increase_J_kgK",gas*s),
                                    ("P02_over_P01",ctx.exp(-s)),
                                    ("stagnation_pressure_loss_fraction",-ctx.expm1(-s))]:
                        actual = ctx.mpf(result[key])
                        self.assertGreater(actual,0)
                        self.assertLess(abs(actual/ref-1), ctx.mpf("2e-15"))
                    def fluxes(p,t,m):
                        rho = p/(gas*t)
                        speed = m*ctx.sqrt(g*gas*t)
                        return (rho*speed, p+rho*speed**2, g*gas*t/(g-1)+speed**2/2)
                    upstream = fluxes(ctx.mpf(73123),ctx.mpf(227),ctx.mpf(mach))
                    downstream = fluxes(*(ctx.mpf(result[k]) for k in ("P2_Pa","T2_K","M2")))
                    for a,b in zip(upstream,downstream):
                        self.assertLess(abs(a/b-1),ctx.mpf("1e-15"))
                    self.assertEqual(result["gamma_effective"], float(g))
                    self.assertEqual(result["R_effective_J_kgK"], float(gas))
                    self.assertEqual(result["cp_J_kgK"], float(g*gas/(g-1)))

    def test_caloric_conflict_and_exact_sonic_endpoint(self):
        for values in [dict(gamma=1.5,cp=700.,R=200.),dict(cp=150.,R=200.)]:
            with self.subTest(values=values):
                self.assertFalse(self.run_shock(2.,**values)["verified"])
        sonic = self.run_shock(1.,cp=700.,R=200.)
        self.assertTrue(sonic["verified"],sonic)
        self.assertFalse(sonic["shock_present"])
        self.assertEqual((sonic["M2"],sonic["P02_over_P01"],sonic["entropy_increase_J_kgK"]),
                         (1.,1.,0.))
