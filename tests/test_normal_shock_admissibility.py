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
