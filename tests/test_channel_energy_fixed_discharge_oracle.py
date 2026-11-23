"""Independent fixed-Q critical-section and energy-branch oracle."""
import math
import unittest
from decimal import Decimal as D, localcontext

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def route(b, z, q, g, e):
    return run_router_conservation_cad_verifier({"mode": "civil_geotech", "spec": {
        "calculation": "channel_specific_energy", "bottom_width_m": b, "side_slope_z": z,
        "Q_m3_s": q, "gravity_m_s2": g, "specific_energy_m": e}})


def bisect(increasing, lo, hi):
    # Each dimensional residual changes sign from negative to positive.
    for _ in range(480):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        f = increasing(mid)
        if f == 0:
            return mid
        if f < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def original_equations(b, z, q, g, e, exact_contact=None):
    b, z, q, g, e = map(D, (b, z, q, g, e))
    def area(y):
        return y * (b + z*y)
    def top(y):
        return b + 2*z*y
    def kinetic(y):
        return (q/area(y))**2 / (2*g)
    # The unique minimum of specific energy at FIXED discharge satisfies
    # g A^3 = Q^2 T. This differs from maximizing Q at fixed energy.
    hi = e
    while g*area(hi)**3 < q*q*top(hi):
        hi *= 2
    yc = (D(exact_contact) if exact_contact is not None else
          bisect(lambda y: g*area(y)**3-q*q*top(y), D(0), hi))
    if exact_contact is not None:
        assert g*area(yc)**3 == q*q*top(yc)
    ec = yc + kinetic(yc)
    if ec > e:
        return []
    if ec == e:
        coords = [(yc, kinetic(yc))]
    else:
        y = bisect(lambda y: e - y - kinetic(y), D(0), yc)
        # Solve directly for h to retain a deep branch whose reported y can
        # equal E. Never obtain the deep kinetic head by subtracting y from E.
        h = bisect(lambda h: h - kinetic(e-h), D(0), e-yc)
        coords = [(y, kinetic(y)), (e-h, h)]
    results = []
    for y, h in coords:
        a = area(y)
        fr2 = (q/a)**2 * top(y)/(g*a)
        results.append([y, a, q/a, h, fr2.sqrt(), fr2-1])
    return results


class FixedDischargeEnergyOracle(unittest.TestCase):
    def check_state(self, b, z, q, g, e, expected_count, exact_contact=None):
        out = route(b,z,q,g,e)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["state"], ["below_minimum", "critical", "two_depths"][expected_count])
        ref = original_equations(b,z,q,g,e,exact_contact)
        self.assertEqual(len(ref), expected_count)
        self.assertEqual(len(out["depth_solutions"]), expected_count)
        fields = ("depth_m","area_m2","velocity_m_s","velocity_head_m","froude","froude_squared_minus_one")
        for point, required in zip(out["depth_solutions"], ref):
            self.assertTrue(out["conservation_verified"])
            for key, target in zip(fields, required):
                v = float(target)
                self.assertLessEqual(abs(point[key]-v), 4*math.ulp(v), (key,point[key],v))

    def test_fixed_discharge_critical_neighbor_controls(self):
        # At each exact contact choose Q=A^2, g=A*T. Both signs at Q and its
        # adjacent magnitudes exercise true two-root, double-root and no-root cases.
        with localcontext() as ctx:
            ctx.prec = 140
            for b,z,y in [(2.,3.,1.), (0.,3.,.5), (.5,0.,2.)]:
                a,t = y*(b+z*y), b+2*z*y
                q,g,e = a*a, a*t, y+a/(2*t)
                for iq,n in [(math.nextafter(q,0.),2), (q,1), (math.nextafter(q,math.inf),0)]:
                    for sign in (-1,1):
                        with self.subTest(b=b,z=z,q=sign*iq):
                            self.check_state(b,z,sign*iq,g,e,n,y if n == 1 else None)

    def test_alternate_depths_and_deep_kinetic_complement(self):
        with localcontext() as ctx:
            ctx.prec = 140
            for b,z,g,e,q in [(1.75,.6,9.81,2.3,.43), (0.,1.3,7.9,3.,.02),
                              (3.,0.,10.,2.,2.**-100), (1.5,1.,4.,1.,2.**-70)]:
                for sign in (-1,1):
                    self.check_state(b,z,sign*q,g,e,2)
