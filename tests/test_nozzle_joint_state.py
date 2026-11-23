"""Independent mass-flux and stagnation-pressure constraints on one nozzle state."""
import math
import unittest
from decimal import Decimal as D, localcontext

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def route(**observations):
    spec = dict(subkind="isentropic", gamma=3., **observations)
    return run_router_conservation_cad_verifier({"mode":"compressible_acoustics","spec":spec})


def measurements(m):
    # For gamma=3, the sonic/local mass-flux ratio is (M+1/M)/2.
    return {"mach":float(m), "area_ratio":float((m+1/m)/2),
            "p0_over_p":float((1+m*m)**D("1.5"))}


def interval(point):
    v = float(point)
    return ((D(v)+D(math.nextafter(v,0.)))/2,
            (D(v)+D(math.nextafter(v,math.inf)))/2)


def common_mach(given, sub):
    lo, hi = interval(given["mach"])
    amin, amax = interval(given["area_ratio"])
    pmin, pmax = interval(given["p0_over_p"])
    def area_root(a):
        r = a+(a*a-1).sqrt()
        return 1/r if sub else r
    def pressure_root(p):
        return (p**(D(2)/3)-1).sqrt()
    ar = sorted(map(area_root,(max(D(1),amin),amax)))
    return max(lo,ar[0],pressure_root(pmin)) <= min(hi,ar[1],pressure_root(pmax))


class CommonNozzleState(unittest.TestCase):
    def test_area_pair_is_disambiguated_by_the_supplied_pressure(self):
        with localcontext() as ctx:
            ctx.prec=125
            for m in map(D,(".5","2")):
                s=measurements(m)
                r=route(area_ratio=s["area_ratio"],p0_over_p=s["p0_over_p"])
                self.assertTrue(r["verified"],r)
                self.assertEqual(r["mach"],float(m))
                self.assertEqual(r["P0_over_P"],s["p0_over_p"])
                self.assertEqual(r["A_over_Astar"],s["area_ratio"])
                t,p,v=map(D,[r["T_static_K"],r["P_static_Pa"],r["velocity_m_s"]])
                flow=p/(D(287.05)*t)*v*D(.001)*D(s["area_ratio"])
                self.assertLess(abs(flow/D(r["mdot_choked_kg_s"])-1),D("5e-15"))

    def test_contradictory_observations_and_branch_never_certify(self):
        pressure=1.25**1.5
        for p in [dict(mach=2.,area_ratio=1.25,p0_over_p=pressure),
                  dict(mach=.5,area_ratio=2.,p0_over_p=pressure),
                  dict(mach=.5,area_ratio=1.25,p0_over_p=pressure,regime="supersonic"),
                  dict(p0_over_p=pressure,regime="sup"),dict(mach=2.,regime="sub")]:
            with self.subTest(p=p):
                self.assertFalse(route(**p)["verified"])

    def test_independently_rounded_measurements_share_one_state(self):
        with localcontext() as ctx:
            ctx.prec=140
            for nominal in (.34,.62,1.25,1.8,3.,8.):
                for side in (-1,1):
                    m=D(nominal)+side*D(".49")*D(math.ulp(nominal))
                    good=measurements(m)
                    with self.subTest(m=nominal,side=side):
                        self.assertTrue(common_mach(good,m<1))
                        r=route(**good)
                        self.assertTrue(r["verified"],r)
                        for key,out in [("mach","mach"),("area_ratio","A_over_Astar"),("p0_over_p","P0_over_P")]:
                            self.assertEqual(good[key],r[out])
                        bad=dict(good,p0_over_p=good["p0_over_p"]+8*math.ulp(good["p0_over_p"]))
                        self.assertFalse(common_mach(bad,m<1))
                        self.assertFalse(route(**bad)["verified"])

    def test_every_supplied_scalar_is_validated(self):
        for p in [dict(mach=.5,area_ratio=.9),dict(mach=2.,p0_over_p=1.),
                  dict(mach=2.,area_ratio=1.25,p0_over_p=float("inf")),
                  dict(p0_over_p=2.,regime="unknown")]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                route(**p)
