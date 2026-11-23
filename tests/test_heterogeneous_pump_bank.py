import math
import unittest
from decimal import Decimal, localcontext

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def table(*points):
    return [{"flow_m3_s": q, "head_m": h} for q, h in points]


A = table((0.,32.),(1/32,0.))
B = table((0.,16.),(1/16,0.))


def bank(static, resistance=4096., curves=None, **opts):
    spec = dict(calculation="pump_operating_point", pump_curves=[A,B] if curves is None else curves,
                check_valves=True, static_head_m=static, system_resistance_s2_m5=resistance)
    return run_router_conservation_cad_verifier({"mode":"thermal_fluid","spec":dict(spec,**opts)})


class HeterogeneousPumpBankTests(unittest.TestCase):
    def test_both_discharging_one_closed_and_exact_opening(self):
        for static, head, flows, states in (
                (6.9375,12.,[.01953125,.015625],["open","open"]),
                (23.75,24.,[.0078125,0.],["open","closed"]),
                (15.,16.,[.015625,0.],["open","at_shutoff"])):
            with self.subTest(head=head):
                r = bank(static)
                self.assertTrue(r["ok"],r)
                point, = r["operating_points"]
                self.assertEqual(point["head_m"],head)
                self.assertEqual(point["total_flow_m3_s"],sum(flows))
                self.assertEqual([p["flow_m3_s"] for p in point["pumps"]],flows)
                self.assertEqual([p["check_valve_state"] for p in point["pumps"]],states)

    def test_rounding_the_header_cannot_close_a_barely_open_valve(self):
        # At H=16, A supplies 1/64 m³/s and K Q²=8. Perturb only Hstatic.
        with localcontext() as ctx:
            ctx.prec=160
            for static, m, k in ((math.nextafter(8.,0.),5,96),
                                  (math.nextafter(8.,math.inf),1,32)):
                s=Decimal.from_float(static)
                disc=Decimal(1024)**2+4*m*32768*(k-m*s)
                total=2*(k-m*s)/(1024+disc.sqrt())
                h=s+32768*total*total
                margin=16-h
                r=bank(static,32768.)
                point,=r["operating_points"]
                self.assertEqual(point["head_m"],16.)
                self.assertEqual(point["total_flow_m3_s"],float(total))
                weak=point["pumps"][1]
                self.assertEqual(weak["flow_m3_s"],float(max(Decimal(0),margin/256)))
                self.assertEqual(weak["shutoff_head_minus_header_m"],float(margin))
                self.assertEqual(weak["check_valve_state"],"open" if static<8 else "closed")

    def test_curve_knots_unequal_pumps_and_permutation(self):
        a=table((0.,32.),(1/128,24.),(1/32,8.))
        c=table((0.,20.),(1/32,12.),(1/16,4.))
        qa=1/128+(1/32-1/128)*.75
        flows=[qa,1/64,1/32]
        static=12-1024*sum(flows)**2
        for curves, expected in (([a,B,c],flows),([c,a,B],[flows[2],*flows[:2]])):
            point,=bank(static,1024.,curves)["operating_points"]
            self.assertEqual(point["head_m"],12.)
            self.assertEqual([p["flow_m3_s"] for p in point["pumps"]],expected)
            self.assertTrue(point["delivering_flow"])

    def test_constant_head_all_closed_and_no_extrapolation(self):
        for static,flows in ((10.,[(32-10)/1024,(16-10)/256]),
                             (32.,[0.,0.]),(40.,[0.,0.])):
            p,=bank(static,0.)["operating_points"]
            self.assertEqual(p["head_m"],static)
            self.assertEqual([unit["flow_m3_s"] for unit in p["pumps"]],flows)
            self.assertIs(p["delivering_flow"],bool(sum(flows)))
        limited=table((0.,32.),(1/64,16.))
        r=bank(5.,0.,[limited,B])
        self.assertTrue(r["ok"])
        self.assertFalse(r["has_rated_intersection"])
        self.assertEqual(r["operating_points"],[])

    def test_requires_rated_monotone_curves_and_check_valves(self):
        for opts in ({"check_valves":False},{"identical_pumps":2},{"curve":A},
                     {"curves":[table((0.,16.),(.01,18.),(.02,12.))]},
                     {"curves":[table((0.,16.),(.01,16.))]},
                     {"curves":[table((.01,16.),(.02,8.))]}):
            with self.subTest(opts=opts):
                self.assertFalse(bank(10.,**opts)["ok"])
