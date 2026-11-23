"""Registered SPICE operating-point and incremental-admittance checks."""
import cmath
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


THERMAL = (1.380649e-23 / 1.602176634e-19) * 300
IS = 1e-12


def route(deck, **extra):
    return run_router_conservation_cad_verifier(
        {"mode": "circuit", "spec": {"netlist": deck, **extra}}
    )


def roots(lo, hi, eq):
    for _ in range(125):
        mid = (lo + hi) / 2
        if eq(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def common(body):
    return ("Junction fixture\n" + body + "\n.model silicon D(IS=1p N=1 TNOM=26.85)"
            "\n.temp 26.85\n.end")


class DiodeOperatingPoint(unittest.TestCase):
    def test_loaded_forward_and_reverse_dc(self):
        for vsrc in (5, -3):
            deck = common(f"V1 in 0 DC {vsrc}\nR1 in out 1k\nD1 out 0 silicon")
            v = roots(min(vsrc,0), max(vsrc,0),
                      lambda x: IS*math.expm1(x/THERMAL)+(x-vsrc)/1000)
            current = IS*math.expm1(v/THERMAL)
            r = route(deck, proposed_node_voltages={"out":v})
            self.assertTrue(r["conservation_verified"])
            self.assertAlmostEqual(r["node_voltages"]["out"],v,places=10)
            states = r["diode_operating_point"]
            self.assertLessEqual(abs(states["D1"]["current_A"]-current), 1e-9*abs(current))
            if vsrc > 0:
                self.assertFalse(route(deck, proposed_node_voltages={"out":vsrc})["conservation_verified"])

    def test_antiparallel_junctions_and_reference_kcl(self):
        for vsrc in (2., -2.):
            deck = common(f"V1 in 0 DC {vsrc}\nR1 in out 330\nDa out 0 silicon\nDb 0 out silicon")
            v = roots(-1.,1., lambda x: 2*IS*math.sinh(x/THERMAL)+(x-vsrc)/330)
            r = route(deck)
            self.assertTrue(r["conservation_verified"])
            self.assertAlmostEqual(r["node_voltages"]["out"],v,places=10)
            currents = {b["name"]: b["current_A"] for b in r["branch_currents"]}
            self.assertAlmostEqual(currents["Da"]-currents["Db"], (vsrc-v)/330, places=12)

    def test_ac_uses_dc_bias_and_zero_bias_derivative(self):
        for bias in (0., .75, -.3):
            for ac in (.03, .15):
                deck = common(f"V1 in 0 DC {bias} AC {ac} 37\nR1 in out 1k\n"
                              "D1 out 0 silicon\nC1 out 0 280n\n.ac lin 1 700 700")
                v = roots(min(bias,0), max(bias,0),
                          lambda x: IS*math.expm1(x/THERMAL)+(x-bias)/1000)
                g = IS*math.exp(v/THERMAL)/THERMAL
                expected = cmath.rect(ac,math.radians(37))/(1+1000*(g+1j*math.tau*700*280e-9))
                r = route(deck)
                self.assertTrue(r["conservation_verified"])
                self.assertAlmostEqual(complex(*r["node_voltages"]["out"]),expected,places=11)
                self.assertAlmostEqual(r["diode_operating_point"]["D1"]["voltage_V"],v,places=10)
                self.assertLessEqual(abs(r["diode_operating_point"]["D1"]["conductance_S"]-g),
                                     1e-9*g)
                diode = next(b for b in r["branch_currents"] if b["name"]=="D1")
                self.assertAlmostEqual(complex(*diode["current_A"]),g*expected,places=12)

    def test_imposed_current_requires_a_finite_junction_bias(self):
        for i in (.001, -IS/2, 0., math.nextafter(-IS, 0.)):
            r = route(common(f"I1 0 out DC {i}\nD1 out 0 silicon"))
            self.assertTrue(r["conservation_verified"])
            expected = THERMAL * (math.log((IS+i)/IS) if i < -IS/2 else math.log1p(i/IS))
            self.assertAlmostEqual(r["node_voltages"]["out"], expected, places=10)
            g = (IS+i)/THERMAL
            self.assertLessEqual(abs(r["diode_operating_point"]["D1"]["conductance_S"]-g),1e-9*g)
        for i in (-IS, -2*IS):
            with self.assertRaises(ValueError):
                route(common(f"I1 0 out DC {i}\nD1 out 0 silicon"))

    def test_unknown_nonlinear_parameters_and_ambiguous_active_networks_refuse(self):
        for changes in [
            ("D1 out 0 missing", ".model silicon D"),
            ("D1 out 0 silicon", ".model silicon D(IS=1p CJO=1n)"),
            ("D1 out 0 silicon", ".model silicon D(IS=1p N=-1)"),
            ("D1 out 0 silicon\nGneg out 0 out 0 -2m", ".model silicon D"),
        ]:
            diode, model = changes
            deck = f"Junction domain\nV1 in 0 DC 1\nR1 in out 1k\n{diode}\n{model}\n.end"
            with self.subTest(diode=diode, model=model), self.assertRaises(ValueError):
                route(deck)
