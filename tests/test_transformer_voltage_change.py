"""Signed regulation from the original secondary-current phasor."""
import math
import unittest
from decimal import Decimal, localcontext

from hamon_dual_verifier import run_dual_verify_spec


def operating_point(**changes):
    spec = dict(S_rated_VA=1000., V_sec_rated_V=100., R_eq_ohm=.2, X_eq_ohm=.4,
                pf=.8, pf_mode="lagging", P_core_W=5., load_fraction=1.)
    spec.update(changes)
    return spec


def report(spec):
    return run_dual_verify_spec({
        "mode": "domain", "kind": "transformer_efficiency_regulation", "spec": spec,
    })


def phasor(spec, precision=180):
    # Retain both Cartesian components before taking the magnitude and
    # subtracting the terminal voltage. No squared-change formula is used here.
    with localcontext() as ctx:
        ctx.prec = precision
        def value(key):
            return Decimal.from_float(float(spec[key]))
        v = value("V_sec_rated_V")
        current = value("load_fraction") * value("S_rated_VA") / v
        cosine = value("pf")
        sine = (1 - cosine*cosine).sqrt()
        if spec["pf_mode"] == "lagging":
            sine = -sine
        real = v + current * (cosine*value("R_eq_ohm") - sine*value("X_eq_ohm"))
        imag = current * (cosine*value("X_eq_ohm") + sine*value("R_eq_ohm"))
        magnitude = (real*real + imag*imag).sqrt()
        return float(magnitude), float(100*(magnitude-v)/v)


class SignedTransformerRegulation(unittest.TestCase):
    def check(self, spec, precision=180):
        result = report(spec)
        self.assertTrue(result["dual_verified"])
        self.assertEqual((result["E2_no_load_V"], result["voltage_regulation_pct"]),
                         phasor(spec, precision))
        return result["voltage_regulation_pct"]

    def test_adjacent_winding_resistances_straddle_zero_with_same_absolute_voltage(self):
        center = .2849260708831916
        for r in (math.nextafter(center, 0.), center, math.nextafter(center, math.inf)):
            with self.subTest(resistance=r):
                spec = operating_point(R_eq_ohm=r, pf_mode="leading")
                expected = self.check(spec)
                self.assertEqual(report(spec)["E2_no_load_V"], 100.)
                self.assertEqual(expected > 0, r > center)
                self.assertNotEqual(expected, 0.)

    def test_zero_contact_cancellation_and_ordinary_phasors(self):
        for direction in ("leading", "lagging"):
            for factor in (0., .375, .8, 1.):
                for reactance in (-.25, .4):
                    with self.subTest(direction=direction, factor=factor, reactance=reactance):
                        self.check(operating_point(pf_mode=direction, pf=factor, X_eq_ohm=reactance))
        self.assertEqual(self.check(operating_point(pf=0., pf_mode="leading", R_eq_ohm=0., X_eq_ohm=20.)), 0.)
        self.assertEqual(self.check(operating_point(pf=0., pf_mode="leading", R_eq_ohm=0., X_eq_ohm=10.)), -100.)
        self.check(operating_point(pf=0., pf_mode="leading", R_eq_ohm=0., X_eq_ohm=math.nextafter(10., math.inf)))

    def test_rounded_voltage_midpoints_and_second_order_reactance(self):
        for resistance in (2.**-53, 3*2.**-53):
            self.check(operating_point(S_rated_VA=1., V_sec_rated_V=1., R_eq_ohm=resistance,
                                       X_eq_ohm=0., pf=1.))
        for x in (1.e-8, 2.**-200, 2.**-530):
            with self.subTest(reactance=x):
                self.assertGreater(self.check(operating_point(S_rated_VA=1., V_sec_rated_V=1.,
                                                   R_eq_ohm=0., X_eq_ohm=x, pf=1.), 850), 0.)
        for x in (2.**-540, 2.**-550):
            with self.assertRaisesRegex(ValueError, "resolution"):
                report(operating_point(S_rated_VA=1., V_sec_rated_V=1., R_eq_ohm=0., X_eq_ohm=x, pf=1.))
