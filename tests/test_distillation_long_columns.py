"""Independent equilibrium and section-balance counts for long binary columns."""
import math
import unittest
from decimal import Decimal as D, ROUND_CEILING, localcontext

from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier


def column(alpha, reflux, **changes):
    p = dict(subkind="distillation", z_F=.5, x_D=.9, x_B=.1, q=1.,
             alpha=float(alpha), R=float(reflux), **changes)
    return dispatch_wave7_verifier("electrochem_separations", p)


def replay(alpha, reflux):
    # Equimolar saturated-liquid feed; D=B=F/2. The stripping operating
    # line follows directly from V' y = L' x - B xB.
    with localcontext() as ctx:
        ctx.prec = 85
        a, r = D(str(alpha)), D(str(reflux))
        xd, xb, z = D(".9"), D(".1"), D(".5")
        vapor, previous, feed = xd, xd, None
        for tray in range(1, 20000):
            liquid = vapor/(a-(a-1)*vapor)
            if liquid <= z and feed is None:
                feed = tray
            if liquid <= xb:
                final = (tray-1)+(previous-xb)/(previous-liquid)
                return tray, float(final), feed
            previous = liquid
            vapor = (r*liquid+xd)/(r+1) if liquid>z else ((r+2)*liquid-xb)/(r+1)
        raise AssertionError("independent short replay did not reach bottoms")


def fixed_point_replay(alpha, reflux):
    with localcontext() as ctx:
        ctx.prec = 100
        a, r = D(str(alpha)), D(str(reflux))
        xd, xb, z = D(".9"), D(".1"), D(".5")
        def section(start, target, slope, intercept):
            c, d = -(a-1)*slope, a-(a-1)*intercept
            disc = ((d-slope)**2+4*c*intercept).sqrt()
            low, high = sorted(((slope-d-disc)/(2*c),(slope-d+disc)/(2*c)))
            ratio = (slope-c*low)/(slope-c*high)
            initial = (start-low)/(start-high)
            final = (target-low)/(target-high)
            n = int(((final/initial).ln()/ratio.ln()).to_integral_value(rounding=ROUND_CEILING))
            def position(k):
                q = initial*((k*ratio.ln()).exp())
                return (low-q*high)/(1-q)
            return n, position(n-1), position(n)
        rect, strip = r/(r+1), (r+2)/(r+1)
        nr, _, feed = section(xd/(a-(a-1)*xd),z,rect,(1-rect)*xd)
        ns, prev, end = section(feed,xb,strip,(1-strip)*xb)
        return 1+nr+ns, float(nr+ns+(prev-xb)/(prev-end)), 1+nr


class LongDistillationColumns(unittest.TestCase):
    def test_feed_and_fractional_tray_by_individual_material_balances(self):
        for alpha, reflux, count in ((1.02,150,316), (1.01,300,628),
                                     (1.006,1000,854), (1.0004,15000,12767)):
            with self.subTest(alpha=alpha):
                integer, fractional, feed = replay(alpha,reflux)
                self.assertEqual(integer,count)
                result = column(alpha,reflux)
                self.assertTrue(result["verified"],result)
                self.assertEqual(result["N_stages_integer"],integer)
                self.assertEqual(result["N_theoretical"],fractional)
                self.assertEqual(result["feed_stage"],feed)

    def test_true_feed_pinch_and_adjacent_finite_reflux(self):
        for reflux in (1.4, math.nextafter(1.4,0.)):
            failure=column(2.,reflux)
            self.assertFalse(failure["verified"])
            self.assertEqual(failure["error"],"reflux_below_minimum_pinch")
        for reflux in (math.nextafter(1.4,math.inf), 1.40000001):
            expected=replay(2.,reflux)
            result=column(2.,reflux)
            self.assertTrue(result["verified"],result)
            self.assertEqual((result["N_stages_integer"],result["N_theoretical"],result["feed_stage"]),expected)

    def test_close_keys_have_a_resolved_underwood_root_and_long_stage_count(self):
        expected = fixed_point_replay(1.00000001,1e9)
        result=column(1.00000001,1e9)
        self.assertTrue(result["verified"],result)
        self.assertEqual(result["N_stages_integer"],478700924)
        self.assertEqual(result["feed_stage"],239350462)
        self.assertEqual((result["N_stages_integer"],result["N_theoretical"],result["feed_stage"]),expected)
        # For saturated liquid, u=(1-z)/(1+(alpha-1)z) from the Underwood law.
        with localcontext() as ctx:
            ctx.prec=60
            self.assertEqual(result["underwood_theta_fraction"],float(D(".5")/(1+D("5e-9"))))
