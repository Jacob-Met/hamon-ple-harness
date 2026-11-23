"""Registered pressure tails after a unit-load response loses binary64 range."""
import math
import unittest

from mpmath import mp

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def bounded_late_pressure(last, start, end, time, scale):
    # Factor the first diffusion eigenmode before multiplying load ordinates.
    # Every omitted mode decays at least nine times as fast. Bound them before
    # rounding; this oracle does not use the production response evaluator.
    ctx = mp.clone()
    ctx.dps = 120
    rate = (ctx.pi / 2)**2
    start, end, time = map(ctx.mpf, (start, end, time))
    unload = ctx.exp(rate * start)
    if end > start:
        unload *= ctx.expm1(rate * (end-start)) / (rate * (end-start))
    leading = scale * (2/rate) * (100000 + (last-100000)*unload) * ctx.exp(-rate*time)
    bound = scale * (100000 + abs(last-100000)) * ctx.exp(-9*rate*(time-end))
    bound += abs(leading) * ctx.mpf("1e-110")
    lower, upper = float(leading-bound), float(leading+bound)
    if lower != upper:
        raise AssertionError("The late-mode bound does not determine binary64 rounding")
    return lower


class LateConsolidationPressureTests(unittest.TestCase):
    def test_steps_ramps_and_final_rounding_boundary(self):
        histories = (
            (85000, .25, .25, (303., 306.4400223776845, 306.44002237768456)),
            (25000, .25, .25, (303., 306.19019928907363, 306.1901992890737)),
            (0, .125, .375, (303., 306.5214021673969, 306.52140216739696)),
        )
        cases = [(last, a, b, time, 1) for last, a, b, times in histories for time in times]
        # These kernels remain nonzero subnormals. Load scaling makes three
        # final pressures normal, so a zero-only recovery cannot pass.
        cases += [(100000, .25, .25, 294., 2**20),
                  (85000, .25, .25, 294.5, 1),
                  (25000, .25, .25, 295., 2**20),
                  (0, .125, .375, 294.75, 2**20)]
        for last, start, end, time, scale in cases:
            with self.subTest(last=last, start=start, end=end, time=time, scale=scale):
                expected = bounded_late_pressure(last, start, end, time, scale)
                spec = dict(calculation="consolidation_history_1d", H_clay_m=1.,
                            cv_m2_s=1., mv_per_Pa=1e-7/scale, drainage="single", time_s=time,
                            load_history=[dict(time_s=t, surcharge_Pa=q*scale)
                                          for t, q in ((0, 100000), (start, 100000), (end, last))])

                def solve(**claim):
                    return run_router_conservation_cad_verifier(
                        {"mode": "auto", "spec": {**spec, **claim}})

                actual = solve(proposed_mean_excess_pore_pressure_Pa=expected)
                self.assertTrue(actual["conservation_verified"], actual)
                self.assertEqual(actual["mean_excess_pore_pressure_Pa"], expected)
                if last == 0:
                    self.assertEqual(actual["mean_effective_increment_Pa"], -expected)
                opposite = -expected if expected else math.ulp(0.)
                self.assertFalse(solve(proposed_mean_excess_pore_pressure_Pa=opposite)
                                 ["conservation_verified"])
                self.assertEqual(solve(proposed_mean_excess_pore_pressure_Pa=0.)
                                 ["conservation_verified"], expected == 0.)

    def test_unloaded_settlement_is_rounded_after_material_and_height_factors(self):
        # In the second case mean stress itself rounds to zero, but settlement
        # remains representable. Multiplying a rounded stress cannot recover it.
        height, time_scale = 64., 2**18
        for scale, mv in ((1, 2**-18), (2**-15, .125)):
            with self.subTest(load_scale=scale, mv=mv):
                expected_pressure = bounded_late_pressure(0, .125, .375, 303., scale)
                expected_settlement = -bounded_late_pressure(0, .125, .375, 303., scale*mv*height)
                self.assertEqual(expected_settlement, math.ulp(0.))
                spec = dict(calculation="consolidation_history_1d", H_clay_m=height,
                            cv_m2_s=2**-6, mv_per_Pa=mv, drainage="single", time_s=303.*time_scale,
                            load_history=[dict(time_s=t*time_scale, surcharge_Pa=q*scale)
                                          for t, q in ((0, 100000), (.125, 100000), (.375, 0))])
                answer = run_router_conservation_cad_verifier({"mode": "auto", "spec": {
                    **spec, "proposed_settlement_m": expected_settlement}})
                self.assertTrue(answer["conservation_verified"], answer)
                self.assertEqual(answer["mean_excess_pore_pressure_Pa"], expected_pressure)
                self.assertEqual(answer["mean_effective_increment_Pa"], -expected_pressure)
                self.assertEqual(answer["settlement_at_time_m"], expected_settlement)
                # The existing absolute settlement tolerance still permits
                # a zero proposal when the computed settlement is positive.
                zero = run_router_conservation_cad_verifier({"mode": "auto", "spec": {
                    **spec, "proposed_settlement_m": 0.}})
                self.assertTrue(zero["conservation_verified"], zero)


if __name__ == "__main__":
    unittest.main()
