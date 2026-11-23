"""Signed mean pressure on ordinary step and ramp unloading histories."""
import math
import unittest

from mpmath import mp
from hamon_consolidation import verify_consolidation_history


def reference_pressures(start, end, observation):
    # Integrate the eigenfunction expansion of mean pressure directly. A step
    # unload multiplies R; a ramp unload uses the integral of R over its ages.
    ctx = mp.clone()
    ctx.dps = 90
    rates = [(ctx.pi * (i + ctx.mpf("0.5")))**2 for i in range(200)]

    def remaining(t):
        if t == 0:
            return ctx.one
        return ctx.fsum(2 * ctx.exp(-rate*t) / rate for rate in rates)

    def integral(t):
        if t == 0:
            return ctx.zero
        return ctx.one/3 - ctx.fsum(2 * ctx.exp(-rate*t) / rate**2 for rate in rates)

    start, end, time = map(ctx.mpf, (start, end, observation))
    if start == end:
        unload = remaining(time-start)
    else:
        unload = (integral(time-start) - integral(max(time-end, 0))) / (end-start)
    initial = 100000
    first = initial * remaining(time)
    crossing = float(initial - first/unload)
    return [(level, float(first + (ctx.mpf(level)-initial)*unload))
            for level in (math.nextafter(crossing, -math.inf), crossing,
                          math.nextafter(crossing, math.inf))]


class SignedUnloadingPressure(unittest.TestCase):
    def test_signed_pressure_and_supplied_claim_at_reversal(self):
        for start, end, time in ((.25, .25, 1), (.125, .125, .375),
                                 (.125, .375, 1), (.25, .75, .625)):
            for level, expected in reference_pressures(start, end, time):
                with self.subTest(start=start, end=end, time=time, level=level):
                    spec = {"H_clay_m": 1, "cv_m2_s": 1, "mv_per_Pa": 1e-7,
                            "drainage": "single", "time_s": time,
                            "load_history": [{"time_s": t, "surcharge_Pa": load}
                                             for t, load in ((0, 100000), (start, 100000), (end, level))]}
                    answer = verify_consolidation_history(spec)
                    self.assertTrue(answer["conservation_verified"], answer)
                    with self.subTest(check="reported pressure"):
                        self.assertTrue(math.isclose(answer["mean_excess_pore_pressure_Pa"], expected,
                                                     rel_tol=1e-10, abs_tol=0), (expected, answer))
                    for claimed, accepted in ((expected, True), (-expected, False), (0.0, False)):
                        result = verify_consolidation_history(
                            {**spec, "proposed_mean_excess_pore_pressure_Pa": claimed})
                        with self.subTest(claim=claimed):
                            self.assertEqual(result["conservation_verified"], accepted, result)
