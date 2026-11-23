"""Independent hydraulic balance for two parallel feeders and a nearly idle outlet."""

import math
import unittest
from mpmath import mp

from hamon_pipe_network import verify_pipe_network


def check_exchange(bias, direction, datum):
    ctx = mp.clone()
    ctx.dps = 80
    diameter = ctx.mpf(0.02)
    gravity, viscosity = ctx.mpf(9.80665), ctx.mpf(1e-6)
    linear = 128 * viscosity / (ctx.pi * gravity * diameter**4)
    quadratic = 8 / (gravity * ctx.pi**2 * diameter**4)

    def feed_loss(q):
        return 2 * linear * q + 20 * quadratic * q * abs(q)

    def outlet_loss(q):
        return linear * q + 5 * quadratic * q * abs(q)

    # Quantize exactly the head and demand that the verifier receives.
    left = ctx.mpf(float(datum + feed_loss(direction * ctx.mpf(2e-5))))
    delta = left - datum
    nominal = 2 * delta / (2 * linear + ctx.sqrt((2*linear)**2 + 80*quadratic*abs(delta)))
    demand = ctx.mpf(float(2 * nominal * (1 + ctx.mpf(bias))))

    # Continuity sets each feed flow. The remaining energy residual decreases
    # strictly with signed outlet flow, so bisection is independent of Newton.
    def residual(out):
        return delta - feed_loss((demand + out)/2) - outlet_loss(out)

    lo, hi = ctx.mpf("-0.001"), ctx.mpf("0.001")
    assert residual(lo) > 0 > residual(hi)
    for _ in range(240):
        middle = (lo + hi)/2
        if residual(middle) > 0:
            lo = middle
        else:
            hi = middle
    out = (lo + hi)/2
    feed = (demand + out)/2
    work = delta * (2*feed) - outlet_loss(out)*demand
    dissipation = 2*feed*feed_loss(feed) + out*outlet_loss(out)
    assert abs(work - dissipation) < ctx.mpf("1e-64")

    case = {
        "nodes": [{"id": "H", "fixed_head_m": float(left)}, {"id": "L", "fixed_head_m": datum},
                  {"id": "J", "demand_m3_s": float(demand)}],
        "pipes": [
            {"id": name, "from": start, "to": end, "length_m": length, "diameter_m": 0.02,
             "minor_loss_K": minor}
            for name, start, end, length, minor in
            [("a", "H", "J", 2, 20), ("b", "H", "J", 2, 20), ("out", "J", "L", 1, 5)]
        ],
    }
    flows = {"a": float(feed), "b": float(feed), "out": float(out)}
    result = verify_pipe_network({**case, "proposed_flows_m3_s": flows})
    assert result["conservation_verified"], result
    assert math.isclose(result["flows_m3_s"]["out"], float(out), rel_tol=1e-9)
    assert math.isclose(result["reservoir_supplies_m3_s"]["L"], float(-out), rel_tol=1e-9)
    for wrong_outlet in [0.0, -float(out)]:
        rejected = verify_pipe_network({**case, "proposed_flows_m3_s": {**flows, "out": wrong_outlet}})
        assert not rejected["proposed_checks"]["flows_match"]
        assert not rejected["conservation_verified"]


class MinorExchangeTest(unittest.TestCase):
    def test_signed_minor_loss_exchange(self):
        for bias in (1e-3, -1e-3, 1e-11, -1e-11, 0.0):
            for direction, datum in ((1, 0.0), (-1, 12.0)):
                with self.subTest(bias=bias, direction=direction, datum=datum):
                    check_exchange(bias, direction, datum)

    def test_small_net_boundary_supply_between_rounded_through_flows(self):
        ctx = mp.clone()
        ctx.dps = 80
        diameter, gravity = ctx.mpf(0.02), ctx.mpf(9.80665)
        r = 256 * ctx.mpf(1e-6) / (ctx.pi * gravity * diameter**4)
        k = 160 / (gravity * ctx.pi**2 * diameter**4)

        def inverse(head_drop):
            return 2 * head_drop / (r + ctx.sqrt(r*r + 4*k*abs(head_drop)))

        for low in (1e-20, -1e-20):
            with self.subTest(low=low):
                heads = {"A": 0.01, "B": 0.005, "C": low}
                inflow = inverse(ctx.mpf(heads["A"]) - ctx.mpf(heads["B"]))
                outflow = inverse(ctx.mpf(heads["B"]) - ctx.mpf(low))
                self.assertEqual(float(outflow), float(inflow))
                expected = float(outflow - inflow)
                self.assertNotEqual(expected, 0.0)
                case = {
                    "nodes": [{"id": name, "fixed_head_m": h} for name, h in heads.items()],
                    "pipes": [{"id": a+b, "from": a, "to": b, "length_m": 2,
                               "diameter_m": 0.02, "minor_loss_K": 20}
                              for a, b in (("A", "B"), ("B", "C"))],
                }
                result = verify_pipe_network(case)
                self.assertTrue(result["conservation_verified"], result)
                self.assertTrue(math.isclose(result["reservoir_supplies_m3_s"]["B"],
                                             expected, rel_tol=1e-12))
