"""Independent hydraulic balance for two parallel feeders and a nearly idle outlet."""

import pytest
from mpmath import mp

from hamon_pipe_network import verify_pipe_network


@pytest.mark.parametrize("bias", [1e-3, -1e-3, 1e-11, -1e-11, 0.0])
@pytest.mark.parametrize("direction,datum", [(1, 0.0), (-1, 12.0)])
def test_signed_minor_loss_exchange(bias, direction, datum):
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
    assert result["flows_m3_s"]["out"] == pytest.approx(float(out), abs=0, rel=1e-9)
    assert result["reservoir_supplies_m3_s"]["L"] == pytest.approx(float(-out), abs=0, rel=1e-9)
    for wrong_outlet in [0.0, -float(out)]:
        rejected = verify_pipe_network({**case, "proposed_flows_m3_s": {**flows, "out": wrong_outlet}})
        assert not rejected["proposed_checks"]["flows_match"]
        assert not rejected["conservation_verified"]
