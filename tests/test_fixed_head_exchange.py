"""Prescribed reservoir exchanges from a direct laminar two-source balance."""
import math
import unittest

from mpmath import mp

from hamon_pipe_network import verify_pipe_network


def two_sources(head, diameter, length, reverse, reorder):
    ctx = mp.clone()
    ctx.dps = 95
    demand = 1e-6
    nodes = [{"id": "A", "fixed_head_m": 0}, {"id": "B", "fixed_head_m": head},
             {"id": "J", "demand_m3_s": demand}]
    pipes = []
    resistance = []
    for reservoir, size, span in (("A", .25, 1), ("B", diameter, length)):
        start, end = ("J", reservoir) if reverse else (reservoir, "J")
        pipes.append({"id": reservoir, "from": start, "to": end, "diameter_m": size, "length_m": span})
        resistance.append(128 * ctx.mpf(1e-6) * span / (ctx.pi * ctx.mpf(9.80665) * ctx.mpf(size)**4))
    ra, rb = resistance
    qb = (ctx.mpf(head) + ra*ctx.mpf(demand)) / (ra+rb)
    qa = ctx.mpf(demand) - qb
    supplies = {"A": float(qa), "B": float(qb)}
    flows = {k: (-v if reverse else v) for k, v in supplies.items()}
    heads = {"A": 0, "B": head, "J": float(-ra*qa)}
    if reorder:
        nodes.reverse()
        pipes.reverse()
    return {"nodes": nodes, "pipes": pipes}, flows, heads, supplies


class FixedHeadExchange(unittest.TestCase):
    def test_each_source_has_its_own_signed_flow(self):
        crossing = -1.06360259114687e-9
        cases = [(2**-29, 2**-13, 64), (-2**-29, 2**-13, 64)]
        # The B pipe is a high-resistance chord in the first geometry and a
        # low-resistance forest edge in the second. Its flow changes sign
        # between adjacent representable boundary heads.
        for head in (crossing, math.nextafter(crossing, math.inf)):
            cases.extend(((head, 2**-13, 64), (head, .5, 1)))
        for head, diameter, length in cases:
            for reverse in (False, True):
                for reorder in (False, True):
                    with self.subTest(head=head, diameter=diameter, reverse=reverse, reorder=reorder):
                        spec, flows, heads, supplies = two_sources(head, diameter, length, reverse, reorder)
                        out = verify_pipe_network({**spec, "proposed_flows_m3_s": flows, "proposed_heads_m": heads})
                        self.assertTrue(out["conservation_verified"], out)
                        self.assertLess(max(out["reynolds"].values()), 2000)
                        for edge, expected in flows.items():
                            self.assertTrue(math.isclose(out["flows_m3_s"][edge], expected, rel_tol=1e-12))
                            self.assertTrue(math.isclose(out["reservoir_supplies_m3_s"][edge], supplies[edge],
                                                         rel_tol=1e-12))
                        for substitute in (0, -flows["B"]):
                            wrong = {**flows, "B": substitute}
                            out = verify_pipe_network({**spec, "proposed_flows_m3_s": wrong})
                            self.assertTrue(out["ok"], out)
                            self.assertFalse(out["proposed_checks"]["flows_match"], out)
                            self.assertFalse(out["conservation_verified"], out)

    def test_small_difference_between_two_roots_survives_a_distant_head_datum(self):
        # The first fixed head exceeds the second-to-third head difference
        # by 2^60. Each connecting pipe has both its heads prescribed.
        for small in (-2**-60, 0, 2**-60):
            nodes = [{"id": n, "fixed_head_m": h} for n, h in zip("ABC", (1, 0, small))]
            pipes = [{"id": a+b, "from": a, "to": b, "length_m": 64, "diameter_m": 2**-13}
                     for a, b in (("A", "B"), ("B", "C"))]
            conductance = math.pi * 9.80665 * (2**-13)**4 / (128e-6 * 64)
            flows = {"AB": conductance, "BC": -small * conductance}
            out = verify_pipe_network({"nodes": nodes, "pipes": pipes, "proposed_flows_m3_s": flows})
            self.assertTrue(out["conservation_verified"], out)
            self.assertTrue(math.isclose(out["flows_m3_s"]["BC"], flows["BC"], rel_tol=1e-12))
            self.assertEqual(out["reservoir_supply_m3_s"], 0.)
