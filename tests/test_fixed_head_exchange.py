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

    def test_independent_turbulent_exchange_keeps_the_laminar_balance_resolved(self):
        spec, flows, heads, _ = two_sources(-1.06360259114687e-9, .5, 1, False, False)
        spec["nodes"].append({"id": "C", "fixed_head_m": 1.})
        side = {"id": "side", "from": "C", "to": "A", "length_m": 10,
                "diameter_m": .025, "minor_loss_K": 2}
        spec["pipes"].append(side)
        # The side line connects two fixed heads and shares no pipe with the
        # A-J-B exchange. It cannot change the two-source closed-form result.
        out = verify_pipe_network(spec)
        self.assertTrue(out["conservation_verified"], out)
        self.assertGreater(out["reynolds"]["side"], 4000)
        self.assertEqual(out["flows_m3_s"]["B"], flows["B"])
        self.assertEqual(out["heads_m"]["J"], heads["J"])
        wrong = {**out["flows_m3_s"], "B": 0}
        self.assertFalse(verify_pipe_network({**spec, "proposed_flows_m3_s": wrong})["conservation_verified"])

    def test_equal_heads_exchange_the_difference_of_junction_demands(self):
        # With equal supply resistances r and bridge resistance 32r, continuity
        # and the path head losses give q_JK = (d_K - d_J) / 34.
        d = 1e-6
        for dk in (math.nextafter(d, -math.inf), d, math.nextafter(d, math.inf)):
            nodes = [{"id": "A", "fixed_head_m": 0}, {"id": "B", "fixed_head_m": 0},
                     {"id": "J", "demand_m3_s": d}, {"id": "K", "demand_m3_s": dk}]
            rows = (("AJ", "A", "J", 1, .125), ("BK", "B", "K", 1, .125),
                    ("JK", "J", "K", 2, .0625))
            pipes = [{"id": name, "from": a, "to": b, "length_m": length, "diameter_m": diameter}
                     for name, a, b, length, diameter in rows]
            cross = (dk - d) / 34
            flows = {"AJ": d + cross, "BK": dk - cross, "JK": cross}
            spec = {"nodes": nodes, "pipes": pipes}
            with self.subTest(dk=dk):
                result = verify_pipe_network({**spec, "proposed_flows_m3_s": flows})
                self.assertTrue(result["conservation_verified"], result)
                self.assertEqual(result["flows_m3_s"]["JK"], cross)
                if cross:
                    for substitute in (0, -cross):
                        wrong = {**flows, "JK": substitute}
                        self.assertFalse(verify_pipe_network(
                            {**spec, "proposed_flows_m3_s": wrong})["conservation_verified"])

    def test_net_and_individual_supply_between_opposing_reservoir_flows(self):
        for demand in (-1e-30, 0., 1e-30):
            for shared_junction in (True, False):
                nodes = [{"id": n, "fixed_head_m": h} for n, h in zip("ABC", (0, 1e-8, -1e-8))]
                nodes.append({"id": "J", "demand_m3_s": demand})
                rows = (("A", "J"), ("B", "J"), ("C", "J"))
                if not shared_junction:
                    nodes.append({"id": "K"})
                    rows = (("A", "J"), ("B", "J"), ("A", "K"), ("C", "K"))
                pipes = [{"id": a+b, "from": a, "to": b, "length_m": 1, "diameter_m": .125}
                         for a, b in rows]
                with self.subTest(demand=demand, shared_junction=shared_junction):
                    out = verify_pipe_network({"nodes": nodes, "pipes": pipes})
                    self.assertTrue(out["conservation_verified"], out)
                    self.assertEqual(out["reservoir_supply_m3_s"], demand)
                    self.assertTrue(math.isclose(out["reservoir_supplies_m3_s"]["A"],
                                                 demand / (3 if shared_junction else 2), rel_tol=1e-12))
