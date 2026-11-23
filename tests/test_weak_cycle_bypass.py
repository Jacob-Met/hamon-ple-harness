"""Check signed flow on a small bypass by exact series/parallel reduction."""
import math
import unittest
from fractions import Fraction

from hamon_pipe_network import verify_pipe_network


def laminar_bypass(demand, triangle, reverse):
    def pipe(name, start, end, length, diameter):
        if reverse:
            start, end = end, start
        return {"id": name, "from": start, "to": end,
                "length_m": length, "diameter_m": diameter}

    nodes = [{"id": "R", "fixed_head_m": 0}, {"id": "J", "demand_m3_s": demand}]
    pipes = [pipe("main", "R", "M" if triangle else "J", 1, 0.25),
             pipe("bypass", "R", "J", 64, 2**-13)]
    if triangle:
        nodes.append({"id": "M"})
        pipes.append(pipe("other", "M", "J", 1, 0.25))

    # Re < 6 for every case. Common 128*nu/(pi*g) cancels in the split.
    # Along the triangle, the two main-pipe resistances add in series.
    main_resistance = Fraction(2 if triangle else 1) / Fraction(0.25)**4
    bypass_resistance = Fraction(64) / Fraction(2**-13)**4
    bypass_flow = Fraction(demand) * main_resistance / (main_resistance + bypass_resistance)
    main_flow = Fraction(demand) - bypass_flow
    sign = -1 if reverse else 1
    expected = {p["id"]: float(sign * (bypass_flow if p["id"] == "bypass" else main_flow))
                for p in pipes}
    head = -float(main_resistance * main_flow) * 128e-6 / (math.pi * 9.80665)
    return {"nodes": nodes, "pipes": pipes}, expected, head


class WeakCycleBypass(unittest.TestCase):
    def test_signed_flows_and_cycle_energy(self):
        for demand in (1e-6, -1e-6, 1e-20, -1e-20):
            for triangle in (False, True):
                for reverse in (False, True):
                    with self.subTest(demand=demand, triangle=triangle, reverse=reverse):
                        spec, expected, head = laminar_bypass(demand, triangle, reverse)
                        result = verify_pipe_network({**spec, "proposed_flows_m3_s": expected})
                        self.assertTrue(result["conservation_verified"], result)
                        for key, flow in expected.items():
                            self.assertTrue(math.isclose(result["flows_m3_s"][key], flow,
                                                         rel_tol=1e-7, abs_tol=0), (key, flow, result))
                        self.assertTrue(math.isclose(result["heads_m"]["J"], head,
                                                     rel_tol=1e-7, abs_tol=0), result)
                        self.assertEqual(result["independent_cycles"], 1)

    def test_an_erased_or_reversed_bypass_fails_its_own_flow_check(self):
        for demand in (1e-6, -1e-6, 1e-20, -1e-20):
            for triangle in (False, True):
                for reverse in (False, True):
                    spec, expected, _ = laminar_bypass(demand, triangle, reverse)
                    for factor in (0, -1):
                        with self.subTest(demand=demand, triangle=triangle, reverse=reverse,
                                          factor=factor):
                            wrong = {**expected, "bypass": factor * expected["bypass"]}
                            result = verify_pipe_network({**spec, "proposed_flows_m3_s": wrong})
                            self.assertTrue(result["ok"], result)
                            self.assertFalse(result["proposed_checks"]["flows_match"], result)
                            self.assertFalse(result["conservation_verified"])
