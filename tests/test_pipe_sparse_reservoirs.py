"""Manufactured physical solutions for large multiple-reservoir networks."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

import hamon_pipe_network as hydraulics


def manufactured_case(count, endpoints, reservoirs):
    # Prescribe physical heads first, then derive each demand from the
    # independent Hagen-Poiseuille law. No production solver sets this oracle.
    heads = [16.0 - 2**-15 * (i + (i % 7) / 64) / count for i in range(count)]
    balance = [[] for _ in heads]
    flows, pipes = {}, []
    for k, (left, right) in enumerate(endpoints):
        if k % 3 == 1:
            left, right = right, left
        length = 8.0 + k % 5
        resistance = 128 * 1e-6 * length / (9.80665 * math.pi * 0.08**4)
        flow = (heads[left] - heads[right]) / resistance
        balance[left].append(-flow)
        balance[right].append(flow)
        flows[str(k)] = flow
        pipes.append({"id": str(k), "from": str(left), "to": str(right),
                      "length_m": length, "diameter_m": 0.08})
    nodes = [{"id": str(i), **({"fixed_head_m": heads[i]} if i in reservoirs else
                              {"demand_m3_s": math.fsum(balance[i])})}
             for i in range(count)]
    supplies = {str(i): -math.fsum(balance[i]) for i in reservoirs}
    return {"nodes": nodes, "pipes": pipes}, flows, heads, supplies


class SparseReservoirTests(unittest.TestCase):
    def check_physics(self, count, endpoints, reservoirs):
        spec, flows, heads, supplies = manufactured_case(count, endpoints, reservoirs)
        with patch.object(np.linalg, "matrix_rank", side_effect=AssertionError("dense rank")), \
             patch.object(np.linalg, "solve", side_effect=AssertionError("dense solve")), \
             patch.object(np.linalg, "lstsq", side_effect=AssertionError("dense least squares")), \
             patch.object(hydraulics, "spsolve", wraps=hydraulics.spsolve) as solves:
            result = hydraulics.verify_pipe_network(spec)
        self.assertTrue(result["conservation_verified"], result)
        for call in solves.call_args_list:
            self.assertLessEqual(call.args[0].shape[0], count - len(reservoirs))
        self.assertEqual(result["independent_cycles"], len(endpoints) - count + 1)
        for key, expected in flows.items():
            self.assertAlmostEqual(result["flows_m3_s"][key], expected, delta=1e-14)
        for i, expected in enumerate(heads):
            self.assertAlmostEqual(result["heads_m"][str(i)], expected, delta=2e-13)
        for key, expected in supplies.items():
            self.assertAlmostEqual(result["reservoir_supplies_m3_s"][key],
                                   expected, delta=1e-14)
        self.assertLess(result["max_mass_residual_m3_s"], 1e-13)
        self.assertLess(result["max_energy_residual_m"], 1e-11)
        self.assertLess(result["max_nodal_cycle_flow_difference_m3_s"], 1e-13)
        return spec, result

    def test_long_ring_with_three_prescribed_heads(self):
        self.check_physics(1025, [(i, (i + 1) % 1025) for i in range(1025)],
                           {0, 341, 1024})

    def test_dense_mesh_with_multiple_boundary_heads(self):
        self.check_physics(33, [(i, j) for i in range(33) for j in range(i + 1, 33)],
                           {0, 11, 32})

    def test_many_parallel_boundary_exchanges_keep_one_pressure_unknown(self):
        self.check_physics(3, [(0, 1), (2, 1)] * 257, {0, 2})

    def test_all_heads_prescribed_need_no_pressure_solve(self):
        self.check_physics(65, [(i, (i + 1) % 65) for i in range(65)], set(range(65)))

    def test_each_disconnected_component_having_a_reservoir_is_still_rejected(self):
        spec, _, _, _ = manufactured_case(
            6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], {0, 3})
        result = hydraulics.verify_pipe_network(spec)
        self.assertFalse(result["ok"])
        self.assertFalse(result["conservation_verified"])
        self.assertIn("connected", result["reason"])

    def test_wrong_boundary_or_local_flow_cannot_pass_the_certificate(self):
        spec, result = self.check_physics(
            17, [(i, (i + 1) % 17) for i in range(17)], {0, 5, 16})
        bad = copy.deepcopy(spec)
        bad["proposed_flows_m3_s"] = dict(result["flows_m3_s"])
        bad["proposed_flows_m3_s"]["3"] *= -1
        self.assertFalse(hydraulics.verify_pipe_network(bad)["conservation_verified"])
        bad = copy.deepcopy(spec)
        bad["proposed_heads_m"] = dict(result["heads_m"])
        bad["proposed_heads_m"]["5"] += 1e-5
        self.assertFalse(hydraulics.verify_pipe_network(bad)["conservation_verified"])


if __name__ == "__main__":
    unittest.main()
