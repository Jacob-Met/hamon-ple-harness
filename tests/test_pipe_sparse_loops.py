"""Independent hydraulic oracles for sparse one-reservoir meshes."""
import math
import unittest
from unittest.mock import patch

import numpy as np

import hamon_pipe_network as hydraulics


def laminar_case(count, endpoints):
    heads = [15.0] + [15.0 - 2**-15 * (1 + (i % 7) / 64) for i in range(1, count)]
    balance = [[] for _ in heads]
    flows, pipes = {}, []
    for k, (left, right) in enumerate(endpoints):
        if k % 3 == 1:
            left, right = right, left
        length = 8.0 + (k % 5)
        resistance = 128 * 1e-6 * length / (9.80665 * math.pi * 0.08**4)
        flow = (heads[left] - heads[right]) / resistance
        balance[left].append(-flow)
        balance[right].append(flow)
        flows[str(k)] = flow
        pipes.append({"id": str(k), "from": str(left), "to": str(right),
                      "length_m": length, "diameter_m": 0.08})
    nodes = [{"id": "0", "fixed_head_m": heads[0]}] + [
        {"id": str(i), "demand_m3_s": math.fsum(balance[i])} for i in range(1, count)]
    return {"nodes": nodes, "pipes": pipes}, flows, heads


class SparsePipeTests(unittest.TestCase):
    def check_physics(self, count, endpoints):
        spec, flows, heads = laminar_case(count, endpoints)
        with patch.object(np.linalg, "matrix_rank", side_effect=AssertionError("dense rank")), \
             patch.object(np.linalg, "solve", side_effect=AssertionError("dense solve")), \
             patch.object(hydraulics, "spsolve", wraps=hydraulics.spsolve) as solves:
            result = hydraulics.verify_pipe_network(spec)
        self.assertTrue(result["conservation_verified"], result)
        self.assertTrue(solves.called)
        for call in solves.call_args_list:
            self.assertLessEqual(call.args[0].shape[0], count - 1)
        self.assertEqual(result["independent_cycles"], len(endpoints) - count + 1)
        for k, expected in flows.items():
            self.assertAlmostEqual(result["flows_m3_s"][k], expected, delta=1e-14)
        for i, expected in enumerate(heads):
            self.assertAlmostEqual(result["heads_m"][str(i)], expected, delta=2e-13)
        self.assertLess(result["max_mass_residual_m3_s"], 1e-13)
        self.assertLess(result["max_energy_residual_m"], 1e-11)
        self.assertLess(result["max_nodal_cycle_flow_difference_m3_s"], 1e-13)

    def test_long_ring(self):
        self.check_physics(129, [(i, (i+1) % 129) for i in range(129)])

    def test_complete_mesh(self):
        self.check_physics(17, [(i, j) for i in range(17) for j in range(i+1, 17)])

    def test_parallel_pipes_keep_the_cycle_step_to_one_unknown(self):
        self.check_physics(2, [(0, 1)] * 513)

    def test_disconnected_loops(self):
        nodes = [{"id": s} for s in "ABCDEF"]
        nodes[0]["fixed_head_m"] = 2
        pipes = [{"id": p, "from": p[0], "to": p[1], "length_m": 1, "diameter_m": 0.1}
                 for p in ("AB", "BC", "CA", "DE", "EF", "FD")]
        result = hydraulics.verify_pipe_network({"nodes": nodes, "pipes": pipes})
        self.assertFalse(result["ok"])
        self.assertFalse(result["conservation_verified"])
        self.assertIn("connected", result["reason"])
