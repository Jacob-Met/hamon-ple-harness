"""Boundary compatibility and independent work balance for a two-bar truss."""
from copy import deepcopy
from fractions import Fraction as Q
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(spec):
    return run_router_conservation_cad_verifier({"mode": "fem", "spec": spec})


def model():
    return {
        "mode": "truss_2d", "nodes": [[0, 0], [1, 0], [2, 0]],
        "elements": [{"i": a, "j": a+1, "E_pa": 200, "A_m2": .5} for a in (0, 1)],
        "supports": [
            {"node": 0, "fix_x": True, "ux_m": .125},
            {"node": 0, "fix_y": True}, {"node": 1, "fix_y": True},
            {"node": 2, "fix_x": True, "ux_m": .375, "fix_y": True},
        ],
        "loads": [{"node": 0, "fx_N": 20}, {"node": 1, "fx_N": 12.5}, {"node": 2, "fx_N": -5}],
    }


class TrussPrescribedSupports(unittest.TestCase):
    def test_settlement_loads_and_free_thermal_extension(self):
        for natural in (Q(0), Q(1, 16)):
            with self.subTest(natural=natural):
                spec = model()
                spec["elements"][0].update(alpha_per_K=float(natural), dT_K=1.)
                # Repeating an identical nonzero x constraint and the default
                # zero y constraint must leave the same physical problem.
                spec["supports"] += [{"node": 2, "fix_x": True, "ux_m": .375},
                                     {"node": 0, "fix_y": True, "uy_m": -0.}]
                # Each member stiffness is 100 N/m. At the free central node
                # the left tension minus the right tension equals its applied load.
                left, right, load = Q(1, 8), Q(3, 8), Q(25, 2)
                center = (left + right + natural + load / 100) / 2
                u = [left, center, right]
                n1, n2 = 100*(center-left-natural), 100*(right-center)
                applied = [20, load, -5]
                reactions = [-n1 - 20, 0, n2 + 5]
                applied_work = sum(f*d for f, d in zip(applied, u)) / 2
                support_work = sum(f*d for f, d in zip(reactions, u)) / 2
                energy = (n1*n1 + n2*n2) / 200
                self.assertEqual(applied_work + support_work - n1*natural/2, energy)
                for supports in (spec["supports"], list(reversed(spec["supports"]))):
                    result = solve({**spec, "supports": supports})
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["displacements_xy_m"], [[float(x), 0.] for x in u])
                    self.assertEqual(result["axial_forces_N"], [float(n1), float(n2)])
                    self.assertEqual(result["reactions_xy_N"], [[float(x), 0.] for x in reactions])
                    self.assertEqual(result["external_work_J"], float(applied_work))
                    self.assertEqual(result["strain_energy_J"], float(energy))

    def test_contradictory_same_axis_is_rejected_in_either_order(self):
        spec = model()
        for axis, first, second in (("x", .375, .125), ("x", .375, math.nextafter(.375, 1.)),
                                    ("y", 0., math.ulp(0.))):
            for reverse in (False, True):
                with self.subTest(axis=axis, first=first, second=second, reverse=reverse):
                    case = deepcopy(spec)
                    case["supports"].append({"node": 2, "fix_"+axis: True, "u"+axis+"_m": second})
                    if reverse:
                        case["supports"].reverse()
                    result = solve(case)
                    self.assertFalse(result["passed"])
                    self.assertIn("Conflicting prescribed u"+axis+"_m at truss node 2", result["reason"])
                    self.assertNotIn("axial_forces_N", result)
