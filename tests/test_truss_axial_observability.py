"""Axial member forces can be unique even when a linear truss has a free mechanism."""
import unittest

from hamon_dual_verifier import run_dual_verify_spec


def solve(nodes, members):
    return run_dual_verify_spec({"mode": "domain", "kind": "truss_2d", "spec": {
        "nodes": nodes, "members": members}})


def bar(label, i, j, ea=1e6):
    return {"id": label, "n1": i, "n2": j, "EA": ea}


class AxialObservableTests(unittest.TestCase):
    def test_rotated_frame_has_axial_forces_and_two_free_transverse_motions(self):
        nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True},
                 "B": {"x": 3, "y": 4}, "C": {"x": 6, "y": 8, "Fx": 60, "Fy": 80}}
        members = [bar("AB", "A", "B"), bar("CB", "C", "B")]
        result = solve(nodes, members)
        self.assertTrue(result["dual_verified"])
        self.assertFalse(result["displacement_unique"])
        self.assertEqual(result["free_mechanism_count"], 2)
        self.assertEqual((result["N_AB_N"], result["N_CB_N"]), (100., 100.))
        for transverse in (1., 1e-30):
            with self.subTest(transverse=transverse):
                nodes["B"].update(Fx=-4*transverse, Fy=3*transverse)
                with self.assertRaisesRegex(ValueError, "virtual work"):
                    solve(nodes, members)

    def test_compatible_redundant_members_follow_stiffness(self):
        nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True},
                 "B": {"x": 1, "y": 0}, "C": {"x": 2, "y": 0, "Fx": 100, "fix_y": True}}
        result = solve(nodes, [bar("a", "A", "B", 1), bar("b", "B", "A", 3), bar("c", "B", "C")])
        self.assertEqual(result["force_solution"], "compatible_mechanism")
        self.assertEqual([result["N_" + j + "_N"] for j in "abc"], [25., 75., 100.])

    def test_constrained_point_load_does_not_extend_a_fully_fixed_bar(self):
        nodes = {i: {"x": x, "y": 0, "fix_x": True, "fix_y": True, "Fx": 75}
                 for i, x in (("A", 0), ("B", 1))}
        self.assertEqual(solve(nodes, [bar("a", "A", "B")])["N_a_N"], 0.)
        for stiffness in (-1, 0):
            with self.assertRaises(ValueError):
                solve(nodes, [bar("a", "A", "B", stiffness)])
        with self.assertRaisesRegex(ValueError, "IDs"):
            solve(nodes, [bar("a", "A", "B"), bar("a", "A", "B")])
