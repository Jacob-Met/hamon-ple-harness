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

    def test_individual_force_rounding_can_hide_a_one_newton_joint_load(self):
        nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True},
                 "B": {"x": 1, "y": 0, "Fx": 1},
                 "C": {"x": 2, "y": 0, "Fx": 2**53, "fix_y": True}}
        result = solve(nodes, [bar("a", "A", "B", 2**61), bar("b", "B", "C", 2**61)])
        self.assertEqual(result["N_a_N"], result["N_b_N"])
        self.assertEqual(result["displayed_free_joint_residual_upper_N"], 1.)
        self.assertEqual(result["equilibrium_basis"], "represented_member_factors_before_force_rounding")

    def test_restrained_settlement_and_thermal_strain(self):
        members = [bar("a", "A", "B", 200000)]
        for motion, change, expected in ((0,0,0), (.001,0,100), (-.001,0,-100),
                                         (0,50,-100), (0,-50,100), (.001,50,0)):
            nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True},
                     "B": {"x": 2, "y": 0, "fix_x": True, "fix_y": True, "ux_m": motion}}
            members[0].update(alpha_per_K=1e-5, dT_K=change)
            with self.subTest(motion=motion, change=change):
                result = solve(nodes, members)
                self.assertTrue(result["displacement_unique"] and result["dual_verified"])
                self.assertAlmostEqual(result["N_a_N"], expected, places=11)

    def test_free_expansion_and_motion_on_an_unrestrained_component(self):
        nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True},
                 "B": {"x": 1, "y": 0}}
        hot = {**bar("a", "A", "B", 64), "alpha_per_K": 1/16, "dT_K": 2}
        result = solve(nodes, [hot])
        self.assertEqual((result["N_a_N"], result["free_mechanism_count"]), (0,1))
        nodes["B"]["ux_m"] = 1/8
        with self.assertRaisesRegex(ValueError, "restrained"):
            solve(nodes, [hot])

    def test_affine_preload_equivalence_with_a_free_mechanism(self):
        nodes = {"A": {"x": 0, "y": 0, "fix_x": True, "fix_y": True, "ux_m": 1/16},
                 "B": {"x": 1, "y": 0, "Fx": 4},
                 "C": {"x": 2, "y": 0, "fix_x": True, "fix_y": True, "ux_m": -1/16}}
        members = [{**bar("a", "A", "B", 64), "alpha_per_K": 1/16, "dT_K": 2},
                   {**bar("b", "B", "C", 128), "alpha_per_K": 1/32, "dT_K": -1}]
        result = solve(nodes, members)
        self.assertEqual((result["N_a_N"], result["N_b_N"]), (-8, -12))
        self.assertEqual(result["free_mechanism_count"], 1)
        # With B temporarily fixed at zero the constrained bars carry -12 N
        # and -4 N. Releasing B adds 8 N to its external load in a homogeneous solve.
        homogeneous = {name: dict(row, ux_m=0) for name,row in nodes.items()}
        homogeneous["B"]["Fx"] += 8
        equivalent = solve(homogeneous, [dict(m, dT_K=0) for m in members])
        self.assertEqual((equivalent["N_a_N"]-12, equivalent["N_b_N"]-4), (-8, -12))
        nodes["B"]["Fy"] = 1/1024
        with self.assertRaisesRegex(ValueError, "virtual work"):
            solve(nodes, members)
