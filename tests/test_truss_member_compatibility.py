"""Check the physical directions and constitutive response lost in BᵀKB assembly."""
import math
import unittest

from hamon_pde_control_psychro_fem import verify_fem_structure


def truss(y, reflect_y=1, reverse=False, reorder=False, split=False, shift=0.0, thermal=0.0):
    # Scale the transverse load with the angle so displacements stay small.
    force = (y - 1.0)**2 * 1e-6 if y != 1.0 else 1e-20
    elements = [{"i": 0, "j": 1, "E_pa": 10.0, "A_m2": 1.0}]
    if split:
        elements.extend({"i": 0, "j": 2, "E_pa": 10.0, "A_m2": 0.5,
                         "alpha_per_K": thermal, "dT_K": dt} for dt in (-1.0, 1.0))
    else:
        elements.append({"i": 0, "j": 2, "E_pa": 10.0, "A_m2": 1.0})
    if reverse:
        for element in elements:
            element["i"], element["j"] = element["j"], element["i"]
    if reorder:
        elements.reverse()
    return {"domain": "fem", "mode": "truss_2d", "nodes": [[0, 0], [1, reflect_y], [1, reflect_y*y]],
            "elements": elements, "loads": [{"node": 0, "fx_N": force, "fy_N": -reflect_y * force}],
            "supports": [{"node": i, "fix_x": True, "fix_y": True, "ux_m": shift, "uy_m": -reflect_y*shift}
                         for i in (1, 2)]}


def analytical(y):
    # Joint statics gives each bar's horizontal force. Compatibility then
    # determines the two free displacements without building a stiffness matrix.
    delta = y - 1.0
    force = delta**2 * 1e-6
    p2 = 2.0 * force / delta
    p1 = -force - p2
    length1, length2 = math.sqrt(2.0), math.hypot(1.0, y)
    n1, n2 = p1 * length1, p2 * length2
    c1, c2 = -n1 * length1**2 / 10.0, -n2 * length2**2 / 10.0
    dy = (c2 - c1) / delta
    return c1 - dy, dy, n1, n2


class TrussCompatibilityTests(unittest.TestCase):
    def test_near_parallel_bars_preserve_motion_force_and_anchor_reactions(self):
        for delta in (1e-6, 1e-7, 1e-8):
            y = 1.0 + delta
            dx, dy, n1, n2 = analytical(y)
            for reflection in (-1, 1):
                for reverse in (False, True):
                    for reorder in (False, True):
                        with self.subTest(delta=delta, reflection=reflection, reverse=reverse, reorder=reorder):
                            result = verify_fem_structure(truss(y, reflection, reverse, reorder))
                            self.assertTrue(result["passed"])
                            expected = [n2, n1] if reorder else [n1, n2]
                            for actual, reference in zip(result["axial_forces_N"], expected):
                                self.assertTrue(math.isclose(actual, reference, rel_tol=1e-7))
                            u = result["displacements_xy_m"][0]
                            self.assertTrue(math.isclose(u[0], dx, rel_tol=1e-7))
                            self.assertTrue(math.isclose(u[1], reflection*dy, rel_tol=1e-7))
                            for i, (force, height) in enumerate(((n1, 1.0), (n2, y)), 1):
                                horizontal = force / math.hypot(1.0, height)
                                reaction = result["reactions_xy_N"][i]
                                self.assertTrue(math.isclose(reaction[0], horizontal, rel_tol=1e-7))
                                self.assertTrue(math.isclose(reaction[1], reflection*height*horizontal, rel_tol=1e-7))

    def test_redundant_members_keep_thermal_self_stress_and_support_motion(self):
        y, shift, thermal = 1.0 + 1e-7, 1e-8, 1e-7
        dx, dy, n1, n2 = analytical(y)
        result = verify_fem_structure(truss(y, split=True, shift=shift, thermal=thermal))
        self.assertTrue(result["passed"])
        u = result["displacements_xy_m"][0]
        self.assertTrue(math.isclose(u[0], dx+shift, rel_tol=1e-7))
        self.assertTrue(math.isclose(u[1], dy-shift, rel_tol=1e-7))
        total1, cold, hot = result["axial_forces_N"]
        self.assertTrue(math.isclose(total1, n1, rel_tol=1e-7))
        self.assertTrue(math.isclose(cold+hot, n2, rel_tol=1e-7))
        self.assertTrue(math.isclose(cold-hot, 10.0*thermal, rel_tol=1e-12))

    def test_force_free_rigid_translation_and_compatible_thermal_expansion(self):
        y = 1.0 + 1e-7
        for split in (False, True):
            for shift in (1e-8, 1e-7, 1.0):
                case = truss(y, split=split, shift=shift)
                case["loads"] = []
                result = verify_fem_structure(case)
                self.assertTrue(result["passed"])
                self.assertEqual(result["displacements_xy_m"][0], [shift, -shift])
                self.assertEqual(result["axial_forces_N"], [0.0] * len(case["elements"]))
                self.assertEqual(result["strain_energy_J"], 0.0)
            strain = 1e-7
            case = truss(y, split=split)
            case["loads"] = []
            for el in case["elements"]:
                el.update(alpha_per_K=strain, dT_K=1)
            result = verify_fem_structure(case)
            self.assertTrue(result["passed"])
            self.assertEqual(result["axial_forces_N"], [0.0] * len(case["elements"]))
            self.assertEqual(result["reactions_xy_N"], [[0.0, 0.0]] * 3)
            dx, dy = result["displacements_xy_m"][0]
            error = math.hypot(dx - strain*(y-1), dy + strain*(y+1))
            self.assertLess(error, 1e-7 * math.hypot(dx, dy))

    def test_unrelated_thermal_self_stress_cannot_mask_a_local_incompatibility(self):
        case = truss(1.0 + 1e-9)
        original = verify_fem_structure(case)
        case["nodes"].extend([[2, 0], [3, 0]])
        case["supports"].extend({"node": i, "fix_x": True, "fix_y": True} for i in (3, 4))
        case["elements"].append({"i": 3, "j": 4, "E_pa": 1e9, "A_m2": 1,
                                 "alpha_per_K": 1e-5, "dT_K": 10})
        self.assertEqual(verify_fem_structure(case)["passed"], original["passed"])

    def test_last_distinct_binary64_anchor_and_half_ulp_collapse(self):
        for y in (math.nextafter(1.0, 2.0), 1.0 + 2*math.ulp(1.0)):
            result = verify_fem_structure(truss(y))
            self.assertTrue(result["passed"])
            dx, dy, n1, n2 = analytical(y)
            values = result["displacements_xy_m"][0] + result["axial_forces_N"]
            for actual, expected in zip(values, (dx, dy, n1, n2)):
                self.assertTrue(math.isclose(actual, expected, rel_tol=1e-14))
        collapsed_y = 1.0 + math.ulp(1.0)/2
        self.assertEqual(collapsed_y, 1.0)
        self.assertFalse(verify_fem_structure(truss(collapsed_y))["passed"])

    def test_exact_transverse_mechanism_is_rejected_in_every_order(self):
        for reflection in (-1, 1):
            for reorder in (False, True):
                result = verify_fem_structure(truss(1.0, reflection, reorder=reorder, split=True))
                self.assertFalse(result["passed"])
