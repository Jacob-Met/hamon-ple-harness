"""Source-authored IFC set operations checked against analytic material moments.

These are geometric controls, not competition-task predictions. Cuboids and
tetrahedra provide independent inclusion/exclusion oracles. The small-gap,
cavity and thin-cut cases distinguish the desired set from a plausible mesh.
"""
import itertools
import math
import sys
import unittest
from unittest import mock

import numpy as np

try:
    import manifold3d
except ImportError as exc:
    raise unittest.SkipTest(f"optional polyhedral Boolean dependency unavailable: {exc}")

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash
from test_ifc_compound_mass import course
from test_ifc_mapped_mass import Scene


def box(scene, low, size, *, unit=1., primitive=False):
    low, size = np.asarray(low, dtype=float), np.asarray(size, dtype=float)
    if primitive:
        frame = scene.frame(np.eye(3), low * unit)
        return scene.add(f"IFCBLOCK(#{frame},{size[0]*unit},{size[1]*unit},{size[2]*unit})")
    profile = scene.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{size[0]*unit},{size[1]*unit})")
    direction = scene.vector("IFCDIRECTION", [0, 0, 1])
    frame = scene.frame(np.eye(3), (low + [size[0]/2, size[1]/2, 0]) * unit)
    return scene.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{frame},#{direction},{size[2]*unit})")


def material_course(operator, *, millimetres=False, mapped=False, oblique=False, primitive=False):
    unit = 1000. if millimetres else 1.
    scene = Scene()
    a = box(scene, [0, 0, 0], [2, 3, 4], unit=unit, primitive=primitive)
    b = box(scene, [.5, .75, 1.25], [2, 1.5, 2.5], unit=unit, primitive=primitive)
    root = scene.add(f"IFCBOOLEANRESULT(.{operator}.,#{a},#{b})")
    root = scene.add(f"IFCCSGSOLID(#{root})")
    transform, shift = np.eye(3), np.zeros(3)
    if mapped:
        transform = np.diag([1.25, -1.25, 1.25])
        shift = np.array([.125, 1.75, -.5])
        root = scene.map(root, np.diag([1., -1., 1.]), shift * unit, 1.25)
    angle = .371 if oblique else 0.
    co, si = math.cos(angle), math.sin(angle)
    rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1.]]) @ np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    translation = np.array([729011.25, -473.5, 16.125]) if oblique else np.array([3., -2., 5.])
    scene.product(root, rotation, translation * unit)
    prefix = ".MILLI." if millimetres else "$"
    length = scene.add(f"IFCSIUNIT(*,.LENGTHUNIT.,{prefix},.METRE.)")
    scene.add(f"IFCUNITASSIGNMENT((#{length}))")
    # A, B, and A intersect B, all in the original local metre frame.
    volumes = np.array([24., 7.5, 5.625])
    centres = np.array([[1.,1.5,2.], [1.5,1.5,2.5], [1.25,1.5,2.5]])
    coefficients = {"UNION":[1,1,-1], "INTERSECTION":[0,0,1], "DIFFERENCE":[1,0,-1]}[operator]
    weights = volumes * coefficients
    centroid = (weights[:, None] * centres).sum(axis=0) / weights.sum()
    return scene.text(), weights.sum() * abs(np.linalg.det(transform)), rotation @ (transform @ centroid + shift) + translation


class PolyhedralBooleans(unittest.TestCase):
    def test_analytic_operations_units_frames_and_primitives(self):
        for operator, millimetres, mapped, oblique, primitive in itertools.product(
                ("UNION", "INTERSECTION", "DIFFERENCE"), (False, True), (False, True), (False, True), (False, True)):
            options = dict(millimetres=millimetres, mapped=mapped, oblique=oblique, primitive=primitive)
            with self.subTest(operator=operator, **options):
                text, volume, centroid = material_course(operator, **options)
                result = run_router_ifc_clash({"ifc_text": text})
                self.assertTrue(result["ok"], result)
                self.assertAlmostEqual(result["total_solid_volume_m3"] / volume, 1., places=9)
                mesh, = load_ifc_solids(text)
                np.testing.assert_allclose(mesh.centroid_m(), centroid, rtol=0, atol=2e-9)

    def test_overlapping_tetrahedra_reversed_and_with_a_partly_filled_cavity(self):
        for reverse, millimetres, swap, cavity in itertools.product((False, True), repeat=4):
            with self.subTest(reverse=reverse, millimetres=millimetres, swap=swap, cavity=cavity):
                text = course(shift=.25, reverse=reverse, millimetres=millimetres, swap=swap, cavity=cavity)
                weights = [1/6, 8/6, -9/128]
                centres = [[.25,.25,.25], [.75,.5,.5], [.4375,.1875,.1875]]
                if cavity:
                    weights += [-1/384, 1/3072]
                    centres += [[3/16]*3, [9/32,5/32,5/32]]
                mesh, = load_ifc_solids(text)
                self.assertAlmostEqual(mesh.volume_m3(), math.fsum(weights), places=12)
                np.testing.assert_allclose(mesh.centroid_m(), np.average(centres, weights=weights, axis=0),
                                           rtol=0, atol=1e-12)

    def test_coincident_independent_items_are_material_union(self):
        for reverse in (False, True):
            mesh, = load_ifc_solids(course(shift=0., right_scale=1., reverse=reverse))
            self.assertAlmostEqual(mesh.volume_m3(), 1/6, places=12)
            np.testing.assert_allclose(mesh.centroid_m(), [.25]*3, rtol=0, atol=1e-12)

    def test_nested_void_and_added_material_control_actual_clash(self):
        for probe, status, distance in [([2.5,2.,2.], "clear", .375), ([1.5,2.,2.], "hard_clash", 0.)]:
            scene = Scene()
            outer = box(scene, [0,0,0], [4,4,4])
            cavity = box(scene, [1,1,1], [2,2,2])
            added = box(scene, [1,1,1], [1,2,2])
            cut = scene.add(f"IFCBOOLEANRESULT(.DIFFERENCE.,#{outer},#{cavity})")
            union = scene.add(f"IFCBOOLEANRESULT(.UNION.,#{cut},#{added})")
            scene.product(union, np.eye(3), [0,0,0])
            small = box(scene, np.asarray(probe)-.125, [.25]*3)
            scene.product(small, np.eye(3), [0,0,0])
            scene.rows[-1] = scene.rows[-1].replace("IFCBEAM(", "IFCDUCTSEGMENT(")
            result = run_router_ifc_clash({"ifc_text":scene.text()})
            self.assertTrue(result["ok"], result)
            pair, = result["all_pairs"]
            self.assertEqual(pair["status"], status, pair)
            self.assertAlmostEqual(pair["min_distance_m"], distance, places=12)
            material = load_ifc_solids(scene.text())[0]
            self.assertAlmostEqual(material.volume_m3(), 60., places=12)
            self.assertAlmostEqual(material.surface_area_m2(), 112., places=12)
            np.testing.assert_allclose(material.centroid_m(), [118/60,2.,2.], rtol=0, atol=1e-12)

    def test_gap_and_thin_cut_do_not_become_a_repaired_solid(self):
        for operator, offset, volume, area in [
                ("UNION", 2.+1e-8, 16., 48.),
                ("DIFFERENCE", 1e-8, 4e-8, 8.+8e-8)]:
            scene = Scene()
            a = box(scene, [0,0,0], [2,2,2])
            b = box(scene, [offset,0,0], [2,2,2])
            root = scene.add(f"IFCBOOLEANRESULT(.{operator}.,#{a},#{b})")
            scene.product(root, np.eye(3), [0,0,0])
            mesh, = load_ifc_solids(scene.text())
            self.assertAlmostEqual(mesh.volume_m3(), volume, delta=1e-14)
            self.assertAlmostEqual(mesh.surface_area_m2(), area, delta=1e-12)

    def test_unresolved_operand_and_missing_kernel_never_produce_clearance(self):
        text, _, _ = material_course("UNION")
        for damaged in [text.replace(".UNION.", ".XOR."),
                        text.replace("IFCCSGSOLID(#", "IFCUNKNOWN(#")]:
            result = run_router_ifc_clash({"ifc_text":damaged})
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "unresolved_solid_geometry")
        with mock.patch.dict(sys.modules, {"manifold3d":None}):
            result = run_router_ifc_clash({"ifc_text":text})
            self.assertFalse(result["ok"], result)
            original = run_router_ifc_clash({"ifc_text":course(shift=.25,probe=True)})
            self.assertTrue(original["ok"], original)
            self.assertIsNone(original["total_solid_volume_m3"])
            self.assertEqual(original["volume_status"], "unresolved")
            self.assertEqual(original["all_pairs"][0]["status"], "clear")

    def test_cyclic_tree_and_empty_result_are_explicit_refusals(self):
        scene = Scene()
        a = box(scene, [0,0,0], [2,2,2])
        next_id = len(scene.rows) + 1
        cyclic = scene.add(f"IFCBOOLEANRESULT(.UNION.,#{a},#{next_id})")
        scene.product(cyclic, np.eye(3), [0,0,0])
        result = run_router_ifc_clash({"ifc_text":scene.text()})
        self.assertFalse(result["ok"], result)
        self.assertIn("cyclic", result["reason"])
        scene = Scene()
        a = box(scene, [0,0,0], [2,2,2])
        empty = scene.add(f"IFCBOOLEANRESULT(.DIFFERENCE.,#{a},#{a})")
        scene.product(empty, np.eye(3), [0,0,0])
        result = run_router_ifc_clash({"ifc_text":scene.text()})
        self.assertFalse(result["ok"], result)
        self.assertIsNone(result["total_solid_volume_m3"])


if __name__ == "__main__":
    unittest.main()
