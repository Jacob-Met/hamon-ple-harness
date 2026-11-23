"""Exercise proper enclosed IFC void shells through the public clash router."""
import re
import unittest

import numpy as np

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash
from test_ifc_faceted_bounds import FacetedScene


def rectangle(low, high):
    x, y = low[:2]
    X, Y = high[:2]
    return [(x, y), (X, y), (X, Y), (x, Y)]


def void_scene(voids, *, matrix=None, shift=None, scale=1., inward=True):
    scene = FacetedScene(matrix=matrix, shift=shift, scale=scale)
    scene.prism(rectangle((0, 0), (4, 4)), z0=0., z1=4.)
    body = next(i for i, row in enumerate(scene.rows) if "=IFCFACETEDBREP(" in row)
    outer = re.search(r"IFCFACETEDBREP\((#\d+)\)", scene.rows[body]).group(1)
    shell_refs = []
    for low, high in voids:
        start = len(scene.rows)
        scene.prism(rectangle(low, high), z0=low[2], z1=high[2])
        shell = next(i for i in range(start, len(scene.rows)) if "=IFCCLOSEDSHELL(" in scene.rows[i])
        del scene.rows[shell + 1:]
        if inward:
            for i in range(start, shell):
                if "=IFCFACEOUTERBOUND(" in scene.rows[i]:
                    scene.rows[i] = scene.rows[i].replace(",.T.)", ",.F.)")
        shell_refs.append(f"#{shell + 1}")
    scene.rows[body] = f"#{body + 1}=IFCFACETEDBREPWITHVOIDS({outer},({','.join(shell_refs)}));"
    return scene


class FacetedVoidShells(unittest.TestCase):
    def test_void_and_wall_probe_have_different_material_membership(self):
        for x, hard, clearance in [(1.75, False, .75), (.25, True, 0.)]:
            with self.subTest(x=x):
                scene = void_scene([((1, 1, 1), (3, 3, 3))])
                scene.prism(rectangle((x, 1.75), (x + .5, 2.25)), z0=1.75, z1=2.25,
                            kind="IFCDUCTSEGMENT", name="probe")
                result = run_router_ifc_clash({"ifc_text": scene.text(), "required_clearance_m": .5})
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["hard_clash_count"], int(hard))
                self.assertAlmostEqual(result["min_clearance_m"], clearance)
                host, _ = load_ifc_solids(scene.text())
                self.assertEqual(host.volume_m3(), 56.)
                self.assertEqual(host.surface_area_m2(), 120.)
                self.assertEqual(tuple(host.centroid_m()), (2., 2., 2.))

    def test_two_voids_under_shear_units_and_survey_translation(self):
        transform = np.array([[1., .5, 0.], [0., 1., .25], [0., 0., 1.]])
        shift = np.array([729000., 9050000., 120.])
        for scale in (1., 1000.):
            with self.subTest(scale=scale):
                scene = void_scene([((.5, .75, .5), (1.5, 1.75, 1.5)),
                                    ((2., 1., 1.), (3.5, 3., 2.))],
                                   matrix=transform, shift=shift, scale=scale)
                host, = load_ifc_solids(scene.text())
                self.assertAlmostEqual(host.volume_m3(), 60.)
                first_center = np.array([1., 1.25, 1.])
                second_center = np.array([2.75, 2., 1.5])
                center = (64 * np.array([2., 2., 2.]) - first_center - 3 * second_center) / 60
                np.testing.assert_allclose(host.centroid_m(), transform @ center + shift, rtol=0, atol=2e-9)

    def test_one_representable_gap_is_distinct_from_touching(self):
        first = ((.5, 1., 1.), (1.5, 2., 2.))
        for x, accepted in [(np.nextafter(1.5, 2.), True), (1.5, False)]:
            with self.subTest(x=x):
                scene = void_scene([first, ((x, 1., 1.), (3., 2., 2.))])
                result = run_router_ifc_clash({"ifc_text": scene.text()})
                self.assertEqual(result["ok"], accepted, result)
                if accepted:
                    self.assertAlmostEqual(result["total_solid_volume_m3"], 61.5)
                else:
                    self.assertIsNone(result.get("total_solid_volume_m3"))

    def test_touching_escaping_nested_and_wrong_winding_stay_unresolved(self):
        first = ((1, 1, 1), (3, 3, 3))
        cases = {
            "touches exterior": ([((0, 1, 1), (2, 2, 2))], True),
            "escapes": ([((3, 1, 1), (5, 2, 2))], True),
            "disjoint outside": ([((5, 1, 1), (6, 2, 2))], True),
            "nested": ([first, ((1.5, 1.5, 1.5), (2, 2, 2))], True),
            "overlap": ([first, ((2.5, 2.5, 2.5), (3.5, 3.5, 3.5))], True),
            "touches other": ([first, ((3, 1, 1), (3.5, 2, 2))], True),
            "outward cavity": ([first], False),
        }
        for name, (voids, inward) in cases.items():
            with self.subTest(name=name):
                result = run_router_ifc_clash({"ifc_text": void_scene(voids, inward=inward).text()})
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "unresolved_solid_geometry")
                self.assertIsNone(result.get("total_solid_volume_m3"))
