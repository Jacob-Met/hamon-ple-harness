"""Concave IFC cavity boundaries checked against independent prism measures."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash
from hamon_shell_relations import certify_void_shells
from test_ifc_faceted_bounds import FacetedScene


SQUARE = [(0., 0.), (6., 0.), (6., 6.), (0., 6.)]
L_VOID = [(1., 1.), (5., 1.), (5., 2.), (2., 2.), (2., 5.), (1., 5.)]
L_OUTER = [(0., 0.), (6., 0.), (6., 2.), (2., 2.), (2., 6.), (0., 6.)]
L_INNER = [(.5, .5), (5.5, .5), (5.5, 1.5), (1.5, 1.5), (1.5, 5.5), (.5, 5.5)]


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def cavity_scene(outer=SQUARE, cavities=None, *, inward=True, **options):
    cavities = [(L_VOID, 1., 3.)] if cavities is None else cavities
    scene = FacetedScene(**options).prism(outer, z0=0., z1=4.)
    body = next(i for i, row in enumerate(scene.rows) if "=IFCFACETEDBREP(" in row)
    exterior = re.search(r"IFCFACETEDBREP\((#\d+)\)", scene.rows[body]).group(1)
    shells = []
    for ring, z0, z1 in cavities:
        start = len(scene.rows)
        scene.prism(ring, z0=z0, z1=z1)
        shell = next(i for i in range(start, len(scene.rows)) if "=IFCCLOSEDSHELL(" in scene.rows[i])
        del scene.rows[shell + 1:]
        if inward:
            for i in range(start, shell):
                if "=IFCFACEOUTERBOUND(" in scene.rows[i]:
                    scene.rows[i] = scene.rows[i].replace(",.T.)", ",.F.)")
        shells.append(f"#{shell + 1}")
    scene.rows[body] = f"#{body + 1}=IFCFACETEDBREPWITHVOIDS({exterior},({','.join(shells)}));"
    return scene


class ConcaveVoidShells(unittest.TestCase):
    def test_concave_cavity_and_its_notch_have_opposite_material_membership(self):
        for x, expected_hard, clearance in [(1.5, 0, .25), (3., 1, 0.)]:
            with self.subTest(x=x):
                scene = cavity_scene()
                scene.prism(rectangle(x - .25, 2.75, x + .25, 3.25), z0=1.75, z1=2.25,
                            kind="IFCDUCTSEGMENT", name="probe")
                report = run_router_ifc_clash({"ifc_text": scene.text(), "required_clearance_m": .3})
                self.assertTrue(report["ok"], report)
                self.assertEqual(report["hard_clash_count"], expected_hard)
                self.assertAlmostEqual(report["min_clearance_m"], clearance)
                host, _ = load_ifc_solids(scene.text())
                self.assertAlmostEqual(host.volume_m3(), 6 * 6 * 4 - 7 * 2)
                self.assertAlmostEqual(host.surface_area_m2(), 168 + 2 * 7 + 16 * 2)
                np.testing.assert_allclose(host.centroid_m(), [399 / 130, 399 / 130, 2.], rtol=0, atol=1e-12)

    def test_concave_exterior_and_cavity_under_shear_survey_origin_and_units(self):
        transform = np.array([[1., .5, 0.], [0., 1., .25], [0., 0., 1.]])
        shift = np.array([729000., 9050000., 120.])
        for scale in (1., 1000.):
            with self.subTest(scale=scale):
                scene = cavity_scene(L_OUTER, [(L_INNER, 1., 3.)], matrix=transform, shift=shift, scale=scale)
                host, = load_ifc_solids(scene.text())
                self.assertAlmostEqual(host.volume_m3(), 20 * 4 - 9 * 2, places=7)
                np.testing.assert_allclose(host.centroid_m(), transform @ [138 / 62, 138 / 62, 2.] + shift,
                                           rtol=0, atol=2e-9)

    def test_disjoint_voids_need_not_have_disjoint_convex_hulls(self):
        elbow = [(1., 1.), (4., 1.), (4., 1.75), (1.75, 1.75), (1.75, 4.), (1., 4.)]
        scene = cavity_scene(cavities=[(elbow, 1., 3.), (rectangle(2.5, 2.5, 3.5, 3.5), 1., 3.)])
        host, = load_ifc_solids(scene.text())
        self.assertAlmostEqual(host.volume_m3(), 144 - 2 * ((3 * .75 * 2 - .75 ** 2) + 1))

    def test_one_representable_gap_is_not_contact_inside_overlapping_bounds(self):
        for x, accepted in [(np.nextafter(2., 3.), True), (2., False), (1.99, False)]:
            with self.subTest(x=x):
                scene = cavity_scene(cavities=[(L_VOID, 1., 3.), (rectangle(x, 2.5, 3.5, 3.5), 1., 3.)])
                report = run_router_ifc_clash({"ifc_text": scene.text()})
                self.assertEqual(report["ok"], accepted, report)
                if not accepted:
                    self.assertIsNone(report.get("total_solid_volume_m3"))

    def test_cavity_crossing_exterior_is_rejected_even_when_every_vertex_is_inside(self):
        u_outer = [(0., 0.), (6., 0.), (6., 6.), (4., 6.), (4., 2.), (2., 2.), (2., 6.), (0., 6.)]
        scene = cavity_scene(u_outer, [(rectangle(1., 3., 5., 5.), 1., 3.)])
        report = run_router_ifc_clash({"ifc_text": scene.text()})
        self.assertFalse(report["ok"], report)
        self.assertIsNone(report.get("total_solid_volume_m3"))

    def test_nested_touching_external_and_outward_cavities_stay_unresolved(self):
        cases = [
            cavity_scene(cavities=[(L_VOID, 1., 3.), (rectangle(1.25, 2.5, 1.75, 3.5), 1.5, 2.5)]),
            cavity_scene(cavities=[(L_VOID, 0., 3.)]),
            cavity_scene(cavities=[([(x + 8, y) for x, y in L_VOID], 1., 3.)]),
            cavity_scene(inward=False),
        ]
        for scene in cases:
            with self.subTest(scene=scene.rows[-1]):
                report = run_router_ifc_clash({"ifc_text": scene.text()})
                self.assertFalse(report["ok"], report)
                self.assertIsNone(report.get("total_solid_volume_m3"))

    def test_predicate_budget_refuses_without_partial_material(self):
        scene = cavity_scene()
        from hamon_ifc_spatial_clash import _faceted_solid_mesh, parse_ifc_entities
        entities = parse_ifc_entities(scene.text())
        _, args = next(value for value in entities.values() if value[0] == "IFCFACETEDBREPWITHVOIDS")
        shells = [_faceted_solid_mesh(entities, [ref], 1.)[0] for ref in [args[0], *args[1]]]
        with self.assertRaisesRegex(ValueError, "budget"):
            certify_void_shells(shells, max_checks=10)

    def test_ifc_round_trip_and_independent_boolean_kernel_measures(self):
        if importlib.util.find_spec("ifcopenshell") is None:
            self.skipTest("IfcOpenShell is installed by the full CI requirements")
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.shape
        for scene, volume, area in [(cavity_scene(), 130., 214.),
                                   (cavity_scene(L_OUTER, [(L_INNER, 1., 3.)]), 62., 194.)]:
            model = ifcopenshell.file.from_string(scene.text())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "concave-cavity.ifc"
                model.write(str(path))
                reopened = ifcopenshell.open(str(path))
                brep, = reopened.by_type("IfcFacetedBrepWithVoids")
                self.assertEqual(len(brep.Voids), 1)
                report = run_router_ifc_clash({"path": str(path)})
                self.assertTrue(report["ok"], report)
                self.assertAlmostEqual(report["total_solid_volume_m3"], volume)
                host, = load_ifc_solids(path.read_text())

                # The 0.9.0.post1 wheel rejects multiple shells in a solid even
                # for a rectangular cavity. Keep the original export above;
                # in a separate model, ask OpenCASCADE to subtract the same
                # independently parsed shell boundaries as ordinary Breps.
                # This verifies the material measures, not direct kernel
                # support for IfcFacetedBrepWithVoids.
                oracle = ifcopenshell.file.from_string(reopened.to_string())
                oracle_brep, = oracle.by_type("IfcFacetedBrepWithVoids")
                material = oracle.create_entity("IfcFacetedBrep", Outer=oracle_brep.Outer)
                for shell in oracle_brep.Voids:
                    for face in shell.CfsFaces:
                        for bound in face.Bounds:
                            bound.Orientation = not bound.Orientation
                    cavity = oracle.create_entity("IfcFacetedBrep", Outer=shell)
                    material = oracle.create_entity("IfcBooleanResult", Operator="DIFFERENCE",
                                                    FirstOperand=material, SecondOperand=cavity)
                product, = oracle.by_type("IfcBeam")
                representation, = product.Representation.Representations
                representation.Items = (material,)
                representation.RepresentationType = "CSG"
                oracle_path = Path(folder) / "independent-boolean-oracle.ifc"
                oracle.write(str(oracle_path))
                oracle_reopened = ifcopenshell.open(str(oracle_path))
                oracle_product, = oracle_reopened.by_type("IfcBeam")
                shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), oracle_product)
                kernel_volume = ifcopenshell.util.shape.get_volume(shape.geometry)
                kernel_area = ifcopenshell.util.shape.get_area(shape.geometry)
                self.assertAlmostEqual(kernel_volume, volume)
                self.assertAlmostEqual(kernel_area, area)
                self.assertAlmostEqual(host.volume_m3(), kernel_volume)
                self.assertAlmostEqual(host.surface_area_m2(), kernel_area)

    def test_actual_registered_tool_reads_and_qualifies_cavity_files(self):
        root = Path(__file__).resolve().parent.parent
        if not (root / "node_modules").is_dir():
            self.skipTest("registered Node runtime is installed by the full CI workflow")
        from test_wired_router_e2e import _start_mock_relay
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / "concave.ifc").write_text(cavity_scene().text())
            u_outer = [(0., 0.), (6., 0.), (6., 6.), (4., 6.), (4., 2.), (2., 2.), (2., 6.), (0., 6.)]
            crossing = cavity_scene(u_outer, [(rectangle(1., 3., 5., 5.), 1., 3.)])
            (work / "crossing.ifc").write_text(crossing.text())
            relay = _start_mock_relay(work, calls)
            try:
                script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(tool) {tools.set(tool.name, tool);}, on() {}});
const reports = [];
for (const path of ['concave.ifc', 'crossing.ifc']) {
  const result = await tools.get('hamon_ifc_spatial_clash').execute(
    path, {path}, null, () => {}, {});
  reports.push(result.details);
}
console.log(JSON.stringify(reports));
"""
                env = {**os.environ, "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}",
                       "TASK_WORKDIR": str(work), "TASK_HARNESS_PYTHON": sys.executable,
                       "TASK_HARNESS_SRC": str(root)}
                result = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                accepted, refused = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertTrue(accepted["ok"], accepted)
                self.assertEqual(accepted["total_solid_volume_m3"], 130.)
                self.assertFalse(refused["ok"], refused)
                self.assertEqual(refused["status"], "unresolved_solid_geometry")
                self.assertIsNone(refused.get("total_solid_volume_m3"))
                self.assertEqual(calls, ["read_file", "read_file"])
            finally:
                relay.shutdown()
                relay.server_close()


if __name__ == "__main__":
    unittest.main()
