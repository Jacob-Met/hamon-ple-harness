"""IFC4 indexed curve sweeps through the public spatial router and OpenCASCADE."""
import importlib.util
import json
import os
import subprocess
import sys
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash
from test_ifc_faceted_bounds import FacetedScene


SEMI = "IFCARCINDEX((1,2,3)),IFCLINEINDEX((3,1))"
POINTS = [(1., 0.), (0., 1.), (-1., 0.)]


def scene(points=POINTS, segments=SEMI, *, scale=1., self_intersect=".F."):
    doc = FacetedScene(scale=scale)
    coords = ",".join(f"({x*scale},{y*scale})" for x, y in points)
    table = doc.add(f"IFCCARTESIANPOINTLIST2D(({coords}))")
    segments = "$" if segments is None else f"({segments})"
    curve = doc.add(f"IFCINDEXEDPOLYCURVE(#{table},{segments},{self_intersect})")
    profile = doc.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{curve})")
    z = doc.add("IFCDIRECTION((0.,0.,1.))")
    solid = doc.add(f"IFCEXTRUDEDAREASOLID(#{profile},#2,#{z},{2.*scale})")
    rep = doc.add(f"IFCSHAPEREPRESENTATION(#{doc.context},'Body','SweptSolid',(#{solid}))")
    body = doc.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{rep}))")
    doc.add(f"IFCBEAM('0AzR0sseD4bBOcPJ50wSE2',$,'Arc beam',$,$,$,#{body},$,$)")
    return doc


class IndexedArcSweeps(unittest.TestCase):
    def test_semicircle_winding_units_and_unused_table_coordinates(self):
        for segments in (SEMI, "IFCLINEINDEX((1,3)),IFCARCINDEX((3,2,1))"):
            for scale in (1., 1000.):
                with self.subTest(segments=segments, scale=scale):
                    text = scene(POINTS + [(96., -70.)], segments, scale=scale).text()
                    mesh, = load_ifc_solids(text)
                    self.assertAlmostEqual(mesh.volume_m3(), math.pi, delta=.0007)
                    self.assertAlmostEqual(mesh.surface_area_m2(), 3*math.pi + 4, delta=.0015)
                    np.testing.assert_allclose(mesh.centroid_m(), [0., 4/(3*math.pi), 1.], atol=.0001)
                    report = run_router_ifc_clash({"ifc_text": text})
                    self.assertAlmostEqual(report["total_solid_volume_m3"], mesh.volume_m3())
                    self.assertIn("chords", report["geometry_basis"])

    def test_major_segment_through_interior_point(self):
        for segments in (SEMI, "IFCLINEINDEX((1,3)),IFCARCINDEX((3,2,1))"):
            text = scene([(1., 0.), (0., -1.), (0., 1.)], segments).text()
            mesh, = load_ifc_solids(text)
            self.assertAlmostEqual(mesh.volume_m3(), 1 + 1.5*math.pi, delta=.0011)

    def test_segments_define_edges_and_implicit_sequence_is_preserved(self):
        points = [(0., 0.), (2., 1.), (2., 0.), (0., 1.), (100., 100.)]
        ordered = scene(points, "IFCLINEINDEX((1,3,2)),IFCLINEINDEX((2,4,1))")
        direct = scene([points[i] for i in (0, 2, 1, 3, 0)], None)
        for example in (ordered, direct):
            self.assertAlmostEqual(run_router_ifc_clash({"ifc_text": example.text()})["total_solid_volume_m3"], 4.)

    def test_real_clash_outside_source_control_triangle(self):
        doc = scene()
        doc.prism([(.61, .61), (.66, .61), (.66, .66), (.61, .66)],
                  z0=.4, z1=.6, kind="IFCDUCTSEGMENT", name="duct")
        result = run_router_ifc_clash({"ifc_text": doc.text()})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["hard_clash_count"], 1, result)

    def test_invalid_or_disconnected_segment_data_abstains(self):
        for example in (
            scene(segments=SEMI.replace("(3,1)", "(2,1)")),
            scene(segments=SEMI.replace("(3,1)", "(3,0)")),
            scene(segments=SEMI.replace("(3,1)", "(3,4)")),
            scene(segments=SEMI.replace("(3,1)", "(3,3,1)")),
            scene(segments=SEMI.replace("(3,1)", "(3,2)")),
            scene(segments=SEMI.replace("(1,2,3)", "(1,2,2)")),
            scene(segments=SEMI.replace("IFCARCINDEX", "UNKNOWNINDEX")),
            scene([(0., 0.), (1., 0.), (2., 0.)]),
            scene(self_intersect=".T."),
            scene([(0., 0.), (1., 1.), (0., 1.), (1., 0.)], "IFCLINEINDEX((1,2,3,4,1))"),
        ):
            with self.subTest(example=example.rows[-8]):
                result = run_router_ifc_clash({"ifc_text": example.text()})
                self.assertFalse(result["ok"], result)
                self.assertIsNone(result["total_solid_volume_m3"])

    @unittest.skipUnless(importlib.util.find_spec("ifcopenshell"), "IfcOpenShell required")
    def test_exported_ifc4_profile_against_independent_kernel(self):
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.shape as quantities
        with tempfile.TemporaryDirectory() as folder:
            for scale in (1., 1000.):
                for points, expected in ((POINTS, math.pi), ([(1., 0.), (0., -1.), (0., 1.)], 1 + 1.5*math.pi)):
                    with self.subTest(scale=scale, expected=expected):
                        model = ifcopenshell.file.from_string(scene(points, scale=scale).text())
                        path = Path(folder) / "indexed-sweep.ifc"
                        model.write(str(path))
                        reopened = ifcopenshell.open(str(path))
                        beam, = reopened.by_type("IfcBeam")
                        settings = ifcopenshell.geom.settings()
                        settings.set("mesher-linear-deflection", .00001)
                        shape = ifcopenshell.geom.create_shape(settings, beam)
                        kernel_volume = quantities.get_volume(shape.geometry)
                        self.assertAlmostEqual(kernel_volume, expected, delta=.0011)
                        report = run_router_ifc_clash({"path": str(path)})
                        self.assertAlmostEqual(report["total_solid_volume_m3"], kernel_volume, delta=.0011)

    @unittest.skipUnless(os.getenv("PLE_GEOMETRY_GYM_SLAB"), "published IFC fixture is staged by CI")
    def test_published_geometry_gym_slab_through_registered_tool(self):
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.shape as quantities
        from test_wired_router_e2e import _start_mock_relay

        source = Path(os.environ["PLE_GEOMETRY_GYM_SLAB"]).resolve()
        root = Path(__file__).resolve().parents[1]
        slab_model = ifcopenshell.open(str(source))
        slab, = slab_model.by_type("IfcSlab")
        settings = ifcopenshell.geom.settings()
        settings.set("mesher-linear-deflection", .00001)
        shape = ifcopenshell.geom.create_shape(settings, slab)
        kernel = quantities.get_volume(shape.geometry)

        # The plan is a 1 m x 4 m rectangle with a 0.4 m sagitta on each
        # long side. The negative extrusion is 200 mm deep.
        radius = (2**2 + .4**2) / (.8)
        sweep = 2 * math.asin(2 / radius)
        expected = .2 * (4 + radius**2 * (sweep - math.sin(sweep)))
        self.assertAlmostEqual(kernel, expected, delta=.0001)
        commands = []
        relay = _start_mock_relay(source.parent, commands)
        try:
            script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(tool) { tools.set(tool.name, tool); }, on() {}});
const result = await tools.get('hamon_ifc_spatial_clash').execute(
  'slab', {path: process.argv[1]}, null, () => {}, {});
console.log(JSON.stringify(result.details));
"""
            env = {**os.environ, "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}",
                   "TASK_WORKDIR": str(source.parent), "TASK_HARNESS_PYTHON": sys.executable,
                   "TASK_HARNESS_SRC": str(root)}
            proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                                   "-e", script, source.name], cwd=root, env=env,
                                  text=True, capture_output=True, timeout=30, check=True)
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["solid_count"], 1)
            self.assertEqual(commands, ["read_file"])
            self.assertAlmostEqual(report["total_solid_volume_m3"], expected, delta=.001)
            print(json.dumps({"source": "Geometry Gym IfcScript Slab.ifc",
                              "registered_volume_m3": report["total_solid_volume_m3"],
                              "kernel_volume_m3": kernel, "analytic_volume_m3": expected,
                              "control_polygon_volume_m3": 1.12}, sort_keys=True))
        finally:
            relay.shutdown()
            relay.server_close()
