"""IFC4 indexed curve sweeps through the public spatial router and OpenCASCADE."""
import importlib.util
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
