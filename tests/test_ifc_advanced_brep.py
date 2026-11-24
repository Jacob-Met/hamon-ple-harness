"""Verify the buildingSMART Annex E twisted B-spline solid against its section integral."""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash


@unittest.skipUnless(os.getenv("PLE_ANNEX_E_TWISTED_CUBE"), "buildingSMART IFC model is staged by CI")
class PublishedAdvancedBrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ["PLE_ANNEX_E_TWISTED_CUBE"]).read_text()

    def test_original_nurbs_and_placement_against_integrated_sections(self):
        # Each of the four spline boundaries interpolates two unit-square
        # corners linearly in height. The upper square has rotated 60 degrees.
        # Horizontal section area = (1-z)^2 + z^2 + 2 z (1-z) cos(60 degrees).
        expected_volume = 5 / 6
        for displacement in ((0., 0., 0.), (234.5, -17., 901.)):
            extra = (f"#900=IFCCARTESIANPOINT(({','.join(map(str, displacement))}));\n"
                     "#901=IFCAXIS2PLACEMENT3D(#900,$,$);\n")
            text = self.source.replace("#184= IFCLOCALPLACEMENT($,#11);",
                                       extra + "#184= IFCLOCALPLACEMENT($,#901);")
            mesh, = load_ifc_solids(text)
            np.testing.assert_allclose(mesh.centroid_m(), np.array(displacement) + [0., 0., .5], atol=.002)
            report = run_router_ifc_clash({"ifc_text": text})
            self.assertTrue(report["ok"], report)
            self.assertAlmostEqual(report["total_solid_volume_m3"], expected_volume, delta=.00167)
            print(json.dumps({"published_model": "buildingSMART Annex E twisted cube",
                              "analytic_m3": expected_volume, "router_m3": report["total_solid_volume_m3"],
                              "faces": len(mesh.faces), "displacement": displacement}, sort_keys=True))

    def test_removed_and_duplicate_source_face_do_not_produce_material(self):
        shell = "#29= IFCCLOSEDSHELL((#104,#115,#131,#147,#163,#179));"
        for altered in (shell.replace(",#179", ""), shell.replace("#179", "#163")):
            self.assertIn(shell, self.source)
            result = run_router_ifc_clash({"ifc_text": self.source.replace(shell, altered)})
            self.assertFalse(result["ok"], result)
            self.assertIsNone(result["total_solid_volume_m3"])
