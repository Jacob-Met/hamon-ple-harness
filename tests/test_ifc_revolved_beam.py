"""Pappus volume controls for an authored revolved IPE600 and its sweep parameters."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_ifc_spatial_clash import run_router_ifc_clash


@unittest.skipUnless(os.environ.get("PLE_REVOLVED_IPE"), "official Annex E file is staged by CI")
class RevolvedBeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(os.environ["PLE_REVOLVED_IPE"])
        cls.text = cls.path.read_text()
        cls.section = "#91= IFCISHAPEPROFILEDEF(.AREA.,'IPE600',#12,0.22,0.6,0.012,0.019,0.024,$,$);"

    def test_public_beam_real_receiver(self):
        from test_wired_router_e2e import _start_mock_relay
        root = Path(__file__).resolve().parents[1]
        calls = []
        relay = _start_mock_relay(self.path.parent, calls)
        try:
            script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(t) {tools.set(t.name,t);}, on() {}});
const result = await tools.get('hamon_ifc_spatial_clash').execute(
    'beam', {path:process.argv[1]}, null, () => {}, {});
console.log(JSON.stringify(result.details));
"""
            env = {**os.environ, "TASK_WORKDIR": str(self.path.parent), "TASK_HARNESS_PYTHON": sys.executable,
                   "TASK_HARNESS_SRC": str(root), "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}"}
            out = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                                  "-e", script, self.path.name], cwd=root, env=env,
                                 text=True, capture_output=True, timeout=30, check=True)
            reply = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertTrue(reply["ok"], reply)
            self.assertEqual(reply["solid_count"], 1)
            self.assertEqual(calls, ["read_file"])
            self.assertAlmostEqual(reply["total_solid_volume_m3"], .17212389993526295, places=8)
            print("registered IPE600 volume m3", reply["total_solid_volume_m3"])
        finally:
            relay.shutdown()
            relay.server_close()

    def test_oblique_angles_symmetric_sections_and_full_torus(self):
        for attributes, area in (
            ("IFCRECTANGLEPROFILEDEF(.AREA.,'rectangle',#12,0.22,0.6)", .22*.6),
            ("IFCCIRCLEPROFILEDEF(.AREA.,'circle',#12,0.3)", math.pi*.3**2),
        ):
            for radians in (.3, math.pi, 1.75*math.pi, math.tau):
                with self.subTest(attributes=attributes, angle=radians):
                    source = self.text.replace(self.section, f"#91= {attributes};").replace(
                        "#86,1.52202550844946)", f"#86,{radians})")
                    out = run_router_ifc_clash({"ifc_text": source})
                    self.assertTrue(out["ok"], out)
                    self.assertAlmostEqual(out["total_solid_volume_m3"], area*7.25*radians, places=8)

    def test_enclosed_duct_and_empty_rotation_axis_have_different_material(self):
        # Put the probes in the beam's local frame; its world placement rotates all three.
        theta, radius = .75, 7.25
        for x, z, touches in ((radius*(1-math.cos(theta)), radius*math.sin(theta), True),
                              (radius, 0., False)):
            with self.subTest(touches=touches):
                records = (
                    f"#900=IFCCARTESIANPOINT(({x},0.,{z-.001}));\n"
                    "#901=IFCAXIS2PLACEMENT3D(#900,$,$);\n"
                    "#902=IFCLOCALPLACEMENT(#73,#901);\n"
                    "#903=IFCRECTANGLEPROFILEDEF(.AREA.,'probe',$,0.002,0.002);\n"
                    "#904=IFCEXTRUDEDAREASOLID(#903,$,#10,0.002);\n"
                    "#905=IFCSHAPEREPRESENTATION(#18,'Body','SweptSolid',(#904));\n"
                    "#906=IFCPRODUCTDEFINITIONSHAPE($,$,(#905));\n"
                    "#907=IFCDUCTSEGMENT('0AzR0sseD4bBOcPJ50wSE9',$,'probe',$,$,#902,#906,$,$);\n"
                )
                end = self.text.rindex("ENDSEC;")
                source = self.text[:end] + records + self.text[end:]
                result = run_router_ifc_clash({"ifc_text": source, "required_clearance_m": .05})
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["pair_count"], 1, result)
                print("revolved beam probe", touches, result["hard_clash_count"], result["min_clearance_m"])
                self.assertEqual(result["hard_clash_count"], int(touches), result)
                if not touches:
                    self.assertGreater(result["min_clearance_m"], 7.)

    def test_degree_conversion_and_profile_axis_shift(self):
        angle_unit = ("#22= IFCCONVERSIONBASEDUNIT(#920,.PLANEANGLEUNIT.,'DEGREE',#921);\n"
                      "#920=IFCDIMENSIONALEXPONENTS(0,0,0,0,0,0,0);\n"
                      "#921=IFCMEASUREWITHUNIT(IFCPLANEANGLEMEASURE(0.017453292519943295),#922);\n"
                      "#922=IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.);")
        source = self.text.replace("#22= IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.);", angle_unit)
        source = source.replace("#86,1.52202550844946)", "#86,90.)").replace(
            "#13= IFCCARTESIANPOINT((0.0,0.0));", "#13= IFCCARTESIANPOINT((1.0,0.0));")
        result = run_router_ifc_clash({"ifc_text": source})
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["total_solid_volume_m3"], .015598442631532278*6.25*math.pi/2, places=8)

    def test_intersecting_axis_and_unqualified_profile_refuse(self):
        for source in (
            self.text.replace("#87= IFCCARTESIANPOINT((7.25,0.0,0.0));", "#87= IFCCARTESIANPOINT((0.01,0.0,0.0));"),
            self.text.replace("0.024,$,$);", "0.024,0.003,$);"),
            self.text.replace("#86,1.52202550844946)", "#86,7.)"),
        ):
            result = run_router_ifc_clash({"ifc_text": source})
            self.assertFalse(result["ok"], result)
            self.assertIsNone(result["total_solid_volume_m3"])
