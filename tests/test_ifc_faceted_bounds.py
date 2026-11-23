"""Real IFC face bounds checked against analytic through-hole and concave prisms."""
import itertools
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash


class FacetedScene:
    def __init__(self, *, reverse=False, holes_first=False, scale=1., matrix=None, shift=None):
        self.rows = []
        self.product_count = 1
        self.reverse, self.holes_first, self.scale = reverse, holes_first, scale
        self.matrix = np.eye(3) if matrix is None else np.asarray(matrix)
        self.shift = np.zeros(3) if shift is None else np.asarray(shift)
        origin = self.add("IFCCARTESIANPOINT((0.,0.,0.))")
        placement = self.add(f"IFCAXIS2PLACEMENT3D(#{origin},$,$)")
        self.context = self.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-7,#{placement},$)")
        unit = self.add("IFCSIUNIT(*,.LENGTHUNIT.," + (".MILLI." if scale == 1000 else "$") + ",.METRE.)")
        units = self.add(f"IFCUNITASSIGNMENT((#{unit}))")
        self.add(f"IFCPROJECT('0AzR0sseD4bBOcPJ50wSE1',$,'bounded-face course',$,$,$,$,(#{self.context}),#{units})")

    def add(self, text):
        self.rows.append(f"#{len(self.rows) + 1}={text};")
        return len(self.rows)

    def prism(self, outer, holes=(), z0=0., z1=1., kind="IFCBEAM", name="host"):
        rings = [list(outer)] + [list(hole)[::-1] for hole in holes]
        points = {}
        for z in (z0, z1):
            for ring in rings:
                for x, y in ring:
                    xyz = (self.matrix @ [x, y, z] + self.shift) * self.scale
                    points[x, y, z] = self.add("IFCCARTESIANPOINT((" + ",".join(repr(float(v)) for v in xyz) + "))")
        faces = []

        def face(loops):
            bounds = []
            for i, loop in enumerate(loops):
                if self.reverse:
                    loop = loop[::-1]
                polyloop = self.add("IFCPOLYLOOP((" + ",".join(f"#{p}" for p in loop) + "))")
                bound = "IFCFACEOUTERBOUND" if i == 0 else "IFCFACEBOUND"
                flag = ".F." if self.reverse else ".T."
                bounds.append(self.add(f"{bound}(#{polyloop},{flag})"))
            if self.holes_first:
                bounds.reverse()
            faces.append(self.add("IFCFACE((" + ",".join(f"#{b}" for b in bounds) + "))"))

        face([[points[x, y, z0] for x, y in ring[::-1]] for ring in rings])
        face([[points[x, y, z1] for x, y in ring] for ring in rings])
        for ring in rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                face([[points[*a, z0], points[*b, z0], points[*b, z1], points[*a, z1]]])
        shell = self.add("IFCCLOSEDSHELL((" + ",".join(f"#{f}" for f in faces) + "))")
        brep = self.add(f"IFCFACETEDBREP(#{shell})")
        shape = self.add(f"IFCSHAPEREPRESENTATION(#{self.context},'Body','Brep',(#{brep}))")
        body = self.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        self.product_count += 1
        self.add(f"{kind}('0AzR0sseD4bBOcPJ50wSE{self.product_count}',$,'{name}',$,$,$,#{body},$,$)")
        return self

    def text(self):
        return ("ISO-10303-21;\nHEADER;FILE_DESCRIPTION(('Faceted bounds'),'2;1');\n"
                "FILE_NAME('bounded.ifc','2026-01-01T00:00:00',('test'),('test'),'course','test','');\n"
                "FILE_SCHEMA(('IFC4'));ENDSEC;\nDATA;\n" + "\n".join(self.rows) + "\nENDSEC;END-ISO-10303-21;")


OUTER = [(0., 0.), (4., 0.), (4., 4.), (0., 4.)]
HOLE = [(1., 1.), (3., 1.), (3., 3.), (1., 3.)]


def through_hole_scene(**options):
    scene = FacetedScene(**options).prism(OUTER, [HOLE])
    scene.prism([(1.8, 1.8), (2.2, 1.8), (2.2, 2.2), (1.8, 2.2)],
                z0=-.25, z1=1.25, kind="IFCDUCTSEGMENT", name="probe")
    return scene


class FacetedBounds(unittest.TestCase):
    def test_true_through_hole_has_no_cap_collision(self):
        text = through_hole_scene().text()
        result = run_router_ifc_clash({"ifc_text": text, "required_clearance_m": .5})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["hard_clash_count"], 0)
        self.assertAlmostEqual(result["min_clearance_m"], .8)
        host, probe = load_ifc_solids(text)
        self.assertAlmostEqual(host.volume_m3(), 12.)
        self.assertAlmostEqual(host.surface_area_m2(), 48.)
        np.testing.assert_allclose(host.centroid_m(), [2., 2., .5], atol=1e-12)

    def test_concave_notch_stays_exterior(self):
        outer = [(0., 0.), (4., 0.), (4., 4.), (3., 4.), (3., 1.), (1., 1.), (1., 4.), (0., 4.)]
        scene = FacetedScene().prism(outer)
        scene.prism([(1.8, 2.8), (2.2, 2.8), (2.2, 3.2), (1.8, 3.2)],
                    z0=-.25, z1=1.25, kind="IFCDUCTSEGMENT", name="probe")
        result = run_router_ifc_clash({"ifc_text": scene.text(), "required_clearance_m": .5})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["hard_clash_count"], 0)
        self.assertAlmostEqual(result["min_clearance_m"], .8)
        host, probe = load_ifc_solids(scene.text())
        self.assertAlmostEqual(host.volume_m3(), 10.)
        self.assertAlmostEqual(host.surface_area_m2(), 42.)

    def test_oriented_bounds_units_and_survey_coordinates(self):
        rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        shift = np.array([729011.25, 9063960.5, 120.])
        for reverse, holes_first, scale in itertools.product((False, True), (False, True), (1., 1000.)):
            with self.subTest(reverse=reverse, holes_first=holes_first, scale=scale):
                scene = through_hole_scene(reverse=reverse, holes_first=holes_first, scale=scale,
                                           matrix=rotation, shift=shift)
                result = run_router_ifc_clash({"ifc_text": scene.text(), "required_clearance_m": .5})
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["hard_clash_count"], 0)
                self.assertAlmostEqual(result["min_clearance_m"], .8, places=7)
                host, probe = load_ifc_solids(scene.text())
                self.assertAlmostEqual(host.volume_m3(), 12., places=7)
                self.assertAlmostEqual(host.surface_area_m2(), 48., places=7)
                np.testing.assert_allclose(host.centroid_m(), rotation @ [2., 2., .5] + shift,
                                           rtol=0, atol=2e-9)

    def test_same_faces_without_optional_geometry_backend(self):
        with patch.dict(sys.modules, {"shapely": None}):
            self.test_true_through_hole_has_no_cap_collision()
            self.test_concave_notch_stays_exterior()

    def test_missing_or_unqualified_boundaries_are_refused(self):
        text = through_hole_scene().text()
        shell = re.search(r"IFCCLOSEDSHELL\(\(([^)]+)\)\)", text).group(1)
        invalid = {
            "missing face": text.replace(shell, ",".join(shell.split(",")[1:]), 1),
            "open shell": text.replace("IFCCLOSEDSHELL", "IFCOPENSHELL", 1),
            "no outer bound": text.replace("IFCFACEOUTERBOUND", "IFCFACEBOUND", 1),
            "two outer bounds": text.replace("IFCFACEBOUND", "IFCFACEOUTERBOUND", 1),
            "unresolved point": re.sub(r"IFCPOLYLOOP\(\(#\d+", "IFCPOLYLOOP((#999999", text, count=1),
            "missing orientation": re.sub(r"(IFCFACEOUTERBOUND\(#\d+,)\.T\.", r"\1$", text, count=1),
            "unsupported curve": text.replace("IFCPOLYLOOP", "IFCPOLYLINE", 1),
            "quoted coordinate": text.replace("(4.0,", "('4.0',"),
            "logical coordinate": text.replace(",1.0))", ",.T.))"),
        }
        for label, content in invalid.items():
            with self.subTest(label=label):
                result = run_router_ifc_clash({"ifc_text": content})
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["status"], "unresolved_solid_geometry")
                self.assertIsNone(result.get("total_solid_volume_m3"))

    def test_file_input_through_actual_json_stdin_entry(self):
        import hamon_ifc_spatial_clash
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hole.ifc"
            path.write_text(through_hole_scene().text())
            result = subprocess.run([sys.executable, hamon_ifc_spatial_clash.__file__, "--json-stdin"],
                                    input=json.dumps({"path": str(path), "required_clearance_m": .5}),
                                    text=True, capture_output=True, timeout=30, check=True)
            report = json.loads(result.stdout)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["hard_clash_count"], 0)
            self.assertAlmostEqual(report["min_clearance_m"], .8)

    def test_independent_ifc_kernel_and_export_reopen(self):
        if importlib.util.find_spec("ifcopenshell") is None:
            self.skipTest("IfcOpenShell is installed by the full CI requirements")
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.shape
        model = ifcopenshell.file.from_string(through_hole_scene().text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "exported.ifc"
            model.write(str(path))
            reopened = ifcopenshell.open(str(path))
            host, = reopened.by_type("IfcBeam")
            shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), host)
            self.assertAlmostEqual(ifcopenshell.util.shape.get_volume(shape.geometry), 12.)
            self.assertAlmostEqual(ifcopenshell.util.shape.get_area(shape.geometry), 48.)
            report = run_router_ifc_clash({"path": str(path), "required_clearance_m": .5})
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["hard_clash_count"], 0)
            self.assertAlmostEqual(report["min_clearance_m"], .8)

    def test_registered_tool_replays_file_and_refusal(self):
        import hamon_ifc_spatial_clash
        root = Path(hamon_ifc_spatial_clash.__file__).parent
        if not (root / "node_modules").is_dir():
            self.skipTest("registered Node runtime is installed by the full CI workflow")
        from test_wired_router_e2e import _start_mock_relay
        commands = []
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            text = through_hole_scene().text()
            (work / "hole.ifc").write_text(text)
            (work / "open.ifc").write_text(text.replace("IFCCLOSEDSHELL", "IFCOPENSHELL", 1))
            relay = _start_mock_relay(work, commands)
            try:
                script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(tool) {tools.set(tool.name, tool);}, on() {}});
const reports = [];
for (const path of ['hole.ifc', 'open.ifc']) {
  const reply = await tools.get('hamon_ifc_spatial_clash').execute(
    path, {path, required_clearance_m: 0.5}, null, () => {}, {});
  reports.push(reply.details);
}
console.log(JSON.stringify(reports));
"""
                env = {**os.environ, "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}",
                       "TASK_WORKDIR": str(work), "TASK_HARNESS_PYTHON": sys.executable,
                       "TASK_HARNESS_SRC": str(root)}
                proc = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", script],
                                      cwd=root, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                good, refused = json.loads(proc.stdout.strip().splitlines()[-1])
                self.assertTrue(good["ok"], good)
                self.assertEqual(good["hard_clash_count"], 0)
                self.assertAlmostEqual(good["min_clearance_m"], .8)
                self.assertFalse(refused["ok"], refused)
                self.assertEqual(refused["status"], "unresolved_solid_geometry")
                self.assertIsNone(refused["total_solid_volume_m3"])
                self.assertEqual(commands, ["read_file", "read_file"])
            finally:
                relay.shutdown()
                relay.server_close()


if __name__ == "__main__":
    unittest.main()
