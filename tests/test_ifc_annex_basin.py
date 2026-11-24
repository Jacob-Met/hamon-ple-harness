"""Compare three member-authored IFC exchanges of the same buildingSMART Annex E basin."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.environ.get("PLE_ANNEX_E_BASINS"), "original IFC files staged by CI")
class AnnexEBasin(unittest.TestCase):
    def test_original_mm_models_through_registered_tool_and_cad_kernel(self):
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.ifcopenshell_wrapper as wrapper
        from test_wired_router_e2e import _start_mock_relay
        root = Path(__file__).resolve().parents[1]
        folder = Path(os.environ["PLE_ANNEX_E_BASINS"])
        calls = []
        relay = _start_mock_relay(folder, calls)
        try:
            script = """
import register from './task-tools.ts';
const tools = new Map();
register({registerTool(t) { tools.set(t.name,t); }, on() {}});
const result = await tools.get('hamon_ifc_spatial_clash').execute(
    'basin', {path:process.argv[1]}, null, () => {}, {});
console.log(JSON.stringify(result.details));
"""
            env = {**os.environ, "TASK_WORKDIR": str(folder), "TASK_HARNESS_PYTHON": sys.executable,
                   "TASK_HARNESS_SRC": str(root),
                   "TASK_RELAY_URL": f"http://127.0.0.1:{relay.server_address[1]}"}
            for kind in ("tessellation", "faceted-brep", "advanced-brep"):
                with self.subTest(kind=kind):
                    path = folder / f"basin-{kind}.ifc"
                    model = ifcopenshell.open(str(path))
                    device, = model.by_type("IfcSanitaryTerminal")
                    settings = ifcopenshell.geom.settings()
                    settings.set("iterator-output", wrapper.NATIVE)
                    element = ifcopenshell.geom.create_shape(settings, device)
                    native = element.geometry.item(0)
                    kernel = native.volume().to_double()
                    out = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                                          "-e", script, path.name], cwd=root, env=env, text=True,
                                         capture_output=True, timeout=150, check=True)
                    reply = json.loads(out.stdout.strip().splitlines()[-1])
                    print(json.dumps({"exchange": kind, "native_volume_m3": kernel,
                                      "native_area_m2": native.area().to_double(),
                                      "native_faces": native.num_faces(), "result": reply}, sort_keys=True))
                    self.assertTrue(reply["ok"], reply)
                    self.assertEqual(reply["solid_count"], 1)
                    self.assertAlmostEqual(reply["total_solid_volume_m3"], kernel, delta=abs(kernel)*.002)
                    if kind == "tessellation":
                        # Signed tetrahedra of the 234 published source triangles (millimetres to SI).
                        self.assertAlmostEqual(reply["total_solid_volume_m3"], .0020268494501826263, places=12)
            self.assertEqual(calls, ["read_file"] * 3)
        finally:
            relay.shutdown()
            relay.server_close()
