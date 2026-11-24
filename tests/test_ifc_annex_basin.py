"""Compare three member-authored IFC exchanges of the same buildingSMART Annex E basin."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import roots_legendre


def integrated_side_volume(surface):
    """Integrate horizontal section areas and dz directly on the tensor-product source."""
    source = np.asarray([[point.Coordinates for point in row] for row in surface.ControlPointsList]) / 1000.
    if not np.allclose(source[:, :, 2], source[:, :1, 2], atol=1.e-12, rtol=0):
        raise ValueError("expected one horizontal control plane per U row")
    def basis(knots, multiplicities, degree, count):
        sequence = np.repeat(knots, multiplicities)
        return BSpline(sequence, np.eye(count), degree)
    ub = basis(surface.UKnots, surface.UMultiplicities, surface.UDegree, source.shape[0])
    vb = basis(surface.VKnots, surface.VMultiplicities, surface.VDegree, source.shape[1])
    def stations(spline):
        low, high = spline.t[spline.k], spline.t[-spline.k-1]
        knots = np.unique(spline.t[(low <= spline.t) & (spline.t <= high)])
        nodes, weights = roots_legendre(8)
        for left, right in zip(knots, knots[1:]):
            for x, w in zip(nodes, weights):
                yield .5*(left+right+x*(right-left)), .5*w*(right-left)
    volume = 0.
    for u, wu in stations(ub):
        row = np.einsum("i,ijk->jk", ub(u), source)
        dz = ub.derivative()(u) @ source[:, 0, 2]
        twice_area = 0.
        for v, wv in stations(vb):
            x, y, _ = vb(v) @ row
            dx, dy, _ = vb.derivative()(v) @ row
            twice_area += wv * (x*dy - y*dx)
        volume += wu * abs(dz * twice_area / 2)
    return volume


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
                    if kind == "advanced-brep":
                        side_volumes = sorted(integrated_side_volume(face.FaceSurface)
                                              for face in model.by_type("IfcAdvancedFace")
                                              if face.FaceSurface.is_a("IfcBSplineSurfaceWithKnots"))
                        self.assertEqual(len(side_volumes), 2)
                        integrated = side_volumes[1] - side_volumes[0]
                        self.assertAlmostEqual(integrated, .002044986344568905, places=13)
                        print(json.dumps({"independent_spline_integral_m3": integrated,
                                          "source_side_volumes_m3": side_volumes,
                                          "native_signed_volume_m3": kernel}, sort_keys=True))
                    out = subprocess.run(["node", "--experimental-strip-types", "--input-type=module",
                                          "-e", script, path.name], cwd=root, env=env, text=True,
                                         capture_output=True, timeout=150, check=True)
                    reply = json.loads(out.stdout.strip().splitlines()[-1])
                    print(json.dumps({"exchange": kind, "native_volume_m3": kernel,
                                      "native_area_m2": native.area().to_double(),
                                      "native_faces": native.num_faces(), "result": reply}, sort_keys=True))
                    if kind == "advanced-brep":
                        self.assertTrue(reply["ok"], reply)
                        self.assertEqual(reply["source_face_count"], 5)
                        self.assertAlmostEqual(reply["total_solid_volume_m3"], integrated,
                                               delta=integrated * .0001)
                        continue
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
