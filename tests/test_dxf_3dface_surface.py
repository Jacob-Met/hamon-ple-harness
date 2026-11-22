"""Physical WCS surface takeoffs are separate from projected closed areas."""
import math
from pathlib import Path
import tempfile
import unittest

from hamon_cad_takeoff import scan_dxf_takeoff


def draw(faces, units):
    pairs = [(0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, units),
             (0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES")]
    for layer, vertices, flags, paper in faces:
        pairs.extend([(0, "3DFACE"), (8, layer), (70, flags), (67, int(paper))])
        for i, point in enumerate(vertices):
            pairs.extend((10 + i + 10 * axis, value) for axis, value in enumerate(point))
    pairs.extend([(0, "ENDSEC"), (0, "EOF")])
    return "".join(f"{code}\n{value}\n" for code, value in pairs)


def takeoff(faces, units=6):
    with tempfile.TemporaryDirectory() as root:
        filename = Path(root) / "fixture.dxf"
        filename.write_text(draw(faces, units), encoding="ascii")
        return scan_dxf_takeoff(filename)


class SurfaceTakeoffTests(unittest.TestCase):
    def test_vertical_triangle_and_tilted_rectangle(self):
        a = [(0, 0, 0), (0, 3, 0), (0, 3, 4), (0, 3, 4)]
        b = [(0, 0, 0), (3, 0, 0), (3, 4, 3), (0, 4, 3)]
        translated = [tuple(x + 10000000 for x in point) for point in reversed(b)]
        result = takeoff([("VERT", a, 0, False), ("TILT", b, 15, False),
                          ("REVERSED", translated, 0, False), ("PAPER", a, 0, True)])
        self.assertEqual(result["paper_space_skipped_count"], 1)
        self.assertEqual(result["closed_area_m2_by_layer"], {})
        self.assertEqual(result["face3d"]["area_m2_by_layer"], {"VERT": 6, "TILT": 15, "REVERSED": 15})
        self.assertEqual(result["face3d"]["boundary_m_by_layer"], {"VERT": 12, "TILT": 16, "REVERSED": 16})

    def test_concave_face_units_and_per_face_accounting(self):
        shape = [(0, 0, 0), (4, 0, 0), (1, 1, 0), (0, 4, 0)]
        faces = [("L", shape, 0, False), ("L", shape, 8, False)]
        calibrated = takeoff(faces, 4)["face3d"]
        self.assertEqual(calibrated["measured_by_layer"], {"L": 2})
        self.assertAlmostEqual(calibrated["area_m2_by_layer"]["L"], 8e-6)
        self.assertAlmostEqual(calibrated["boundary_m_by_layer"]["L"], (16 + 4 * math.sqrt(10)) / 1000)
        unitless = takeoff(faces, 0)["face3d"]
        self.assertEqual(unitless["area_drawing2_by_layer"], {"L": 8})
        self.assertIsNone(unitless["area_m2_by_layer"])
        self.assertIsNone(unitless["boundary_m_by_layer"])

    def test_unsupported_faces_are_counted_without_measure(self):
        shapes = [[(0,0,0), (2,0,0), (2,2,0), (0,2,1)],
                  [(0,0,0), (2,2,0), (0,2,0), (2,0,0)],
                  [(0,0,0), (1,0,0), (1,1,0), (0,1)],
                  [(0,0,0), (1,0,0), (2,0,0), (2,0,0)]]
        obs = takeoff([("L", s, 0, False) for s in shapes])
        self.assertEqual(obs["entity_counts"]["3DFACE"], 4)
        self.assertEqual(obs["face3d"]["unmeasured_by_layer"], {"L": 4})
        self.assertEqual(obs["face3d"]["measured_by_layer"], {})
        self.assertEqual(obs["face3d"]["area_m2_by_layer"], {})


if __name__ == "__main__":
    unittest.main()
