import tempfile
import unittest
from pathlib import Path

from hamon_cad_takeoff import scan_dxf_takeoff


class SolidTakeoffTests(unittest.TestCase):
    def takeoff(self, vertices, unit=6, extra=()):
        pairs = [(0, "SECTION"), (2, "HEADER"), (9, "$INSUNITS"), (70, unit),
                 (0, "ENDSEC"), (0, "SECTION"), (2, "ENTITIES"), (0, "SOLID"),
                 (100, "AcDbEntity"), (8, "FACE"), (100, "AcDbTrace"), *extra]
        for i, (x, y) in enumerate(vertices):
            pairs.extend([(10+i, x), (20+i, y), (30+i, 0)])
        pairs.extend([(0, "ENDSEC"), (0, "EOF")])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "face.dxf"
            path.write_text("".join(f"{k}\n{v}\n" for k, v in pairs), encoding="ascii")
            return scan_dxf_takeoff(path)

    def test_four_corner_storage_order_and_scale(self):
        # Physical outline (0,0),(8,0),(5,4),(0,4) has trapezoid area 26.
        r = self.takeoff([(0,0),(8,0),(0,4),(5,4)], unit=4)
        self.assertEqual(r["entity_counts"], {"SOLID": 1})
        self.assertAlmostEqual(r["closed_area_m2_by_layer"]["FACE"], 26e-6, places=14)
        self.assertAlmostEqual(r["polyline_perimeter_m_by_layer"]["FACE"], .022)

    def test_repeated_last_corner_is_triangle(self):
        h = 2**38
        r = self.takeoff([(h,h),(h+5,h),(h,h+12),(h,h+12)])
        self.assertEqual(r["closed_area_m2_by_layer"], {"FACE": 30})
        self.assertEqual(r["polyline_perimeter_m_by_layer"], {"FACE": 30})

    def test_tilted_plane_keeps_face_area(self):
        r = self.takeoff([(0,0),(3,0),(0,2),(3,2)], extra=[(210,1),(220,0),(230,0)])
        self.assertEqual(r["closed_area_m2_by_layer"], {"FACE": 6})
        self.assertEqual(r["polyline_perimeter_m_by_layer"], {"FACE": 10})

    def test_paper_space_remains_excluded(self):
        r = self.takeoff([(0,0),(1,0),(0,1),(1,1)], extra=[(67,1)])
        self.assertEqual(r["closed_area_m2_by_layer"], {})
        self.assertEqual(r["paper_space_skipped_count"], 1)

    def test_unitless_preserves_count_only(self):
        r = self.takeoff([(0,0),(1,0),(0,1),(1,1)], unit=0)
        self.assertIsNone(r["closed_area_m2_by_layer"])
        self.assertIsNone(r["polyline_perimeter_m_by_layer"])
        self.assertEqual(r["entity_counts"], {"SOLID": 1})
