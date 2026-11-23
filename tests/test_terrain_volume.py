"""Public terrain-volume API against independently frozen synthetic CAD drawings."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import unittest

from hamon_terrain_volume import scan_dxf_terrain_volume


COURSE = Path(__file__).resolve().with_name("terrain_course")
ORACLE_SHA256 = "d918ce8199dfbb1d3ae7153adcbeec9741bb047d4c2fe4e7968a1c974be0bf09"


class TerrainVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = (COURSE / "oracle.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != ORACLE_SHA256:
            raise AssertionError("Frozen analytic oracle digest changed")
        cases = json.loads(payload)["cases"]
        cls.cases = {case["id"]: case for case in cases}
        if len(cases) != 13 or len(cls.cases) != 13:
            raise AssertionError("Expected thirteen distinct cases")

    def check_case(self, case_id):
        case = self.cases[case_id]
        drawing = COURSE / case["file"]
        self.assertEqual(hashlib.sha256(drawing.read_bytes()).hexdigest(), case["sha256"])
        try:
            result = scan_dxf_terrain_volume(
                str(drawing), case["existing_layer"], case["proposed_layer"]
            )
        except ValueError as exc:
            if not case["allow_parse_refusal"]:
                raise
            self.assertTrue(str(exc).strip(), "A parse refusal must include a diagnostic")
            return

        self.assertIsInstance(result, dict)
        for key, expected in case["expected"].items():
            with self.subTest(field=key):
                self.assertIn(key, result)
                actual = result[key]
                if expected is None or isinstance(expected, bool):
                    self.assertIs(actual, expected)
                else:
                    self.assertIsInstance(actual, (float, int))
                    self.assertNotIsInstance(actual, bool)
                    self.assertTrue(math.isfinite(actual))
                    self.assertAlmostEqual(actual, float(Fraction(expected)), delta=1e-8)

    def test_crossing(self):
        self.check_case("crossing")

    def test_opposite_diagonals(self):
        self.check_case("opposite_diagonals")

    def test_opposite_diagonals_reversed(self):
        self.check_case("opposite_diagonals_reversed")

    def test_rotated_translated_mm(self):
        self.check_case("rotated_translated_mm")

    def test_partial_footprints(self):
        self.check_case("partial_footprints")

    def test_coincident_duplicate(self):
        self.check_case("coincident_duplicate")

    def test_overlap_different_elevation(self):
        self.check_case("overlap_different_elevation")

    def test_paper_decoys(self):
        self.check_case("paper_decoys")

    def test_paper_only(self):
        self.check_case("paper_only")

    def test_vertical_face(self):
        self.check_case("vertical_face")

    def test_malformed_coordinate(self):
        self.check_case("malformed_coordinate")

    def test_unitless(self):
        self.check_case("unitless")

    def test_unknown_units(self):
        self.check_case("unknown_units")


if __name__ == "__main__":
    unittest.main()
