"""Changed-input controls for coordinate dimensions in model space."""
from pathlib import Path
import tempfile
import unittest

from hamon_dxf import scan_dxf


def entity(*, flags=6, datum=(7, 3, 2), feature=(19, -1, 2), leader=(30, 60, 2), extra=()):
    points = ((10, datum), (13, feature), (14, leader))
    pairs = [(0, "SECTION"), (2, "ENTITIES"), (0, "DIMENSION"), (5, "A"), (70, flags)]
    for code, point in points:
        pairs.extend((code + 10*i, x) for i, x in enumerate(point))
    pairs.extend(extra)
    pairs.extend(((0, "ENDSEC"), (0, "EOF")))
    return "".join(f"{code}\n{value}\n" for code, value in pairs)


class OrdinateTests(unittest.TestCase):
    def read(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drawing.dxf"
            path.write_text(entity(**kwargs))
            return scan_dxf(path)

    def test_datum_axis_projection_and_leader_independence(self):
        for flags, axis, signed in ((6, "y", -4), (102, "x", 12)):
            for opts in ({}, {"leader": (350, -90, 2)},
                         {"datum": (107, 203, 2), "feature": (119, 199, 2), "leader": (130, 260, 2)}):
                with self.subTest(flags=flags, options=opts):
                    row = self.read(flags=flags, **opts)["dimensions"][0]
                    self.assertEqual(row["ordinate"]["axis"], axis)
                    self.assertEqual(row["ordinate"]["signed_coordinate_model_units"], signed)
                    self.assertEqual(row["usable_qualified_value"], abs(signed))
                    self.assertEqual(row["geometry_status"], "geometry_only")

    def test_cache_must_agree_and_hidden_layout_is_excluded(self):
        for value, expected in ((4, 4), (5, None), (-4, None), ("nan", None)):
            with self.subTest(value=value):
                row = self.read(extra=((42, value), (1, "user text")))["dimensions"][0]
                self.assertEqual(row["usable_qualified_value"], expected)
                self.assertEqual(row["text_override"], "user text")
                self.assertTrue(row["text_override_is_unverified"])
        self.assertEqual(self.read(extra=((67, 1), (410, "Layout1")))["dimensions"], [])

    def test_unresolved_frames_and_valid_zero_axis(self):
        for kwargs in ({"extra": ((51, 45),)}, {"extra": ((210, 1), (230, 0))},
                       {"leader": (30, 60, 3)}, {"extra": ((13, 19),)},
                       {"extra": ((102, "{META"), (102, "}"))}, {"flags": 22}):
            with self.subTest(kwargs=kwargs):
                row = self.read(**kwargs)["dimensions"][0]
                self.assertEqual(row["geometry_status"], "geometry_unavailable")
                self.assertIsNone(row["usable_qualified_value"])
        row = self.read(feature=(20, 3, 2))["dimensions"][0]
        self.assertEqual(row["usable_qualified_value"], 0)
