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


class OrdinateAnnotationTests(unittest.TestCase):
    @staticmethod
    def block(name, records):
        return [(0, "BLOCK"), (2, name), *records, (0, "ENDBLK")]

    def read(self, blocks, *, second_owner=False, coincident=False):
        pairs = [(0, "SECTION"), (2, "BLOCKS"), *blocks, (0, "ENDSEC")]
        drawing = "".join(f"{code}\n{value}\n" for code, value in pairs)
        drawing += entity(feature=(7, 3, 2) if coincident else (19, -1, 2),
                          extra=[(2, "*d1"), (1, "<>")])
        if second_owner:
            # A paper-space reference after the output limit still shares the picture.
            drawing = drawing.removesuffix("0\nENDSEC\n0\nEOF\n")
            drawing += "0\nDIMENSION\n2\n*D1\n67\n1\n0\nENDSEC\n0\nEOF\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "drawing.dxf")
            p.write_text(drawing)
            row = scan_dxf(p, limit=1)["dimensions"][0]
        self.assertEqual(row["usable_qualified_value"], None if coincident else 4)
        return row["ordinate"]["stored_block_annotation"]

    def test_direct_and_nested_annotation_keep_literal_text_and_separate_geometry(self):
        text = [(0, "TEXT"), (5, "AF"), (1, "400,00")]
        insert = [(0, "INSERT"), (2, "*u2")]
        decoy = self.block("*U3", [(0, "TEXT"), (1, "999")])
        for blocks, path in [(self.block("*D1", text), ["*D1"]),
                             (self.block("*U2", text) + self.block("*D1", insert), ["*D1", "*U2"])]:
            with self.subTest(path=path):
                self.assertEqual(self.read(decoy + blocks), {
                    "block_path": path, "text": "400,00", "text_handle": "AF"})
        zero = self.block("*D1", [(0, "MTEXT"), (5, "AB"), (1, "0")])
        self.assertEqual(self.read(zero, coincident=True)["text"], "0")

    def test_ambiguous_incomplete_and_unrelated_blocks_withhold_the_annotation(self):
        text = [(0, "TEXT"), (5, "AF"), (1, "400")]
        insert = [(0, "INSERT"), (2, "*U2")]
        direct = self.block("*D1", text)
        for blocks in (self.block("*D2", text), direct + direct, self.block("*D1", text + text),
                       self.block("*D1", insert), self.block("*D1", insert + text),
                       self.block("*D1", insert) + self.block("*U2", insert),
                       self.block("*D1", [(0, "MTEXT"), (3, "1"), (1, "400")]),
                       self.block("*D1", text + [(60, "1")]),
                       self.block("*D1", [(0, "TEXT"), (102, "{META"), (1, "400"), (102, "}")])):
            with self.subTest(blocks=blocks):
                self.assertIsNone(self.read(blocks))
        self.assertIsNone(self.read(direct, second_owner=True))
