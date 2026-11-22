import unittest
from unittest import mock

import hamon_drawing as drawing


class SourceRepaintTest(unittest.TestCase):
    def inventory(self, arms, root_attributes="", body_prefix=""):
        svg = f'<svg viewBox="0 0 50 50" {root_attributes}>{body_prefix}{arms}</svg>'
        kept = set()
        result = drawing._svg_inventory(svg.encode(), source_repaints=kept)
        return kept, [item["id"] for item in result["reported_vectors"]], svg.encode()

    def path(self, color="black", attrs=""):
        return f'<path fill="none" stroke="{color}" d="M 2 8 L 20 8" {attrs}/>'

    def test_exact_source_repaint_preserves_original_ids_and_destination_inventory(self):
        line = self.path()
        kept, ids, svg = self.inventory(line + line + self.path(attrs='transform="translate(0 2)"'))
        self.assertEqual(kept, {"v1"})
        self.assertEqual(ids, ["v1", "v3"])
        normal = drawing._svg_inventory(svg)
        self.assertEqual([v["id"] for v in normal["reported_vectors"]], ["v1", "v2", "v3"])

    def test_distinct_or_inherited_paint_remains_distinct(self):
        line = self.path()
        for alternate in (self.path(attrs='transform="translate(0 .04)"'),
                          self.path(attrs='stroke-width="3"'), self.path(color="#666"),
                          self.path(attrs='stroke-opacity=".4"'), f"<g>{line}</g>"):
            with self.subTest(alternate=alternate):
                self.assertEqual(self.inventory(line + alternate)[:2], (set(), ["v1", "v2"]))
        for root, child in [('opacity=".5"', ""), ("", "<style>path {opacity:.5}</style>")]:
            self.assertEqual(self.inventory(line + line, root, child)[:2], (set(), ["v1", "v2"]))

    def test_wrong_visible_numerator_or_sheet_withdraws_repaint(self):
        source = {"numerator_evidence": [{"text": "6", "box": [0, 0, 8, 8]}],
                  "source": {"text": "A5.1", "box": [0, 8, 8, 16]}}
        expected = [[("6", 98)], [("6", 95)], [("A5.1", 89)], [("A5.1", 85)]]
        for index, observed, accepted in ((0, [("6", 98)], True), (0, [("9", 98)], False),
                                          (1, [], False), (2, [("A5.2", 99)], False),
                                          (3, [("A5.1", 40)], False)):
            reads = expected.copy()
            reads[index] = observed
            with self.subTest(index=index, observed=observed), \
                    mock.patch.object(drawing, "_caption_raster", return_value=None), \
                    mock.patch.object(drawing, "_read_caption_words", side_effect=reads):
                self.assertEqual(bool(drawing._read_repainted_source("plan.pdf", 2, source)), accepted)
