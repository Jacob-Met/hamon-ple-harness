import unittest

from unittest.mock import patch

from hamon_eagle_schematic import _graph, _ink, _stitch_pages, document_connectivity


def word(name, box):
    return {"text": name, "page_bbox_pt": box}


class EagleSchematicTests(unittest.TestCase):
    def test_two_distinct_wires_cross_without_an_electrical_join(self):
        wires = [[(0, 0), (10, 0)], [(5, -5), (5, 5)], [(10, 0), (12, 2)]]
        groups, marks, anchors = _graph(wires, [], [])
        self.assertEqual(groups, [0, 1, 0])
        self.assertFalse(marks)
        self.assertFalse(anchors)

    def test_supply_stem_bar_and_name_are_all_required(self):
        wires = [[(0, -10), (0, 0)], [(20, -10), (20, 0)], [(40, -10), (40, 0)]]
        lines = []
        words = []
        for x in (0, 20, 40):
            lines.append([(x, 0), (x, 8)])
            words.append(word("GND", (x - 4, 9, x + 4, 14)))
        for x in (0, 20):
            lines.append([(x - 6, 8), (x + 6, 8)])
        groups, marks, anchors = _graph(wires, lines, words)
        self.assertEqual(groups, [0, 0, 1])
        self.assertEqual(len(marks), 2)
        self.assertEqual(len(anchors["GND"]), 2)

    def test_repeated_signal_label_and_unrelated_nearby_text(self):
        wires = [[(0, 10), (20, 10)], [(50, 40), (75, 40)], [(100, 75), (120, 75)]]
        words = [word("SCK", (0, 3, 11, 11.4)), word("SCK", (50, 33, 61, 41.4)),
                 word("SCK", (102, 77, 113, 86))]
        self.assertEqual(_graph(wires, [], words)[0], [0, 0, 1])

    def test_curve_and_unknown_transform_remain_unresolved(self):
        raw = b'<svg xmlns="http://www.w3.org/2000/svg"><g stroke="rgb(0%, 50%, 0%)" fill="none"><path d="M 1 2 C 1 3 4 6 7 8"/><path transform="wibble(8)" d="M 0 0 L 10 0"/></g></svg>'
        wires, red, marks, unknown = _ink(raw)
        self.assertEqual((wires, red, marks), ([], [], 0))
        self.assertEqual(unknown, {"green_nonsegment": 1, "green_appearance_or_transform": 1})

    def test_document_requires_single_unambiguous_name_and_matching_supply_glyph(self):
        def page(number, names, glyphs):
            return {"page": number, "wire_groups": [0, 1, 2, 3, 4, 5, 6, 6],
                    "printed_equivalence_anchors": names,
                    "supply_glyphs": [{"name": name, "wire_index": index} for name, index in glyphs]}
        pages = [
            page(1, {"CLK": [1], "GND": [2], "+5V": [3], "D0": [4], "ALIAS": [4], "REF": [5]}, [("GND", 2), ("+5V", 3)]),
            page(2, {"CLK": [0], "GND": [1], "+5V": [2], "D0": [3], "REF": [4, 5]}, [("+5V", 2)]),
        ]
        membership, links = _stitch_pages(pages)
        self.assertEqual([link["name"] for link in links], ["+5V", "CLK"])
        a, b = membership
        self.assertEqual(a[1], b[0])
        self.assertEqual(a[3], b[2])
        self.assertNotEqual(a[2], b[1])  # bare GND on page 2
        self.assertNotEqual(a[4], b[3])  # conflicting local names
        self.assertNotEqual(a[5], b[4])  # two candidate components on page 2
        self.assertNotEqual(a[6], b[6])  # unnamed groups with the same local index
        self.assertEqual(a[6], a[7])
        self.assertEqual(_stitch_pages(pages[:1]), ([[0, 1, 2, 3, 4, 5, 6, 6]], []))

    def test_unsupported_sheet_withholds_whole_document_membership(self):
        pages = [{"pdf_sha256": "test", "page": i, "status": status}
                 for i, status in enumerate(("candidate_evidence", "unresolved_profile"), 1)]
        with patch("hamon_eagle_schematic.drawing._page_geometry", return_value=(2, {})), \
             patch("hamon_eagle_schematic.page_connectivity", side_effect=pages):
            result = document_connectivity("design.pdf")
        self.assertEqual(result["status"], "unresolved_profile")
        self.assertIsNone(result["document_wire_groups"])
