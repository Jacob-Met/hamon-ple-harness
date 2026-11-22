import unittest

from hamon_eagle_schematic import _graph, _ink


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
