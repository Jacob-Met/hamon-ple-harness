"""Physical stroke evidence for wire labels that need not sit on a routing grid."""
import unittest
from xml.etree.ElementTree import Element

from hamon_eagle_schematic import _outlined_supplies


# Hand-drawn T, X and A in four-by-six cells. These fixtures do not use the
# recognizer's alphabet or generate expected words from its glyph dictionary.
TX = [
    ((0, 6), (4, 6)), ((2, 0), (2, 6)),
    ((6, 0), (10, 6)), ((6, 6), (10, 0)),
]
A_SUFFIX = [
    ((12, 0), (12, 4)), ((12, 4), (14, 6)),
    ((14, 6), (16, 4)), ((16, 4), (16, 0)),
    ((12, 3), (16, 3)),
]


def lettering(edges, origin, wire_y, *, vertical=False):
    paths = []
    for edge in edges:
        points = [(origin + 1.1*x, wire_y - .3 - 1.1*y) for x, y in edge]
        if vertical:
            points = [(y, -x) for x, y in points]
        (x0, y0), (x1, y1) = points
        element = Element("path", {
            "d": f"M {x0} {y0} L {x1} {y1}",
            "stroke-width": ".66",
        })
        paths.append((element, (1, 0, 0, 1, 0, 0)))
    return paths


def connect(wires, paths):
    return _outlined_supplies(
        {"rgb(50%,50%,50%)": paths}, "rgb(0%,50%,0%)", 10,
        wires, list(range(len(wires))),
    )


class OutlinedWireLabelTests(unittest.TestCase):
    def test_full_labels_translate_continuously_along_wires(self):
        for first, second in ((2.3, 7.7), (4.9, 10.1), (9.25, 19.6)):
            for vertical in (False, True):
                with self.subTest(origins=(first, second), vertical=vertical):
                    wires = [[(0, y), (40, y)] for y in (0, 20)]
                    if vertical:
                        wires = [[(y, -x) for x, y in wire] for wire in wires]
                    paths = (lettering(TX, first, 0, vertical=vertical)
                             + lettering(TX, second, 20, vertical=vertical))
                    groups, _, anchors = connect(wires, paths)
                    self.assertEqual(groups, [0, 0])
                    self.assertEqual(anchors, {"TX": [0, 1]})

    def test_missing_stroke_or_letter_cannot_supply_the_same_name(self):
        wires = [[(0, y), (40, y)] for y in (0, 20)]
        for damaged in (TX[:-1], TX[:2]):
            with self.subTest(strokes=len(damaged)):
                paths = lettering(TX, 2.3, 0) + lettering(damaged, 7.7, 20)
                groups, _, anchors = connect(wires, paths)
                self.assertEqual(groups, [0, 1])
                self.assertNotIn(1, anchors.get("TX", []))

    def test_complete_suffix_is_not_trimmed_to_a_shorter_signal(self):
        wires = [[(0, y), (40, y)] for y in (0, 20)]
        paths = lettering(TX, 2.3, 0) + lettering(TX + A_SUFFIX, 7.7, 20)
        groups, _, anchors = connect(wires, paths)
        self.assertEqual(groups, [0, 1])
        self.assertEqual(anchors, {"TX": [0], "TXA": [1]})

    def test_overhang_requires_a_real_endpoint(self):
        for origin in (.3, -6.3):
            with self.subTest(origin=origin):
                wires = [[(0, 0), (5, 0)], [(0, 20), (40, 20)]]
                groups, _, _ = connect(
                    wires, lettering(TX, origin, 0) + lettering(TX, 4.9, 20))
                self.assertEqual(groups, [0, 0])
        # A shifted origin and an interior routing-grid position are neither
        # endpoint evidence nor sufficient whole-word overlap.
        for length, origin in ((5, .8), (15, 10.3)):
            with self.subTest(length=length, origin=origin):
                wires = [[(0, 0), (length, 0)], [(0, 20), (40, 20)]]
                groups, _, _ = connect(
                    wires, lettering(TX, origin, 0) + lettering(TX, 4.9, 20))
                self.assertEqual(groups, [0, 1])

    def test_one_word_between_two_wires_is_not_two_observations(self):
        wires = [[(0, y), (40, y)] for y in (0, -7, 20)]
        paths = lettering(TX, 2.3, 0) + lettering(TX, 7.7, 20)
        groups, _, anchors = connect(wires, paths)
        self.assertEqual(groups, [0, 1, 2])
        self.assertEqual(anchors, {"TX": [2]})


class OutlinedSignedSupplyLabelTests(unittest.TestCase):
    def test_signed_supply_requires_an_actual_wire_endpoint(self):
        from test_eagle_outlined_voltages import THREE_V_THREE
        for vertical in (False, True):
            wires = [[(0, y), (40, y)] for y in (0, 20)]
            if vertical:
                wires = [[(y, -x) for x, y in wire] for wire in wires]
            for origin, expected in ((.3, [0, 0]), (2.3, [0, 1]), (10.3, [0, 1])):
                with self.subTest(vertical=vertical, origin=origin):
                    paths = (lettering(THREE_V_THREE, origin, 0, vertical=vertical)
                             + lettering(THREE_V_THREE, .3, 20, vertical=vertical))
                    groups, _, _ = connect(wires, paths)
                    self.assertEqual(groups, expected)

    def test_signed_word_cannot_belong_to_an_arrow_and_another_wire(self):
        from test_eagle_outlined_voltages import THREE_V_THREE
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                wires = [[(0, 0), (40, 0)], [(12.4, 13), (12.4, 33)], [(0, 50), (40, 50)]]
                paths = (lettering(THREE_V_THREE, .3, 0, vertical=vertical)
                         + lettering(THREE_V_THREE, .3, 50, vertical=vertical))
                arrow = [((12.4, 13), (12.4, 3)), ((12.4, 3), (7.4, 10.5)),
                         ((12.4, 3), (17.4, 10.5))]
                if vertical:
                    wires = [[(y, -x) for x, y in wire] for wire in wires]
                    arrow = [tuple((y, -x) for x, y in edge) for edge in arrow]
                for (x0, y0), (x1, y1) in arrow:
                    element = Element("path", {"d": f"M {x0} {y0} L {x1} {y1}", "stroke-width": ".66"})
                    paths.append((element, (1, 0, 0, 1, 0, 0)))
                groups, marks, anchors = connect(wires, paths)
                self.assertEqual(groups, [0, 1, 2])
                self.assertEqual(anchors, {"+3V3": [2]})
                self.assertEqual(marks, [])
