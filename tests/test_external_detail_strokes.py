import unittest
from unittest.mock import patch

import hamon_drawing as drawing


class ExternalDetailStrokeTests(unittest.TestCase):
    def setUp(self):
        self.alphabet = drawing._simplex_strokes()
        self.shapes = {symbol: edges for symbol, _, edges in self.alphabet}

    def glyphs(self, token, stretches):
        result = []
        for symbol, stretch in zip(token, stretches):
            segments = tuple((x0 * stretch, y0, x1 * stretch, y1)
                             for x0, y0, x1, y1 in self.shapes[symbol])
            result.append((segments, len(result)))
        return result

    def test_complete_compact_label_and_unknown_suffix(self):
        glyphs = self.glyphs("12B", [.86, .86, .86])
        self.assertEqual(drawing._recognize_stroke_label(glyphs, self.alphabet), "12B")
        without_b = [record for record in self.alphabet if record[0] != "B"]
        self.assertIsNone(drawing._recognize_stroke_label(glyphs, without_b))

    def test_inconsistent_text_stretch(self):
        glyphs = self.glyphs("12B", [.86, 1.12, .96])
        self.assertIsNone(drawing._recognize_stroke_label(glyphs, self.alphabet))

    def test_distinct_edges_are_required(self):
        first = (0., 0., 1., 0.)
        second = (.1, 0., 1., 1.)
        self.assertFalse(drawing._same_stroke_graph([first, first], [first, second], 1.))
        self.assertTrue(drawing._same_stroke_graph([first, (1., 1., .1, 0.)], [first, second], 1.))

    def test_complete_stroke_requires_visible_ink_separate_from_the_border(self):
        scale = 8
        size = 10 * scale
        region = {"ring": {"box": (0, 0, 10, 10)}}
        glyphs = [((), "glyph")]
        vector = {"id": "glyph", "svg_pen_diameter": .3,
                  "svg_points": [(2, 5), (5, 5), (7, 3)],
                  "svg_segment_vertex_pairs": [(0, 1), (1, 2)]}
        printed = {(x, 40) for x in range(16, 41)}
        printed.update((40 + i, 40 - i) for i in range(17))
        border = {(1, y) for y in range(29, 52)}

        def visible(ink):
            pixels = bytearray([255]) * (size * size)
            for x, y in ink | border:
                pixels[y * size + x] = 0
            image = f"P5\n{size} {size}\n255\n".encode() + pixels
            with patch.object(drawing, "_caption_raster", return_value=image):
                return drawing._visible_complete_label(None, 1, region, glyphs, [vector])

        self.assertTrue(visible(printed))
        erased = {(40 + i, 40 - i) for i in range(6, 13)}
        self.assertFalse(visible(printed - erased))
        connected_to_border = {(x, 40) for x in range(1, 17)}
        self.assertFalse(visible(printed | connected_to_border))

    def test_other_printable_glyph_vetoes_complete_token(self):
        glyphs = self.glyphs("9B", [1., 1.])
        same_geometry = next(record for record in self.alphabet if record[0] == "B")
        alternate = ("?", same_geometry[1], same_geometry[2])
        self.assertIsNone(drawing._recognize_stroke_label(glyphs, [*self.alphabet, alternate]))


if __name__ == "__main__":
    unittest.main()
