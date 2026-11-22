import math
import unittest
from unittest import mock
import hamon_drawing as drawing


class MixedCaptionGeometryTests(unittest.TestCase):
    def test_rule_mask_is_applied_to_the_competing_glyph(self):
        # The rival differs exclusively in a horizontal stroke along the rule.
        vertical = ((0., 0.), (0., 1.))
        top = ((0., 0.), (.5, 0.))
        groups = [[.5, [vertical, top], ["ink"]]]
        route = [("A", 0, 1, ["A"])]
        face = {"A": [(0., 0., 0., 1.), (0., 0., .5, 0.)],
                "B": [(0., 0., 0., 1.), (0., 0., .5, 0.), (0., 1., .5, 1.)]}
        self.assertTrue(drawing._rule_hides_competitor(route, groups, face, 1., 0., 1.,
                                                       {"ink"}, {8}))
        self.assertFalse(drawing._rule_hides_competitor(route, groups, face, 1., 0., 1.,
                                                        {"ink"}, set()))

    def test_numeric_and_unit_competitors_veto_an_otherwise_accepted_token(self):
        groups = [[1., [((0., 0.), (1., 1.))], ["digit"]]]
        for char, rival in [("0", "O"), ("1", "l"), ("”", "°")]:
            self.assertIsNone(drawing._caption_token_route(char, groups, {(0, 1): {char, rival}}, 1.))

    def test_unmatched_group_cannot_be_dropped_from_a_unit(self):
        groups = [[.1, [((0., 0.), (.1, .1))], ["left"]],
                  [.4, [((.3, 0.), (.4, .1))], ["right"]]]
        options = {(0, 1): {"’"}, (1, 1): {"’"}, (0, 2): {"”"}}
        self.assertIsNone(drawing._caption_token_route("’", groups, options, 1.))
        self.assertEqual(drawing._caption_token_route("”", groups, options, 1.)[0][:3],
                         ("”", 0, 2))


    def test_each_foreign_pixel_across_full_title_span_withdraws_evidence(self):
        region = {"ring": {"box": [0, 0, 8, 8]}, "caption_box": [8, 7, 30, 18]}
        objects = [({"id": "stroke", "svg_pen_diameter": .4},
                    [((12, 10), (14, 10)), ((14, 10), (14, 12))])]
        bounds = [12, 10, 14, 12]

        def raster(extra):
            def paint(source, page, box, scale):
                left, top = math.floor(box[0]*scale), math.floor(box[1]*scale)
                width = math.ceil(box[2]*scale)-left
                height = math.ceil(box[3]*scale)-top
                pixels = bytearray([255]) * (width*height)
                points = {(x, 10*scale) for x in range(12*scale, 14*scale+1)}
                points.update((14*scale, y) for y in range(10*scale, 12*scale+1))
                if extra:
                    points.add((23*scale, 11*scale))
                for x, y in points:
                    pixels[(y-top)*width+x-left] = 0
                return f"P5\n{width} {height}\n255\n".encode() + pixels
            return paint

        for extra in (False, True):
            with self.subTest(extra=extra), mock.patch.object(drawing, "_caption_raster", raster(extra)):
                result = drawing._caption_edge_ink(None, 1, region, bounds, objects)
                self.assertEqual(result is None, extra)

    def test_confident_conflicting_token_vetoes_tight_crop_consensus(self):
        groups = [[2., [((1., 1.), (2., 2.))], ["character"]]]
        for wrong_confidence, accepts in ((74, True), (92, False)):
            line_a = [("Al", wrong_confidence, (4, 4, 4, 4)), ("B", 96, (20, 4, 5, 5))]
            line_b = [("AT", 96, (6, 6, 6, 6)), ("B", 96, (30, 6, 6, 6))]
            readings = [line_a, line_b, [("AT", 95)], [("AT", 96)]]
            with self.subTest(confidence=wrong_confidence), mock.patch.object(
                    drawing, "_caption_raster", return_value=b"view"), mock.patch.object(
                    drawing, "_read_caption_words", side_effect=readings):
                result = drawing._caption_token_views(None, 1, [0, 0, 8, 4], groups, 1.)
                self.assertEqual(result is not None, accepts)
                if accepts:
                    self.assertEqual([row[0] for row in result], ["AT", "B"])


    def test_one_tracking_must_fit_measured_stems_and_published_spaces(self):
        # A space consumes its own advance (16 font units) and tracking.
        fit = drawing._shared_tracking_partitions([1, 2, 20, 0, 1], 16)
        self.assertEqual(fit, {(False, False, True, False, False)})
        self.assertEqual(drawing._shared_tracking_partitions([1, 2, 37, 0, 1], 16), set())
        # Preserve competing word boundaries when the measured displacements
        # are genuinely compatible with more than one common tracking value.
        self.assertEqual(drawing._shared_tracking_partitions([6], 16),
                         {(False,), (True,)})

    def test_external_advance_and_sidebearings_match_proportional_letters(self):
        face = drawing._caption_face()
        metrics = drawing._external_caption_metrics()
        self.assertEqual(metrics[" "][0], 16)
        self.assertNotEqual(metrics["I"], metrics["W"])
        words = ("WILL", "FIT")
        caps = 21.0
        position = 100.0
        groups = []
        expected = []
        for word in words:
            first = len(groups)
            for char in word:
                advance, ink_origin = metrics[char]
                edges = [((position + ink_origin + x*caps, y*caps),
                          (position + ink_origin + z*caps, w*caps)) for x,y,z,w in face[char]]
                groups.append([max(x for edge in edges for x,_ in edge), edges, [str(len(groups))]])
                position += advance + 1
            points = [p for group in groups[first:] for edge in group[1] for p in edge]
            expected.append([min(x for x,y in points), min(y for x,y in points),
                             max(x for x,y in points), max(y for x,y in points)])
            position += metrics[" "][0] + 1
        self.assertEqual(drawing._font_spaced_words(groups, caps, 0), expected)


    def test_annotation_control_cannot_erase_a_dark_correction(self):
        def image(pixels):
            return b"P5\n7 4\n255\n" + bytes(pixels)
        dark = [255] * 28
        for i in (8, 15, 22):
            dark[i] = 0
        gray = [209 if v == 255 else 30 for v in dark]
        self.assertTrue(drawing._annotation_preserves_ink(image(gray), image(dark)))
        for i in (3, 9):
            changed = gray.copy()
            changed[i] = 0
            self.assertFalse(drawing._annotation_preserves_ink(image(changed), image(dark)))
        self.assertFalse(drawing._annotation_preserves_ink(image(dark), image(dark)))


if __name__ == "__main__":
    unittest.main()
