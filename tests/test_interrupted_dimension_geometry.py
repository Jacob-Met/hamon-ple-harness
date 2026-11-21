"""Source-fragment controls for dimensions interrupted at the native word box."""
import unittest

import hamon_callout


def drawing(segments):
    vectors = [
        {"id": f"v{i}", "svg_points": [a, b], "svg_segment_vertex_pairs": [[0, 1]]}
        for i, (a, b) in enumerate(segments, 1)
    ]
    return {
        "page": 1, "document_pages": 1, "pdf_page_size_pt": [240, 240],
        "viewbox_svg_units": [0, 0, 240, 240], "word_boxes_reported": 1,
        "words": [{"text": "100mm", "page_bbox_pt": [80, 95, 120, 105]}],
        "reported_vectors": vectors, "vector_primitives": len(vectors),
        "vector_limit_hit": False, "unlocalized_vectors": 0,
    }


class InterruptedDimensionGeometry(unittest.TestCase):
    def setUp(self):
        self.fragments = [((100, 20), (100, 90)), ((100, 110), (100, 180))]
        self.ticks = [((94, 20), (106, 20)), ((94, 180), (106, 180))]

    def test_source_edges_and_gap_have_distinct_provenance(self):
        sheet = drawing(self.fragments + self.ticks)
        result = hamon_callout.associate_page(sheet)
        self.assertEqual(len(result["matches"]), 1)
        match = result["matches"][0]
        self.assertEqual(match["annotation_axis_svg_units"], 160)
        self.assertEqual(match["source_vector_ids"], ["v1", "v2", "v3", "v4"])
        self.assertEqual(match["source_fragments"], [
            {"outer": {"vector": "v1", "vertex": 0}, "inner": {"vector": "v1", "vertex": 1}},
            {"outer": {"vector": "v2", "vertex": 1}, "inner": {"vector": "v2", "vertex": 0}},
        ])
        span = hamon_callout._stroke_for_match(match, hamon_callout._strokes(sheet))
        self.assertEqual((span["a"], span["b"]), ((100, 20), (100, 180)))

    def test_reject_extensions_missing_witnesses_and_misaligned_fragments(self):
        controls = [
            self.fragments + self.ticks[:1],
            self.fragments + [((100, 20), (108, 20)), self.ticks[1]],
            self.fragments + self.ticks + [((25, 90), (165, 90))],
            [self.fragments[0], ((105, 110), (105, 180))] + self.ticks,
            self.fragments + self.ticks + [((94, 90), (106, 90)), ((94, 110), (106, 110))],
        ]
        for segments in controls:
            with self.subTest(segments=segments):
                result = hamon_callout.associate_page(drawing(segments))
                self.assertEqual(result["matches"], [])
