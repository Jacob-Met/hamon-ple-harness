import unittest

from unittest.mock import patch

from hamon_eagle_schematic import _graph, _ink, _stitch_pages, _word_directions, document_connectivity


def word(name, box):
    return {"text": name, "page_bbox_pt": box, "baseline_direction": (1, 0)}


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

    def test_native_text_overhanging_a_free_wire_end(self):
        wires = [[(0, 10), (30, 10)], [(50, 40), (80, 40)]]
        words = [word("CLK", (1, 3, 13, 11)), word("CLK", (45, 33, 58, 41))]
        # First word qualifies under the ordinary printed-label rule.
        # The second begins before its wire and only qualifies through the actual glyph baseline.
        _word_directions(words, [((x, 10), "font") for x in (1, 5, 9)]
                               + [((x, 40), "font") for x in (45, 49, 53)])
        witnesses = []
        self.assertEqual(_graph(wires, [], words, wire_end_witnesses=witnesses)[0], [0, 0])
        self.assertEqual(witnesses, [{"name": "CLK", "wire_index": 1, "endpoint": (50, 40),
                                      "baseline_origin": (45, 40)}])
        for extra_wire, symbol in (([(50, 39), (50, 44)], None),
                                   (None, [(50.1, 35), (50.1, 45)]),
                                   ([(44, 40), (57, 40)], None)):
            observed = _graph(wires + ([extra_wire] if extra_wire else []),
                              [symbol] if symbol else [], words)[0]
            self.assertNotEqual(observed[0], observed[1])

        for label_index in (0, 1):
            conflicting = [dict(w) for w in words]
            x, y = ((20, 10), (65, 40))[label_index]
            conflicting.append(word("ALT", (x, y-7, x+10, y+1)))
            self.assertEqual(_graph(wires, [], conflicting)[0], [0, 1])

        for origin in ((45, 40.2), None):
            shifted = [dict(w) for w in words]
            if origin is None:
                shifted[1].pop("baseline_origin")
            else:
                shifted[1]["baseline_origin"] = origin
            self.assertEqual(_graph(wires, [], shifted)[0], [0, 1])

    def test_wire_end_supply_label_requires_a_physical_same_page_anchor(self):
        wires = [[(0, 10), (30, 10)], [(50, 40), (80, 40)]]
        labels = [word("VCC", (1, 3, 13, 11)), word("VCC", (45, 33, 58, 41))]
        labels[1]["baseline_origin"] = (45, 40)
        self.assertEqual(_graph(wires, [], labels)[0], [0, 1])

    def test_ink_at_a_proposed_free_end(self):
        wires = '<g stroke="rgb(0%, 50%, 0%)" fill="none"><path d="M 0 10 L 30 10"/><path d="M 50 40 L 80 40"/></g>'
        labels = [
            {"text": "CLK", "baseline_direction": (1, 0), "baseline_origin": (1, 10),
             "page_bbox_pt": (1, 3, 13, 11)},
            {"text": "CLK", "baseline_direction": (1, 0), "baseline_origin": (45, 40),
             "page_bbox_pt": (45, 33, 58, 41)},
        ]
        for name, shape, joined in (
            ("clear", "", True),
            ("crossing cubic", '<path d="M 45 30 C 57 35 43 45 55 50"/>', False),
            ("crossing polyline", '<path d="M 48 30 L 50 40 L 55 50"/>', False),
            ("clear L shape", '<path d="M 30 20 L 30 60 L 70 60"/>', True),
            ("disconnected strokes", '<path d="M 30 40 L 35 40 M 65 40 L 70 40"/>', True),
            ("clear cubic", '<path d="M 55 30 C 65 35 55 45 65 50"/>', True),
            ("red circle bounds", '<circle cx="50" cy="40" r="1"/>', False),
            ("unknown bounds", '<path d="M 0 0 A 1 1 0 0 0 2 0"/>', False),
        ):
            svg = f'<svg>{wires}<g stroke="rgb(90%, 0%, 0%)" fill="none">{shape}</g></svg>'
            with self.subTest(name=name):
                obstacles = ([], [])
                green, red, _, _ = _ink(svg.encode(), symbol_obstacles=obstacles)
                groups, _, _ = _graph(green, red, labels, symbol_obstacles=obstacles)
                self.assertEqual(groups[0] == groups[1], joined)

    def test_one_occurrence_cannot_name_an_ordinary_arrow_and_a_second_wire(self):
        for separate in (False, True):
            dx = 40 if separate else 0
            wires = [[(2.5+dx, 29), (2.5+dx, 19)], [(2.8, 11), (2.8, 1)]]
            foot, tip = (2.5+dx, 19), (2.5+dx, 9)
            red = [[foot, tip], [tip, (-2.5+dx, 16.5)], [tip, (7.5+dx, 16.5)]]
            words = [{"text": "+5V", "page_bbox_pt": (x+2, -1, x+3, 9), "baseline_origin": (x+2.8, 9),
                      "baseline_direction": (0, -1)} for x in ([0, dx] if separate else [0])]
            evidence = []
            groups, _, _ = _graph(wires, red, words, wire_end_witnesses=evidence)
            self.assertEqual(groups[0] == groups[1], separate)
            self.assertEqual(len(evidence), int(separate))

    def test_one_occurrence_cannot_name_a_parallel_wire_and_a_second_loose_end(self):
        for separate in (False, True):
            dx = 40 if separate else 0
            wires = [[(dx, 40.7), (dx+30, 40.7)], [(5, 40), (30, 40)]]
            words = [{"text": "CLK", "page_bbox_pt": (x, 33, x+13, 41),
                      "baseline_origin": (x, 40), "baseline_direction": (1, 0)}
                     for x in ([0, dx] if separate else [0])]
            evidence = []
            groups, _, _ = _graph(wires, [], words, wire_end_witnesses=evidence)
            self.assertEqual(groups[0] == groups[1], separate)
            self.assertEqual(len(evidence), int(separate))

    def test_curve_and_unknown_transform_remain_unresolved(self):
        raw = b'<svg xmlns="http://www.w3.org/2000/svg"><g stroke="rgb(0%, 50%, 0%)" fill="none"><path d="M 1 2 C 1 3 4 6 7 8"/><path transform="wibble(8)" d="M 0 0 L 10 0"/></g></svg>'
        wires, red, marks, unknown = _ink(raw)
        self.assertEqual((wires, red, marks), ([], [], 0))
        self.assertEqual(unknown, {"green_nonsegment": 1, "green_appearance_or_transform": 1})

    def test_document_requires_single_unambiguous_name_and_matching_supply_glyph(self):
        def page(number, names, glyphs):
            return {"page": number, "wire_groups": [0, 1, 2, 3, 4, 5, 6, 6],
                    "printed_equivalence_anchors": names,
                    "supply_glyphs": [{"name": name, "wire_index": index,
                                       "shape": "ground_bar" if name == "GND" else "supply_arrow"}
                                      for name, index in glyphs]}
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

    def test_ground_requires_independent_glyph_islands_and_single_incidental_label(self):
        def ground_page(number):
            return {"page": number, "wire_groups": [0, 0, 0, 1],
                    "printed_equivalence_anchors": {"GND": [0, 1]},
                    "supply_glyphs": [{"name": "GND", "shape": "ground_bar", "wire_index": i}
                                      for i in (0, 1)]}

        a, b = ground_page(1), ground_page(2)
        a["printed_equivalence_anchors"]["U12"] = [1]
        members, links = _stitch_pages([a, b])
        self.assertEqual(members[0][0], members[1][1])
        self.assertEqual(links[0]["ground_bar_witnesses"][0]["incidental_labels"], ["U12"])
        for trial in ("same island twice", "bare label away from stem", "recurrent label",
                      "second supply glyph", "different ground family"):
            a, b = ground_page(1), ground_page(2)
            a["printed_equivalence_anchors"]["U12"] = [1]
            if trial == "same island twice":
                b["supply_glyphs"][1]["wire_index"] = 0
            elif trial == "bare label away from stem":
                a["printed_equivalence_anchors"]["U12"] = [2]
            elif trial == "recurrent label":
                b["printed_equivalence_anchors"]["U12"] = [3]
            elif trial == "second supply glyph":
                b["supply_glyphs"].append({"name": "VCC", "shape": "supply_arrow", "wire_index": 1})
            else:
                a["printed_equivalence_anchors"]["AGND"] = [1]
            with self.subTest(trial=trial):
                left, right = _stitch_pages([a, b])[0]
                self.assertNotEqual(left[0], right[0])

    def test_wire_anchor_uses_actual_glyph_direction_under_quarter_turns(self):
        wires = [[(0, 10), (22, 10)], [(50, 40), (72, 40)]]
        boxes = [(0, 3, 14, 11), (50, 33, 64, 41)]
        good = [(0, 9), (4, 9), (9, 9)]
        for rotation in range(4):
            def rotate(point):
                x, y = point
                for _ in range(rotation):
                    x, y = -y, x
                return x, y

            paths = [[rotate(a), rotate(b)] for a, b in wires]
            for perpendicular in (False, True):
                words = []
                for x, y, u, v in boxes:
                    corners = [rotate((x, y)), rotate((u, v))]
                    lo, hi = zip(*corners)
                    words.append({"text": "CLK", "page_bbox_pt": (min(lo), min(hi), max(lo), max(hi))})
                second = [(53, 40), (53, 37), (53, 34)] if perpendicular else [(50, 39), (54, 39), (59, 39)]
                _word_directions(words, [(rotate(pt), "font") for pt in good + second])
                with self.subTest(rotation=rotation, perpendicular=perpendicular):
                    self.assertEqual(_graph(paths, [], words)[0], [0, int(perpendicular)])

    def test_ambiguous_incomplete_or_bent_glyph_runs_abstain(self):
        for glyphs in (
            [((0, 1), "f"), ((4, 1), "f")],
            [((0, 1), "f"), ((4, 1), "g"), ((8, 1), "f")],
            [((0, 1), "f"), ((4, 1), "f"), ((8, 3), "f")],
            [((0, 1), "f"), ((42, 40), "f"), ((4, 1), "f"), ((8, 1), "f")],
            [((0, 1), "f"), ((4, 1), "f"), ((8, 1), "f"), ((4, 2), "f")],
        ):
            words = [{"text": "CLK", "page_bbox_pt": (0, 0, 12, 7)}]
            _word_directions(words, glyphs)
            self.assertNotIn("baseline_direction", words[0])


    @staticmethod
    def _side_supply_course(kind):
        # Synthetic drawing geometry; no manufacturer fixture coordinates.
        def arrow(x, y):
            foot, tip = (x, y + 8), (x, y)
            return [foot, (x, y + 24)], [[foot, tip], [(x + 4, y + 6), tip],
                                                 [tip, (x - 4, y + 6)]]

        side = {"text": "+5V", "page_bbox_pt": (3, -2, 13, 13)}
        side_glyphs = [((10, y), "font") for y in (13, 8, 3)]
        a, lines_a = arrow(0, 0)
        b, lines_b = arrow(8, 10.5) if kind == "shared_tip" else arrow(80, 0)
        words, glyphs = [side], side_glyphs
        if kind != "shared_tip":
            words.append({"text": "+5V", "page_bbox_pt": (76, -8, 84, -2)})
            glyphs += [((x, -2), "font") for x in (76, 79, 82)]
        wires, lines = [b, a], lines_b + lines_a
        if kind == "shared_end":
            wires.append([(10, 15), (10, 1)])
        elif kind == "two_tip_names":
            words += [{"text": name, "page_bbox_pt": box}
                      for name, box in (("VIN", (-4, -8, 4, -2)), ("VCC", (-4, -9, 4, -3)))]
        elif kind in ("negative_rail", "unoriented_rail"):
            words.append({"text": "-5V" if kind == "negative_rail" else "VIN",
                          "page_bbox_pt": (-13, -2, -3, 13)})
            glyphs += [((-10, y), "different" if kind == "unoriented_rail" and i == 1 else "font")
                       for i, y in enumerate((13, 8, 3))]
        elif kind == "missing_seed":
            lines = lines_a
            words, glyphs = words[:1], glyphs[:3]
        elif kind == "incomplete_arrow":
            lines.pop()
        return wires, lines, words, glyphs

    def test_supply_side_labels_keep_distinct_printed_occurrences(self):
        for kind, expected in (("separate", 1), ("shared_tip", 2), ("shared_end", 3)):
            for turn in range(4):
                def rotate(point):
                    x, y = point
                    for _ in range(turn):
                        x, y = -y, x
                    return (x, y)

                wires, lines, words, glyphs = self._side_supply_course(kind)
                for w in words:
                    x, y, u, v = w["page_bbox_pt"]
                    corners = [rotate(p) for p in ((x, y), (u, y), (u, v), (x, v))]
                    xs, ys = zip(*corners)
                    w["page_bbox_pt"] = (min(xs), min(ys), max(xs), max(ys))
                _word_directions(words, [(rotate(p), font) for p, font in glyphs])
                witnesses = []
                groups, marks, _ = _graph([[rotate(p) for p in wire] for wire in wires],
                                          [[rotate(p) for p in line] for line in lines],
                                          words, wire_end_witnesses=witnesses)
                with self.subTest(kind=kind, turn=turn):
                    self.assertEqual(len(set(groups)), expected)
                    self.assertEqual(sum(m.get("label_placement") == "side" for m in marks),
                                     int(kind == "separate"))
                    self.assertFalse(witnesses)
                    self.assertFalse(any(k.startswith("_") for m in marks for k in m))

    def test_supply_side_labels_reject_missing_or_competing_evidence(self):
        for kind in ("two_tip_names", "negative_rail", "unoriented_rail",
                     "missing_seed", "incomplete_arrow"):
            wires, lines, words, glyphs = self._side_supply_course(kind)
            _word_directions(words, glyphs)
            with self.subTest(kind=kind):
                groups, marks, _ = _graph(wires, lines, words)
                self.assertEqual(groups, [0, 1])
                self.assertFalse(any(m.get("label_placement") == "side" for m in marks))

    def test_power_only_baselines_do_not_broaden_legacy_wire_names(self):
        for name in ("-5V", "+3.3V", "vcc"):
            words = [{"text": name, "page_bbox_pt": (20, 0, 28, 22)}]
            _word_directions(words, [((26, 20 - 4 * i), "font") for i in range(len(name))])
            with self.subTest(name=name):
                self.assertNotIn("baseline_direction", words[0])
                self.assertEqual(words[0]["power_baseline_direction"], (0, -1))
                self.assertEqual(words[0]["power_baseline_origin"], (26, 20))
