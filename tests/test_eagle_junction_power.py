import unittest

from hamon_eagle_schematic import _graph, _ink


class JunctionPowerTests(unittest.TestCase):
    def setUp(self):
        self.wires = [[(100, 70), (100, 50)]]
        self.red = [[(100, 50), (100, 30)], [(90, 45), (100, 30)], [(110, 45), (100, 30)]]
        self.words = [self.word("+5V", (95, 17, 105, 23), (1, 0), (95, 22))]
        self.dots = []
        self.add_junction(0)

    @staticmethod
    def word(name, box, axis, origin):
        return {"text": name, "page_bbox_pt": box, "baseline_direction": axis, "baseline_origin": origin}

    def add_junction(self, y):
        self.wires.extend([[(0, y), (-40, y)], [(0, y), (36, y)], [(0, y), (0, 25+y)]])
        self.red.append([(0, 25+y), (0, 34+y)])
        self.words.extend([self.word("+5V", (18, y-12, 32, y-5), (1, 0), (18, y-7)),
                           self.word("+5V", (-7, y+16, 1, y+32), (0, 1), (0, y+16))])
        self.dots.append(((0, y), 3))

    def receive(self):
        witnesses = []
        groups, _, _ = _graph(self.wires, self.red, self.words,
                              junction_dots=self.dots, junction_witnesses=witnesses)
        return groups[0] == groups[1], witnesses

    def test_requires_both_arm_roles_dot_and_independent_supply_glyph(self):
        joined, witnesses = self.receive()
        self.assertTrue(joined)
        self.assertEqual([a["role"] for a in witnesses[0]["arms"]], ["caption", "pin"])
        for source, index in ((self.dots, 0), (self.red, 0),
                              (self.words, 0), (self.words, 1), (self.words, 2)):
            prior = source.copy()
            source.pop(index)
            self.assertEqual(self.receive(), (False, []))
            source[:] = prior

    def test_absent_red_pin_leaves_the_original_free_end_rule_available(self):
        self.red.pop()
        end, junction = [], []
        groups, _, _ = _graph(self.wires, self.red, self.words, junction_dots=self.dots,
                              wire_end_witnesses=end, junction_witnesses=junction)
        self.assertEqual(groups[0], groups[1])
        self.assertFalse(junction)
        self.assertEqual([w["name"] for w in end], ["+5V"])

    def test_competing_occurrences_on_either_role(self):
        for original in self.words[1:]:
            for name, oriented in (("+5V", True), ("VIN", True), ("GND", True),
                                   ("NET_A", True), ("-1.2V", False)):
                other = dict(original, text=name)
                if not oriented:
                    other.pop("baseline_direction")
                    other.pop("baseline_origin")
                self.words.append(other)
                self.assertEqual(self.receive(), (False, []))
                self.words.pop()

    def test_separate_caption_arm_in_either_direction(self):
        for wire, joined in (([(12, -14), (36, -14)], False), ([(36, -14), (12, -14)], False),
                             ([(12, -30), (36, -30)], True), ([(12, -14), (16, -14)], True)):
            self.wires.append(wire)
            self.assertEqual(self.receive()[0], joined)
            self.wires.pop()

    def test_two_individually_valid_junctions_on_one_island_are_ambiguous(self):
        self.add_junction(80)
        self.wires.append([(-40, 0), (-40, 80)])
        self.assertEqual(self.receive(), (False, []))

    def test_shared_free_end_occurrence_has_no_new_join_in_either_direction(self):
        self.wires.append([(10, -7), (29, -7)])
        witnesses = []
        groups, _, _ = _graph(self.wires, self.red, self.words, junction_dots=self.dots,
                              wire_end_witnesses=witnesses)
        self.assertEqual(len({groups[0], groups[1], groups[-1]}), 3)
        self.assertFalse(witnesses)

    def test_dot_requires_visible_closed_round_fill_in_page_coordinates(self):
        arc = ("M 1 0 C 1 .5523 .5523 1 0 1 C -.5523 1 -1 .5523 -1 0 "
               "C -1 -.5523 -.5523 -1 0 -1 C .5523 -1 1 -.5523 1 0 Z M 1 0")
        for attrs, expected in (("", True), ('transform="scale(1,2)"', False),
                                ('clip-path="url(#hidden)"', False), ('fill-opacity="0"', False)):
            dots = []
            _ink(f'<svg><g {attrs}><path fill="rgb(0%, 50%, 0%)" d="{arc}"/></g></svg>', junction_dots=dots)
            self.assertEqual(bool(dots), expected)
        dot = f'<path fill="rgb(0%, 50%, 0%)" d="{arc}"/>'
        cover = '<rect x="-2" y="-2" width="4" height="4" fill="white"/>'
        stripe = '<path d="M -5 0 L 5 0" fill="none" stroke="white" stroke-width="4"/>'
        for shapes, expected in ((cover + dot, True), (dot + cover, False),
                                 (stripe + dot, True), (dot + stripe, False)):
            dots = []
            _ink(f"<svg>{shapes}</svg>", junction_dots=dots)
            self.assertEqual(bool(dots), expected)
