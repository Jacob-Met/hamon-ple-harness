"""Physical complete-word controls for tightly packed outlined supply labels."""
import unittest

from hamon_eagle_outlined import LetterStrokes, anchored_voltage, paired_voltages


# Independently drawn plus, 5, 3 and V. No recognizer alphabet is imported.
PLUS = [((0, 3), (4, 3)), ((2, 1), (2, 5))]
FIVE = [(0, 1), (1, 0), (3, 0), (4, 1), (4, 3),
        (3, 4), (2, 4), (0, 3), (0, 6), (4, 6)]
THREE_A = [(0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (2, 3)]
THREE_B = [(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3)]
VEE = [(4, 6), (4, 2), (2, 0), (0, 2), (0, 6)]
DEE = [(4, 1), (3, 0), (0, 0), (0, 6), (3, 6), (4, 5), (4, 1)]
LOWER_W = [(0, 4), (0, 1), (1, 0), (2, 1), (3, 0), (4, 1), (4, 4)]


def shift_stroke(points, x):
    return [((a[0] + x, a[1]), (b[0] + x, b[1]))
            for a, b in zip(points, points[1:])]


FIVE_V = PLUS + shift_stroke(FIVE, 6) + shift_stroke(VEE, 12)
THREE_V_THREE = (PLUS + shift_stroke(THREE_A, 6) + shift_stroke(THREE_B, 6)
                + shift_stroke(VEE, 12) + shift_stroke(THREE_A, 18)
                + shift_stroke(THREE_B, 18))


def vertical_word(edges, plus_y):
    return [tuple((-1.1 * (y - 3), plus_y - 1.1 * (x - 2)) for x, y in edge)
            for edge in edges]


def packed_words():
    marks = [{"foot": (16.5, y), "tip": (6.5, y)} for y in (0, -20)]
    words = [vertical_word(FIVE_V, 9.4), vertical_word(THREE_V_THREE, -6.7)]
    return marks, words


def names(marks, edges, components=None):
    return [word[0] if word else None
            for word in paired_voltages(marks, LetterStrokes(edges), range(len(marks)) if components is None else components)]


class OutlinedVoltageTests(unittest.TestCase):
    def test_complete_neighbors_explain_each_other_without_sharing_strokes(self):
        marks, words = packed_words()
        for rotation in (False, True):
            for scale, tx, ty in ((1, 0, 0), (.7, 193.7, -81.2), (1.3, -50, 67.3)):
                with self.subTest(rotation=rotation, scale=scale, offset=(tx, ty)):
                    def transform(point):
                        x, y = point
                        if rotation:
                            x, y = -y, x
                        return tx + scale*x, ty + scale*y
                    shifted_marks = [{k: transform(v) for k, v in mark.items()} for mark in marks]
                    edges = [tuple(transform(p) for p in edge) for word in words for edge in word]
                    self.assertEqual(names(shifted_marks, edges), ["+5V", "+3V3"])
                    self.assertEqual(names(shifted_marks[::-1], edges[::-1]), ["+3V3", "+5V"])

    def test_every_missing_stroke_removes_the_dependent_interpretations(self):
        marks, words = packed_words()
        edges = words[0] + words[1]
        for i in range(len(edges)):
            with self.subTest(erased_stroke=i):
                self.assertEqual(names(marks, edges[:i] + edges[i+1:]), [None, None])

    def test_two_arrows_cannot_split_one_concatenated_word(self):
        marks, words = packed_words()
        for transform in (lambda p: p, lambda p: (-p[1], p[0])):
            arrows = [{key: transform(p) for key, p in mark.items()} for mark in marks]
            second = vertical_word(THREE_V_THREE, 9.4 - 18*1.1)
            phrase = [tuple(transform(p) for p in edge) for edge in words[0] + second]
            self.assertEqual(names(arrows, phrase), [None, None])
            self.assertEqual(names(arrows[::-1], phrase[::-1]), [None, None])

    def test_unexplained_suffix_or_extra_ink_is_not_ignored(self):
        marks, words = packed_words()
        # x=25 is the middle stem of a damaged next-cell I, beyond the old
        # narrow margin. The whole neighboring cell must remain observable.
        extras = [vertical_word([((x, 0), (x, 6))], -6.7) for x in (24, 25, 28)]
        extras.extend((vertical_word([((24, -1), (28, -1))], -6.7),
                       vertical_word([((28, 2), (28, 6))], -6.7)))
        extras.append([((-.2, 7.3), (.8, 8.1))])
        for extra in extras:
            with self.subTest(extra=extra):
                self.assertEqual(names(marks, words[0] + words[1] + extra), [None, None])

    def test_complete_orthogonal_glyphs_explain_only_their_own_ink(self):
        marks, words = packed_words()
        for glyph in (shift_stroke(DEE, 0), shift_stroke(LOWER_W, 0),
                      shift_stroke(THREE_A, 0) + shift_stroke(THREE_B, 0)):
            annotation = [tuple((-6+1.1*x, -30-1.1*y) for x, y in edge) for edge in glyph]
            self.assertEqual(names(marks, words[0] + words[1] + annotation), ["+5V", "+3V3"])
            for i in range(len(annotation)):
                with self.subTest(glyph=glyph, erased_stroke=i):
                    damaged = annotation[:i] + annotation[i+1:]
                    self.assertEqual(names(marks, words[0] + words[1] + damaged), [None, None])
            # Extra attached ink invalidates the whole orthogonal component,
            # including ink outside the voltage word's immediate crop.
            end = annotation[0][0]
            attached = [(end, (end[0]-1.7, end[1]+2.3))]
            self.assertEqual(names(marks, words[0] + words[1] + annotation + attached), [None, None])

    def test_orthogonal_annotation_cannot_supply_a_missing_word_stroke(self):
        marks, words = packed_words()
        # The D's four-unit right side coincides with the removed plus bar.
        # Its complete orthogonal geometry proves ownership by the annotation.
        annotation = [tuple((-4.4+1.1*x, 12.7-1.1*y) for x, y in edge)
                      for edge in shift_stroke(DEE, 0)]
        self.assertIsNone(anchored_voltage(marks[0], LetterStrokes(words[0][1:] + annotation)))

    def test_neighbor_without_its_own_arrow_cannot_explain_extra_ink(self):
        marks, words = packed_words()
        for i, expected in ((0, "+5V"), (1, "+3V3")):
            with self.subTest(word=expected):
                self.assertEqual(anchored_voltage(marks[i], LetterStrokes(words[i]))[0], expected)
                self.assertIsNone(anchored_voltage(marks[i], LetterStrokes(words[0] + words[1])))

    def test_ambiguous_complete_words_cannot_explain_each_other(self):
        marks, _ = packed_words()
        # Overlay both fully drawn names at one plus. Shared strokes do not
        # provide independent readings or erase their differing digits.
        edges = set(vertical_word(FIVE_V, 9.4) + vertical_word(THREE_V_THREE, 9.4))
        self.assertEqual(names([marks[0]], edges), [None])

    def test_duplicate_painted_stroke_is_not_a_second_complete_word(self):
        marks, words = packed_words()
        self.assertEqual(names(marks, words[0] + words[1] + [words[0][0]]), [None, None])

    def test_word_on_two_components_cannot_justify_a_dependent_neighbor(self):
        marks, words = packed_words()
        marks.append({k: (x, y+2) for k, (x, y) in marks[0].items()})
        edges = words[0] + words[1]
        self.assertEqual(names(marks, edges), [None, None, None])


if __name__ == "__main__":
    unittest.main()
