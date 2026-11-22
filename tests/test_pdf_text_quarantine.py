import unittest
import xml.etree.ElementTree as ET

import hamon_drawing as drawing


def layout(lines):
    parts = [b'<doc><page width="50" height="80">']
    for index, words in enumerate(lines):
        bounds = f' xMin="1" xMax="10" yMin="{4 * index + 1}" yMax="{4 * index + 3}"'.encode()
        parts.append(b"<line" + bounds + b">")
        parts.extend(b"<word" + bounds + b">" + word + b"</word>" for word in words)
        parts.append(b"</line>")
    return b"".join([*parts, b"</page></doc>"])


class PopplerControlByteTests(unittest.TestCase):
    def test_corrupt_identifier_quarantines_the_whole_line_but_preserves_its_position(self):
        raw = layout([[b"SHEET", b"S1"], [b"DETAIL", b"1\x03", b"2", b"ON", b"SHEET", b"S1"], [b"3/S2"]])
        page = drawing._poppler_text_xml(raw).find("page")
        rows = drawing._drawing_text_lines(page, 1)
        self.assertEqual([(row["line"], row["text"]) for row in rows],
                         [(1, "SHEET S1"), (2, "\ufffd"), (3, "3/S2")])
        words, _ = drawing._word_inventory(raw, 99)
        self.assertEqual([row["text"] for row in words if row["source_line_id"] == 1],
                         ["\ufffd"] * 6)

    def test_control_inside_a_number_or_a_caption_cannot_join_its_fragments(self):
        for words in ([b"1\x1e2", b"S4"], [b"DETAIL", b"9", b"FORT\x1dY", b"WALL"]):
            with self.subTest(words=words):
                page = drawing._poppler_text_xml(layout([words])).find("page")
                self.assertEqual(drawing._drawing_text_lines(page, 1)[0]["text"], "\ufffd")

    def test_control_bytes_in_markup_and_malformed_trees_still_fail(self):
        cases = [b"<doc><wo\x03rd>2</word></doc>", b'<doc a="\x1d">2</doc>',
                 b"<doc><word>2\x1e3</word></bad>", b"<doc>outside\x11word</doc>"]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ET.ParseError):
                drawing._poppler_text_xml(raw)


if __name__ == "__main__":
    unittest.main()
