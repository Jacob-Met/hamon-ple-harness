import unittest
from unittest import mock

import hamon_drawing as drawing


class DetailDigitFramingTest(unittest.TestCase):
    def setUp(self):
        pixels = bytearray([255] * 400)
        for y in range(5, 15):
            pixels[y * 20 + 8:y * 20 + 11] = b'\0' * 3
        self.crop = b'P5\n20 20\n255\n' + pixels
        self.region = {'ring': {'box': [0, 0, 20, 20]}, 'caption_box': [20, 0, 180, 10]}

    def read(self, words):
        with mock.patch.object(drawing, '_caption_raster', return_value=self.crop), \
                mock.patch.object(drawing, '_read_caption_words', side_effect=words):
            return drawing._read_ruled_detail('page.pdf', 1, self.region)

    def test_confirmed_confidence_retry(self):
        evidence = self.read([[('6', 86)], [('6', 77)], [('6', 98)], [('6', 95)],
                              [('WALL', 93), ('SECTION', 94)]])
        self.assertEqual(evidence['detail'], '6')
        self.assertEqual(evidence['digit_confidences'], [98, 95])

    def test_disagreement_and_suffix_are_final(self):
        for opposite in ('8', '6B'):
            self.assertIsNone(self.read([[('6', 99)], [(opposite, 99)]]))

    def test_retry_cannot_change_number_or_skip_confidence(self):
        for bad in ([('8', 99)], [('6', 71)], [('6B', 97)], []):
            self.assertIsNone(self.read([[('6', 99)], [('6', 77)], [('6', 98)], bad]))

    def test_blank_raster_stays_blank(self):
        self.assertIsNone(drawing._frame_detail_digits(b'P5\n20 20\n255\n' + b'\xff' * 400))


if __name__ == '__main__':
    unittest.main()
