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

    def test_number_only_vetoes_uniqueness_without_creating_a_link(self):
        source = {'source': {'page': 1}, 'target_sheet': 'A5.1', 'target_detail': '6',
                  'reference_form': 'stacked_text_candidate', 'status': 'sheet_only'}
        target = {'A5.1': [{'sheet_ids_on_page': ['A5.1'], 'page': 2}]}
        region = {'ring': {'box': [0, 0, 20, 20]}}
        for reads, expected in [
            ([{'detail': '6', 'caption': 'WALL SECTION'}, {'detail': '6'}], 'ambiguous_detail'),
            ([{'detail': '6'}, {'detail': '6'}], 'sheet_only'),
            ([{'detail': '6', 'caption': 'WALL SECTION'}, {'detail': '8'}], 'candidate_detail'),
        ]:
            ref = dict(source)
            with mock.patch.object(drawing, '_detail_symbol_inventory', return_value=([], [])), \
                    mock.patch.object(drawing, '_divided_source_mark', return_value=region), \
                    mock.patch.object(drawing, '_ink_window', return_value=True), \
                    mock.patch.object(drawing, '_printed_mark', return_value=True), \
                    mock.patch.object(drawing, '_printed_source_tokens', return_value=True), \
                    mock.patch.object(drawing, '_ruled_detail_regions', return_value=[region, region]), \
                    mock.patch.object(drawing.shutil, 'which', return_value='tool'), \
                    mock.patch.object(drawing, '_read_ruled_detail', side_effect=reads):
                drawing._resolve_rendered_stacked({'references': [ref], 'sheets': target}, 'A5.1', 'page.pdf', '.')
            self.assertEqual(ref['status'], expected)
            self.assertEqual(bool(ref.get('rendered_number_conflicts')), expected == 'ambiguous_detail')


if __name__ == '__main__':
    unittest.main()
