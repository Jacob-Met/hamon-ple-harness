import unittest
from unittest import mock
import hamon_drawing as h


class PrintedSourceSymbolTest(unittest.TestCase):
    def setUp(self):
        self.ref = {'target_detail': '6', 'source': {'line': 21, 'box': [94,102,106,108]},
                    'numerator_evidence': [{'line': 20, 'text': '6', 'box':[97,92,103,98]}]}
        self.circle = {'id': 'v1', 'box': [88,88,112,112]}
        self.lines = [(88,112,100,'v2')]

    def read(self, ref=None, markers=None, lines=None):
        return h._divided_source_mark(self.ref if ref is None else ref,
                [self.circle] if markers is None else markers, self.lines if lines is None else lines)

    def test_same_circle_and_diameter(self):
        self.assertEqual(self.read()['divider'], 'v2')
        touching = {'id':'v3','box':[112,88,136,112]}
        self.assertEqual(self.read(markers=[self.circle,touching])['ring']['id'], 'v1')

    def test_two_touching_circles_cannot_cross_bind(self):
        ref=dict(self.ref,numerator_evidence=[{'text':'6','line':20,'box':[121,92,127,98]}])
        self.assertIsNone(self.read(ref=ref,markers=[self.circle,{'id':'v3','box':[112,88,136,112]}]))

    def test_missing_partial_or_displaced_divider(self):
        for lines in ([],[(96,112,100,'v2')],[(88,112,104,'v2')],self.lines+self.lines):
            self.assertIsNone(self.read(lines=lines))

    def test_disallowed_hypotheses(self):
        self.assertIsNone(self.read(markers=[]))
        self.assertIsNone(self.read(markers=[self.circle,self.circle]))
        self.assertIsNone(self.read(ref=dict(self.ref,target_detail='8')))
        self.assertIsNone(self.read(ref=dict(self.ref,numerator_evidence=self.ref['numerator_evidence']*2)))

    def test_ruled_caption_requires_right_edge_not_split_diameter(self):
        correct=[(112,270,100,'v17')]
        self.assertEqual(h._ruled_detail_regions([self.circle],correct)[0]['rule'],'v17')
        for lines in ([],self.lines,[(119,270,100,'v18')],[(112,140,100,'v19')],correct+correct):
            self.assertEqual(h._ruled_detail_regions([self.circle],lines),[])

    def test_open_alphabet_rejects_suffix_and_blank_title(self):
        region={'ring':self.circle,'rule':'v17','caption_box':[112,80,270,99]}
        for words in ([[('15',97)],[('15B',98)]],[[('15',97)],[('15',98)],[]],
                      [[('15',97)],[('16',98)]],[[('15',97)],[('15',12)]]):
            with mock.patch.object(h,'_caption_raster',return_value=b'pixels'),\
                    mock.patch.object(h,'_read_caption_words',side_effect=words)as reader:
                self.assertIsNone(h._read_ruled_detail('pdf',1,region))
                self.assertFalse(reader.call_args_list[1].args[-1])

    def test_optical_hits_do_not_override_direct_native_read(self):
        src=dict(self.ref,reference_form='explicit_text',status='resolved_detail',target_sheet='S5')
        result={'references':[src],'sheets':{'S5':[{'page':2,'sheet_ids_on_page':['S5']}]}}
        with mock.patch.object(h,'_detail_symbol_inventory',side_effect=AssertionError):
            h._resolve_rendered_stacked(result,'S5','pdf','.');self.assertEqual(src['status'],'resolved_detail')


class PrintedTextCellTest(unittest.TestCase):
    def read(self, mode):
        width = height = 80
        pixels = bytearray([255]) * (width * height)
        for cy in (32, 48):
            for y in range(cy - 4, cy + 4):
                for x in range(36, 44):
                    dark = (x in (37, 38) or y in (cy - 3, cy + 2)) and 37 <= x <= 42
                    if mode == 'numerator_blank' and cy == 32:
                        dark = False
                    if mode == 'sheet_blank' and cy == 48:
                        dark = False
                    if mode == 'straight_line':
                        dark = y == cy
                        if dark:
                            for xx in range(30, 50):
                                pixels[y * width + xx] = 0
                    if dark:
                        pixels[y * width + x] = 0
        ref = {'numerator_evidence':[{'box':[36,27,44,36]}], 'source':{'box':[36,44,44,53]}}
        return h._printed_source_tokens((0,0,1,width,height,pixels), {'box':[24,24,56,56]}, ref)

    def test_two_printed_tokens(self):
        self.assertEqual(set(self.read('characters')), {'numerator','sheet'})

    def test_rejects_blank_token_and_rule_only(self):
        for mode in ('numerator_blank','sheet_blank','straight_line'):
            self.assertIsNone(self.read(mode), mode)


if __name__=='__main__':unittest.main()
