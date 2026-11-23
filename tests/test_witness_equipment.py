import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from xml.sax.saxutils import escape
from zipfile import ZipFile

from engineering_toolkit import _equipment_catalog
from hamon_drawing_reader import build_minimal_vector_pdf
from hamon_witness_package import match_installation


def workbook(path, rows):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pkg = 'http://schemas.openxmlformats.org/package/2006/relationships'
    with ZipFile(path, 'w') as archive:
        with archive.open('xl/workbook.xml', 'w') as part:
            part.write((f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
                        '<sheet name="Issue" sheetId="1" r:id="r1"/>'
                        '<sheet name="Equipment" sheetId="2" r:id="r2"/>'
                        '</sheets></workbook>').encode())
        refs = ''.join(f'<Relationship Id="r{i}" Type="{rel}/worksheet" Target="worksheets/s{i}.xml"/>'
                       for i in (1,2))
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{pkg}">{refs}</Relationships>')
        for i, data in enumerate(([('Doc no.', 'Issue-B')], rows), 1):
            cells = []
            for n, pair in enumerate(data, 1):
                values = ''.join(f'<c r="{col}{n}" t="inlineStr"><is><t>{escape(val)}</t></is></c>'
                                 for col,val in zip('AB', pair))
                cells.append(f'<row r="{n}">{values}</row>')
            archive.writestr(f'xl/worksheets/s{i}.xml', f'<worksheet xmlns="{ns}"><sheetData>'
                             + ''.join(cells) + '</sheetData></worksheet>')


class EquipmentJoin(unittest.TestCase):
    def test_two_physical_pages_to_native_cells(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'assembly.pdf'
            pages = ['Inverter model (per cover sheet): North M-8',
                     'Inverter model (as installed): North M-8B']
            pdf.write_bytes(build_minimal_vector_pdf([{'content_stream':
                'BT /F1 12 Tf 25 500 Td ('+line.replace('(',r'\(').replace(')',r'\)')+') Tj ET'} for line in pages]))
            wb = root / 'equipment.xlsx'
            workbook(wb, [('Manufacturer','Model'),('North','M-8'),('North','M-8B')])
            tool = Path(__file__).resolve().parents[1] / 'engineering_toolkit.py'
            cmd = [sys.executable,str(tool),'witness-equipment', str(pdf),str(wb),
                   '--edition','Issue-B','--title-sheet','Issue','--sheet','Equipment']
            result = json.loads(subprocess.check_output(cmd, text=True))
            self.assertEqual(result['status'], 'matched_installed_equipment')
            self.assertEqual([(x['role'],x['page']) for x in result['evidence']], [('reference',1),('installed',2)])
            self.assertEqual(result['catalog_rows'][0]['cells'], ['Equipment!A3','Equipment!B3'])
            self.assertEqual(result['title_cell'], 'Issue!B1')
            with self.assertRaisesRegex(ValueError,'edition'):
                _equipment_catalog(wb,'Issue-A','Equipment','Issue')
            workbook(wb, [('Manufacturer','Model'),('North','M-8'),('','M-8B')])
            with self.assertRaisesRegex(ValueError,'missing literal'):
                _equipment_catalog(wb,'Issue-B','Equipment','Issue')

    def test_unresolved_inputs_and_no_prefix_match(self):
        line='Inverter model (as installed): North M-8B'
        catalog={'north m-8':[{'model':'M-8'}]}
        self.assertEqual(match_installation(line,catalog)['status'],'no_exact_catalog_row')
        self.assertEqual(match_installation(line.replace('as installed','cover sheet'),catalog)['status'],'installation_unresolved')
        self.assertEqual(match_installation(line+'\f'+line.replace('M-8B','M-8'),catalog)['status'],'conflicting_installation_evidence')
        self.assertEqual(match_installation(line,{'north m-8b':[{},{}]})['status'],'ambiguous_catalog_rows')

if __name__ == '__main__':
    unittest.main()
