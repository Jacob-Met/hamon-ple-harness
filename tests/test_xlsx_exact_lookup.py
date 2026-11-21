"""Independent OOXML lookup controls, with literal engineering catalog rows."""
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from engineering_toolkit import xlsx_audit

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
OFFICE = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def worksheet(rows):
    root = ET.Element('worksheet', xmlns=MAIN)
    data = ET.SubElement(root, 'sheetData')
    for row_number, cells in enumerate(rows, 1):
        row = ET.SubElement(data, 'row', r=str(row_number))
        for col, formula, numeric in cells:
            cell = ET.SubElement(row, 'c', r=f'{col}{row_number}')
            if formula is not None:
                ET.SubElement(cell, 'f').text = formula
            if numeric is not None:
                ET.SubElement(cell, 'v').text = str(numeric)
    return ET.tostring(root)


def package(path):
    table = "'Ratings 2026'!$A$1:$C$4"
    formulas = [
        (f'VLOOKUP(A1,{table},2,FALSE)', 25),
        (f'VLOOKUP(A2,{table},3,0)', 99),
        (f'HLOOKUP(A3,{table},2,FALSE)', 15),
        (f'HLOOKUP(A4,{table},2,FALSE)', 10),
        (f'VLOOKUP(A5,{table},3,TRUE)', 17),
        (f'VLOOKUP(A6,{table},3,FALSE)', 42),
        (f'VLOOKUP(A7,{table},2,FALSE)', None),
        (f'VLOOKUP(A8,{table},2,FALSE)', 42),
        (f'VLOOKUP(A9,{table},100,FALSE)', 42),
        ('VLOOKUP(A10,Missing!A1:C4,2,FALSE)', 25),
    ]
    with ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{MAIN}" xmlns:r="{OFFICE}"><sheets>'
                   '<sheet name="Results" sheetId="1" r:id="s1"/>'
                   '<sheet name="Ratings 2026" sheetId="2" r:id="s2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
                   ''.join(f'<Relationship Id="s{i}" Type="{OFFICE}/worksheet" Target="worksheets/{i}.xml"/>'
                           for i in (1, 2)) + '</Relationships>')
        keys = [1, 1, 25, 25, 1, 3, 1, 1, 1, 1]
        z.writestr('xl/worksheets/1.xml', worksheet([
            [('A', '1+0' if i == 7 else None, keys[i]), ('B', f, v)]
            for i, (f, v) in enumerate(formulas)]))
        z.writestr('xl/worksheets/2.xml', worksheet([
            [('A', None, 1), ('B', None, 25), ('C', None, 0)],
            [('A', None, 2), ('B', None, 15), ('C', None, 11)],
            [('A', None, 1), ('B', None, 30), ('C', None, 99)],
            [('A', None, 3), ('B', None, 42), ('C', '2*21', 42)]]))


class LookupTests(unittest.TestCase):
    def test_first_exact_match_from_saved_literals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'catalog.xlsx'
            package(path)
            audit = xlsx_audit(str(path)).get('lookup_audit', {})
            self.assertEqual(audit.get('counts'), {'matched': 2, 'mismatch': 2, 'unverified': 6})
            output = {r['cell']: r for r in audit['examples']}
            self.assertEqual(output['B2']['source_cell'], 'C1')
            self.assertEqual(output['B2']['expected'], 0)
            self.assertEqual(output['B4']['source_cell'], 'B2')
            self.assertEqual(output['B4']['source_sheet'], 'Ratings 2026')

    def test_formula_range_followers_are_not_literals(self):
        cases = [('needle', 1, 'A1', 'A1:A2', 'unverified'),
                 ('prior-key', 2, 'A1', 'A1:A2', 'unverified'),
                 ('returned', 2, 'B2', 'B2:B3', 'unverified'),
                 ('disjoint', 2, 'D2', 'D2:D3', 'matched'),
                 ('missing', 2, 'D2', None, 'unverified'),
                 ('malformed', 2, 'D2', 'not-a-range', 'unverified'),
                 ('reversed', 2, 'D2', 'D3:D2', 'unverified'),
                 ('whole-grid', 2, 'A1', 'A1:XFD1048576', 'unverified')]
        for mode in ('array', 'dataTable'):
            for label, sheet, master, area, expected in cases:
                with self.subTest(mode=mode, case=label), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'range.xlsx'
                    package(path)
                    with ZipFile(path) as z:
                        parts = {name: z.read(name) for name in z.namelist()}
                    parts['xl/worksheets/1.xml'] = worksheet([
                        [('A', None, 1)], [('A', None, 2),
                        ('B', "VLOOKUP(A2,'Ratings 2026'!A2:B4,2,FALSE)", 20)]])
                    parts['xl/worksheets/2.xml'] = worksheet([
                        [('A', None, 1)],
                        [('A', None, 1), ('B', None, 10), ('D', None, 2)],
                        [('A', None, 2), ('B', None, 20), ('D', None, 3)],
                        [('A', None, 3), ('B', None, 30)]])
                    member = f'xl/worksheets/{sheet}.xml'
                    root = ET.fromstring(parts[member])
                    cell = next(c for c in root.iter(f'{{{MAIN}}}c') if c.get('r') == master)
                    attrs = {'t': mode, **({'ref': area} if area is not None else {})}
                    ET.SubElement(cell, f'{{{MAIN}}}f', attrs).text = 'ROW(A1:A2)'
                    # The master is deliberately serialized after its followers.
                    data = root.find(f'{{{MAIN}}}sheetData')
                    data[:] = reversed(list(data))
                    parts[member] = ET.tostring(root)
                    with ZipFile(path, 'w') as z:
                        for name, content in parts.items():
                            z.writestr(name, content)
                    report = xlsx_audit(str(path))
                    self.assertEqual(report['lookup_audit']['examples'][0]['status'], expected)

    def test_lookup_endpoints_respect_worksheet_bounds(self):
        for kind, endpoint, expected in [('VLOOKUP', 'XFE4', 'unverified'),
                                         ('VLOOKUP', 'XFD4', 'matched'),
                                         ('HLOOKUP', 'C1048577', 'unverified'),
                                         ('HLOOKUP', 'C1048576', 'matched')]:
            with self.subTest(kind=kind, endpoint=endpoint), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bounds.xlsx'
                package(path)
                with ZipFile(path) as z:
                    parts = {name: z.read(name) for name in z.namelist()}
                root = ET.fromstring(parts['xl/worksheets/1.xml'])
                address, key = ('B1', 'A1') if kind == 'VLOOKUP' else ('B3', 'A3')
                cell = next(c for c in root.iter(f'{{{MAIN}}}c') if c.get('r') == address)
                cell.find(f'{{{MAIN}}}f').text = f"{kind}({key},'Ratings 2026'!A1:{endpoint},2,FALSE)"
                parts['xl/worksheets/1.xml'] = ET.tostring(root)
                with ZipFile(path, 'w') as z:
                    for name, content in parts.items():
                        z.writestr(name, content)
                examples = xlsx_audit(str(path))['lookup_audit']['examples']
                self.assertEqual(next(x for x in examples if x['cell'] == address)['status'], expected)
