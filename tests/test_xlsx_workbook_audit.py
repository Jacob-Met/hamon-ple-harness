"""Workbook relationship and typed-cache controls for the provisioned helper."""
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from engineering_intake import REMOTE_PROBE
from engineering_toolkit import xlsx_audit


class WorkbookAuditTests(unittest.TestCase):
    def package(self, root, strict=False):
        main = ("http://purl.oclc.org/ooxml/spreadsheetml/main" if strict else
                "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
        office = ("http://purl.oclc.org/ooxml/officeDocument/relationships" if strict else
                  "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
        rels = "http://schemas.openxmlformats.org/package/2006/relationships"
        workbook = (f'<workbook xmlns="{main}" xmlns:r="{office}"><sheets>'
                    '<sheet name="Forces" sheetId="7" r:id="loads"/>'
                    '<sheet name="Summary" sheetId="2" r:id="summary" state="hidden"/>'
                    '</sheets></workbook>')
        def relationships(items):
            return (f'<Relationships xmlns="{rels}">' + ''.join(
                f'<Relationship Id="{key}" Type="{office}/{kind}" Target="{target}"/>'
                for key,kind,target in items) + '</Relationships>')
        sheet = (f'<worksheet xmlns="{main}"><sheetData><row r="1">'
                 '<c r="A1"><f>2+2</f><v/></c><c r="B1"><f>1-1</f><v>0</v></c>'
                 '<c r="C1" t="b"><f>FALSE()</f><v>0</v></c>'
                 '<c r="D1" t="str"><f>""</f><v/></c>'
                 '<c r="E1"><f t="shared" si="0"/><v>4</v></c>'
                 '<c r="F1"><f>3*2</f></c></row></sheetData></worksheet>')
        path = Path(root)/'workbook.xlsx'
        with ZipFile(path,'w') as z:
            z.writestr('_rels/.rels',relationships([('root','officeDocument','analysis/book.xml')]))
            z.writestr('analysis/book.xml',workbook)
            z.writestr('analysis/_rels/book.xml.rels',relationships([
                ('summary','worksheet','/values/summary.xml'), ('loads','worksheet','../values/load.xml')]))
            z.writestr('values/load.xml',sheet)
            z.writestr('values/summary.xml',f'<worksheet xmlns="{main}"><sheetData/></worksheet>')
            z.writestr('xl/worksheets/sheet1.xml',sheet)
        return path

    def test_declared_order_relative_absolute_and_orphan_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for strict in (False,True):
                with self.subTest(strict=strict):
                    path = self.package(tmp,strict)
                    result = xlsx_audit(str(path))
                    self.assertEqual(result['workbook_part'],'analysis/book.xml')
                    self.assertEqual([s['part'] for s in result['sheets']],['values/load.xml','values/summary.xml'])
                    self.assertEqual([s['name'] for s in result['sheets']],['Forces','Summary'])
                    self.assertEqual(result['sheets'][1]['state'],'hidden')
                    scope={'__name__':'inventory_test'}
                    exec(REMOTE_PROBE,scope)
                    self.assertEqual(scope['probe_xlsx'](path)['sheets'],['Forces','Summary'])

    def test_empty_strings_and_numeric_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=xlsx_audit(str(self.package(tmp)))['sheets'][0]
            self.assertEqual(row['formulas'],6)
            self.assertEqual(row['formulas_without_cached_values'],2)
            self.assertEqual(row['formulas_with_empty_string_cache'],1)
            self.assertEqual(row['shared_formula_followers'],1)
