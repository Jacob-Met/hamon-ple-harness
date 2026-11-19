"""Deterministic engineering CLI: exact dimensions and file invariants."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engineering_toolkit as t
from engineering_intake import provision_engineering_cli


class FakeSession:
    def __init__(self): self.path = ""; self.payload = b""; self.files = {}
    async def write_file(self,path,payload):
        self.path=path;self.payload=payload;self.files[path]=payload
        return len(payload)


class ToolkitContracts(unittest.TestCase):
    def test_exact_shear_and_pressure_with_units(self):
        a=t.conversion("12*kip/(8*in_**2)","ksi")
        self.assertEqual(t.D(a["value"]),t.D("1.5"))
        b=t.conversion("50*kN/(200*mm**2)","MPa")
        self.assertEqual(t.D(b["value"]),t.D("250"))
        self.assertAlmostEqual(float(t.conversion("100*psf","Pa")["value"]),
                               4788.025898033584, places=9)
        self.assertEqual(t.D(t.conversion("1000*psi","ksi")["value"]),t.D("1"))
        self.assertEqual(t.conversion("sqrt(9*mm**2)","mm")["value"],"3")

    def test_dimensional_mismatch_and_security(self):
        for expr,unit in [("20*ft + 5*s","ft"),
                          ("3*lb","N"),("__import__('os').system('id')","one"),
                          ("1/(0*m)","one"),("2**1000","one"),
                          ("sqrt(9*m)","m")]:
            with self.subTest(expr=expr):
                with self.assertRaises((ValueError,SyntaxError)):
                    t.conversion(expr,unit)
        self.assertTrue(t.equal_dimensions("1000*N","1*kN","0.00001")["passed"])
        self.assertFalse(t.equal_dimensions("900*N","1*kN","0.00001")["passed"])

    def test_broad_phase_overlap_is_not_mesh_verdict(self):
        a=([0,0,0],[1,1,1]);b=([0.9,0,0],[2,1,1])
        self.assertTrue(t.overlaps(a,b))
        self.assertFalse(t.overlaps(a,([2,2,2],[3,3,3])))

    def test_xlsx_formula_cache_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"calculation.xlsx"
            sheet=('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<sheetData><row r="1"><c r="A1"><f>SUM(B1:C1)</f></c>'
                   '<c r="B1" t="e"><v>#REF!</v></c></row></sheetData></worksheet>')
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("xl/worksheets/sheet1.xml",sheet)
            result=t.xlsx_audit(str(p))
            self.assertEqual(result["sheets"][0]["formulas"],1)
            self.assertEqual(result["sheets"][0]["formulas_without_cached_values"],1)
            self.assertEqual(result["sheets"][0]["error_count"],1)

    def test_cli_error_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(__file__).resolve().parents[1]/"engineering_toolkit.py"
            cmd=[sys.executable,str(src),"quantity","12*kip","--to","m"]
            cp=subprocess.run(cmd,capture_output=True,text=True,check=False)
            self.assertEqual(cp.returncode,2)
            self.assertFalse(json.loads(cp.stdout)["ok"])

    def test_provision_only_copies_code_and_writes_no_task_content(self):
        fake=FakeSession()
        instructions=asyncio.run(provision_engineering_cli(fake))
        self.assertEqual(fake.path,"/tmp/hamon-engineering-toolkit.py")
        self.assertEqual(fake.payload,(Path(__file__).resolve().parents[1]/
                                        "engineering_toolkit.py").read_bytes())
        self.assertIn("in the task container",instructions)
        self.assertIn("/tmp/hamon_callout.py",fake.files)
        self.assertEqual(fake.files["/tmp/hamon_callout.py"],
                         (Path(__file__).resolve().parents[1]/"hamon_callout.py").read_bytes())
        self.assertIn("/tmp/hamon_drawing.py",fake.files)
        self.assertIn("pdf-dimension-links",instructions)
        self.assertIn("pdf-auto-measure",instructions)


if __name__=="__main__":
    unittest.main()
