"""Changed-input controls for source-native DXF dimensions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import hamon_dxf

def fixture(path,units=4,kind=1,angle=None,p2=(120,0),reported=120,override="<>"):
    lines=["0","SECTION","2","HEADER","9","$INSUNITS","70",str(units),
           "0","ENDSEC","0","SECTION","2","ENTITIES",
           "0","DIMENSION","5","A2","8","DIM","70",str(kind),
           "13","0","23","0","14",str(p2[0]),"24",str(p2[1]),
           "42",str(reported),"1",override]
    if angle is not None:lines+=["50",str(angle)]
    lines+=["0","ENDSEC","0","EOF"]
    path.write_text("\n".join(lines)+"\n")

class DxfEvidenceTests(unittest.TestCase):
    def test_source_and_live_cli(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";fixture(p)
            d=hamon_dxf.scan_dxf(p)
            self.assertEqual(d["declared_unit"],"mm")
            self.assertEqual(d["total_dimensions"],1)
            self.assertEqual(d["dimensions"][0]["usable_qualified_value"],120)
            proc=subprocess.run([sys.executable,"engineering_toolkit.py","dxf-dimensions",str(p)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["total_dimensions"],1)
    def test_source_change_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";fixture(p);before=hamon_dxf.scan_dxf(p)
            fixture(p,reported=119,override="119")
            after=hamon_dxf.scan_dxf(p)
            self.assertNotEqual(before["source_sha256"],after["source_sha256"])
            self.assertEqual(after["dimensions"][0]["geometry_status"],"cad_measurement_mismatch")
            self.assertIsNone(after["dimensions"][0]["usable_qualified_value"])
    def test_rotated_and_unsupported_kind(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";fixture(p,kind=0,angle=90,p2=(120,30),reported=30)
            self.assertAlmostEqual(hamon_dxf.scan_dxf(p)["dimensions"][0]["usable_qualified_value"],30)
            fixture(p,kind=2,p2=(120,30),reported=30)
            self.assertIsNone(hamon_dxf.scan_dxf(p)["dimensions"][0]["usable_qualified_value"])
    def test_declared_units_and_missing_cad_measurement(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";fixture(p,units=0)
            self.assertEqual(hamon_dxf.scan_dxf(p)["unit_status"],"unknown")
            p.write_text(p.read_text().replace("42\n120\n",""))
            self.assertEqual(hamon_dxf.scan_dxf(p)["dimensions"][0]["geometry_status"],"geometry_only")
    def test_native_radius_diameter_and_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"radial.dxf"
            fixture(p,kind=3)
            text=p.read_text().replace("42\n120\n",
                "10\n1\n20\n1\n15\n13\n25\n1\n42\n12\n")
            p.write_text(text)
            d=hamon_dxf.scan_dxf(p)["dimensions"][0]
            self.assertEqual(d["kind"],"diameter")
            self.assertEqual(d["usable_qualified_value"],12)
            p.write_text(text.replace("70\n3\n","70\n4\n"))
            self.assertEqual(hamon_dxf.scan_dxf(p)["dimensions"][0]["kind"],"radius")
            self.assertEqual(hamon_dxf.scan_dxf(p)["dimensions"][0]["usable_qualified_value"],12)
            p.write_text(p.read_text().replace("42\n12\n","42\n11\n"))
            self.assertEqual(hamon_dxf.scan_dxf(p)["dimensions"][0]["geometry_status"],
                             "cad_measurement_mismatch")
            self.assertIsNone(hamon_dxf.scan_dxf(p)["dimensions"][0]["usable_qualified_value"])

    def test_nondefault_linear_extrusion_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"nondefault.dxf"
            fixture(p,kind=0,angle=0)
            p.write_text(p.read_text().replace("50\n0\n",
                "50\n0\n210\n0\n220\n1\n230\n0\n"))
            self.assertIsNone(hamon_dxf.scan_dxf(p)["dimensions"][0]["usable_qualified_value"])

    def test_bad_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";p.write_bytes(b"AutoCAD Binary DXF\x00")
            with self.assertRaisesRegex(ValueError,"binary"):hamon_dxf.scan_dxf(p)
            p.write_bytes(b"0\nSECTION\n2\n")
            with self.assertRaisesRegex(ValueError,"malformed"):hamon_dxf.scan_dxf(p)
    def test_truncation_is_loud(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a.dxf";fixture(p)
            p.write_text(p.read_text().replace("0\nENDSEC\n0\nEOF",
              "0\nDIMENSION\n70\n1\n13\n0\n23\n0\n14\n10\n24\n0\n42\n10\n0\nENDSEC\n0\nEOF"))
            d=hamon_dxf.scan_dxf(p,limit=1)
            self.assertEqual(d["total_dimensions"],2)
            self.assertEqual(d["reported_dimensions"],1)
            self.assertTrue(d["dimension_limit_hit"])

if __name__=="__main__":unittest.main()
