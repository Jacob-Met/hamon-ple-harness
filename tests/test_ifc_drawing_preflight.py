"""Cold unit + Wave-3 holdout verification suite for 3D IFC4/STL/OBJ clash,
multi-sheet vector PDF drawing reader, and PLE deliverable preflight.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    from ship.competitions.ple.ifc_spatial_clash import (
        detect_ifc_clashes,
        evaluate_pair_clash,
        load_ifc_solids,
        parse_obj_mesh,
        parse_stl_mesh,
    )
    from ship.competitions.ple.drawing_reader import (
        PT_TO_M,
        build_minimal_vector_pdf,
        check_deliverable_preflight,
        parse_drawing_scale,
        query_sheet_region,
        read_vector_pdf,
        sanitize_scalar_value,
    )
except ImportError:
    from hamon_ifc_spatial_clash import (
        detect_ifc_clashes,
        evaluate_pair_clash,
        load_ifc_solids,
        parse_obj_mesh,
        parse_stl_mesh,
    )
    from hamon_drawing_reader import (
        PT_TO_M,
        build_minimal_vector_pdf,
        check_deliverable_preflight,
        parse_drawing_scale,
        query_sheet_region,
        read_vector_pdf,
        sanitize_scalar_value,
    )


def _clean_ifc_beam_duct(duct_z_mm: float = 3000.0, unit_prefix: str = ".MILLI.") -> str:
    return f"""ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1 = IFCSIUNIT(*,.LENGTHUNIT.,{unit_prefix},.METRE.);
#2 = IFCUNITASSIGNMENT((#1));
#10 = IFCCARTESIANPOINT((0.,0.,0.));
#11 = IFCDIRECTION((0.,0.,1.));
#12 = IFCDIRECTION((1.,0.,0.));
#13 = IFCAXIS2PLACEMENT3D(#10,#11,#12);
#14 = IFCLOCALPLACEMENT($,#13);
#20 = IFCCARTESIANPOINT((0.,0.,3000.));
#21 = IFCAXIS2PLACEMENT3D(#20,#11,#12);
#22 = IFCLOCALPLACEMENT(#14,#21);
#30 = IFCCARTESIANPOINT((-2000.,0.,0.));
#31 = IFCDIRECTION((1.,0.,0.));
#32 = IFCDIRECTION((0.,1.,0.));
#33 = IFCAXIS2PLACEMENT3D(#30,#31,#32);
#34 = IFCLOCALPLACEMENT(#22,#33);
#35 = IFCISHAPEPROFILEDEF(.AREA.,'W310x97',$,300.,310.,10.,16.);
#36 = IFCEXTRUDEDAREASOLID(#35,#13,#11,4000.);
#37 = IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#36));
#38 = IFCPRODUCTDEFINITIONSHAPE($,$,(#37));
#39 = IFCBEAM('BEAM_W310_01',$,'Main Steel Girder',$,$,#34,#38,$,$);
#50 = IFCCARTESIANPOINT((0.,-1500.,{duct_z_mm:.3f}));
#51 = IFCDIRECTION((0.,1.,0.));
#52 = IFCDIRECTION((1.,0.,0.));
#53 = IFCAXIS2PLACEMENT3D(#50,#51,#52);
#54 = IFCLOCALPLACEMENT(#14,#53);
#55 = IFCRECTANGLEPROFILEDEF(.AREA.,'SupplyDuct_400x250',$,400.,250.);
#56 = IFCEXTRUDEDAREASOLID(#55,#13,#11,3000.);
#57 = IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#56));
#58 = IFCPRODUCTDEFINITIONSHAPE($,$,(#57));
#59 = IFCDUCTSEGMENT('DUCT_SUP_01',$,'HVAC Supply Duct',$,$,#54,#58,$);
ENDSEC;
END-ISO-10303-21;"""


class TestIfcSpatialClashAndDrawingPreflight(unittest.TestCase):
    def test_01_ifc_ishape_and_duct_exact_volume_and_unit_scale(self):
        solids = load_ifc_solids(_clean_ifc_beam_duct(duct_z_mm=3000.0))
        self.assertEqual(len(solids), 2)
        by_id = {s.element_id: s for s in solids}
        self.assertAlmostEqual(by_id["BEAM_W310_01"].volume_m3(), 0.04952, places=6)
        self.assertAlmostEqual(by_id["DUCT_SUP_01"].volume_m3(), 0.30000, places=6)

    def test_02_ifc_hard_clash_detection_beam_vs_duct(self):
        rep = detect_ifc_clashes(_clean_ifc_beam_duct(duct_z_mm=3000.0), required_clearance_m=0.05, cross_discipline_only=True)
        self.assertEqual(rep["hard_clash_count"], 1)
        self.assertEqual(rep["min_clearance_m"], 0.0)

    def test_cross_discipline_filter_uses_declared_distribution_element(self):
        base = _clean_ifc_beam_duct().replace("IFCBEAM(", "IFCBUILDINGELEMENTPROXY(")
        base = base.replace("Main Steel Girder", "Panel A").replace("HVAC Supply Duct", "Run 1")
        for kind in ("IFCCABLECARRIERSEGMENT", "IFCFIRESUPPRESSIONTERMINAL",
                     "IFCELECTRICDISTRIBUTIONBOARD", "IFCUNITARYEQUIPMENT", "IFCSANITARYTERMINAL"):
            with self.subTest(kind=kind):
                text = base.replace("IFCDUCTSEGMENT(", kind + "(")
                result = detect_ifc_clashes(text, cross_discipline_only=True)
                self.assertEqual((result["pair_count"], result["hard_clash_count"]), (1, 1))

        # A cooled beam and a fan are both MEP, even with a structural word in the name.
        both_mep = base.replace("IFCBUILDINGELEMENTPROXY(", "IFCCOOLEDBEAM(")
        both_mep = both_mep.replace("Panel A", "Cooled beam").replace("IFCDUCTSEGMENT(", "IFCFAN(")
        self.assertEqual(detect_ifc_clashes(both_mep, cross_discipline_only=True)["pair_count"], 0)
        self.assertEqual(detect_ifc_clashes(both_mep, cross_discipline_only=False)["hard_clash_count"], 1)

    def test_03_ifc_exact_vertical_clearance_violation_and_clear(self):
        rep_viol = detect_ifc_clashes(_clean_ifc_beam_duct(duct_z_mm=3320.0), required_clearance_m=0.050, cross_discipline_only=True)
        self.assertEqual(rep_viol["hard_clash_count"], 0)
        self.assertEqual(rep_viol["clearance_violation_count"], 1)
        self.assertAlmostEqual(rep_viol["min_clearance_m"], 0.040, places=5)

        rep_ok = detect_ifc_clashes(_clean_ifc_beam_duct(duct_z_mm=3380.0), required_clearance_m=0.050, cross_discipline_only=True)
        self.assertEqual(rep_ok["hard_clash_count"], 0)
        self.assertEqual(rep_ok["clearance_violation_count"], 0)
        self.assertAlmostEqual(rep_ok["min_clearance_m"], 0.100, places=5)

    def test_04_ifc_hollow_rectangular_section(self):
        ifc = """ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1 = IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#2 = IFCUNITASSIGNMENT((#1));
#10 = IFCCARTESIANPOINT((0.,0.,0.));
#11 = IFCDIRECTION((0.,0.,1.));
#12 = IFCDIRECTION((1.,0.,0.));
#13 = IFCAXIS2PLACEMENT3D(#10,#11,#12);
#14 = IFCLOCALPLACEMENT($,#13);
#20 = IFCRECTANGLEHOLLOWPROFILEDEF(.AREA.,'HSS200x200x10',$,0.20,0.20,0.01);
#21 = IFCEXTRUDEDAREASOLID(#20,#13,#11,5.0);
#22 = IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#21));
#23 = IFCPRODUCTDEFINITIONSHAPE($,$,(#22));
#24 = IFCCOLUMN('COL_HSS_1',$,'Hollow Steel Column',$,$,#14,#23,$,$);
ENDSEC;
END-ISO-10303-21;"""
        solids = load_ifc_solids(ifc)
        self.assertEqual(len(solids), 1)
        self.assertAlmostEqual(solids[0].volume_m3(), 0.038, places=6)

    def test_05_ifc_imperial_inch_units(self):
        ifc = """ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC2X3'));ENDSEC;
DATA;
#1 = IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#2 = IFCMEASUREWITHUNIT(IFCLENGTHMEASURE(0.0254),#1);
#3 = IFCCONVERSIONBASEDUNIT(#4,.LENGTHUNIT.,'INCH',#2);
#5 = IFCUNITASSIGNMENT((#3));
#10 = IFCCARTESIANPOINT((0.,0.,0.));
#11 = IFCDIRECTION((0.,0.,1.));
#12 = IFCDIRECTION((1.,0.,0.));
#13 = IFCAXIS2PLACEMENT3D(#10,#11,#12);
#14 = IFCLOCALPLACEMENT($,#13);
#20 = IFCRECTANGLEPROFILEDEF(.AREA.,'Pad',$,12.,24.);
#21 = IFCEXTRUDEDAREASOLID(#20,#13,#11,36.);
#22 = IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#21));
#23 = IFCPRODUCTDEFINITIONSHAPE($,$,(#22));
#24 = IFCFOOTING('FTG_IMP_01',$,'Imperial Footing',$,$,#14,#23,$);
ENDSEC;
END-ISO-10303-21;"""
        solids = load_ifc_solids(ifc)
        self.assertAlmostEqual(solids[0].volume_m3(), 0.3048 * 0.6096 * 0.9144, places=6)

    def test_06_stl_binary_and_obj_containment_clash(self):
        verts = [(0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2)]
        tris = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(2,3,7),(2,7,6),(1,2,6),(1,6,5),(3,0,4),(3,4,7)]
        buf = bytearray(b"PLE_BINARY_STL".ljust(80, b" ") + struct.pack("<I", len(tris)))
        for (i0, i1, i2) in tris:
            buf.extend(struct.pack("<12fH", 0, 0, 0, *verts[i0], *verts[i1], *verts[i2], 0))
        outer_cube = parse_stl_mesh(bytes(buf), element_id="outer_cube")
        self.assertAlmostEqual(outer_cube.volume_m3(), 8.0, places=5)
        self.assertAlmostEqual(outer_cube.surface_area_m2(), 24.0, places=5)

        obj_txt = (
            "v 0.5 0.5 0.5\nv 1.5 0.5 0.5\nv 1.5 1.5 0.5\nv 0.5 1.5 0.5\n"
            "v 0.5 0.5 1.5\nv 1.5 0.5 1.5\nv 1.5 1.5 1.5\nv 0.5 1.5 1.5\n"
            "f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 3 4 8 7\nf 2 3 7 6\nf 4 1 5 8\n"
        )
        inner_cube = parse_obj_mesh(obj_txt, element_id="inner_cube")
        self.assertAlmostEqual(inner_cube.volume_m3(), 1.0, places=6)
        res = evaluate_pair_clash(outer_cube, inner_cube)
        self.assertEqual(res.status, "hard_clash")
        self.assertEqual(res.min_distance_m, 0.0)

    def test_07_drawing_scale_parser_metric_and_imperial(self):
        self.assertAlmostEqual(parse_drawing_scale("SCALE 1:50"), 50.0)
        self.assertAlmostEqual(parse_drawing_scale("Scale: 1 : 200"), 200.0)
        self.assertAlmostEqual(parse_drawing_scale("DETAIL 2:1"), 0.5)
        self.assertAlmostEqual(parse_drawing_scale("""1/4" = 1'-0" """), 48.0)
        self.assertAlmostEqual(parse_drawing_scale("""SHEET A-201 1/4" = 1'-0" """), 48.0)
        self.assertAlmostEqual(parse_drawing_scale("""3/16" = 1'-0" """), 64.0)
        self.assertAlmostEqual(parse_drawing_scale("""1 1/2" = 1'-0" """), 8.0)
        self.assertAlmostEqual(parse_drawing_scale("""1" = 20' """), 240.0)
        self.assertAlmostEqual(parse_drawing_scale("1 cm = 0.5 m"), 50.0)

    def test_08_multi_sheet_flatedecode_pdf_reader_and_crop_measurement(self):
        len_pt_s0 = 0.100 / PT_TO_M
        s0_stream = (
            "q 1 0 0 1 100 300 cm "
            f"0 0 m {len_pt_s0:.6f} 0 l S Q "
            "BT /F1 10 Tf 650 50 Td (SHEET S-101 | SCALE 1:50) Tj ET"
        )
        s1_stream = (
            "q 200 400 144 72 re S Q "
            """BT /F1 10 Tf 620 45 Td [(SHEET A-201 ) -20 (1/4" = 1'-0")] TJ ET"""
        )
        pdf_bytes = build_minimal_vector_pdf([
            {"width_pt": 841.89, "height_pt": 595.28, "content_stream": s0_stream},
            {"width_pt": 864.00, "height_pt": 648.00, "content_stream": s1_stream},
        ], compress=True)
        sheets = read_vector_pdf(pdf_bytes)
        self.assertEqual(len(sheets), 2)
        self.assertAlmostEqual(sheets[0].detected_scale_ratio, 50.0)
        self.assertAlmostEqual(sheets[1].detected_scale_ratio, 48.0)

        q0 = query_sheet_region(sheets[0], norm_bbox=(0.05, 0.35, 0.70, 0.75))
        self.assertEqual(q0["segment_count"], 1)
        self.assertAlmostEqual(q0["max_real_span_m"], 5.0, places=5)

        q1 = query_sheet_region(sheets[1], norm_bbox=(0.15, 0.50, 0.55, 0.85))
        self.assertEqual(q1["segment_count"], 4)
        self.assertAlmostEqual(q1["max_real_span_m"], 2.4384, places=5)
        self.assertAlmostEqual(q1["total_real_length_m"], 7.3152, places=5)

    def test_09_deliverable_preflight_valid_and_rejection_gates(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            good_zip = td_p / "good_submission.zip"
            with zipfile.ZipFile(good_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("agent/runner.py", "def solve(task): return 42.0\n")
                zf.writestr("predictions.json", json.dumps({"ple_001": {"answer": 12.5}, "ple_002": -3.25}))
            rep_ok = check_deliverable_preflight(good_zip, expected_task_ids=["ple_001", "ple_002"])
            self.assertTrue(rep_ok["ok"], msg=rep_ok)
            self.assertEqual(rep_ok["scalar_payloads_checked"], 1)

            bad_zip = td_p / "bad_submission.zip"
            with zipfile.ZipFile(bad_zip, "w") as zf:
                zf.writestr("../escape.txt", "bad")
                zf.writestr("predictions.json", '{"ple_001": "12.5 kN", "ple_002": NaN}')
                sym = zipfile.ZipInfo("link.py")
                sym.external_attr = (0o120777 << 16)
                zf.writestr(sym, "/etc/passwd")
            rep_bad = check_deliverable_preflight(bad_zip, expected_task_ids=["ple_001", "ple_002", "ple_003"])
            self.assertFalse(rep_bad["ok"])
            joined_err = " | ".join(rep_bad["errors"])
            self.assertIn("Unsafe archive member path", joined_err)
            self.assertIn("Symlink prohibited", joined_err)
            self.assertIn("Non-standard JSON constant", joined_err)

            with self.assertRaises(ValueError):
                sanitize_scalar_value("42.0 MPa")
            with self.assertRaises(ValueError):
                sanitize_scalar_value(float("inf"))
            self.assertAlmostEqual(sanitize_scalar_value(" -1.25e2 "), -125.0)


if __name__ == "__main__":
    unittest.main()
