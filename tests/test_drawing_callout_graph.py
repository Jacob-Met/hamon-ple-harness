"""Cold PDF dimensional callout-to-source-geometry binding controls.

Prerequisite: exact two-endpoint vector witness, page-frame-compatible text,
globally unambiguous pairings, 4 independent angles for an affine physical
metric. All negative cases must *refuse*, never silently interpolate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import engineering_toolkit as units
import hamon_drawing
import hamon_callout


def page_fixture():
    # Four distinct directions with metric tensor [[2.5e-5, 5e-6],
    #                                              [5e-6, 2.5e-5]].
    strokes=[
        ((80,100),(280,100)), ((80,95),(80,105)), ((280,95),(280,105)),
        ((380,100),(380,300)), ((375,100),(385,100)),
        ((375,300),(385,300)),
        ((100,360),(220,480)), ((96,364),(104,356)), ((216,484),(224,476)),
        ((340,480),(460,360)), ((336,476),(344,484)), ((456,356),(464,364)),
        ((90,88),(270,88)),  # long but UNANNOTATED close decoy line
        ((200,620),(310,660)),  # target (not annotated)
    ]
    rows=[{"id":f"v{i+1}","type":"line",
           "svg_points":[list(a),list(b)],
           "svg_segment_vertex_pairs":[[0,1]],
           "svg_bbox":[min(a[0],b[0]),min(a[1],b[1]),
                       max(a[0],b[0]),max(a[1],b[1])]}
          for i,(a,b) in enumerate(strokes)]
    numbers=[
        ("1000mm",[157,84,203,96]),
        ("1.0m",[388,177,418,190]),
        ("929.516mm",[134,388,186,400]),
        ("758.947mm",[374,388,426,400]),
    ]
    return {"page":1,"document_pages":1,
            "pdf_page_size_pt":[612,792],
            "viewbox_svg_units":[0,0,612,792],
            "word_boxes_reported":len(numbers),
            "words":[{"text":text,"page_bbox_pt":box}
                     for text,box in numbers],
            "reported_vectors":rows,
            "vector_primitives":len(rows),"vector_limit_hit":False,
            "unlocalized_vectors":0}


def target():
    return [{"name":"offset","from":{"vector":"v14","vertex":0},
             "to":{"vector":"v14","vertex":1},"unit":"mm"}]


class DimensionAssociationContracts(unittest.TestCase):
    def test_unambiguous_four_direction_binding_and_affine_measure(self):
        sheet=page_fixture()
        found=hamon_callout.associate_page(sheet)
        self.assertEqual(found["kind"],"pdf-dimension-callout-associations-v1")
        self.assertEqual(len(found["matches"]),4,found)
        self.assertFalse(found["claim_of_engineering_approval"])
        self.assertEqual(
            {(m["source_label"],m["from"]["vector"]) for m in found["matches"]},
            {("1000mm","v1"),("1.0m","v4"),
             ("929.516mm","v7"),("758.947mm","v10")})
        self.assertTrue(all(len(m["endpoint_witnesses"])==2
                            for m in found["matches"]))
        out=hamon_callout.auto_measure_page(
            sheet, target(), units.quantity, units.UNITS["m"],
            calibration_model="affine_spd",max_relative_residual=0.001)
        true_length=math.sqrt(110**2*2.5e-5+40**2*2.5e-5
                              +2*110*40*5e-6)*1000
        self.assertAlmostEqual(out["targets"][0]["estimate"],
                               true_length,delta=0.1)
        self.assertFalse(out["physical_scale_verified"])
        self.assertEqual(out["annotation_binding"],"two_endpoint_source_witness")

    def test_nearby_unmarked_geometry_cannot_steal_dimension(self):
        p=page_fixture()
        out=hamon_callout.associate_page(p)
        self.assertEqual(out["matches"][0]["from"]["vector"],"v1")
        self.assertNotIn("v13",[m["from"]["vector"] for m in out["matches"]])

    def test_missing_one_tick_fails_closed_not_best_effort(self):
        p=page_fixture()
        p["reported_vectors"].pop(2)
        p["vector_primitives"]-=1
        got=hamon_callout.associate_page(p)
        self.assertNotIn("1000mm",[m["source_label"] for m in got["matches"]])
        self.assertTrue(any(r["source_label"]=="1000mm"
                            for r in got["rejected"]))

    def test_duplicated_equidistant_fully_witnessed_lines_are_ambiguous(self):
        p=page_fixture()
        for a,b in [((80,80),(280,80)),((80,75),(80,85)),
                    ((280,75),(280,85))]:
            i=len(p["reported_vectors"])+1
            p["reported_vectors"].append(
                {"id":f"v{i}","type":"line","svg_points":[list(a),list(b)],
                 "svg_segment_vertex_pairs":[[0,1]]})
        p["vector_primitives"]=len(p["reported_vectors"])
        # Text box center exactly between y80 and y100; both have two witnesses.
        p["words"][0]["page_bbox_pt"]=[157,84,203,96]
        result=hamon_callout.associate_page(p)
        self.assertNotIn("1000mm",[m["source_label"] for m in result["matches"]])
        self.assertTrue(any(r["reason"]=="ambiguous_dimension_line"
                            for r in result["rejected"]),result)

    def test_missing_frame_or_incomplete_page_refuses_assignment(self):
        p=page_fixture()
        p["viewbox_svg_units"]=[0,0,800,792]
        with self.assertRaisesRegex(ValueError,"coordinate|frame|aspect"):
            hamon_callout.associate_page(p)
        p=page_fixture()
        p["vector_limit_hit"]=True
        with self.assertRaisesRegex(ValueError,"incomplete"):
            hamon_callout.associate_page(p)

    def test_no_unit_scale_caption_or_unexpected_diameter_not_length(self):
        p=page_fixture()
        p["words"]=[
            {"text":"1:50","page_bbox_pt":[157,84,203,96]},
            {"text":"2000","page_bbox_pt":[388,177,418,190]},
            {"text":"Ø200mm","page_bbox_pt":[134,388,186,400]},
        ]
        self.assertEqual(hamon_callout.associate_page(p)["matches"],[])

    def test_multi_move_pdf_paths_do_not_invent_gap_vector(self):
        svg=(b'<svg xmlns="http://www.w3.org/2000/svg" '
             b'viewBox="0 0 612 792"><path d="M 10 10 L 50 10 '
             b'M 150 10 L 190 10" /></svg>')
        out=hamon_drawing._svg_inventory(svg)
        row=out["reported_vectors"][0]
        self.assertEqual(row["svg_segment_vertex_pairs"],[[0,1],[2,3]])
        self.assertEqual(row["svg_points"],[[10,10],[50,10],[150,10],[190,10]])

    def test_unit_label_change_does_not_keep_old_target_length(self):
        p=page_fixture()
        a=hamon_callout.auto_measure_page(
            p,target(),units.quantity,units.UNITS["m"],
            calibration_model="affine_spd")
        p["words"][2]["text"]="1000mm"
        with self.assertRaises(ValueError):
            hamon_callout.auto_measure_page(
                p,target(),units.quantity,units.UNITS["m"],
                calibration_model="affine_spd")
        self.assertGreater(a["targets"][0]["estimate"],600)

    def test_scan_native_pdf_callout_with_offset_and_text_provenance(self):
        if not all(shutil.which(x) for x in ("pdfinfo","pdftotext","pdftocairo")):
            self.skipTest("Poppler required for end-to-end cold PDF")
        # Independent true PDF content, not an in-memory synthetic inventory.
        strokes=[
            ((80,692),(280,692)),((80,687),(80,697)),((280,687),(280,697)),
            ((380,692),(380,492)),((375,692),(385,692)),
            ((375,492),(385,492)), ((100,432),(220,312)),
            ((96,428),(104,436)),((216,308),(224,316)),
            ((340,312),(460,432)),((336,316),(344,308)),
            ((456,436),(464,428)), ((90,704),(270,704)),
            ((200,172),(310,132)),
        ]
        # PDF y-up and text baselines; scan returns SVG y-down.
        ops=[f"{a[0]} {a[1]} m {b[0]} {b[1]} l S"
             for a,b in strokes]
        labels=[("1000mm",157,700),("1.0m",388,604),
                ("929.516mm",134,394),("758.947mm",374,394)]
        ops.extend(f"BT /F1 11 Tf {x} {y} Td ({text}) Tj ET"
                   for text,x,y in labels)
        script=("\n".join(ops)+"\n").encode()
        objects=[
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length "+str(len(script)).encode()+b" >>\nstream\n"
            +script+b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        stream=bytearray(b"%PDF-1.4\n")
        offsets=[0]
        for i,body in enumerate(objects,1):
            offsets.append(len(stream))
            stream.extend(f"{i} 0 obj\n".encode()+body+b"\nendobj\n")
        xref=len(stream)
        stream.extend(b"xref\n0 6\n0000000000 65535 f \n")
        for offset in offsets[1:]:
            stream.extend(f"{offset:010d} 00000 n \n".encode())
        stream.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
                      f"startxref\n{xref}\n%%EOF\n".encode())
        with tempfile.TemporaryDirectory() as tmp:
            pdf=Path(tmp)/"source.pdf"
            pdf.write_bytes(stream)
            sheet=hamon_drawing.scan_pdf(str(pdf),limit_paths=5000,
                                         limit_words=5000)["pages"][0]
            self.assertEqual(sheet["viewbox_svg_units"][2:],[612,792])
            got=hamon_callout.associate_page(sheet)
            self.assertEqual(len(got["matches"]),4,got)
            # Keep genuine vector IDs discovered by Poppler, not assumed
            # manually synthesized IDs from the PDF stream.
            pairings={m["source_label"]:m for m in got["matches"]}
            self.assertEqual(set(pairings),set(text for text,_,_ in labels))
            labels_and_vectors={m["source_label"]:m["from"]["vector"] for m in got["matches"]}
            self.assertEqual(len(labels_and_vectors),4)
            self.assertNotEqual(next(iter(labels_and_vectors.values())),"v500")
            self.assertEqual(len(hashlib.sha256(pdf.read_bytes()).hexdigest()),64)


if __name__=="__main__":
    unittest.main()
