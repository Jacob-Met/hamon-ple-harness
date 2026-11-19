"""Held-out dimensional controls for source-anchored PDF measurement.

A printed caption, invented points, one-axis calibration, inconsistent source
dimensions or missing text must never yield engineering measurement approval.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import engineering_toolkit as tool
import hamon_drawing
import hamon_scale


def endpoint(vector, vertex):
    return {"vector": vector, "vertex": vertex}


def segment(a, b, **other):
    return {"from": a, "to": b, **other}


def inventory():
    return {
        "page": 2,
        "word_boxes_reported": 3,
        "words": [{"text": "1000mm"}, {"text": "1.0m"}, {"text": "NTS"}],
        "vector_limit_hit": False, "unlocalized_vectors": 0,
        "reported_vectors": [
            {"id": "v1", "svg_points": [[0., 0.], [100., 0.]]},
            {"id": "v2", "svg_points": [[0., 0.], [0., 100.]]},
            {"id": "v3", "svg_points": [[20., 20.], [50., 60.]]},
            {"id": "v4", "svg_points": None},
        ],
    }


def specification():
    return {
        "schema": "hamon.pdf.measure.v1",
        "page": 2,
        "references": [
            segment(endpoint("v1", 0), endpoint("v1", 1),
                    length="1000*mm", source_label="1000mm"),
            segment(endpoint("v2", 0), endpoint("v2", 1),
                    length="1*m", source_label="1.0m"),
        ],
        "targets": [
            segment(endpoint("v3", 0), endpoint("v3", 1),
                    name="duct-offset", unit="mm"),
        ],
    }


class AnchoredMeasureContracts(unittest.TestCase):
    def calculate(self, page=None, spec=None):
        return hamon_scale.calibrate_inventory(
            inventory() if page is None else page,
            specification() if spec is None else spec,
            tool.quantity, tool.UNITS["m"])

    def test_nonparallel_consistent_anchors_and_units(self):
        result = self.calculate()
        self.assertFalse(result["physical_scale_verified"])
        self.assertEqual(result["reference_count"], 2)
        self.assertAlmostEqual(result["conditional_scale_m_per_svg_unit"], 0.01)
        self.assertAlmostEqual(result["targets"][0]["estimate"], 500)
        self.assertEqual(result["targets"][0]["status"],
                         "conditional_source_anchored_length")
        self.assertEqual(result["reference_max_relative_residual"], 0.0)

    def test_printed_nts_does_not_invent_or_invalidate_independent_dimensions(self):
        self.assertIn("NTS", inventory()["words"][-1]["text"])
        self.assertFalse(self.calculate()["physical_scale_verified"])

    def test_conflicting_scale_fails_closed(self):
        spec = specification()
        spec["references"][1]["length"] = "2*m"
        with self.assertRaisesRegex(ValueError, "inconsistent dimension"):
            self.calculate(spec=spec)

    def test_same_axis_cannot_validate_page(self):
        page = inventory()
        page["reported_vectors"][1]["svg_points"] = [[0, 20], [100, 20]]
        with self.assertRaisesRegex(ValueError, "nonparallel"):
            self.calculate(page=page)

    def test_model_invented_coordinate_and_unlocalized_curve_rejected(self):
        spec = specification()
        spec["targets"][0]["to"] = {"x": 0, "y": 0}
        with self.assertRaisesRegex(ValueError, "endpoint"):
            self.calculate(spec=spec)
        spec = specification()
        spec["targets"][0]["from"] = endpoint("v4", 0)
        with self.assertRaisesRegex(ValueError, "exact, bounded vertex"):
            self.calculate(spec=spec)

    def test_dimensional_confusion_and_stale_text_rejected(self):
        spec = specification()
        spec["references"][0]["length"] = "1*kN"
        with self.assertRaisesRegex(ValueError, "physical length"):
            self.calculate(spec=spec)
        spec = specification()
        spec["references"][0]["source_label"] = "fictitious"
        with self.assertRaisesRegex(ValueError, "not found"):
            self.calculate(spec=spec)

    def test_partial_or_curved_vector_inventory_fails_closed(self):
        page = inventory()
        page["vector_limit_hit"] = True
        with self.assertRaisesRegex(ValueError, "listing truncated"):
            self.calculate(page=page)
        page = inventory()
        del page["reported_vectors"][0]["id"]
        with self.assertRaisesRegex(ValueError, "identifiers"):
            self.calculate(page=page)

    def test_no_duplicate_labels_or_names(self):
        spec = specification()
        spec["references"][1]["source_label"] = "1000mm"
        with self.assertRaisesRegex(ValueError, "independent"):
            self.calculate(spec=spec)
        spec = specification()
        spec["targets"].append(dict(spec["targets"][0]))
        with self.assertRaisesRegex(ValueError, "unique name"):
            self.calculate(spec=spec)

    def test_svg_source_vertices_survive_affine_transform(self):
        svg = (b'<svg xmlns="http://www.w3.org/2000/svg" '
               b'viewBox="0 0 200 200"><defs><path d="M 0 0 H 100"/></defs>'
               b'<g transform="translate(12,34)">'
               b'<path d="M 0 0 L 100 0"/>'
               b'<line x1="0" y1="0" x2="0" y2="100"/></g></svg>')
        got = hamon_drawing._svg_inventory(svg)
        self.assertEqual(got["vector_primitives"], 2)
        self.assertEqual(got["reported_vectors"][0]["id"], "v1")
        self.assertEqual(got["reported_vectors"][0]["svg_points"],
                         [[12.,34.],[112.,34.]])
        self.assertEqual(got["reported_vectors"][1]["svg_points"],
                         [[12.,34.],[12.,134.]])

    def test_source_anchor_uses_real_pdf_geometry_when_poppler_available(self):
        # Minimal native vector PDF: a real cold Poppler path, not hand-built
        # inventory used as a substitute for source evidence.
        import shutil
        if not all(shutil.which(x) for x in ("pdfinfo","pdftotext","pdftocairo")):
            self.skipTest("Poppler not installed in test environment")
        drawing = (
            b"q 1 w 20 100 m 220 100 l S 20 120 m 20 320 l S Q\n"
            b"BT /F1 12 Tf 400 100 Td (1000mm) Tj ET\n"
            b"BT /F1 12 Tf 400 120 Td (1.0m) Tj ET\n"
        )
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length " + str(len(drawing)).encode() + b" >>\nstream\n"
            + drawing + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, payload in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf.extend(f"{i} 0 obj\n".encode()+payload+b"\nendobj\n")
        end = len(pdf)
        pdf.extend(b"xref\n0 6\n0000000000 65535 f \n")
        for o in offsets[1:]:
            pdf.extend(f"{o:010d} 00000 n \n".encode())
        pdf.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{end}\n%%EOF\n".encode())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"plans.pdf"
            spec_path = Path(tmp)/"anchors.json"
            path.write_bytes(pdf)
            page = hamon_drawing.scan_pdf(str(path))["pages"][0]
            vectors = [x for x in page["reported_vectors"]
                       if x.get("svg_points") and
                       any(math.dist(x["svg_points"][i], x["svg_points"][i+1]) > 50
                           for i in range(len(x["svg_points"])-1))]
            self.assertTrue(vectors, page)
            # Select two verified extracted segments with different directions
            # even when Poppler folds the two lines into one SVG path.
            refs = []
            for v in vectors:
                pts = v["svg_points"]
                for i in range(len(pts)-1):
                    p, q = pts[i:i+2]
                    dx, dy = q[0]-p[0], q[1]-p[1]
                    if abs(dx)>100 and abs(dy)<1:
                        refs.append(segment(endpoint(v["id"],i),endpoint(v["id"],i+1),
                                            length="1000*mm",source_label="1000mm"))
                    if abs(dy)>100 and abs(dx)<1:
                        refs.append(segment(endpoint(v["id"],i),endpoint(v["id"],i+1),
                                            length="1*m",source_label="1.0m"))
            self.assertGreaterEqual(len(refs),2,page)
            spec = {"schema":"hamon.pdf.measure.v1", "page":1,
                    "references":refs[:2],
                    "targets":[segment(refs[0]["from"],refs[0]["to"],
                                       name="baseline",unit="mm")]}
            spec_path.write_text(json.dumps(spec))
            out=hamon_scale.measure_pdf(path,spec_path,tool.quantity,tool.UNITS["m"])
            self.assertEqual(out["pdf_sha256"], __import__("hashlib").sha256(pdf).hexdigest())
            self.assertEqual(out["page"],1)
            self.assertAlmostEqual(out["targets"][0]["estimate"],1000,delta=1)


if __name__ == "__main__":
    unittest.main()
