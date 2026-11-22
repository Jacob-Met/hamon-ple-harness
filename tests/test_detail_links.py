"""Controls for the native-text cross-sheet lookup. Coordinates are Poppler page space."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import hamon_drawing


def sheet(page, lines):
    return [{"page": page, "line": i + 1, "box": [10, i * 15, 140, i * 15 + 12], "text": text}
            for i, text in enumerate(lines)]


class DocumentDetailTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("pdfinfo") and shutil.which("pdftotext"), "requires Poppler")
    def test_native_cli_binds_the_physical_sheet_and_source(self):
        from hamon_drawing_reader import build_minimal_vector_pdf
        pages = [["Sheet No. A1", "DETAIL 3/S5"], ["Sheet No. S4", "DETAIL 3 - Other"],
                 ["Sheet No. S5", "DETAIL 3 - Sill"]]
        data = build_minimal_vector_pdf([{"content_stream": "\n".join(
            f"BT /F1 12 Tf 80 {500 - 30*i} Td ({line}) Tj ET" for i, line in enumerate(lines))}
            for lines in pages])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "drawing.pdf")
            path.write_bytes(data)
            cli = Path(__file__).resolve().parents[1] / "engineering_toolkit.py"
            result = json.loads(subprocess.check_output([sys.executable, cli, "pdf-detail-links", path]))
        self.assertEqual(result["source_sha256"], hashlib.sha256(data).hexdigest())
        ref, = result["references"]
        self.assertEqual((ref["candidate_pages"], ref["status"]), ([3], "resolved_detail"))
        self.assertEqual(ref["detail_evidence"][0]["text"], "DETAIL 3 - Sill")

    def test_identity_and_evidence_across_sheets(self):
        source = sheet(1, ["Sheet No. A1", "DETAILS 3 & 4 ON SHEET S4.", "3/S5", "3/S4",
                           "3/S4", "SEE DETAIL 8/S4", "SEE DETAIL 2/A9"])
        targets = [sheet(2, ["Sheet No. S4", "DETAIL 3 - Roof", "DETAIL 4 - Door", "8"]),
                   sheet(3, ["Sheet No. S5", "DETAIL 3 - Sill"])]
        refs = hamon_drawing._detail_graph([source, *targets])["references"]
        self.assertEqual([r["status"] for r in refs], ["resolved_detail"] * 5 + ["sheet_only", "sheet_not_found"])
        self.assertEqual([r["detail_evidence"][0]["page"] for r in refs[:5]], [2, 2, 3, 2, 2])
        self.assertEqual([r["source"]["line"] for r in refs[3:5]], [4, 5])
        self.assertEqual([r["target_detail"] for r in refs[:2]], ["3", "4"])

    def test_duplicate_page_and_heading_remain_ambiguous(self):
        src = sheet(1, ["SHEET NO. A1", "DETAIL 2/S4"])
        page = sheet(2, ["SHEET NO. S4", "DETAIL 2: DOOR", "DETAIL 2: WINDOW"])
        separate = sheet(3, ["SHEET NO. S4", "DETAIL 2: SILL"])
        ref, = hamon_drawing._detail_graph([src, page])["references"]
        self.assertEqual(ref["status"], "ambiguous_detail")
        ref, = hamon_drawing._detail_graph([src, page, separate])["references"]
        self.assertEqual((ref["status"], ref["candidate_pages"]), ("ambiguous_sheet", [2, 3]))

    def test_stamps_do_not_bind_distant_identifiers_or_other_labels(self):
        source = sheet(1, ["SHEET NO.", "S4", "Detail 1/A1"])
        source[1]["box"] = [800, 300, 820, 320]
        self.assertEqual(hamon_drawing._sheet_stamps(source), {})
        source[1]["box"] = [145, 1, 166, 16]
        self.assertEqual(list(hamon_drawing._sheet_stamps(source)), ["S4"])

    def test_stacked_text_requires_mutually_unique_alignment(self):
        source = sheet(1, ["2", "S5"])
        target = sheet(2, ["Sheet No. S5", "DETAIL 2 - Deck"])
        ref, = hamon_drawing._detail_graph([source, target])["references"]
        self.assertEqual(ref["status"], "candidate_detail")
        self.assertEqual([r["line"] for r in ref["numerator_evidence"]], [1])
        source.append({**source[0], "line": 3, "text": "3"})
        ref, = hamon_drawing._detail_graph([source, target])["references"]
        self.assertEqual(ref["status"], "ambiguous_stacked_source")
        self.assertIsNone(ref["target_detail"])

    def test_caption_needs_one_enclosing_marker_and_touching_title_rule(self):
        number = {"text": "4", "box": [5, 5, 15, 15]}
        title = {"text": "COLUMN PLAN", "box": [23, 0, 75, 11]}
        marker = {"id": "v1", "type": "circle", "svg_bbox": [0, 0, 20, 20],
                  "stroke": "black", "bbox_exact_for_centerline": True}
        rule = {"id": "v2", "type": "line", "svg_bbox": [20, 10, 100, 10],
                "stroke": "black", "bbox_exact_for_centerline": True, "svg_segment_vertex_pairs": [[0, 1]]}
        self.assertIsNotNone(hamon_drawing._caption_geometry([marker, rule], number, title))
        for vectors in ([marker], [rule], [marker, rule, {**rule, "id": "v3"}],
                        [{**marker, "type": "rect"}, rule], [marker, {**rule, "svg_bbox": [30, 10, 100, 10]}]):
            self.assertIsNone(hamon_drawing._caption_geometry(vectors, number, title))

    @unittest.skipUnless(all(shutil.which(p) for p in ("pdfinfo", "pdftotext", "pdftocairo")), "requires Poppler")
    def test_numbered_vector_caption_is_scoped_and_repeated_titles_stay_ambiguous(self):
        from hamon_drawing_reader import build_minimal_vector_pdf
        # Four cubic quadrants form the numbered title marker. A horizontal
        # rule touches its right side and underlines the adjacent printed title.
        caption = """123.3 264 m
        129.10 264 133.8 259.30 133.8 253.5 c
        133.8 247.70 129.10 243 123.3 243 c
        117.50 243 112.8 247.70 112.8 253.5 c
        112.8 259.30 117.50 264 123.3 264 c S
        133.8 253.5 m 280 253.5 l S
        BT /F1 12 Tf 120 250 Td (4) Tj ET
        BT /F1 12 Tf 138 257 Td (COLUMN SECTION) Tj ET"""
        source = "BT /F1 12 Tf 80 500 Td (Sheet No. A1) Tj 0 -25 Td (SEE DETAIL 4/S4) Tj ET"
        target = "BT /F1 12 Tf 80 500 Td (Sheet No. S4) Tj ET\n" + caption
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "views.pdf")
            for repeated in (False, True):
                content = target + ("\nq 1 0 0 1 0 -70 cm\n" + caption + "\nQ" if repeated else "")
                path.write_bytes(build_minimal_vector_pdf([{"content_stream": source}, {"content_stream": content}]))
                graph = hamon_drawing.detail_links(path, "S4")
                ref, = graph["references"]
                self.assertEqual(ref["status"], "ambiguous_detail" if repeated else "resolved_detail")
                self.assertEqual(len(ref["ruled_caption_evidence"]), 2 if repeated else 1)
                for proof in ref["ruled_caption_evidence"]:
                    self.assertEqual((proof["number"]["page"], proof["caption"]["text"]), (2, "COLUMN SECTION"))
                    self.assertEqual(len(proof["vectors"]), 2)
