"""Generated two-page native PDF control for Poppler text/vector extraction."""
from pathlib import Path
import shutil
import tempfile
import unittest
import hamon_drawing as d


def fixture(path, page_boxes=b"", parent_boxes=b"", extra_objects=(), first_contents=b"4 0 R"):
    page1=b"q 0.7 w 0 0 0 RG 60 90 340 220 re S Q\nBT /F1 15 Tf 400 90 Td (SCALE 1:50) Tj ET\n"
    page2=b"q 0.8 w 0 0 0 RG 80 100 m 330 100 l 330 310 l S Q\nBT /F1 12 Tf 400 100 Td (SCALE 1:100) Tj ET\n"
    def contents(content):
        return (f"<< /Length {len(content)} >>\nstream\n".encode("ascii")+
                content+b"endstream")
    objects=[
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 " + parent_boxes + b" >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] " + page_boxes
        + b" /Resources << /Font << /F1 7 0 R >> >> /Contents " + first_contents + b" >>",
        contents(page1),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
        contents(page2),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    objects.extend(extra_objects)
    size = len(objects) + 1
    stream=bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets=[0]
    for index,payload in enumerate(objects,start=1):
        offsets.append(len(stream))
        stream.extend(f"{index} 0 obj\n".encode()+payload+b"\nendobj\n")
    xref=len(stream)
    stream.extend(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        stream.extend(f"{offset:010d} 00000 n \n".encode())
    stream.extend(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(stream)


class DrawingContracts(unittest.TestCase):
    def test_straight_svg_path_bounds_and_coplanar_transform(self):
        p=d._path_points("M 0 0 h 10 v 10 h -10 Z")
        self.assertEqual(p,[(0,0),(10,0),(10,10),(0,10),(0,0)])
        self.assertIsNone(d._path_points("M 0 0 C 1 0 2 4 4 5"))
        svg=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><path d="M 0 0 L 100 100"/></defs><g transform="matrix(1,0,0,1,10,20)"><path d="M 0 0 L 10 0 L 10 10 Z"/></g></svg>'
        result=d._svg_inventory(svg)
        self.assertEqual(result["vector_primitives"],1)
        self.assertEqual(result["reported_vectors"][0]["svg_bbox"],[10,20,20,30])
        self.assertEqual(d._svg_inventory(svg,region=[80,80,90,90])["reported_count"],0)

    def test_multisheet_native_poppler_and_scale_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"two_sheets.pdf"
            fixture(path)
            result=d.scan_pdf(str(path),first=1,last=2,limit_paths=100,limit_words=100)
            self.assertEqual(result["document_pages"],2)
            self.assertEqual(len(result["pages"]),2)
            self.assertFalse(result["physical_drawing_scale_verified"])
            first,second=result["pages"]
            self.assertGreater(first["vector_primitives"],0,first)
            self.assertGreater(second["vector_primitives"],0,second)
            self.assertEqual(first["embedded_images"],0)
            self.assertTrue(any("1:50" in a["text"] for a in first["scale_annotations"]),first)
            self.assertTrue(any("1:100" in a["text"] for a in second["scale_annotations"]),second)
            self.assertEqual(first["pdf_page_size_pt"],[612,792])
            self.assertEqual(first["viewbox_svg_units"][2:],[612,792])

    def test_word_bounds_use_rotated_cropbox(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shifted.pdf"
            fixture(path, b"/CropBox [30 40 570 740] /Rotate 90")
            page = d.scan_pdf(str(path))["pages"][0]
            self.assertEqual(page["pdf_page_size_pt"], [700, 540])
            word = next(w for w in page["words"] if w["text"] == "SCALE")
            left, top, right, bottom = word["page_bbox_pt"]
            # Text starts at PDF (400, 90); clockwise 90 maps it to (90-40, 400-30).
            self.assertLessEqual(left, 50)
            self.assertGreaterEqual(right, 50)
            self.assertLessEqual(top, 370)
            self.assertGreaterEqual(bottom, 370)

    def test_lightweight_reader_uses_visible_lower_left_for_text_and_vectors(self):
        from hamon_drawing_reader import read_vector_pdf, query_sheet_region

        # Crop the inherited, indirect rectangle to the MediaBox. The first
        # page overrides the parent's indirect rotation; the second inherits it.
        parent = b"/CropBox 8 0 R /Rotate 9 0 R"
        objects = (b"[30 40 570 810]", b"-90", b"[4 0 R]")
        expected = {
            0: ((540, 752), (370, 50), (30, 50)),
            90: ((752, 540), (50, 170), (50, 510)),
            180: ((540, 752), (170, 702), (510, 702)),
            270: ((752, 540), (702, 370), (702, 30)),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "display.pdf"
            for rotation, (size, anchor, line_start) in expected.items():
                with self.subTest(rotation=rotation):
                    fixture(path, f"/Rotate {rotation}".encode(), parent, objects, first_contents=b"10 0 R")
                    first, second = read_vector_pdf(path.read_bytes())
                    self.assertEqual((first.width_pt, first.height_pt), size)
                    self.assertEqual((second.width_pt, second.height_pt), (752, 540))
                    span = next(w for w in first.texts if "SCALE" in w.text)
                    self.assertEqual((span.x_pt, span.y_pt), anchor)
                    segment = first.segments[0]
                    self.assertEqual((segment.x0_pt, segment.y0_pt), line_start)
                    self.assertEqual((segment.x1_pt, segment.y1_pt), anchor)
                    self.assertEqual(segment.length_pt, 340)
                    x, y = anchor
                    w, h = size
                    actual = query_sheet_region(first, ((x-2)/w, (y-2)/h, (x+2)/w, (y+2)/h))
                    self.assertIn("SCALE 1:50", actual["texts"])

    def test_invalid_page_and_crop_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"source.pdf";fixture(path)
            with self.assertRaisesRegex(ValueError,"outside PDF"):
                d.scan_pdf(str(path),first=3,last=3)
            with self.assertRaisesRegex(ValueError,"region"):
                d.scan_pdf(str(path),region=[0,0,-1,5])
            with self.assertRaisesRegex(ValueError,"20-sheet"):
                d.scan_pdf(str(path),first=1,last=21)


if __name__=="__main__":unittest.main()
