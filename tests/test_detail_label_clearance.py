"""Use the actual Poppler raster to check a narrow physical lettering/rim gap."""
import pathlib
import shutil
import tempfile
import unittest

import hamon_drawing as drawing


def draw_contact(path, tip, erase, bridge):
    # The number stroke and marker are separate PDF painting operations. The
    # optional final white rectangle erases the printed stroke while preserving
    # all the earlier vector and spelling evidence.
    arc = "15 10 m 15 12.7614 12.7614 15 10 15 c 7.2386 15 5 12.7614 5 10 c 5 7.2386 7.2386 5 10 5 c 12.7614 5 15 7.2386 15 10 c S"
    commands = f"0 G 1 J .4 w {arc} .45 w 10 10 m {tip} 10 l S\n"
    if erase:
        commands += "1 g 12 9 1 2 re f\n"
    if bridge:
        commands += "0 G .3 w 14.5 10 m 15 10 l S\n"
    stream = commands.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 20] /Resources << >> /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    positions = []
    for i, obj in enumerate(objects, 1):
        positions.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in positions:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(data)


@unittest.skipUnless(shutil.which("pdftoppm"), "Poppler is required for physical ink tests")
class LabelClearanceTests(unittest.TestCase):
    def visible(self, tip, erase=False, bridge=False):
        with tempfile.TemporaryDirectory() as folder:
            pdf = pathlib.Path(folder) / "drawing.pdf"
            draw_contact(pdf, tip, erase, bridge)
            mark = {"id": "ring", "svg_pen_diameter": .4}
            line = {"id": "glyph", "svg_pen_diameter": .45,
                    "svg_points": [(10,10),(tip,10)], "svg_segment_vertex_pairs": [(0,1)]}
            region = {"ring": {"id": "ring", "box": [5,5,15,15]}}
            self.assertFalse(drawing._visible_label_at_scale(pdf, 1, region["ring"]["box"], [line], 8))
            return drawing._visible_complete_label(pdf, 1, region, [(None, "glyph")], [mark,line])

    def test_subpixel_gap_recovers_a_complete_separate_stroke(self):
        self.assertTrue(self.visible(14.53))

    def test_erasure_and_true_contact_remain_unqualified(self):
        self.assertFalse(self.visible(14.53, erase=True))
        self.assertFalse(self.visible(14.7))

    def test_unmodeled_printed_bridge_vetoes_nominal_gap(self):
        self.assertFalse(self.visible(14.53, bridge=True))
