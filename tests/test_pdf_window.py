"""Native PDF raster windows, independent fiducials, and immutable source files."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import engineering_intake
import hamon_drawing


CASES = [
    ("media", [0, 0, 300, 400], [0, 0, 300, 400], 0, 1),
    ("nonzero", [20, 30, 320, 430], [20, 30, 320, 430], 0, 1),
    ("crop", [0, 0, 300, 400], [30, 40, 260, 330], 0, 1),
    ("rotate90", [0, 0, 300, 400], [0, 0, 300, 400], 90, 1),
    ("rotate180", [0, 0, 300, 400], [0, 0, 300, 400], 180, 1),
    ("rotate270", [0, 0, 300, 400], [0, 0, 300, 400], 270, 1),
    ("crop_rotate90", [20, 30, 320, 430], [30, 40, 260, 330], 90, 1),
    ("fractional", [0.25, 0.5, 300.75, 400.25], [0.25, 0.5, 300.75, 400.25], 0, 1),
    ("userunit2", [0, 0, 300, 400], [0, 0, 300, 400], 0, 2),
]


def make_pdf(path, pages):
    """Literal PDF geometry, independent of the drawing-window implementation."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    page_ids = []
    for media, crop, rotation, user_unit, offset in pages:
        page_id = len(objects) + 1
        page_ids.append(page_id)
        stream = (f"q 1 0 0 rg {60 + offset} 80 30 20 re f "
                  "0 0 1 rg 150 160 20 40 re f Q\n").encode()
        page = ("<< /Type /Page /Parent 2 0 R /MediaBox [%s] /CropBox [%s] "
                "/Rotate %d /UserUnit %s /Resources << >> /Contents %d 0 R >>" %
                (" ".join(map(str, media)), " ".join(map(str, crop)),
                 rotation, user_unit, page_id + 1)).encode()
        objects.extend([page, b"<< /Length " + str(len(stream)).encode() +
                        b" >>\nstream\n" + stream + b"endstream"])
    objects[1] = ("<< /Type /Pages /Kids [%s] /Count %d >>" %
                  (" ".join(f"{n} 0 R" for n in page_ids), len(page_ids))).encode()
    data = bytearray(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend((f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                 f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(data)


def read_rgb(path):
    """Small independent PNG decoder for native 8-bit RGB/RGBA test fixtures."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, compressed = 8, bytearray()
    while pos < len(data):
        size = int.from_bytes(data[pos:pos + 4], "big")
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + size]
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            assert (depth, compression, filtering, interlace) == (8, 0, 0, 0)
            channels = {2: 3, 6: 4}[color]
        elif kind == b"IDAT":
            compressed.extend(body)
        pos += size + 12
    raw = zlib.decompress(compressed)
    stride = width * channels
    assert len(raw) == height * (stride + 1)
    previous, rows = bytearray(stride), []
    for y in range(height):
        mode = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = previous[x]
            upper_left = previous[x - channels] if x >= channels else 0
            if mode == 1:
                predictor = left
            elif mode == 2:
                predictor = up
            elif mode == 3:
                predictor = (left + up) // 2
            elif mode == 4:
                estimate = left + up - upper_left
                distances = [abs(estimate - value) for value in (left, up, upper_left)]
                predictor = (left, up, upper_left)[distances.index(min(distances))]
            else:
                assert mode == 0
                predictor = 0
            row[x] = (row[x] + predictor) & 255
        rows.append(bytes(value for x in range(0, stride, channels) for value in row[x:x + 3]))
        previous = row
    return width, height, rows


def expected_red_box(crop, rotation, offset=0):
    points = []
    for x, y in [(60 + offset, 80), (90 + offset, 80),
                 (60 + offset, 100), (90 + offset, 100)]:
        if rotation == 0:
            point = (x - crop[0], crop[3] - y)
        elif rotation == 90:
            point = (y - crop[1], x - crop[0])
        elif rotation == 180:
            point = (crop[2] - x, y - crop[1])
        else:
            point = (crop[3] - y, crop[2] - x)
        points.append(point)
    return [min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points)]


def red_bounds(width, height, rows):
    bounds = [width, height, -1, -1]
    for y, row in enumerate(rows):
        for x in range(width):
            r, g, b = row[x * 3:x * 3 + 3]
            if r > 245 and g < 10 and b < 10:
                bounds = [min(bounds[0], x), min(bounds[1], y),
                          max(bounds[2], x + 1), max(bounds[3], y + 1)]
    return bounds


class PdfWindowContracts(unittest.TestCase):
    def test_known_fiducials_across_native_page_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, media, crop, rotation, user_unit in CASES:
                for dpi in (72, 300):
                    with self.subTest(case=name, dpi=dpi):
                        pdf, png = root / f"{name}.pdf", root / f"{name}-{dpi}.png"
                        make_pdf(pdf, [(media, crop, rotation, user_unit, 0)])
                        before = pdf.read_bytes()
                        box = expected_red_box(crop, rotation)
                        region = [box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8]
                        result = hamon_drawing.render_window(pdf, png, region=region, dpi=dpi)
                        width, height, pixels = read_rgb(png)
                        bounds = red_bounds(width, height, pixels)
                        left, top = math.floor(region[0] * dpi / 72), math.floor(region[1] * dpi / 72)
                        expected = [box[0] * dpi / 72 - left, box[1] * dpi / 72 - top,
                                    box[2] * dpi / 72 - left, box[3] * dpi / 72 - top]
                        self.assertLessEqual(max(abs(x - y) for x, y in zip(bounds, expected)), 1)
                        self.assertEqual(result["source_pdf_sha256"], hashlib.sha256(before).hexdigest())
                        self.assertEqual(result["image"]["sha256"], hashlib.sha256(png.read_bytes()).hexdigest())
                        self.assertEqual(result["image"]["width"], width)
                        self.assertEqual(result["image"]["height"], height)
                        self.assertEqual(result["coordinate_frame"]["pixel_to_svg"],
                                         [72 / dpi, 0, 0, 72 / dpi, left * 72 / dpi, top * 72 / dpi])
                        self.assertFalse(result["physical_drawing_scale_verified"])
                        self.assertFalse(result["ocr_performed"])
                        self.assertEqual(pdf.read_bytes(), before)
                        inventory = hamon_drawing.scan_pdf(pdf)["pages"][0]
                        for actual, expected_value in zip(inventory["reported_vectors"][0]["svg_bbox"], box):
                            self.assertLessEqual(abs(actual - expected_value) * dpi / 72, 1)

    def test_window_is_exact_crop_of_independent_full_native_render(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf, png = root / "page.pdf", root / "window.png"
            make_pdf(pdf, [([20, 30, 320, 430], [30, 40, 260, 330], 90, 1, 0)])
            region, dpi = [22.5, 15.25, 177.125, 180.75], 144
            result = hamon_drawing.render_window(pdf, png, region=region, dpi=dpi)
            subprocess.run(["pdftocairo", "-png", "-cropbox", "-singlefile", "-r", str(dpi),
                            str(pdf), str(root / "reference")], check=True, capture_output=True, timeout=30)
            full_width, _, full = read_rgb(root / "reference.png")
            width, height, cropped = read_rgb(png)
            left, top = math.floor(region[0] * 2), math.floor(region[1] * 2)
            right, bottom = math.ceil(region[2] * 2), math.ceil(region[3] * 2)
            self.assertEqual((width, height), (right - left, bottom - top))
            self.assertLessEqual(right, full_width)
            self.assertEqual(cropped, [row[left * 3:right * 3] for row in full[top:bottom]])
            self.assertEqual(result["coordinate_frame"]["requested_region_svg"], region)

    def test_selected_sheet_and_existing_output_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "two-pages.pdf"
            page = ([0, 0, 300, 400], [0, 0, 300, 400], 0, 1)
            make_pdf(pdf, [(*page, 0), (*page, 40)])
            one = hamon_drawing.render_window(pdf, root / "one.png", page=1, dpi=72)
            two = hamon_drawing.render_window(pdf, root / "two.png", page=2, dpi=72)
            self.assertEqual(two["document_pages"], 2)
            self.assertEqual(two["page"], 2)
            self.assertNotEqual(one["image"]["sha256"], two["image"]["sha256"])
            before = (root / "one.png").read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                hamon_drawing.render_window(pdf, root / "one.png", page=2, dpi=72)
            self.assertEqual((root / "one.png").read_bytes(), before)
            source = pdf.read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                hamon_drawing.render_window(pdf, pdf)
            self.assertEqual(pdf.read_bytes(), source)

    def test_invalid_requests_fail_without_emitting_an_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf = root / "page.pdf"
            make_pdf(pdf, [([0, 0, 300, 400], [0, 0, 300, 400], 0, 1, 0)])
            invalid = [{"page": 0}, {"page": 2}, {"dpi": 601}, {"dpi": float("nan")},
                       {"region": [0, 0, 0, 1]}, {"region": [0, 0, 301, 400]},
                       {"region": [-1, 0, 30, 30]}, {"region": [0, 0, float("inf"), 40]},
                       {"dpi": 600}]
            for index, arguments in enumerate(invalid):
                with self.subTest(arguments=arguments):
                    output = root / f"invalid-{index}.png"
                    with self.assertRaises(ValueError):
                        hamon_drawing.render_window(pdf, output, **arguments)
                    self.assertFalse(output.exists())

    def test_provisioned_cli_emits_image_and_machine_readable_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            class Session:
                async def write_file(self, path, payload):
                    (root / Path(path).name).write_bytes(payload)
                    return len(payload)
            instructions = asyncio.run(engineering_intake.provision_engineering_cli(Session()))
            self.assertIn("pdf-window", instructions)
            self.assertIn("read", instructions)
            self.assertEqual((root / "hamon_drawing.py").read_bytes(), Path(hamon_drawing.__file__).read_bytes())
            pdf, png = root / "input with spaces.pdf", root / "proof.png"
            make_pdf(pdf, [([0, 0, 300, 400], [0, 0, 300, 400], 0, 1, 0)])
            command = [sys.executable, str(root / "hamon-engineering-toolkit.py"), "pdf-window",
                       str(pdf), "--output", str(png), "--page", "1", "--dpi", "72",
                       "--region", "40", "280", "110", "340"]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            result = json.loads(proc.stdout)
            self.assertTrue(result["ok"])
            self.assertEqual(result["image"]["path"], str(png))
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertLess(len(png.read_bytes()), 8_000_000)


if __name__ == "__main__":
    unittest.main()
