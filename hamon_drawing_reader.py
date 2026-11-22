"""Multi-sheet vector PDF content-stream reader, engineering scale parser,
and PLE competition deliverable pre-flight validator.
"""
from __future__ import annotations

import io
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hamon_preflight import (
    MAX_PLE_BYTES,
    MAX_PLE_FILES,
    _validate_scalar_json_text,
    check_bundle_preflight as check_deliverable_preflight,
    sanitize_scalar_value,
)


PT_TO_MM = 25.4 / 72.0
PT_TO_M = 0.0254 / 72.0


@dataclass
class PdfTextSpan:
    page_index: int
    text: str
    x_pt: float
    y_pt: float


@dataclass
class PdfVectorSegment:
    page_index: int
    x0_pt: float
    y0_pt: float
    x1_pt: float
    y1_pt: float

    @property
    def length_pt(self) -> float:
        return math.hypot(self.x1_pt - self.x0_pt, self.y1_pt - self.y0_pt)

    @property
    def length_mm_paper(self) -> float:
        return self.length_pt * PT_TO_MM


@dataclass
class PdfSheetPage:
    page_index: int
    width_pt: float
    height_pt: float
    texts: List[PdfTextSpan] = field(default_factory=list)
    segments: List[PdfVectorSegment] = field(default_factory=list)
    detected_scale_ratio: Optional[float] = None


def parse_drawing_scale(text: str) -> Optional[float]:
    """Return real_length / paper_length scale ratio (e.g. '1:50' -> 50.0, '1/4" = 1\'-0"' -> 48.0)."""
    s = text.strip()
    m_arch = re.search(
        r'(?<![A-Za-z0-9-])(?:([1-9]|1[01])\s+)?(\d+)\s*/\s*(\d+)\s*(?:"|in\b)\s*=\s*(\d+)\s*(?:\'|ft\b)(?:\s*-?\s*(\d+(?:\.\d+)?)\s*(?:"|in\b))?',
        s,
        flags=re.IGNORECASE,
    )
    if m_arch:
        whole = float(m_arch.group(1) or 0.0)
        num = float(m_arch.group(2))
        den = float(m_arch.group(3))
        paper_in = whole + (num / den if den else 0.0)
        real_ft = float(m_arch.group(4))
        real_extra_in = float(m_arch.group(5) or 0.0)
        real_in = real_ft * 12.0 + real_extra_in
        if paper_in > 0:
            return real_in / paper_in

    m_civ = re.search(
        r'(?<![A-Za-z0-9/-])(\d+(?:\.\d+)?)\s*(?:"|in\b)\s*=\s*(\d+(?:\.\d+)?)\s*(?:\'|ft\b)',
        s,
        flags=re.IGNORECASE,
    )
    if m_civ:
        paper_in = float(m_civ.group(1))
        real_in = float(m_civ.group(2)) * 12.0
        if paper_in > 0:
            return real_in / paper_in

    unit_mm = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    m_unit = re.search(
        r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\b\s*=\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m_unit:
        p_val = float(m_unit.group(1)) * unit_mm[m_unit.group(2).lower()]
        r_val = float(m_unit.group(3)) * unit_mm[m_unit.group(4).lower()]
        if p_val > 0:
            return r_val / p_val

    m_rat = re.search(r"(?:SCALE\s*[:=]?\s*)?\b(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b", s, flags=re.IGNORECASE)
    if m_rat:
        lhs = float(m_rat.group(1))
        rhs = float(m_rat.group(2))
        if lhs > 0 and rhs > 0:
            return rhs / lhs

    return None


def _mat_mul_pdf(m1: Tuple[float, ...], m2: Tuple[float, ...]) -> Tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _apply_pdf_ctm(x: float, y: float, m: Tuple[float, float, float, float, float, float]) -> Tuple[float, float]:
    return (x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5])


def _clip_segment(segment: PdfVectorSegment, box: Tuple[float, ...]):
    """Clip a centerline in form space before the form's affine placement."""
    x0, y0, x1, y1 = segment.x0_pt, segment.y0_pt, segment.x1_pt, segment.y1_pt
    dx, dy = x1 - x0, y1 - y0
    enter, leave = 0.0, 1.0
    for delta, margin in ((-dx, x0-box[0]), (dx, box[2]-x0),
                          (-dy, y0-box[1]), (dy, box[3]-y0)):
        if delta == 0:
            if margin < 0:
                return None
        elif delta < 0:
            enter = max(enter, margin / delta)
        else:
            leave = min(leave, margin / delta)
    if enter >= leave:
        return None
    return (x0 + enter*dx, y0 + enter*dy, x0 + leave*dx, y0 + leave*dy)


def _interpret_pdf_stream(
    stream, resources, pdf, page_index: int, ctm: Tuple[float, ...], ancestors=(),
) -> Tuple[List[PdfTextSpan], List[PdfVectorSegment]]:
    from pypdf.generic import ByteStringObject, ContentStream, TextStringObject

    def text_operand(value):
        if isinstance(value, TextStringObject):
            return str(value)
        if isinstance(value, ByteStringObject):
            return bytes(value).decode("latin-1")
        return ""

    stack: List[Tuple[float, float, float, float, float, float]] = []
    tm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    tlm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    leading = 12.0

    texts: List[PdfTextSpan] = []
    segments: List[PdfVectorSegment] = []
    pending_segs: List[PdfVectorSegment] = []
    cur_pt: Optional[Tuple[float, float]] = None
    subpath_start: Optional[Tuple[float, float]] = None

    paint_ops = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
    for operands, operator in ContentStream(stream, pdf).operations:
        op = operator.decode("ascii")
        if op == "q":
            stack.append(ctm)
        elif op == "Q":
            if stack:
                ctm = stack.pop()
        elif op == "cm" and len(operands) >= 6:
            vals = tuple(float(x) for x in operands[-6:])
            ctm = _mat_mul_pdf(vals, ctm)
        elif op == "Do" and operands:
            xobjects = resources.get("/XObject")
            if xobjects is None:
                continue
            form_ref = xobjects.get_object().get(operands[-1])
            if form_ref is None or form_ref in ancestors:
                continue
            form = form_ref.get_object()
            if form.get("/Subtype") != "/Form":
                continue
            matrix = form["/Matrix"] if "/Matrix" in form else (1, 0, 0, 1, 0, 0)
            form_matrix = tuple(map(float, matrix))
            nested_resources = form.get("/Resources")
            nested_text, nested_segments = _interpret_pdf_stream(
                form, nested_resources.get_object() if nested_resources is not None else resources,
                pdf, page_index, (1, 0, 0, 1, 0, 0), (*ancestors, form_ref),
            )
            box = tuple(map(float, form["/BBox"]))
            placement = _mat_mul_pdf(form_matrix, ctm)
            for text in nested_text:
                # The API reports text origins, not every painted glyph pixel.
                if box[0] <= text.x_pt <= box[2] and box[1] <= text.y_pt <= box[3]:
                    x, y = _apply_pdf_ctm(text.x_pt, text.y_pt, placement)
                    texts.append(PdfTextSpan(page_index, text.text, x, y))
            for segment in nested_segments:
                clipped = _clip_segment(segment, box)
                if clipped is not None:
                    a = _apply_pdf_ctm(*clipped[:2], placement)
                    b = _apply_pdf_ctm(*clipped[2:], placement)
                    segments.append(PdfVectorSegment(page_index, *a, *b))
        elif op == "m" and len(operands) >= 2:
            x, y = _apply_pdf_ctm(float(operands[-2]), float(operands[-1]), ctm)
            cur_pt = (x, y)
            subpath_start = (x, y)
        elif op == "l" and len(operands) >= 2 and cur_pt is not None:
            x, y = _apply_pdf_ctm(float(operands[-2]), float(operands[-1]), ctm)
            pending_segs.append(PdfVectorSegment(page_index, cur_pt[0], cur_pt[1], x, y))
            cur_pt = (x, y)
        elif op == "c" and len(operands) >= 6 and cur_pt is not None:
            x, y = _apply_pdf_ctm(float(operands[-2]), float(operands[-1]), ctm)
            pending_segs.append(PdfVectorSegment(page_index, cur_pt[0], cur_pt[1], x, y))
            cur_pt = (x, y)
        elif op == "re" and len(operands) >= 4:
            rx, ry, rw, rh = (float(operands[-4]), float(operands[-3]), float(operands[-2]), float(operands[-1]))
            p0 = _apply_pdf_ctm(rx, ry, ctm)
            p1 = _apply_pdf_ctm(rx + rw, ry, ctm)
            p2 = _apply_pdf_ctm(rx + rw, ry + rh, ctm)
            p3 = _apply_pdf_ctm(rx, ry + rh, ctm)
            pending_segs.extend([
                PdfVectorSegment(page_index, p0[0], p0[1], p1[0], p1[1]),
                PdfVectorSegment(page_index, p1[0], p1[1], p2[0], p2[1]),
                PdfVectorSegment(page_index, p2[0], p2[1], p3[0], p3[1]),
                PdfVectorSegment(page_index, p3[0], p3[1], p0[0], p0[1]),
            ])
            cur_pt = p0
            subpath_start = p0
        elif op == "h":
            if cur_pt is not None and subpath_start is not None and math.hypot(cur_pt[0] - subpath_start[0], cur_pt[1] - subpath_start[1]) > 1e-6:
                pending_segs.append(PdfVectorSegment(page_index, cur_pt[0], cur_pt[1], subpath_start[0], subpath_start[1]))
                cur_pt = subpath_start
        elif op in paint_ops:
            if op in ("s", "b", "b*") and cur_pt is not None and subpath_start is not None:
                if math.hypot(cur_pt[0] - subpath_start[0], cur_pt[1] - subpath_start[1]) > 1e-6:
                    pending_segs.append(PdfVectorSegment(page_index, cur_pt[0], cur_pt[1], subpath_start[0], subpath_start[1]))
            segments.extend(s for s in pending_segs if s.length_pt > 1e-6)
            pending_segs.clear()
            cur_pt = None
            subpath_start = None
        elif op == "n":
            pending_segs.clear()
            cur_pt = None
            subpath_start = None
        elif op == "BT":
            tm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            tlm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        elif op == "TL" and len(operands) >= 1:
            leading = float(operands[-1])
        elif op == "Tm" and len(operands) >= 6:
            tm = tuple(float(x) for x in operands[-6:])
            tlm = tm
        elif op in ("Td", "TD") and len(operands) >= 2:
            tx, ty = float(operands[-2]), float(operands[-1])
            if op == "TD":
                leading = -ty
            tlm = _mat_mul_pdf((1.0, 0.0, 0.0, 1.0, tx, ty), tlm)
            tm = tlm
        elif op == "T*":
            tlm = _mat_mul_pdf((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm)
            tm = tlm
        elif op == "Tj" and len(operands) >= 1:
            last = operands[-1]
            if text_operand(last):
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, text_operand(last), px, py))
        elif op == "'" and len(operands) >= 1:
            tlm = _mat_mul_pdf((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm)
            tm = tlm
            last = operands[-1]
            if text_operand(last):
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, text_operand(last), px, py))
        elif op == "TJ" and len(operands) >= 1 and isinstance(operands[-1], list):
            parts = [text_operand(item) for item in operands[-1]]
            joined = "".join(parts)
            if joined:
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, joined, px, py))

    return texts, segments


def read_vector_pdf(pdf_bytes: bytes, page_index: Optional[int] = None) -> List[PdfSheetPage]:
    from pypdf import PdfReader

    return _read_pdf_pages(PdfReader(io.BytesIO(pdf_bytes)), page_index)


def _read_pdf_pages(pdf, page_index: Optional[int]) -> List[PdfSheetPage]:
    if page_index is not None and (type(page_index) is not int or not 0 <= page_index < len(pdf.pages)):
        raise ValueError("page_index outside PDF")
    sheets: List[PdfSheetPage] = []
    page_indexes = range(len(pdf.pages)) if page_index is None else (page_index,)
    for p_idx in page_indexes:
        page = pdf.pages[p_idx]
        media, crop = (tuple(map(float, box)) for box in (page.mediabox, page.cropbox))
        x0, y0 = max(media[0], crop[0]), max(media[1], crop[1])
        x1, y1 = min(media[2], crop[2]), min(media[3], crop[3])
        rotation = page.rotation % 360
        if x0 >= x1 or y0 >= y1 or rotation not in (0, 90, 180, 270):
            raise ValueError("Invalid PDF display box or page rotation")
        width_pt, height_pt = x1 - x0, y1 - y0
        if rotation in (90, 270):
            width_pt, height_pt = height_pt, width_pt
        # Display coordinates keep the reader's lower-left origin. Install this
        # as the initial CTM so content, text and path matrices compose normally.
        page_ctm = {
            0: (1, 0, 0, 1, -x0, -y0),
            90: (0, -1, 1, 0, -y0, x1),
            180: (-1, 0, 0, -1, x1, y1),
            270: (0, 1, -1, 0, y1, -x0),
        }[rotation]

        resources = page.get("/Resources")
        texts, segs = _interpret_pdf_stream(
            page.get_contents(), resources.get_object() if resources is not None else {},
            pdf, p_idx, page_ctm,
        )

        detected_scale: Optional[float] = None
        tb_texts = [t for t in texts if t.x_pt >= 0.55 * width_pt and t.y_pt <= 0.35 * height_pt]
        for candidate_pool in (tb_texts, texts):
            for t in candidate_pool:
                if "SCALE" in t.text.upper() or ":" in t.text or "=" in t.text:
                    sc = parse_drawing_scale(t.text)
                    if sc is not None:
                        detected_scale = sc
                        break
            if detected_scale is not None:
                break

        sheets.append(PdfSheetPage(
            page_index=p_idx,
            width_pt=width_pt,
            height_pt=height_pt,
            texts=texts,
            segments=segs,
            detected_scale_ratio=detected_scale,
        ))
    return sheets


def query_sheet_region(
    sheet: PdfSheetPage,
    norm_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    scale_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Query normalized coordinates measured from the displayed CropBox's lower-left."""
    u0, v0, u1, v1 = norm_bbox
    bx0, by0 = u0 * sheet.width_pt, v0 * sheet.height_pt
    bx1, by1 = u1 * sheet.width_pt, v1 * sheet.height_pt
    scale = scale_override if scale_override is not None else (sheet.detected_scale_ratio or 1.0)

    in_texts = [
        t for t in sheet.texts
        if bx0 - 1e-6 <= t.x_pt <= bx1 + 1e-6 and by0 - 1e-6 <= t.y_pt <= by1 + 1e-6
    ]
    in_segs = [
        s for s in sheet.segments
        if (bx0 - 1e-6 <= s.x0_pt <= bx1 + 1e-6 and by0 - 1e-6 <= s.y0_pt <= by1 + 1e-6
            and bx0 - 1e-6 <= s.x1_pt <= bx1 + 1e-6 and by0 - 1e-6 <= s.y1_pt <= by1 + 1e-6)
    ]
    real_lengths_m = [s.length_pt * PT_TO_M * scale for s in in_segs]
    return {
        "page_index": sheet.page_index,
        "scale_ratio": scale,
        "texts": [t.text for t in in_texts],
        "segment_count": len(in_segs),
        "total_real_length_m": float(sum(real_lengths_m)),
        "max_real_span_m": float(max(real_lengths_m, default=0.0)),
        "real_lengths_m": real_lengths_m,
    }


def build_minimal_vector_pdf(pages_spec: Sequence[Dict[str, Any]], compress: bool = True) -> bytes:
    objects: List[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_placeholder_id = add_obj(b"")
    page_ids: List[int] = []

    for spec in pages_spec:
        w = float(spec.get("width_pt", 841.89))
        h = float(spec.get("height_pt", 595.28))
        stream_str: str = spec["content_stream"]
        raw_stream = stream_str.encode("latin-1")
        if compress:
            comp = zlib.compress(raw_stream)
            c_obj = f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode("latin-1") + comp + b"\nendstream"
        else:
            c_obj = f"<< /Length {len(raw_stream)} >>\nstream\n".encode("latin-1") + raw_stream + b"\nendstream"
        cid = add_obj(c_obj)
        p_obj = (
            f"<< /Type /Page /Parent {pages_placeholder_id} 0 R "
            f"/MediaBox [0 0 {w:.4f} {h:.4f}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {cid} 0 R >>"
        ).encode("latin-1")
        page_ids.append(add_obj(p_obj))

    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_placeholder_id - 1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>".encode("latin-1")
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_placeholder_id} 0 R >>".encode("latin-1"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("latin-1"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return bytes(out)




# --- Wave-3 Live Router Endpoint & Vector PDF Spec Adapter ---
def _wave3_pdf_to_jsonable(obj):
    import dataclasses as _dc
    from pathlib import Path as _Path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, _Path):
        return str(obj)
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        return {str(k): _wave3_pdf_to_jsonable(v) for k, v in _dc.asdict(obj).items()}
    if hasattr(obj, "_asdict") and callable(obj._asdict):
        return {str(k): _wave3_pdf_to_jsonable(v) for k, v in obj._asdict().items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _wave3_pdf_to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _wave3_pdf_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_wave3_pdf_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _wave3_pdf_to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return _wave3_pdf_to_jsonable(obj.item())
    if hasattr(obj, "__dict__"):
        return {str(k): _wave3_pdf_to_jsonable(v) for k, v in vars(obj).items() if not str(k).startswith("_")}
    return str(obj)


if "build_minimal_vector_pdf" in globals():
    _orig_build_minimal_vector_pdf = build_minimal_vector_pdf

    def _pdf_esc(txt: str) -> str:
        return str(txt).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def build_minimal_vector_pdf(pages, *args, **kwargs):
        try:
            return _orig_build_minimal_vector_pdf(pages, *args, **kwargs)
        except Exception:
            norm_pages = []
            for pg in pages:
                if isinstance(pg, dict) and "content_stream" not in pg:
                    ops = []
                    for seg in pg.get("segments", ()):
                        if isinstance(seg, dict):
                            x0, y0, x1, y1 = seg["x0"], seg["y0"], seg["x1"], seg["y1"]
                        else:
                            x0, y0, x1, y1 = seg[:4]
                        ops.append(f"{float(x0):.4f} {float(y0):.4f} m {float(x1):.4f} {float(y1):.4f} l S")
                    for tx in pg.get("texts", ()):
                        if isinstance(tx, dict):
                            x, y, s = tx["x"], tx["y"], tx["text"]
                            fs = float(tx.get("font_size", 10.0))
                        else:
                            x, y, s = tx[0], tx[1], tx[2]
                            fs = float(tx[3]) if len(tx) > 3 else 10.0
                        ops.append(f"BT /F1 {fs:.2f} Tf {float(x):.4f} {float(y):.4f} Td ({_pdf_esc(s)}) Tj ET")
                    pg = {
                        "width_pt": pg.get("width_pt", 612.0),
                        "height_pt": pg.get("height_pt", 792.0),
                        "content_stream": "\n".join(ops),
                    }
                norm_pages.append(pg)
            return _orig_build_minimal_vector_pdf(norm_pages, *args, **kwargs)


def run_router_drawing_reader(payload: dict) -> dict:
    import base64 as _b64
    path = payload.get("path")
    if path and Path(path).is_file():
        pdf_bytes = Path(path).read_bytes()
    elif payload.get("pdf_b64"):
        pdf_bytes = _b64.b64decode(payload["pdf_b64"])
    elif isinstance(payload.get("pdf_bytes"), (bytes, bytearray)):
        pdf_bytes = bytes(payload["pdf_bytes"])
    else:
        raise FileNotFoundError(f"PDF file not found: {path}")
    from pypdf import PdfReader

    pdf = PdfReader(io.BytesIO(pdf_bytes))
    pages = _read_pdf_pages(pdf, payload.get("page_index"))
    return {"pages": _wave3_pdf_to_jsonable(pages), "page_count": len(pdf.pages),
            "ok": True, "tool": "hamon_drawing_reader"}


if __name__ == "__main__" and "--json-stdin" in __import__("sys").argv:
    import json as _cli_json, sys as _cli_sys
    _req = _cli_json.loads(_cli_sys.stdin.read() or "{}")
    _resp = run_router_drawing_reader(_req)
    print(_cli_json.dumps(_wave3_pdf_to_jsonable(_resp)))
    raise SystemExit(0)

