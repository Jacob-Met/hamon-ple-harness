"""Multi-sheet vector PDF content-stream reader, engineering scale parser,
and PLE competition deliverable pre-flight validator.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _decode_pdf_literal_string(raw: str) -> str:
    out: List[str] = []
    i = 0
    n = len(raw)
    escape_map = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "(": "(", ")": ")", "\\": "\\"}
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt in "01234567":
                oct_digits = nxt
                j = i + 2
                while j < n and len(oct_digits) < 3 and raw[j] in "01234567":
                    oct_digits += raw[j]
                    j += 1
                out.append(chr(int(oct_digits, 8)))
                i = j
                continue
            out.append(escape_map.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _tokenize_content_stream(stream_text: str) -> List[Any]:
    tokens: List[Any] = []
    i = 0
    n = len(stream_text)
    while i < n:
        ch = stream_text[i]
        if ch in " \t\r\n\x00\x0c":
            i += 1
            continue
        if ch == "%":
            while i < n and stream_text[i] not in "\r\n":
                i += 1
            continue
        if ch == "(":
            depth = 1
            i += 1
            buf: List[str] = []
            while i < n and depth > 0:
                c = stream_text[i]
                if c == "\\" and i + 1 < n:
                    buf.append(c)
                    buf.append(stream_text[i + 1])
                    i += 2
                    continue
                if c == "(":
                    depth += 1
                    buf.append(c)
                elif c == ")":
                    depth -= 1
                    if depth > 0:
                        buf.append(c)
                else:
                    buf.append(c)
                i += 1
            tokens.append(("str", _decode_pdf_literal_string("".join(buf))))
            continue
        if ch == "<" and (i + 1 >= n or stream_text[i + 1] != "<"):
            j = stream_text.find(">", i + 1)
            if j == -1:
                j = n
            hex_s = re.sub(r"\s+", "", stream_text[i + 1:j])
            if len(hex_s) % 2 == 1:
                hex_s += "0"
            try:
                raw_b = bytes.fromhex(hex_s)
                txt = raw_b.decode("utf-16-be") if raw_b.startswith(b"\xfe\xff") else raw_b.decode("latin-1")
            except Exception:
                txt = ""
            tokens.append(("str", txt))
            i = j + 1
            continue
        if ch == "[":
            tokens.append("[")
            i += 1
            continue
        if ch == "]":
            tokens.append("]")
            i += 1
            continue
        j = i
        while j < n and stream_text[j] not in " \t\r\n\x00\x0c()[]<>%":
            j += 1
        tok = stream_text[i:j]
        i = j
        try:
            tokens.append(float(tok))
        except ValueError:
            tokens.append(tok)
    return tokens


def _interpret_pdf_stream(stream_text: str, page_index: int) -> Tuple[List[PdfTextSpan], List[PdfVectorSegment]]:
    tokens = _tokenize_content_stream(stream_text)
    ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    stack: List[Tuple[float, float, float, float, float, float]] = []
    tm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    tlm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    leading = 12.0

    texts: List[PdfTextSpan] = []
    segments: List[PdfVectorSegment] = []
    pending_segs: List[PdfVectorSegment] = []
    cur_pt: Optional[Tuple[float, float]] = None
    subpath_start: Optional[Tuple[float, float]] = None

    operands: List[Any] = []
    arr_stack: List[List[Any]] = []
    paint_ops = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}

    for tok in tokens:
        if tok == "[":
            arr_stack.append([])
            continue
        if tok == "]":
            arr = arr_stack.pop() if arr_stack else []
            if arr_stack:
                arr_stack[-1].append(arr)
            else:
                operands.append(arr)
            continue
        if arr_stack:
            arr_stack[-1].append(tok)
            continue
        if isinstance(tok, (float, int, tuple, list)) or (isinstance(tok, str) and tok.startswith("/")):
            operands.append(tok)
            continue

        op = tok
        if op == "q":
            stack.append(ctm)
        elif op == "Q":
            if stack:
                ctm = stack.pop()
        elif op == "cm" and len(operands) >= 6:
            vals = tuple(float(x) for x in operands[-6:])
            ctm = _mat_mul_pdf(vals, ctm)
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
            if isinstance(last, tuple) and last[0] == "str":
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, last[1], px, py))
        elif op == "'" and len(operands) >= 1:
            tlm = _mat_mul_pdf((1.0, 0.0, 0.0, 1.0, 0.0, -leading), tlm)
            tm = tlm
            last = operands[-1]
            if isinstance(last, tuple) and last[0] == "str":
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, last[1], px, py))
        elif op == "TJ" and len(operands) >= 1 and isinstance(operands[-1], list):
            parts = [item[1] for item in operands[-1] if isinstance(item, tuple) and item[0] == "str"]
            joined = "".join(parts)
            if joined:
                px, py = _apply_pdf_ctm(0.0, 0.0, _mat_mul_pdf(tm, ctm))
                texts.append(PdfTextSpan(page_index, joined, px, py))
        operands.clear()

    return texts, segments


def read_vector_pdf(pdf_bytes: bytes) -> List[PdfSheetPage]:
    objs: Dict[int, bytes] = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b(.*?)endobj", pdf_bytes, flags=re.DOTALL):
        objs[int(m.group(1))] = m.group(3)

    def extract_stream(obj_body: bytes) -> bytes:
        sm = re.search(rb"<<(.*?)>>\s*stream(?:\r\n|\n)(.*?)endstream", obj_body, flags=re.DOTALL)
        if not sm:
            return b""
        hdr = sm.group(1)
        raw = sm.group(2)
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        elif raw.endswith(b"\n"):
            raw = raw[:-1]
        if b"/FlateDecode" in hdr:
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw.strip(b"\r\n"))
        return raw

    def parse_box(body: bytes, inherited_box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        bm = re.search(rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*\]", body)
        if bm:
            return tuple(float(bm.group(k)) for k in range(1, 5))  # type: ignore[return-value]
        return inherited_box

    ordered_page_ids: List[int] = []
    inherited_boxes: Dict[int, Tuple[float, float, float, float]] = {}

    def walk_pages(oid: int, cur_box: Tuple[float, float, float, float], visited: set) -> None:
        if oid in visited or oid not in objs:
            return
        visited.add(oid)
        body = objs[oid]
        hdr_only = body.split(b"stream", 1)[0]
        box = parse_box(hdr_only, cur_box)
        if re.search(rb"/Type\s*/Pages\b", hdr_only):
            km = re.search(rb"/Kids\s*\[(.*?)\]", hdr_only, flags=re.DOTALL)
            if km:
                for rm in re.finditer(rb"(\d+)\s+\d+\s+R", km.group(1)):
                    walk_pages(int(rm.group(1)), box, visited)
        elif re.search(rb"/Type\s*/Page\b", hdr_only):
            ordered_page_ids.append(oid)
            inherited_boxes[oid] = box

    for oid, body in sorted(objs.items()):
        hdr = body.split(b"stream", 1)[0]
        if re.search(rb"/Type\s*/Catalog\b", hdr):
            pm = re.search(rb"/Pages\s+(\d+)\s+\d+\s+R", hdr)
            if pm:
                walk_pages(int(pm.group(1)), (0.0, 0.0, 612.0, 792.0), set())
                break

    if not ordered_page_ids:
        for oid, body in sorted(objs.items()):
            hdr = body.split(b"stream", 1)[0]
            if re.search(rb"/Type\s*/Page\b", hdr):
                ordered_page_ids.append(oid)
                inherited_boxes[oid] = parse_box(hdr, (0.0, 0.0, 612.0, 792.0))

    sheets: List[PdfSheetPage] = []
    for p_idx, oid in enumerate(ordered_page_ids):
        body = objs[oid]
        hdr = body.split(b"stream", 1)[0]
        x0, y0, x1, y1 = inherited_boxes.get(oid, (0.0, 0.0, 612.0, 792.0))
        width_pt, height_pt = abs(x1 - x0), abs(y1 - y0)

        content_ids: List[int] = []
        cm_arr = re.search(rb"/Contents\s*\[(.*?)\]", hdr, flags=re.DOTALL)
        if cm_arr:
            content_ids = [int(rm.group(1)) for rm in re.finditer(rb"(\d+)\s+\d+\s+R", cm_arr.group(1))]
        else:
            cm_single = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R", hdr)
            if cm_single:
                content_ids = [int(cm_single.group(1))]

        combined = b"\n".join(extract_stream(objs[cid]) for cid in content_ids if cid in objs)
        texts, segs = _interpret_pdf_stream(combined.decode("latin-1", errors="replace"), p_idx)

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


MAX_PLE_BYTES = 512 * 1024 * 1024
MAX_PLE_FILES = 10_000


def sanitize_scalar_value(val: Any) -> float:
    if isinstance(val, bool) or val is None:
        raise ValueError(f"Invalid boolean/null scalar answer: {val!r}")
    if isinstance(val, (int, float)):
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite scalar answer: {val!r}")
        return f
    if isinstance(val, str):
        s = val.strip()
        if not re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$", s):
            raise ValueError(f"Scalar string contains prose or unstripped units: {val!r}")
        f = float(s)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite parsed scalar: {val!r}")
        return f
    raise ValueError(f"Unsupported scalar type: {type(val).__name__}")


def check_deliverable_preflight(
    bundle_path: str | Path,
    expected_task_ids: Optional[Sequence[str]] = None,
    max_bytes: int = MAX_PLE_BYTES,
    max_files: int = MAX_PLE_FILES,
) -> Dict[str, Any]:
    path = Path(bundle_path)
    errors: List[str] = []
    total_uncompressed = 0
    compressed_bytes = 0
    file_count = 0
    seen_names: set = set()
    scalar_payloads_checked = 0

    if not path.exists():
        return {"ok": False, "errors": [f"Missing bundle: {path}"], "file_count": 0, "total_bytes": 0}

    if path.is_file() and path.suffix.lower() == ".zip":
        compressed_bytes = path.stat().st_size
        if compressed_bytes > max_bytes:
            errors.append(f"Compressed zip size {compressed_bytes} exceeds {max_bytes} bytes")
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if unix_mode and stat.S_ISLNK(unix_mode):
                    errors.append(f"Symlink prohibited in archive: {name}")
                if name.startswith("/") or name.startswith("\\") or ".." in Path(name).parts:
                    errors.append(f"Unsafe archive member path: {name}")
                if info.is_dir():
                    continue
                norm = name.replace("\\", "/").lstrip("./")
                if norm in seen_names:
                    errors.append(f"Duplicate archive member path: {norm}")
                seen_names.add(norm)
                file_count += 1
                total_uncompressed += info.file_size
                if norm.endswith(("predictions.json", "answers.json", "scalar_answers.json", "wave3_scalar_predictions.json")):
                    raw_txt = zf.read(info).decode("utf-8")
                    _validate_scalar_json_text(raw_txt, norm, expected_task_ids, errors)
                    scalar_payloads_checked += 1
    elif path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            root_p = Path(root)
            for d in dirs:
                dp = root_p / d
                if dp.is_symlink():
                    errors.append(f"Symlink directory prohibited: {dp.relative_to(path)}")
            for fn in files:
                fp = root_p / fn
                rel = fp.relative_to(path).as_posix()
                if fp.is_symlink():
                    errors.append(f"Symlink file prohibited: {rel}")
                    continue
                file_count += 1
                total_uncompressed += fp.stat().st_size
                if rel.endswith(("predictions.json", "answers.json", "scalar_answers.json", "wave3_scalar_predictions.json")):
                    _validate_scalar_json_text(fp.read_text(encoding="utf-8"), rel, expected_task_ids, errors)
                    scalar_payloads_checked += 1
    else:
        errors.append(f"Unsupported deliverable path type: {path}")

    if file_count > max_files:
        errors.append(f"File count {file_count} exceeds {max_files}")
    if total_uncompressed > max_bytes:
        errors.append(f"Uncompressed size {total_uncompressed} exceeds {max_bytes} bytes")
    if file_count == 0:
        errors.append("Deliverable archive contains 0 files")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "file_count": file_count,
        "compressed_bytes": compressed_bytes,
        "total_uncompressed_bytes": total_uncompressed,
        "scalar_payloads_checked": scalar_payloads_checked,
    }


def _validate_scalar_json_text(
    raw_txt: str,
    label: str,
    expected_task_ids: Optional[Sequence[str]],
    errors: List[str],
) -> None:
    def bad_const(tok: str) -> Any:
        raise ValueError(f"Non-standard JSON constant {tok} in {label}")

    try:
        data = json.loads(raw_txt, parse_constant=bad_const)
    except Exception as exc:
        errors.append(f"Invalid JSON in {label}: {exc}")
        return

    if not isinstance(data, dict):
        errors.append(f"Scalar JSON root in {label} must be an object")
        return

    if expected_task_ids is not None:
        missing = [tid for tid in expected_task_ids if tid not in data]
        if missing:
            errors.append(f"Missing {len(missing)} required task IDs in {label}: {missing[:5]}")

    for k, v in data.items():
        val = v.get("answer") if isinstance(v, dict) and "answer" in v else v
        try:
            sanitize_scalar_value(val)
        except ValueError as exc:
            errors.append(f"Task {k} in {label}: {exc}")



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
    import inspect as _ins
    import os as _os
    import tempfile as _tf
    from pathlib import Path as _Path

    tmp_file = None
    try:
        if payload.get("path") and _Path(payload["path"]).is_file():
            pdf_path = _Path(payload["path"])
        else:
            fd, tname = _tf.mkstemp(suffix=".pdf", prefix="ple-pdf-")
            _os.close(fd)
            tmp_file = _Path(tname)
            if "pdf_b64" in payload and payload["pdf_b64"]:
                tmp_file.write_bytes(_b64.b64decode(payload["pdf_b64"]))
            elif "pdf_bytes" in payload and isinstance(payload["pdf_bytes"], (bytes, bytearray)):
                tmp_file.write_bytes(bytes(payload["pdf_bytes"]))
            else:
                raise FileNotFoundError(f"PDF file not found: {payload.get('path')}")
            pdf_path = tmp_file

        preferred = (
            "read_engineering_drawing",
            "analyze_drawing_pdf",
            "read_drawing_pdf",
            "analyze_engineering_drawing",
            "extract_drawing_takeoff",
            "parse_drawing_pdf",
            "read_vector_pdf",
            "inspect_drawing_pdf",
            "analyze_pdf",
            "extract_pdf_pages",
            "parse_pdf_sheets",
        )
        g = globals()
        cands = []
        for nm in preferred:
            if nm in g and callable(g[nm]):
                cands.append((nm, g[nm]))
        for nm, fn in g.items():
            if callable(fn) and not nm.startswith("_") and nm not in ("build_minimal_vector_pdf", "run_router_drawing_reader"):
                if any(k in nm.lower() for k in ("read", "analyze", "extract", "parse", "takeoff", "inspect", "sheet")):
                    if (nm, fn) not in cands:
                        cands.append((nm, fn))
        pdf_bytes = pdf_path.read_bytes()
        for nm, fn in cands:
            try:
                sig = _ins.signature(fn)
            except Exception:
                continue
            params = list(sig.parameters.values())
            if not params:
                continue
            kw = {}
            for pk in sig.parameters:
                if pk in ("scale_override", "scale", "override_scale") and payload.get("scale_override") is not None:
                    kw[pk] = payload["scale_override"]
                elif pk in ("page_index", "page") and payload.get("page_index") is not None:
                    kw[pk] = int(payload["page_index"])
            for arg0 in (pdf_path, str(pdf_path), pdf_bytes):
                try:
                    raw_out = fn(arg0, **kw)
                    if raw_out is None:
                        continue
                    out = _wave3_pdf_to_jsonable(raw_out)
                    if isinstance(out, list):
                        out = {"pages": out, "page_count": len(out)}
                    if isinstance(out, dict) and len(out) >= 1:
                        out["ok"] = True
                        out.setdefault("tool", "hamon_drawing_reader")
                        return out
                except Exception:
                    continue
        raise RuntimeError("No compatible PDF drawing reader entrypoint matched")
    finally:
        if tmp_file is not None:
            tmp_file.unlink(missing_ok=True)


if __name__ == "__main__" and "--json-stdin" in __import__("sys").argv:
    import json as _cli_json, sys as _cli_sys
    _req = _cli_json.loads(_cli_sys.stdin.read() or "{}")
    _resp = run_router_drawing_reader(_req)
    print(_cli_json.dumps(_wave3_pdf_to_jsonable(_resp)))
    raise SystemExit(0)

