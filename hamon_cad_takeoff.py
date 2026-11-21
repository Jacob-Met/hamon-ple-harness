"""Native ASCII DXF / CAD Spatial Coordinate, Closed-Polygon Area & Block BOM Takeoff.

Extracts model-space LINE, LWPOLYLINE, CIRCLE, ARC, and INSERT entities with
exact $INSUNITS -> SI meter conversion and Shoelace area/perimeter cross-checks.
Zero task-specific hardcoding.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any

INSUNITS_TO_METERS: dict[int, float] = {
    1: 0.0254,       # inches
    2: 0.3048,       # feet
    4: 0.001,        # millimeters
    5: 0.01,         # centimeters
    6: 1.0,          # meters
    7: 1000.0,       # kilometers
    10: 0.9144,      # yards
}


def _iter_dxf_pairs(data: bytes):
    if b"\x00" in data[:100] or data.startswith(b"AutoCAD Binary DXF"):
        raise ValueError("binary DXF not supported; supply ASCII DXF")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    lines = text.splitlines()
    if len(lines) % 2 != 0 or len(lines) // 2 > 1_000_000:
        raise ValueError("malformed DXF group pairs or exceeds bound")
    for i in range(0, len(lines), 2):
        yield int(lines[i].strip()), lines[i + 1].strip()


def shoelace_area_and_perimeter(pts: list[tuple[float, float]], closed: bool) -> tuple[float, float]:
    """Compute unsigned polygon area (if closed & >=3 vertices) and polyline length."""
    if len(pts) < 2:
        return 0.0, 0.0
    perim = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    if closed:
        perim += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
    if not closed or len(pts) < 3:
        return 0.0, perim
    twice_a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        twice_a += x1 * y2 - x2 * y1
    return 0.5 * abs(twice_a), perim


def scan_dxf_takeoff(filename: str | Path) -> dict[str, Any]:
    """Extract SI-calibrated spatial quantities (lengths, areas, block counts) from ASCII DXF."""
    p = Path(filename)
    if not p.is_file() or p.stat().st_size > 32_000_000:
        raise ValueError("missing or oversized DXF file")
    data = p.read_bytes()
    pairs = list(_iter_dxf_pairs(data))

    insunits: int | None = None
    section: str | None = None
    i = 0
    n_pairs = len(pairs)

    line_len_by_layer: dict[str, float] = {}
    poly_area_by_layer: dict[str, float] = {}
    poly_perim_by_layer: dict[str, float] = {}
    insert_by_layer: Counter[str] = Counter()
    insert_by_block: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()

    def flush_entity(etype: str, groups: list[tuple[int, str]]) -> None:
        entity_counts[etype] += 1
        layer = next((v for c, v in groups if c == 8), "0")
        if etype == "LINE":
            pts = {c: float(v) for c, v in groups if c in (10, 20, 30, 11, 21, 31)}
            if 10 in pts and 20 in pts and 11 in pts and 21 in pts:
                dz = pts.get(31, 0.0) - pts.get(30, 0.0)
                seg = math.sqrt((pts[11] - pts[10]) ** 2 + (pts[21] - pts[20]) ** 2 + dz * dz)
                line_len_by_layer[layer] = line_len_by_layer.get(layer, 0.0) + seg
        elif etype == "LWPOLYLINE":
            flag = int(next((v for c, v in groups if c == 70), "0"))
            closed = bool(flag & 1)
            verts: list[tuple[float, float]] = []
            cur_x: float | None = None
            for c, v in groups:
                if c == 10:
                    cur_x = float(v)
                elif c == 20 and cur_x is not None:
                    verts.append((cur_x, float(v)))
                    cur_x = None
            area, perim = shoelace_area_and_perimeter(verts, closed)
            if closed and area > 0:
                poly_area_by_layer[layer] = poly_area_by_layer.get(layer, 0.0) + area
            poly_perim_by_layer[layer] = poly_perim_by_layer.get(layer, 0.0) + perim
        elif etype == "CIRCLE":
            rad = next((float(v) for c, v in groups if c == 40), 0.0)
            if rad > 0.0:
                poly_area_by_layer[layer] = poly_area_by_layer.get(layer, 0.0) + math.pi * rad * rad
                poly_perim_by_layer[layer] = poly_perim_by_layer.get(layer, 0.0) + 2.0 * math.pi * rad
        elif etype == "ARC":
            rad = next((float(v) for c, v in groups if c == 40), 0.0)
            a0 = next((float(v) for c, v in groups if c == 50), 0.0)
            a1 = next((float(v) for c, v in groups if c == 51), 0.0)
            sweep_deg = (a1 - a0) % 360.0
            if sweep_deg == 0.0 and a1 != a0:
                sweep_deg = 360.0
            if rad > 0.0 and sweep_deg > 0.0:
                line_len_by_layer[layer] = line_len_by_layer.get(layer, 0.0) + rad * math.radians(sweep_deg)
        elif etype == "INSERT":
            bname = next((v for c, v in groups if c == 2), "UNNAMED")
            insert_by_layer[layer] += 1
            insert_by_block[bname] += 1

    cur_ent: str | None = None
    cur_groups: list[tuple[int, str]] = []
    hdr_var: str | None = None

    while i < n_pairs:
        code, val = pairs[i]
        uval = val.upper() if code in (0, 2, 9) else val
        if code == 0 and uval == "SECTION" and i + 1 < n_pairs and pairs[i + 1][0] == 2:
            if cur_ent:
                flush_entity(cur_ent, cur_groups)
                cur_ent, cur_groups = None, []
            section = pairs[i + 1][1].upper()
            i += 2
            continue
        if code == 0 and uval in ("ENDSEC", "EOF"):
            if cur_ent:
                flush_entity(cur_ent, cur_groups)
                cur_ent, cur_groups = None, []
            section = None
            if uval == "EOF":
                break
            i += 1
            continue
        if section == "HEADER":
            if code == 9:
                hdr_var = uval
            elif code == 70 and hdr_var == "$INSUNITS":
                try:
                    insunits = int(val)
                except ValueError:
                    insunits = None
                hdr_var = None
        elif section == "ENTITIES":
            if code == 0:
                if cur_ent:
                    flush_entity(cur_ent, cur_groups)
                cur_ent, cur_groups = uval, []
            elif cur_ent is not None:
                cur_groups.append((code, val))
        i += 1

    scale_m = INSUNITS_TO_METERS.get(insunits or 0)
    return {
        "kind": "dxf-spatial-takeoff",
        "schema": "hamon.dxf.takeoff.v1",
        "file": str(p),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "declared_insunits": insunits,
        "scale_to_m": scale_m,
        "entity_counts": dict(entity_counts),
        "line_length_m_by_layer": {k: v * scale_m for k, v in line_len_by_layer.items()} if scale_m else None,
        "closed_area_m2_by_layer": {k: v * (scale_m**2) for k, v in poly_area_by_layer.items()} if scale_m else None,
        "polyline_perimeter_m_by_layer": {k: v * scale_m for k, v in poly_perim_by_layer.items()} if scale_m else None,
        "insert_count_by_layer": dict(insert_by_layer),
        "insert_count_by_block": dict(insert_by_block),
        "total_insert_blocks": sum(insert_by_block.values()),
    }
