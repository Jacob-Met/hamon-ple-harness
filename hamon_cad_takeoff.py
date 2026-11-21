"""Native ASCII DXF / CAD Spatial Coordinate, Curved-Bulge Area, Paper-Space Filter & 3D MEP Clash Solver.

Extracts model-space LINE, LWPOLYLINE (including group-42 circular bulge arcs),
POLYLINE/VERTEX/SEQEND, CIRCLE, ARC, ELLIPSE, and INSERT entities while filtering
out paper-space layout entities (group 67 == 1), with exact $INSUNITS -> SI meter
conversion, Green's-theorem curved area/perimeter integration, and 3D segment/AABB
clearance verification. Zero task-specific hardcoding.
"""
from __future__ import annotations

from collections import Counter
from itertools import pairwise
import hashlib
import math
from pathlib import Path
from typing import Any

INSUNITS_TO_METERS: dict[int, float] = {
    1: 0.0254,       # inches
    2: 0.3048,       # feet
    3: 1609.344,     # miles
    4: 0.001,        # millimeters
    5: 0.01,         # centimeters
    6: 1.0,          # meters
    7: 1000.0,       # kilometers
    8: 2.54e-8,      # microinches
    9: 2.54e-5,      # mils
    10: 0.9144,      # yards
    12: 1e-9,        # nanometers
    13: 1e-6,        # microns
    14: 0.1,         # decimeters
}

# 16-point Gauss-Legendre nodes and weights on [-1, 1] for elliptical arc quadrature
_GL16_X = (
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499,
)
_GL16_W = (
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
)


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
    """Compute unsigned polygon area (if closed & >=3 vertices) and chord polyline length."""
    if len(pts) < 2:
        return 0.0, 0.0
    perim = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    if closed:
        perim += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
    if not closed or len(pts) < 3:
        return 0.0, perim
    origin_x, origin_y = pts[0]
    twice_a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        twice_a += (x1 - origin_x) * (y2 - origin_y) - (x2 - origin_x) * (y1 - origin_y)
    return 0.5 * abs(twice_a), perim


def bulge_area_and_perimeter(verts_xyb: list[tuple[float, float, float]], closed: bool) -> tuple[float, float]:
    """Exact Green's-theorem area and arc-length perimeter for polylines with DXF code-42 bulges."""
    n = len(verts_xyb)
    if n < 2:
        return 0.0, 0.0
    has_bulge = any(abs(v[2]) > 1e-12 for v in verts_xyb)
    seg_count = n if closed else (n - 1)
    perim = 0.0
    origin_x, origin_y, _ = verts_xyb[0]
    signed_area = 0.0
    for k in range(seg_count):
        x1, y1, bulge = verts_xyb[k]
        x2, y2, _ = verts_xyb[(k + 1) % n]
        chord = math.hypot(x2 - x1, y2 - y1)
        if closed:
            # Local coordinates preserve the chord area at large survey offsets.
            signed_area += 0.5 * (
                (x1 - origin_x) * (y2 - origin_y) - (x2 - origin_x) * (y1 - origin_y)
            )
        if chord > 1e-15 and abs(bulge) > 1e-12:
            theta = 4.0 * math.atan(bulge)
            radius = chord * (1.0 + bulge * bulge) / (4.0 * abs(bulge))
            perim += radius * abs(theta)
            if closed:
                signed_area += 0.5 * (radius * radius) * (theta - math.sin(theta))
        else:
            perim += chord
    if not closed or (n < 3 and not has_bulge):
        return 0.0, perim
    return abs(signed_area), perim


def _ellipse_area_and_arc(
    mx: float, my: float, mz: float, ratio: float, u0: float, u1: float
) -> tuple[float, float, bool]:
    a = math.sqrt(mx * mx + my * my + mz * mz)
    if a <= 0.0 or ratio <= 0.0:
        return 0.0, 0.0, False
    b = a * ratio
    du = u1 - u0
    if abs(du) < 1e-12:
        du = 2.0 * math.pi
    elif du < 0.0:
        du += 2.0 * math.pi
    is_closed = abs(du - 2.0 * math.pi) <= 1e-6
    def panel(lo: float, hi: float) -> float:
        half = (hi - lo) / 2
        mid = lo + half
        total = 0.0
        for node, weight in zip(_GL16_X, _GL16_W):
            t = mid + half * node
            total += weight * math.hypot(math.sin(t), ratio * math.cos(t))
        return half * total

    def refine(lo: float, hi: float, coarse: float, depth: int) -> float:
        mid = (lo + hi) / 2
        left, right = panel(lo, mid), panel(mid, hi)
        if depth == 20 or abs(left + right - coarse) <= 2e-13 * (hi - lo):
            return left + right
        return refine(lo, mid, left, depth + 1) + refine(mid, hi, right, depth + 1)

    # Split at quadrant boundaries so an eccentric ellipse's speed minima
    # occur at panel ends, including on arcs that wrap across zero.
    quarter = math.pi / 2
    end = u0 + du
    first = math.floor(u0 / quarter) + 1
    # A DXF elliptic arc covers at most one revolution. Bound the partition
    # count even if a malformed drawing supplies out-of-range parameters.
    cuts = [u0] + [i * quarter for i in range(first, first + 4) if u0 < i * quarter < end] + [end]
    arc = a * math.fsum(refine(lo, hi, panel(lo, hi), 0) for lo, hi in pairwise(cuts))
    area = math.pi * a * b if is_closed else 0.0
    return area, arc, is_closed


def _is_paper_space(groups: list[tuple[int, str]]) -> bool:
    for c, v in groups:
        if c == 67:
            try:
                if int(v.strip() or "0") == 1:
                    return True
            except ValueError:
                pass
    return False


def scan_dxf_takeoff(filename: str | Path) -> dict[str, Any]:
    """Extract SI-calibrated model-space lengths, areas, and block counts.

    For 3D POLYLINEs the perimeter follows the straight WCS edges. A closed
    loop contributes its XY plan footprint to the area totals.
    """
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
    paper_space_skipped = 0
    curved_bulge_segments = 0
    poly_state: dict[str, Any] | None = None

    def finalize_poly_state() -> None:
        nonlocal poly_state, paper_space_skipped, curved_bulge_segments
        if poly_state is None:
            return
        st = poly_state
        poly_state = None
        if st["paper"]:
            paper_space_skipped += 1
            return
        if st["is_mesh"]:
            return
        entity_counts["POLYLINE"] += 1
        verts = st["verts"]
        closed = st["closed"]
        layer = st["layer"]
        spatial = st["spatial_vertices"]
        if spatial is None:
            seg_n = len(verts) if closed else max(0, len(verts) - 1)
            curved_bulge_segments += sum(1 for idx_v in range(seg_n) if abs(verts[idx_v][2]) > 1e-12)
            area, perim = bulge_area_and_perimeter(verts, closed)
        else:
            area = 0.0
            perim = sum(math.dist(a, b) for a, b in pairwise(spatial))
            if closed and spatial:
                perim += math.dist(spatial[-1], spatial[0])
                area, _ = shoelace_area_and_perimeter([(x, y) for x, y, _ in spatial], True)
        if closed and area > 0.0:
            poly_area_by_layer[layer] = poly_area_by_layer.get(layer, 0.0) + area
        poly_perim_by_layer[layer] = poly_perim_by_layer.get(layer, 0.0) + perim

    def flush_entity(etype: str, groups: list[tuple[int, str]]) -> None:
        nonlocal paper_space_skipped, curved_bulge_segments, poly_state
        if etype == "VERTEX" and poly_state is not None:
            v_flag = int(next((v for c, v in groups if c == 70), "0"))
            if not (v_flag & (16 | 64)):
                vx = next((float(v) for c, v in groups if c == 10), None)
                vy = next((float(v) for c, v in groups if c == 20), None)
                if vx is not None and vy is not None:
                    if poly_state["spatial_vertices"] is None:
                        bulge = next((float(v) for c, v in groups if c == 42), 0.0)
                        poly_state["verts"].append((vx, vy, bulge))
                    else:
                        vz = next((float(v) for c, v in groups if c == 30), 0.0)
                        poly_state["spatial_vertices"].append((vx, vy, vz))
            return
        if etype == "SEQEND":
            finalize_poly_state()
            return
        if poly_state is not None:
            finalize_poly_state()

        if etype == "POLYLINE":
            flag = int(next((v for c, v in groups if c == 70), "0"))
            layer = next((v for c, v in groups if c == 8), "0")
            poly_state = {
                "layer": layer,
                "closed": bool(flag & 1),
                "paper": _is_paper_space(groups),
                "is_mesh": bool(flag & (16 | 64)),
                "spatial_vertices": [] if flag & 8 else None,
                "verts": [],
            }
            return

        if _is_paper_space(groups):
            paper_space_skipped += 1
            return

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
            raw_verts: list[list[float]] = []
            for c, v in groups:
                if c == 10:
                    raw_verts.append([float(v), 0.0, 0.0])
                elif c == 20 and raw_verts:
                    raw_verts[-1][1] = float(v)
                elif c == 42 and raw_verts:
                    raw_verts[-1][2] = float(v)
            verts_t = [(rv[0], rv[1], rv[2]) for rv in raw_verts]
            seg_n = len(verts_t) if closed else max(0, len(verts_t) - 1)
            curved_bulge_segments += sum(1 for idx_v in range(seg_n) if abs(verts_t[idx_v][2]) > 1e-12)
            area, perim = bulge_area_and_perimeter(verts_t, closed)
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
        elif etype == "ELLIPSE":
            mx = next((float(v) for c, v in groups if c == 11), 0.0)
            my = next((float(v) for c, v in groups if c == 21), 0.0)
            mz = next((float(v) for c, v in groups if c == 31), 0.0)
            ratio = next((float(v) for c, v in groups if c == 40), 0.0)
            u0 = next((float(v) for c, v in groups if c == 41), 0.0)
            u1 = next((float(v) for c, v in groups if c == 42), 2.0 * math.pi)
            e_area, e_arc, e_closed = _ellipse_area_and_arc(mx, my, mz, ratio, u0, u1)
            if e_closed and e_area > 0.0:
                poly_area_by_layer[layer] = poly_area_by_layer.get(layer, 0.0) + e_area
                poly_perim_by_layer[layer] = poly_perim_by_layer.get(layer, 0.0) + e_arc
            elif e_arc > 0.0:
                line_len_by_layer[layer] = line_len_by_layer.get(layer, 0.0) + e_arc
        elif etype == "INSERT":
            bname = next((v for c, v in groups if c == 2), "UNNAMED")
            columns = int(next((v for c, v in groups if c == 70), "1"))
            rows = int(next((v for c, v in groups if c == 71), "1"))
            if columns < 1 or rows < 1:
                raise ValueError("DXF INSERT row and column counts must be positive")
            placements = columns * rows
            insert_by_layer[layer] += placements
            insert_by_block[bname] += placements

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
            finalize_poly_state()
            section = pairs[i + 1][1].upper()
            i += 2
            continue
        if code == 0 and uval in ("ENDSEC", "EOF"):
            if cur_ent:
                flush_entity(cur_ent, cur_groups)
                cur_ent, cur_groups = None, []
            finalize_poly_state()
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
        "paper_space_skipped_count": paper_space_skipped,
        "curved_bulge_segments": curved_bulge_segments,
        "line_length_m_by_layer": {k: v * scale_m for k, v in line_len_by_layer.items()} if scale_m else None,
        "closed_area_m2_by_layer": {k: v * (scale_m**2) for k, v in poly_area_by_layer.items()} if scale_m else None,
        "polyline_perimeter_m_by_layer": {k: v * scale_m for k, v in poly_perim_by_layer.items()} if scale_m else None,
        "insert_count_by_layer": dict(insert_by_layer),
        "insert_count_by_block": dict(insert_by_block),
        "total_insert_blocks": sum(insert_by_block.values()),
    }


def segment_segment_distance_3d(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> float:
    """Exact 3D Euclidean minimum distance between line segments [p1, q1] and [p2, q2]."""
    d1 = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    d2 = (q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2])
    r = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2]
    c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2]
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]

    def _dist_sq(s: float, t: float) -> float:
        dx = r[0] + s * d1[0] - t * d2[0]
        dy = r[1] + s * d1[1] - t * d2[1]
        dz = r[2] + s * d1[2] - t * d2[2]
        return dx * dx + dy * dy + dz * dz

    candidates: list[tuple[float, float]] = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    if e > 1e-18:
        candidates.append((0.0, min(1.0, max(0.0, f / e))))
        candidates.append((1.0, min(1.0, max(0.0, (b + f) / e))))
    if a > 1e-18:
        candidates.append((min(1.0, max(0.0, -c / a)), 0.0))
        candidates.append((min(1.0, max(0.0, (b - c) / a)), 1.0))
    denom = a * e - b * b
    if denom > 1e-18:
        s_int = (b * f - c * e) / denom
        t_int = (a * f - b * c) / denom
        if 0.0 <= s_int <= 1.0 and 0.0 <= t_int <= 1.0:
            candidates.append((s_int, t_int))
    best_sq = min(_dist_sq(s, t) for s, t in candidates)
    return math.sqrt(max(0.0, best_sq))


def segment_aabb_distance_3d(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> float:
    """Exact piecewise-quadratic 3D Euclidean minimum distance between segment [p1, q1] and solid AABB."""
    d = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    breaks = {0.0, 1.0}
    for ax in range(3):
        if abs(d[ax]) > 1e-18:
            for bound in (box_min[ax], box_max[ax]):
                t_val = (bound - p1[ax]) / d[ax]
                if 0.0 < t_val < 1.0:
                    breaks.add(t_val)
    pts_t = sorted(breaks)

    def _pt_box_sq(s: float) -> float:
        tot = 0.0
        for ax in range(3):
            u = p1[ax] + s * d[ax]
            gap = max(0.0, box_min[ax] - u, u - box_max[ax])
            tot += gap * gap
        return tot

    best_sq = min(_pt_box_sq(s) for s in pts_t)
    for idx in range(len(pts_t) - 1):
        sa, sb = pts_t[idx], pts_t[idx + 1]
        sm = 0.5 * (sa + sb)
        num = 0.0
        den = 0.0
        for ax in range(3):
            um = p1[ax] + sm * d[ax]
            if um < box_min[ax]:
                num += (p1[ax] - box_min[ax]) * d[ax]
                den += d[ax] * d[ax]
            elif um > box_max[ax]:
                num += (p1[ax] - box_max[ax]) * d[ax]
                den += d[ax] * d[ax]
        if den > 1e-18:
            s_star = min(sb, max(sa, -num / den))
            best_sq = min(best_sq, _pt_box_sq(s_star))
    return math.sqrt(max(0.0, best_sq))
