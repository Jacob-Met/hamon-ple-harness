"""Exact cut/fill over two named layers of model-space planar DXF faces."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path

from hamon_cad_takeoff import INSUNITS_TO_METERS, _iter_dxf_pairs


MAX_BYTES = 8_000_000
MAX_FACES_PER_LAYER = 2048
MAX_X_PAIRS = 200_000
MAX_CLIPPED_PAIRS = 32_000
MAX_ACCUMULATOR_BITS = 8192
_DXF_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_COORD_CODES = {10 + i + 10 * axis for i in range(4) for axis in range(3)}
_SURVEY = {21: Q(1200, 3937), 22: Q(100, 3937),
           23: Q(3600, 3937), 24: Q(5280 * 1200, 3937)}


class _Unqualified(ValueError):
    """Input is readable, but cannot establish a whole-site volume."""


def _coord(text):
    if len(text) > 80:
        raise _Unqualified("terrain coordinate exceeds text budget")
    if not _DXF_DECIMAL.fullmatch(text):
        raise _Unqualified("invalid terrain coordinate")
    try:
        v = Decimal(text)
    except InvalidOperation as exc:
        raise _Unqualified("invalid terrain coordinate") from exc
    if not v.is_finite() or len(v.as_tuple().digits) > 40 or abs(v.as_tuple().exponent) > 80:
        raise _Unqualified("nonfinite or out-of-budget terrain coordinate")
    return Q(v)


def _orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


@dataclass(frozen=True)
class _Face:
    footprint: tuple
    origin_z: Q
    dz_dx: Q
    dz_dy: Q
    area: Q
    xmin: Q
    xmax: Q
    ymin: Q
    ymax: Q

    def height(self, p):
        x, y = self.footprint[0]
        return self.origin_z + self.dz_dx*(p[0]-x) + self.dz_dy*(p[1]-y)


def _face(groups):
    coords = {}
    for code, value in groups:
        if code in _COORD_CODES:
            if code in coords:
                raise _Unqualified("duplicate terrain 3DFACE coordinate")
            coords[code] = _coord(value)
    if coords.keys() != _COORD_CODES:
        raise _Unqualified("terrain 3DFACE needs four explicit complete WCS corners")
    a, b, c, d = (tuple(coords[10+i+10*j] for j in range(3)) for i in range(4))
    cross = _orient(a, b, c)
    if not cross:
        raise _Unqualified("vertical or degenerate terrain face")
    ux, uy, uz = (b[i]-a[i] for i in range(3))
    vx, vy, vz = (c[i]-a[i] for i in range(3))
    dz_dx = (uz*vy-uy*vz)/cross
    dz_dy = (ux*vz-uz*vx)/cross
    pts = (a[:2], b[:2], c[:2])
    if d != c:
        if d[2] != a[2] + dz_dx*(d[0]-a[0]) + dz_dy*(d[1]-a[1]):
            raise _Unqualified("nonplanar terrain quadrilateral")
        pts += (d[:2],)
        # Strict turns in boundary order also reject bow ties and duplicate XYs.
        for i in range(1, 4):
            if _orient(pts[i], pts[(i+1) % 4], pts[(i+2) % 4])*cross <= 0:
                raise _Unqualified("terrain quadrilateral needs a strictly convex XY boundary")
    if cross < 0:
        pts = (pts[0], *reversed(pts[1:]))
    xs, ys = zip(*pts)
    return _Face(pts, a[2], dz_dx, dz_dy, _integrate(pts),
                 min(xs), max(xs), min(ys), max(ys))


def _records(data):
    kind = None
    groups = []
    for code, value in _iter_dxf_pairs(data):
        if code == 999:
            continue
        if code == 0:
            if kind is not None:
                yield kind, groups
            kind, groups = value.upper(), []
        elif kind is None:
            raise ValueError("DXF data before first record")
        else:
            groups.append((code, value))
    if kind is not None:
        yield kind, groups


def _paper(groups):
    flags = [v for c, v in groups if c == 67]
    layouts = [v.casefold() for c, v in groups if c == 410]
    if len(flags) > 1 or len(layouts) > 1 or (layouts and not layouts[0]):
        raise _Unqualified("ambiguous terrain entity layout")
    if flags:
        flag = int(flags[0])
        if flag not in (0, 1) or (layouts and bool(flag) != (layouts[0] != "model")):
            raise _Unqualified("contradictory or invalid terrain entity layout")
        return bool(flag)
    return bool(layouts and layouts[0] != "model")


def _read(data, names, report):
    surfaces = {name: [] for name in names}
    section, units, ended = None, None, False
    seen = set()
    for kind, groups in _records(data):
        if ended:
            raise ValueError("DXF data after EOF")
        if kind == "SECTION":
            if section is not None or not groups or groups[0][0] != 2:
                raise ValueError("invalid DXF SECTION")
            section = groups[0][1].upper()
            if section in seen and section in {"HEADER", "ENTITIES"}:
                raise ValueError("duplicate DXF HEADER or ENTITIES section")
            seen.add(section)
            if section == "HEADER":
                for i, pair in enumerate(groups):
                    if pair == (9, "$INSUNITS"):
                        if (units is not None or i + 1 >= len(groups) or groups[i + 1][0] != 70
                                or (i + 2 < len(groups) and groups[i + 2][0] != 9)):
                            raise ValueError("ambiguous or malformed $INSUNITS")
                        units = int(groups[i+1][1])
                        report["declared_insunits"] = units
        elif kind == "ENDSEC":
            if section is None or groups:
                raise ValueError("invalid DXF ENDSEC")
            section = None
        elif kind == "EOF":
            if section is not None or groups:
                raise ValueError("invalid DXF EOF")
            ended = True
        elif section == "ENTITIES":
            tags = [v.casefold() for c, v in groups if c == 8] or ["0"]
            if any(name in surfaces for name in tags):
                if len(tags) != 1:
                    raise _Unqualified("ambiguous terrain entity layer")
                if _paper(groups):
                    continue
                if kind != "3DFACE":
                    raise _Unqualified("selected terrain layer contains a non-3DFACE entity")
                dest = surfaces[tags[0]]
                if len(dest) >= MAX_FACES_PER_LAYER:
                    raise _Unqualified("terrain face budget exceeded")
                dest.append(_face(groups))
        elif section is None:
            raise ValueError("DXF record outside section")
    if not ended or "ENTITIES" not in seen:
        raise ValueError("incomplete DXF document")
    if units not in INSUNITS_TO_METERS:
        raise _Unqualified("terrain needs known $INSUNITS")
    if not all(surfaces.values()):
        raise _Unqualified("each terrain layer needs model-space terrain faces")
    return units, surfaces


def _clip(poly, side):
    """Clip a convex boundary by side(p) >= 0, retaining point/line contact."""
    if not poly:
        return []
    result = []
    prev = poly[-1]
    before = side(prev)
    for curr in poly:
        after = side(curr)
        if (before < 0 < after) or (after < 0 < before):
            t = before/(before-after)
            result.append(tuple(a+t*(b-a) for a, b in zip(prev, curr)))
        if after >= 0:
            result.append(curr)
        prev, before = curr, after
    return result


def _intersection(left, right):
    poly = left.footprint
    for i, a in enumerate(right.footprint):
        b = right.footprint[(i+1) % len(right.footprint)]
        poly = _clip(poly, lambda p: _orient(a, b, p))
    return poly


def _integrate(poly, height=None):
    # The average of an affine function on a triangle is its vertex mean.
    total = Q(0)
    for i in range(1, len(poly)-1):
        a, b, c = poly[0], poly[i], poly[i+1]
        area = _orient(a, b, c)/2
        total += area if height is None else area*(height(a)+height(b)+height(c))/3
    return total


def _checked(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_ACCUMULATOR_BITS:
        raise _Unqualified("terrain exact accumulator budget exceeded")
    return value


def _as_float(value):
    try:
        out = float(value)
    except OverflowError as exc:
        raise _Unqualified("terrain quantity outside representable output range") from exc
    if not math.isfinite(out) or (value and not out):
        raise _Unqualified("terrain quantity outside representable output range")
    return out


def _measure(layers, old):
    tagged = sorted(((f, k) for k, faces in layers.items() for f in faces),
                    key=lambda fk: fk[0].xmin)
    covered = cut = fill = Q(0)
    x_pairs = clipped = 0
    for i, (a, key_a) in enumerate(tagged):
        for j in range(i+1, len(tagged)):
            b, key_b = tagged[j]
            if b.xmin > a.xmax:
                break
            x_pairs += 1
            if x_pairs > MAX_X_PAIRS:
                raise _Unqualified("terrain broad-phase pair budget exceeded")
            if b.ymin > a.ymax or a.ymin > b.ymax:
                continue
            clipped += 1
            if clipped > MAX_CLIPPED_PAIRS:
                raise _Unqualified("terrain intersection budget exceeded")
            poly = _intersection(a, b)
            area = _integrate(poly)
            if key_a == key_b:
                if area:
                    raise _Unqualified("positive-area overlap inside one terrain layer")
                if any(a.height(p) != b.height(p) for p in poly):
                    raise _Unqualified("terrain elevations disagree at a shared boundary")
            elif area:
                existing, proposed = (a, b) if key_a == old else (b, a)

                def delta(p):
                    return proposed.height(p)-existing.height(p)

                covered = _checked(covered+area)
                fill = _checked(fill+_integrate(_clip(poly, delta), delta))
                cut = _checked(cut-_integrate(_clip(poly, lambda p: -delta(p)), delta))
    return covered, cut, fill


def scan_dxf_terrain_volume(path, existing_layer, proposed_layer):
    """Integrate proposed minus existing on an equal, verified XY footprint."""
    if not all(isinstance(x, str) and x.strip() for x in (existing_layer, proposed_layer)):
        raise ValueError("two non-empty terrain layer names required")
    old, new = existing_layer.casefold(), proposed_layer.casefold()
    if old == new:
        raise ValueError("terrain layers must be distinct")
    filename = Path(path)
    with filename.open("rb") as stream:
        data = stream.read(MAX_BYTES+1)
    if len(data) > MAX_BYTES:
        raise ValueError("terrain DXF byte budget exceeded")
    report = {
        "kind": "dxf-terrain-volume", "schema": "hamon.dxf.terrain-volume.v1",
        "file": str(filename), "source_sha256": sha256(data).hexdigest(),
        "existing_layer": existing_layer, "proposed_layer": proposed_layer,
        "declared_insunits": None, "scale_to_m": None, "qualified": False,
        "common_area_m2": None, "existing_area_m2": None, "proposed_area_m2": None,
        "uncovered_existing_area_m2": None, "uncovered_proposed_area_m2": None,
        "cut_m3": None, "fill_m3": None, "net_fill_m3": None, "reasons": [],
    }
    try:
        units, layers = _read(data, (old, new), report)
        scale = _SURVEY[units] if units in _SURVEY else Q(str(INSUNITS_TO_METERS[units]))
        report["scale_to_m"] = _as_float(scale)
        report["existing_face_count"] = len(layers[old])
        report["proposed_face_count"] = len(layers[new])
        covered, cut, fill = _measure(layers, old)
        areas = {k: sum((f.area for f in faces), Q(0)) for k, faces in layers.items()}
        area_unit = scale**2
        report.update(
            existing_area_m2=_as_float(areas[old]*area_unit),
            proposed_area_m2=_as_float(areas[new]*area_unit),
            common_area_m2=_as_float(covered*area_unit),
            uncovered_existing_area_m2=_as_float((areas[old]-covered)*area_unit),
            uncovered_proposed_area_m2=_as_float((areas[new]-covered)*area_unit),
        )
        if areas[old] != covered or areas[new] != covered:
            raise _Unqualified("incomplete common coverage of terrain footprints")
        volume = scale**3
        report.update(qualified=True, cut_m3=_as_float(cut*volume),
                      fill_m3=_as_float(fill*volume), net_fill_m3=_as_float((fill-cut)*volume))
    except _Unqualified as exc:
        report["reasons"] = [str(exc)]
    return report
