"""3D IFC4/IFC2x3 STEP-P21, STL & OBJ solid geometry + exact spatial clash solver.

Provides zero-dependency 3D solid extraction (extruded rectangular, hollow,
circular, pipe, I-beam/wide-flange, arbitrary closed polyline profiles, and
explicit B-Rep/triangulated face sets), hierarchical SE(3) IFCLOCALPLACEMENT
transforms, divergence-theorem volume/area takeoff, and exact 3D triangle-mesh
intersection + minimum surface-to-surface clearance for Physical-Engineering's
Last Exam (PLE).
"""
from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hamon_preflight import check_deliverable_preflight, run_router_preflight


@dataclass
class SolidMesh3D:
    element_id: str
    ifc_type: str
    name: str
    discipline: str
    vertices: np.ndarray  # (V, 3) float64 in meters
    faces: np.ndarray     # (F, 3) int32
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def aabb_min(self) -> np.ndarray:
        return np.min(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)

    @property
    def aabb_max(self) -> np.ndarray:
        return np.max(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)

    def _local_tetrahedra(self):
        # The tetrahedra share a point on the mesh. Using the global origin
        # makes their signed moments cancel at survey-coordinate magnitudes.
        origin = self.vertices[self.faces[0, 0]]
        local = self.vertices - origin
        a, b, c = (local[self.faces[:, i]] for i in range(3))
        six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))
        return origin, a, b, c, six_volumes

    def volume_m3(self) -> float:
        if len(self.faces) == 0:
            return 0.0
        _, _, _, _, six_volumes = self._local_tetrahedra()
        return float(abs(np.sum(six_volumes)) / 6.0)

    def surface_area_m2(self) -> float:
        if len(self.faces) == 0:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return float(np.sum(areas))

    def centroid_m(self) -> Tuple[float, float, float]:
        if len(self.faces) == 0:
            c = np.mean(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)
            return (float(c[0]), float(c[1]), float(c[2]))
        origin, a, b, c_local, six_volumes = self._local_tetrahedra()
        total = np.sum(six_volumes)
        if abs(total) < 1e-15:
            c = np.mean(self.vertices, axis=0)
        else:
            weighted = (a + b + c_local) * six_volumes[:, None]
            c = origin + np.sum(weighted, axis=0) / (4.0 * total)
        return (float(c[0]), float(c[1]), float(c[2]))


@dataclass
class ClashResult:
    element_a: str
    element_b: str
    status: str  # "hard_clash" | "clearance_violation" | "clear"
    min_distance_m: float
    overlap_aabb_volume_m3: float
    witness_point_m: Tuple[float, float, float]
    required_clearance_m: float = 0.0


_STRUCTURAL_TYPES = {
    "IFCBEAM", "IFCCOLUMN", "IFCSLAB", "IFCWALL", "IFCWALLSTANDARDCASE",
    "IFCFOOTING", "IFCMEMBER", "IFCPLATE", "IFCPILE",
}
# IfcDistributionElement and its IFC2X3, IFC4 and IFC4X3_ADD2 subtypes.
# The STEP reader runs without the optional IfcOpenShell schema library.
_MEP_TYPES = {
    "IFCACTUATOR", "IFCAIRTERMINAL", "IFCAIRTERMINALBOX", "IFCAIRTOAIRHEATRECOVERY",
    "IFCALARM", "IFCAUDIOVISUALAPPLIANCE", "IFCBOILER", "IFCBURNER", "IFCCABLECARRIERFITTING",
    "IFCCABLECARRIERSEGMENT", "IFCCABLEFITTING", "IFCCABLESEGMENT", "IFCCHILLER", "IFCCOIL",
    "IFCCOMMUNICATIONSAPPLIANCE", "IFCCOMPRESSOR", "IFCCONDENSER", "IFCCONTROLLER",
    "IFCCONVEYORSEGMENT", "IFCCOOLEDBEAM", "IFCCOOLINGTOWER", "IFCDAMPER",
    "IFCDISTRIBUTIONBOARD", "IFCDISTRIBUTIONCHAMBERELEMENT", "IFCDISTRIBUTIONCONTROLELEMENT",
    "IFCDISTRIBUTIONELEMENT", "IFCDISTRIBUTIONFLOWELEMENT", "IFCDUCTFITTING", "IFCDUCTSEGMENT",
    "IFCDUCTSILENCER", "IFCELECTRICAPPLIANCE", "IFCELECTRICDISTRIBUTIONBOARD",
    "IFCELECTRICDISTRIBUTIONPOINT", "IFCELECTRICFLOWSTORAGEDEVICE",
    "IFCELECTRICFLOWTREATMENTDEVICE", "IFCELECTRICGENERATOR", "IFCELECTRICMOTOR",
    "IFCELECTRICTIMECONTROL", "IFCENERGYCONVERSIONDEVICE", "IFCENGINE", "IFCEVAPORATIVECOOLER",
    "IFCEVAPORATOR", "IFCFAN", "IFCFILTER", "IFCFIRESUPPRESSIONTERMINAL", "IFCFLOWCONTROLLER",
    "IFCFLOWFITTING", "IFCFLOWINSTRUMENT", "IFCFLOWMETER", "IFCFLOWMOVINGDEVICE",
    "IFCFLOWSEGMENT", "IFCFLOWSTORAGEDEVICE", "IFCFLOWTERMINAL", "IFCFLOWTREATMENTDEVICE",
    "IFCHEATEXCHANGER", "IFCHUMIDIFIER", "IFCINTERCEPTOR", "IFCJUNCTIONBOX", "IFCLAMP",
    "IFCLIGHTFIXTURE", "IFCLIQUIDTERMINAL", "IFCMEDICALDEVICE",
    "IFCMOBILETELECOMMUNICATIONSAPPLIANCE", "IFCMOTORCONNECTION", "IFCOUTLET",
    "IFCPIPEFITTING", "IFCPIPESEGMENT", "IFCPROTECTIVEDEVICE",
    "IFCPROTECTIVEDEVICETRIPPINGUNIT", "IFCPUMP", "IFCSANITARYTERMINAL", "IFCSENSOR",
    "IFCSIGNAL", "IFCSOLARDEVICE", "IFCSPACEHEATER", "IFCSTACKTERMINAL", "IFCSWITCHINGDEVICE",
    "IFCTANK", "IFCTRANSFORMER", "IFCTUBEBUNDLE", "IFCUNITARYCONTROLELEMENT",
    "IFCUNITARYEQUIPMENT", "IFCVALVE", "IFCWASTETERMINAL",
}


def _infer_discipline(ifc_type: str, name: str = "") -> str:
    ut = ifc_type.upper()
    un = name.upper()
    if ut in _MEP_TYPES:
        return "mep"
    if ut in _STRUCTURAL_TYPES or any(k in un for k in ("BEAM", "COL", "SLAB", "GIRDER", "FOOTING")):
        return "structural"
    if any(k in un for k in ("DUCT", "PIPE", "HVAC", "CONDUIT", "SPRINKLER")):
        return "mep"
    return "architectural"


def _strip_step_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _parse_step_args(arg_str: str) -> List[Any]:
    n = len(arg_str)

    def parse_list(idx: int) -> Tuple[List[Any], int]:
        items: List[Any] = []
        cur: List[str] = []
        in_str = False
        while idx < n:
            ch = arg_str[idx]
            if in_str:
                cur.append(ch)
                if ch == "'":
                    if idx + 1 < n and arg_str[idx + 1] == "'":
                        cur.append("'")
                        idx += 2
                        continue
                    in_str = False
                idx += 1
                continue
            if ch == "'":
                in_str = True
                cur.append(ch)
                idx += 1
            elif ch == "(":
                sub, idx = parse_list(idx + 1)
                token_prefix = "".join(cur).strip()
                cur.clear()
                if token_prefix:
                    items.append((token_prefix.upper(), sub))
                else:
                    items.append(sub)
            elif ch == ")":
                tok = "".join(cur).strip()
                if tok:
                    items.append(_coerce_step_atom(tok))
                return items, idx + 1
            elif ch == ",":
                tok = "".join(cur).strip()
                cur.clear()
                # A nested or typed value was already appended at its closing parenthesis.
                if tok:
                    items.append(_coerce_step_atom(tok))
                idx += 1
            else:
                cur.append(ch)
                idx += 1
        tok = "".join(cur).strip()
        if tok:
            items.append(_coerce_step_atom(tok))
        return items, idx

    out, _ = parse_list(0)
    return out


def _coerce_step_atom(tok: str) -> Any:
    if not tok or tok == "$":
        return None
    if tok == "*":
        return "*"
    if tok.startswith("'") and tok.endswith("'"):
        return tok[1:-1].replace("''", "'")
    if tok.startswith("#"):
        try:
            return ("ref", int(tok[1:]))
        except ValueError:
            return tok
    if tok.startswith(".") and tok.endswith("."):
        v = tok[1:-1].upper()
        if v == "T":
            return True
        if v == "F":
            return False
        return v
    try:
        if any(c in tok for c in (".", "E", "e")):
            return float(tok)
        return int(tok)
    except ValueError:
        return tok


def parse_ifc_entities(ifc_text: str) -> Dict[int, Tuple[str, List[Any]]]:
    cleaned = _strip_step_comments(ifc_text)
    m = re.search(r"DATA\s*;(.*?)ENDSEC\s*;", cleaned, flags=re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else cleaned
    entities: Dict[int, Tuple[str, List[Any]]] = {}
    stmts: List[str] = []
    buf: List[str] = []
    in_str = False
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if in_str:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and body[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_str = False
            i += 1
        elif ch == "'":
            in_str = True
            buf.append(ch)
            i += 1
        elif ch == ";":
            s = "".join(buf).strip()
            buf.clear()
            if s:
                stmts.append(s)
            i += 1
        else:
            buf.append(ch)
            i += 1

    stmt_re = re.compile(r"^#(\d+)\s*=\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$", re.DOTALL)
    for st in stmts:
        mm = stmt_re.match(st)
        if not mm:
            continue
        eid = int(mm.group(1))
        etype = mm.group(2).upper()
        args = _parse_step_args(mm.group(3))
        entities[eid] = (etype, args)
    return entities


def _detect_length_unit_scale(entities: Dict[int, Tuple[str, List[Any]]]) -> float:
    prefix_scales = {
        "MILLI": 1e-3,
        "CENTI": 1e-2,
        "DECI": 1e-1,
        "KILO": 1e3,
        "MICRO": 1e-6,
    }
    conv_names = {
        "INCH": 0.0254,
        "FOOT": 0.3048,
        "FEET": 0.3048,
        "YARD": 0.9144,
        "MM": 0.001,
        "CM": 0.01,
    }
    candidate_ids: List[int] = []
    for eid, (etype, args) in entities.items():
        if etype == "IFCUNITASSIGNMENT" and args and isinstance(args[0], list):
            for u in args[0]:
                if isinstance(u, tuple) and u[0] == "ref":
                    candidate_ids.append(u[1])
    if not candidate_ids:
        candidate_ids = list(entities.keys())

    for cid in candidate_ids:
        if cid not in entities:
            continue
        etype, args = entities[cid]
        if etype == "IFCCONVERSIONBASEDUNIT" and len(args) >= 4 and args[1] == "LENGTHUNIT":
            uname = str(args[2] or "").upper()
            if uname in conv_names:
                return conv_names[uname]
            mref = args[3]
            if isinstance(mref, tuple) and mref[0] == "ref" and mref[1] in entities:
                _, margs = entities[mref[1]]
                if margs:
                    val = margs[0]
                    if isinstance(val, tuple) and len(val) == 2 and val[1]:
                        val = val[1][0]
                    if isinstance(val, (int, float)):
                        base_s = 1.0
                        if len(margs) > 1 and isinstance(margs[1], tuple) and margs[1][0] == "ref":
                            bu = entities.get(margs[1][1])
                            if bu and bu[0] == "IFCSIUNIT" and len(bu[1]) >= 4:
                                base_s = prefix_scales.get(str(bu[1][2] or "").upper(), 1.0)
                        return float(val) * base_s
        elif etype == "IFCSIUNIT" and len(args) >= 4 and args[1] == "LENGTHUNIT":
            pref = str(args[2] or "").upper()
            return prefix_scales.get(pref, 1.0)
    return 1.0


def _vec3_from_ref(entities: Dict[int, Tuple[str, List[Any]]], ref: Any, default: Sequence[float]) -> np.ndarray:
    if isinstance(ref, tuple) and ref[0] == "ref" and ref[1] in entities:
        _, args = entities[ref[1]]
        if args and isinstance(args[0], list):
            coords = [float(x) for x in args[0]]
            if len(coords) == 2:
                return np.array([coords[0], coords[1], 0.0], dtype=np.float64)
            if len(coords) >= 3:
                return np.array(coords[:3], dtype=np.float64)
    return np.array(default, dtype=np.float64)


def _axis2placement3d_matrix(entities: Dict[int, Tuple[str, List[Any]]], ref: Any, unit_scale: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if not (isinstance(ref, tuple) and ref[0] == "ref" and ref[1] in entities):
        return mat
    etype, args = entities[ref[1]]
    if etype == "IFCAXIS2PLACEMENT3D":
        loc = _vec3_from_ref(entities, args[0] if len(args) > 0 else None, (0.0, 0.0, 0.0)) * unit_scale
        z_axis = _vec3_from_ref(entities, args[1] if len(args) > 1 else None, (0.0, 0.0, 1.0))
        x_ref = _vec3_from_ref(entities, args[2] if len(args) > 2 else None, (1.0, 0.0, 0.0))
        zn = np.linalg.norm(z_axis)
        z_axis = z_axis / zn if zn > 1e-12 else np.array([0.0, 0.0, 1.0])
        x_proj = x_ref - np.dot(x_ref, z_axis) * z_axis
        xn = np.linalg.norm(x_proj)
        if xn < 1e-12:
            fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_proj = fallback - np.dot(fallback, z_axis) * z_axis
            xn = np.linalg.norm(x_proj)
        x_axis = x_proj / xn
        y_axis = np.cross(z_axis, x_axis)
        mat[:3, 0] = x_axis
        mat[:3, 1] = y_axis
        mat[:3, 2] = z_axis
        mat[:3, 3] = loc
    elif etype == "IFCAXIS2PLACEMENT2D":
        loc = _vec3_from_ref(entities, args[0] if len(args) > 0 else None, (0.0, 0.0, 0.0)) * unit_scale
        x_ref = _vec3_from_ref(entities, args[1] if len(args) > 1 else None, (1.0, 0.0, 0.0))
        xn = np.linalg.norm(x_ref[:2])
        x2 = x_ref[:2] / xn if xn > 1e-12 else np.array([1.0, 0.0])
        y2 = np.array([-x2[1], x2[0]])
        mat[:2, 0] = x2
        mat[:2, 1] = y2
        mat[:3, 3] = np.array([loc[0], loc[1], 0.0])
    return mat


def _resolve_local_placement(entities: Dict[int, Tuple[str, List[Any]]], ref: Any, unit_scale: float, depth: int = 0) -> np.ndarray:
    if depth > 32 or not (isinstance(ref, tuple) and ref[0] == "ref" and ref[1] in entities):
        return np.eye(4, dtype=np.float64)
    etype, args = entities[ref[1]]
    if etype != "IFCLOCALPLACEMENT":
        return _axis2placement3d_matrix(entities, ref, unit_scale)
    parent = _resolve_local_placement(entities, args[0] if len(args) > 0 else None, unit_scale, depth + 1)
    rel = _axis2placement3d_matrix(entities, args[1] if len(args) > 1 else None, unit_scale)
    return parent @ rel


def _signed_area_2d(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_tri_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-15:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-10 and v >= -1e-10 and (u + v) <= 1.0 + 1e-10


def _triangulate_simple_polygon_2d(pts: np.ndarray) -> List[Tuple[int, int, int]]:
    n = len(pts)
    if n < 3:
        return []
    idxs = list(range(n))
    if _signed_area_2d(pts) < 0:
        idxs.reverse()
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idxs) > 3 and guard < n * n + 10:
        guard += 1
        ear_found = False
        m = len(idxs)
        for k in range(m):
            i0 = idxs[(k - 1) % m]
            i1 = idxs[k]
            i2 = idxs[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:
                continue
            inside = False
            for j in idxs:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri_2d(pts[j], a, b, c):
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idxs.pop(k)
                ear_found = True
                break
        if not ear_found:
            for k in range(1, len(idxs) - 1):
                tris.append((idxs[0], idxs[k], idxs[k + 1]))
            return tris
    if len(idxs) == 3:
        tris.append((idxs[0], idxs[1], idxs[2]))
    return tris


def _extrude_polygon_2d(outer_pts: np.ndarray, dir_vec: np.ndarray, depth: float, inner_pts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if _signed_area_2d(outer_pts) < 0:
        outer_pts = outer_pts[::-1].copy()
    dn = np.linalg.norm(dir_vec)
    d = (dir_vec / dn if dn > 1e-12 else np.array([0.0, 0.0, 1.0])) * depth

    if inner_pts is not None and len(inner_pts) >= 3:
        if _signed_area_2d(inner_pts) < 0:
            inner_pts = inner_pts[::-1].copy()
        if len(outer_pts) == len(inner_pts):
            k = len(outer_pts)
            b_out = np.column_stack([outer_pts, np.zeros(k)])
            b_in = np.column_stack([inner_pts, np.zeros(k)])
            t_out = b_out + d
            t_in = b_in + d
            verts = np.vstack([b_out, b_in, t_out, t_in])
            faces: List[Tuple[int, int, int]] = []
            for i in range(k):
                nxt = (i + 1) % k
                bo0, bo1 = i, nxt
                bi0, bi1 = k + i, k + nxt
                to0, to1 = 2 * k + i, 2 * k + nxt
                ti0, ti1 = 3 * k + i, 3 * k + nxt
                faces.append((bo0, bi1, bo1))
                faces.append((bo0, bi0, bi1))
                faces.append((to0, to1, ti1))
                faces.append((to0, ti1, ti0))
                faces.append((bo0, bo1, to1))
                faces.append((bo0, to1, to0))
                faces.append((bi0, ti1, bi1))
                faces.append((bi0, ti0, ti1))
            if d[2] < 0:
                faces = [(a, c, b) for (a, b, c) in faces]
            return verts, np.array(faces, dtype=np.int32)

    k = len(outer_pts)
    cap_tris = _triangulate_simple_polygon_2d(outer_pts)
    bottom = np.column_stack([outer_pts, np.zeros(k)])
    top = bottom + d
    verts = np.vstack([bottom, top])
    faces = []
    for (a, b, c) in cap_tris:
        faces.append((a, c, b))
        faces.append((k + a, k + b, k + c))
    for i in range(k):
        nxt = (i + 1) % k
        faces.append((i, nxt, k + nxt))
        faces.append((i, k + nxt, k + i))
    if d[2] < 0:
        faces = [(a, c, b) for (a, b, c) in faces]
    return verts, np.array(faces, dtype=np.int32)


def _profile_polygons_2d(entities: Dict[int, Tuple[str, List[Any]]], pref: Any, unit_scale: float, circle_segments: int = 64) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    if not (isinstance(pref, tuple) and pref[0] == "ref" and pref[1] in entities):
        return None, None, np.eye(4, dtype=np.float64)
    ptype, args = entities[pref[1]]
    pos_mat = _axis2placement3d_matrix(entities, args[2] if len(args) > 2 else None, unit_scale)

    if ptype == "IFCRECTANGLEPROFILEDEF" and len(args) >= 5:
        hx = 0.5 * float(args[3]) * unit_scale
        hy = 0.5 * float(args[4]) * unit_scale
        outer = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
        return outer, None, pos_mat

    if ptype == "IFCRECTANGLEHOLLOWPROFILEDEF" and len(args) >= 6:
        hx = 0.5 * float(args[3]) * unit_scale
        hy = 0.5 * float(args[4]) * unit_scale
        wt = float(args[5]) * unit_scale
        outer = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
        ix, iy = max(hx - wt, 1e-6), max(hy - wt, 1e-6)
        inner = np.array([[-ix, -iy], [ix, -iy], [ix, iy], [-ix, iy]], dtype=np.float64)
        return outer, inner, pos_mat

    if ptype == "IFCCIRCLEPROFILEDEF" and len(args) >= 4:
        r = float(args[3]) * unit_scale
        ang = np.linspace(0.0, 2.0 * math.pi, max(circle_segments, 64), endpoint=False)
        outer = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        return outer, None, pos_mat

    if ptype == "IFCCIRCLEHOLLOWPROFILEDEF" and len(args) >= 5:
        r = float(args[3]) * unit_scale
        wt = float(args[4]) * unit_scale
        ri = max(r - wt, 1e-6)
        ang = np.linspace(0.0, 2.0 * math.pi, max(circle_segments, 64), endpoint=False)
        outer = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        inner = np.column_stack([ri * np.cos(ang), ri * np.sin(ang)])
        return outer, inner, pos_mat

    if ptype == "IFCISHAPEPROFILEDEF" and len(args) >= 7:
        w = float(args[3]) * unit_scale
        d = float(args[4]) * unit_scale
        tw = float(args[5]) * unit_scale
        tf = float(args[6]) * unit_scale
        hw, hd, htw = 0.5 * w, 0.5 * d, 0.5 * tw
        outer = np.array([
            [-hw, -hd], [hw, -hd], [hw, -hd + tf], [htw, -hd + tf],
            [htw, hd - tf], [hw, hd - tf], [hw, hd], [-hw, hd],
            [-hw, hd - tf], [-htw, hd - tf], [-htw, -hd + tf], [-hw, -hd + tf],
        ], dtype=np.float64)
        return outer, None, pos_mat

    if ptype == "IFCARBITRARYCLOSEDPROFILEDEF" and len(args) >= 3:
        pos_mat = np.eye(4, dtype=np.float64)
        cref = args[2]
        if isinstance(cref, tuple) and cref[0] == "ref" and cref[1] in entities:
            ctype, cargs = entities[cref[1]]
            if ctype == "IFCPOLYLINE" and cargs and isinstance(cargs[0], list):
                pts = []
                for pref_pt in cargs[0]:
                    v = _vec3_from_ref(entities, pref_pt, (0.0, 0.0, 0.0)) * unit_scale
                    pts.append([v[0], v[1]])
                if len(pts) >= 2 and np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 1e-9:
                    pts.pop()
                if len(pts) >= 3:
                    return np.array(pts, dtype=np.float64), None, pos_mat
            elif ctype == "IFCCARTESIANPOINTLIST2D" and cargs and isinstance(cargs[0], list):
                pts = [[float(p[0]) * unit_scale, float(p[1]) * unit_scale] for p in cargs[0] if len(p) >= 2]
                if len(pts) >= 2 and np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 1e-9:
                    pts.pop()
                if len(pts) >= 3:
                    return np.array(pts, dtype=np.float64), None, pos_mat
            elif ctype == "IFCINDEXEDPOLYCURVE" and cargs:
                lref = cargs[0]
                if isinstance(lref, tuple) and lref[0] == "ref" and lref[1] in entities:
                    _, largs = entities[lref[1]]
                    if largs and isinstance(largs[0], list):
                        pts = [[float(p[0]) * unit_scale, float(p[1]) * unit_scale] for p in largs[0] if len(p) >= 2]
                        if len(pts) >= 2 and np.linalg.norm(np.array(pts[0]) - np.array(pts[-1])) < 1e-9:
                            pts.pop()
                        if len(pts) >= 3:
                            return np.array(pts, dtype=np.float64), None, pos_mat
    return None, None, pos_mat


def _apply_transform(verts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if len(verts) == 0:
        return verts
    ones = np.ones((len(verts), 1), dtype=np.float64)
    hom = np.hstack([verts, ones])
    return (mat @ hom.T).T[:, :3]


def _mapped_operator_3d(entities, ref, unit_scale):
    """Read inherited IFC operator attributes in STEP order and construct its orthogonal basis."""
    if not isinstance(ref, tuple) or ref[0] != "ref" or ref[1] not in entities:
        return None
    operator, args = entities[ref[1]]
    if operator not in ("IFCCARTESIANTRANSFORMATIONOPERATOR3D",
                        "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM") or len(args) < 5:
        return None
    axis1, axis2, origin, uniform_scale, axis3 = args[:5]
    z = _vec3_from_ref(entities, axis3, (0, 0, 1))
    norm = np.linalg.norm(z)
    if norm <= 1e-12:
        return None
    z /= norm
    default_x = (0, 1, 0) if abs(z[0]) > 1 - 1e-12 else (1, 0, 0)
    x = _vec3_from_ref(entities, axis1, default_x)
    x -= np.dot(x, z) * z
    norm = np.linalg.norm(x)
    if norm <= 1e-12:
        return None
    x /= norm
    y = np.cross(z, x)
    if np.dot(y, _vec3_from_ref(entities, axis2, (0, 1, 0))) < 0:
        y = -y
    scales = [float(uniform_scale) if uniform_scale is not None else 1.0] * 3
    if operator == "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM":
        for i, value in enumerate(args[5:7], start=1):
            if value is not None:
                scales[i] = float(value)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x, y, z)) * scales
    matrix[:3, 3] = _vec3_from_ref(entities, origin, (0, 0, 0)) * unit_scale
    return matrix


class UnresolvedSolidGeometry(ValueError):
    """A physical Body was present but could not supply verified solid geometry."""


def _polygonal_solid_mesh(entities, args, unit_scale):
    """Tessellate bounded planar regions; preserve all holes and oriented shell edges."""
    from collections import Counter
    try:
        from shapely import Polygon, constrained_delaunay_triangles
        from shapely.errors import GEOSException
    except ImportError as exc:
        raise UnresolvedSolidGeometry("polygonal Body requires constrained polygon triangulation") from exc

    def fail(reason):
        raise UnresolvedSolidGeometry("polygonal Body: " + reason)

    def reference(ref, accepted):
        if not isinstance(ref, tuple) or ref[0] != "ref" or ref[1] not in entities:
            fail("missing referenced IFC entity")
        kind, attrs = entities[ref[1]]
        if kind not in accepted:
            fail("unexpected referenced " + kind)
        return kind, attrs

    if len(args) < 3 or args[1] is False or not args[2]:
        fail("open or empty face set cannot establish solid quantities")
    _, points = reference(args[0], {"IFCCARTESIANPOINTLIST3D"})
    if not points:
        fail("missing point list")
    try:
        vertices = np.asarray(points[0], dtype=np.float64) * unit_scale
    except (ValueError, TypeError) as exc:
        raise UnresolvedSolidGeometry("malformed polygonal point list") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        fail("non-finite or non-3D point list")
    # STEP PnIndex, when present, maps both outer and inner loop indices.
    index_map = args[3] if len(args) > 3 and args[3] is not None else range(1, len(vertices) + 1)
    if not isinstance(index_map, (list, range)) or any(type(i) is not int or not 1 <= i <= len(vertices) for i in index_map):
        fail("invalid point-index indirection")
    def indices(source):
        if not isinstance(source, list) or len(source) < 3 or any(type(i) is not int or not 1 <= i <= len(index_map) for i in source):
            fail("invalid face vertex indices")
        return [index_map[i - 1] - 1 for i in source]

    triangles = []
    directed = Counter()
    for ref in args[2]:
        kind, bounds = reference(ref, {"IFCINDEXEDPOLYGONALFACE", "IFCINDEXEDPOLYGONALFACEWITHVOIDS"})
        if not bounds:
            fail("missing exterior ring")
        rings = [indices(bounds[0])]
        if kind == "IFCINDEXEDPOLYGONALFACEWITHVOIDS":
            if len(bounds) < 2 or not isinstance(bounds[1], list) or not bounds[1]:
                fail("missing inner rings")
            rings.extend(indices(loop) for loop in bounds[1])

        # Coordinates are shifted before projection to keep survey origin out of the fit.
        exterior = vertices[rings[0]] - vertices[rings[0][0]]
        normal = np.cross(exterior, np.roll(exterior, -1, axis=0)).sum(axis=0)
        norm = np.linalg.norm(normal)
        if norm == 0 or not np.isfinite(norm):
            fail("degenerate exterior ring")
        normal /= norm
        omit = int(np.argmax(np.abs(normal)))
        axes = [(omit + 1) % 3, (omit + 2) % 3]
        faces_2d = []
        vertex_for = {}
        for ring in rings:
            local = vertices[ring] - vertices[rings[0][0]]
            span = np.max(np.abs(local))
            if np.max(np.abs(local @ normal)) > 1e-9 * max(1., span):
                fail("non-planar polygonal face")
            projected = local[:, axes]
            area = _signed_area_2d(projected)
            if (area * normal[omit] > 0) != (len(faces_2d) == 0) or area == 0:
                fail("inconsistent ring winding")
            for uv, idx in zip(projected, ring):
                key = tuple(uv)
                if key in vertex_for:
                    fail("repeated or overlapping boundary vertex")
                vertex_for[key] = idx
            faces_2d.append(projected)
        try:
            polygon = Polygon(faces_2d[0], faces_2d[1:])
            if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                fail("invalid polygon or inner ring nesting")
            tessellation = constrained_delaunay_triangles(polygon)
            area_sum = 0.
            for triangle in tessellation.geoms:
                coords = list(triangle.exterior.coords)[:-1]
                if len(coords) != 3 or any(c not in vertex_for for c in coords) or not polygon.covers(triangle):
                    fail("constrained tessellation escaped polygon boundaries")
                area_sum += triangle.area
                tri = [vertex_for[c] for c in coords]
                if _signed_area_2d(np.asarray(coords)) * normal[omit] < 0:
                    tri.reverse()
                triangles.append(tri)
                for a, b in zip(tri, tri[1:] + tri[:1]):
                    directed[(tuple(vertices[a]), tuple(vertices[b]))] += 1
            if not math.isclose(area_sum, polygon.area, rel_tol=1e-10, abs_tol=0.):
                fail("incomplete constrained triangulation")
        except (GEOSException, ValueError) as exc:
            if isinstance(exc, UnresolvedSolidGeometry):
                raise
            raise UnresolvedSolidGeometry("invalid planar polygonal Body") from exc
    if not triangles or any(count != 1 or directed[(b, a)] != 1 for (a, b), count in directed.items()):
        fail("shell is open, non-manifold or inconsistently oriented")
    # Drop unreferenced coordinates: PnIndex may select only one part of a shared list.
    used, inverse = np.unique(np.asarray(triangles, dtype=np.int32), return_inverse=True)
    return [(vertices[used], inverse.reshape(-1, 3).astype(np.int32))]


def _extract_solid_item(entities: Dict[int, Tuple[str, List[Any]]], iref: Any, unit_scale: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not (isinstance(iref, tuple) and iref[0] == "ref" and iref[1] in entities):
        return []
    itype, args = entities[iref[1]]
    if itype == "IFCEXTRUDEDAREASOLID" and len(args) >= 4:
        outer, inner, prof_mat = _profile_polygons_2d(entities, args[0], unit_scale)
        if outer is None:
            return []
        solid_mat = _axis2placement3d_matrix(entities, args[1], unit_scale)
        ext_dir = _vec3_from_ref(entities, args[2], (0.0, 0.0, 1.0))
        depth = float(args[3]) * unit_scale
        verts, faces = _extrude_polygon_2d(outer, ext_dir, depth, inner_pts=inner)
        verts = _apply_transform(verts, solid_mat @ prof_mat)
        return [(verts, faces)]

    if itype == "IFCMAPPEDITEM" and len(args) >= 2:
        sref, tref = args[0], args[1]
        map_mat = _mapped_operator_3d(entities, tref, unit_scale)
        if map_mat is None:
            return []
        out_items: List[Tuple[np.ndarray, np.ndarray]] = []
        if isinstance(sref, tuple) and sref[0] == "ref" and sref[1] in entities:
            _, sargs = entities[sref[1]]
            if len(sargs) >= 2:
                orig_mat = _axis2placement3d_matrix(entities, sargs[0], unit_scale)
                rep_ref = sargs[1]
                if isinstance(rep_ref, tuple) and rep_ref[0] == "ref" and rep_ref[1] in entities:
                    _, rargs = entities[rep_ref[1]]
                    if len(rargs) >= 4 and isinstance(rargs[3], list):
                        for sub in rargs[3]:
                            sub_meshes = _extract_solid_item(entities, sub, unit_scale)
                            if not sub_meshes:
                                raise UnresolvedSolidGeometry("unresolved physical item in mapped representation")
                            for (vv, ff) in sub_meshes:
                                out_items.append((_apply_transform(vv, map_mat @ orig_mat), ff))
        return out_items

    if itype == "IFCTRIANGULATEDFACESET" and len(args) >= 4:
        cref = args[0]
        if isinstance(cref, tuple) and cref[0] == "ref" and cref[1] in entities:
            _, cargs = entities[cref[1]]
            if cargs and isinstance(cargs[0], list):
                verts = np.array([[float(c) * unit_scale for c in pt] for pt in cargs[0]], dtype=np.float64)
                faces = np.array([[int(idx) - 1 for idx in tri] for tri in args[3]], dtype=np.int32)
                return [(verts, faces)]

    if itype == "IFCPOLYGONALFACESET":
        return _polygonal_solid_mesh(entities, args, unit_scale)

    if itype == "IFCFACETEDBREP" and len(args) >= 1:
        sref = args[0]
        if isinstance(sref, tuple) and sref[0] == "ref" and sref[1] in entities:
            _, sargs = entities[sref[1]]
            if sargs and isinstance(sargs[0], list):
                vert_map: Dict[int, int] = {}
                verts_list: List[np.ndarray] = []
                faces_list: List[Tuple[int, int, int]] = []
                for fref in sargs[0]:
                    if not (isinstance(fref, tuple) and fref[0] == "ref" and fref[1] in entities):
                        continue
                    _, fargs = entities[fref[1]]
                    if not (fargs and isinstance(fargs[0], list)):
                        continue
                    for bref in fargs[0]:
                        if not (isinstance(bref, tuple) and bref[0] == "ref" and bref[1] in entities):
                            continue
                        _, bargs = entities[bref[1]]
                        if not bargs:
                            continue
                        lref = bargs[0]
                        ori = bool(bargs[1]) if len(bargs) > 1 and bargs[1] is not None else True
                        if not (isinstance(lref, tuple) and lref[0] == "ref" and lref[1] in entities):
                            continue
                        _, largs = entities[lref[1]]
                        if not (largs and isinstance(largs[0], list)):
                            continue
                        loop_idxs: List[int] = []
                        for p_ref in largs[0]:
                            if isinstance(p_ref, tuple) and p_ref[0] == "ref":
                                pid = p_ref[1]
                                if pid not in vert_map:
                                    vert_map[pid] = len(verts_list)
                                    verts_list.append(_vec3_from_ref(entities, p_ref, (0.0, 0.0, 0.0)) * unit_scale)
                                loop_idxs.append(vert_map[pid])
                        if not ori:
                            loop_idxs.reverse()
                        for k in range(1, len(loop_idxs) - 1):
                            faces_list.append((loop_idxs[0], loop_idxs[k], loop_idxs[k + 1]))
                if verts_list and faces_list:
                    return [(np.array(verts_list, dtype=np.float64), np.array(faces_list, dtype=np.int32))]
    return []


class UnresolvedOpening(ValueError):
    """The tessellation does not establish the net host geometry."""


def _box_bounds(vertices, faces):
    """Prove these twelve triangles tile all six sides of one cuboid."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    extent = hi-lo
    tol = max(1e-10, float(extent.max())*1e-10)
    if min(extent) <= 4*tol or faces.shape != (12,3):
        raise UnresolvedOpening("opening subtraction requires a verified cuboid")
    bits = np.abs(vertices-hi) <= tol
    if not np.all(bits | (np.abs(vertices-lo) <= tol)):
        raise UnresolvedOpening("opening or host is not aligned to the host frame")
    tiles = {}
    for face in bits[faces]:
        fixed = np.where(np.all(face == face[0],axis=0))[0]
        if len(fixed) != 1:
            raise UnresolvedOpening("degenerate or internal cuboid triangle")
        axis = fixed[0]
        face_key = (axis, bool(face[0,axis]))
        projected = {tuple(np.delete(point,axis)) for point in face}
        tiles.setdefault(face_key,[]).append(projected)
    if len(tiles) != 6:
        raise UnresolvedOpening("incomplete cuboid boundary")
    for side in tiles.values():
        if len(side) != 2 or any(len(tri) != 3 for tri in side):
            raise UnresolvedOpening("cuboid side lacks two triangles")
        diagonal = list(side[0] & side[1])
        if len(diagonal) != 2 or not all(a != b for a,b in zip(*diagonal)):
            raise UnresolvedOpening("cuboid triangles must share an interior diagonal")
    return lo, hi


def _subtract_opening_boxes(host, openings, frame):
    # Work in the host placement, retaining the original SI/world registration.
    origin, basis = frame[:3,3], frame[:3,:3]
    def bounds(mesh):
        return _box_bounds((mesh.vertices-origin) @ basis, mesh.faces)
    lo, hi = bounds(host)
    boxes = []
    for opening in openings:
        low, high = bounds(opening)
        low, high = np.maximum(lo,low), np.minimum(hi,high)
        if np.all(low < high):
            boxes.append((low,high))
    if not boxes:
        return host
    axes = [sorted({lo[d],hi[d]} | {corner[d] for box in boxes for corner in box})
            for d in range(3)]
    shape = tuple(len(axis)-1 for axis in axes)
    if math.prod(shape) > 64000:
        raise UnresolvedOpening("opening union exceeds bounded cell count")
    material = np.ones(shape,dtype=bool)
    for low,high in boxes:
        region = tuple(slice(axes[d].index(low[d]),axes[d].index(high[d])) for d in range(3))
        material[region] = False
    vertices = []
    indices = {}
    triangles = []
    for cell in zip(*np.nonzero(material)):
        for d in range(3):
            u,v = (d+1)%3,(d+2)%3
            for side in (0,1):
                neighbor = list(cell)
                neighbor[d] += 2*side-1
                if 0 <= neighbor[d] < shape[d] and material[tuple(neighbor)]:
                    continue
                quad = []
                for du,dv in ((0,0),(1,0),(1,1),(0,1)):
                    corner = list(cell)
                    corner[d] += side
                    corner[u] += du
                    corner[v] += dv
                    key = tuple(corner)
                    if key not in indices:
                        indices[key] = len(vertices)
                        vertices.append([axes[i][corner[i]] for i in range(3)])
                    quad.append(indices[key])
                if side == 0:
                    quad.reverse()
                triangles.extend(((quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3])))
    if not triangles:
        return None
    return SolidMesh3D(host.element_id,host.ifc_type,host.name,host.discipline,
                       np.asarray(vertices) @ basis.T + origin,np.asarray(triangles,dtype=np.int32),
                       {**host.metadata,"opening_ids":[opening.element_id for opening in openings]})


def _require_single_swept_body(entities, step):
    """Refuse partial bodies and curves the vertex reader cannot certify."""
    def resolve(ref):
        if not isinstance(ref, tuple) or ref[0] != "ref" or ref[1] not in entities:
            raise UnresolvedOpening("missing source geometry for opening subtraction")
        return entities[ref[1]]
    _, shape = resolve(entities[step][1][6])
    items = []
    for ref in shape[2]:
        kind, rep = resolve(ref)
        if kind != "IFCSHAPEREPRESENTATION" or len(rep) < 4:
            raise UnresolvedOpening("unverified opening representation")
        if str(rep[1]).upper() in ("AXIS","BOX","FOOTPRINT"):
            continue
        if str(rep[1]).upper() != "BODY" or not isinstance(rep[3], list):
            raise UnresolvedOpening("unverified opening body")
        items.extend(rep[3])
    if len(items) != 1:
        raise UnresolvedOpening("expected one complete swept opening or host body")
    kind, solid = resolve(items[0])
    if kind != "IFCEXTRUDEDAREASOLID":
        raise UnresolvedOpening("only direct cuboid extrusions are qualified")
    profile, attrs = resolve(solid[0])
    if profile == "IFCRECTANGLEPROFILEDEF":
        return
    if profile == "IFCARBITRARYCLOSEDPROFILEDEF":
        curve, args = resolve(attrs[2])
        if curve == "IFCPOLYLINE" and len(args[0]) == 5 and args[0][0] == args[0][-1]:
            if all(resolve(ref)[0] == "IFCCARTESIANPOINT" for ref in args[0]):
                return
    raise UnresolvedOpening("unsupported swept opening or host profile")


def _opening_relationships(entities):
    related = {}
    for kind,args in entities.values():
        if kind != "IFCRELVOIDSELEMENT":
            continue
        if len(args) < 6 or any(not isinstance(ref,tuple) or ref[0]!="ref" for ref in args[4:6]):
            raise UnresolvedOpening("malformed IFC opening relationship")
        host, opening = args[4][1],args[5][1]
        related.setdefault(host,set()).add(opening)
    return related


def _physical_occurrences(solids,entities,unit_scale,related):
    by_step = {s.metadata["step_id"]:s for s in solids}
    for host,openings in related.items():
        if host not in by_step or any(i not in by_step or
                by_step[i].ifc_type not in ("IFCOPENINGELEMENT","IFCOPENINGSTANDARDCASE")
                for i in openings):
            raise UnresolvedOpening("missing or unsupported host/opening body")
        frame = _resolve_local_placement(entities,entities[host][1][5],unit_scale)
        by_step[host] = _subtract_opening_boxes(by_step[host],[by_step[i] for i in sorted(openings)],frame)
    return [s for s in by_step.values() if s is not None and
            s.ifc_type not in ("IFCOPENINGELEMENT","IFCOPENINGSTANDARDCASE")]


def load_ifc_solids(ifc_text: str) -> List[SolidMesh3D]:
    entities = parse_ifc_entities(ifc_text)
    unit_scale = _detect_length_unit_scale(entities)
    solids: List[SolidMesh3D] = []
    related = _opening_relationships(entities)
    participants = set(related).union(*related.values())

    for eid, (etype, args) in sorted(entities.items()):
        if len(args) < 7 or not etype.startswith("IFC"):
            continue
        rep_ref = args[6]
        if not (isinstance(rep_ref, tuple) and rep_ref[0] == "ref" and rep_ref[1] in entities):
            continue
        rtype, rargs = entities[rep_ref[1]]
        if rtype != "IFCPRODUCTDEFINITIONSHAPE" or len(rargs) < 3 or not isinstance(rargs[2], list):
            continue

        if eid in participants:
            _require_single_swept_body(entities, eid)
        global_id = str(args[0] or f"#{eid}")
        name = str(args[2] or f"{etype}_{eid}")
        world_mat = _resolve_local_placement(entities, args[5], unit_scale)

        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        v_offset = 0
        for srep in rargs[2]:
            if not (isinstance(srep, tuple) and srep[0] == "ref" and srep[1] in entities):
                continue
            sr_type, sr_args = entities[srep[1]]
            if sr_type != "IFCSHAPEREPRESENTATION" or len(sr_args) < 4 or not isinstance(sr_args[3], list):
                continue
            rep_ident = str(sr_args[1] or "").upper()
            if rep_ident in ("AXIS", "FOOTPRINT", "BOX"):
                continue
            for item_ref in sr_args[3]:
                item_meshes = _extract_solid_item(entities, item_ref, unit_scale)
                if rep_ident == "BODY" and not item_meshes:
                    raise UnresolvedSolidGeometry(f"unsupported or empty physical Body item on #{eid}")
                for (v_local, f_local) in item_meshes:
                    v_world = _apply_transform(v_local, world_mat)
                    all_verts.append(v_world)
                    all_faces.append(f_local + v_offset)
                    v_offset += len(v_world)

        if all_verts and all_faces:
            solids.append(SolidMesh3D(
                element_id=global_id,
                ifc_type=etype,
                name=name,
                discipline=_infer_discipline(etype, name),
                vertices=np.vstack(all_verts),
                faces=np.vstack(all_faces),
                metadata={"step_id": eid, "unit_scale_m": unit_scale},
            ))
    return _physical_occurrences(solids,entities,unit_scale,related)


def parse_stl_mesh(data: bytes | str, element_id: str = "stl_0", name: str = "stl_part", scale_to_m: float = 1.0) -> SolidMesh3D:
    if isinstance(data, str):
        raw_bytes = data.encode("utf-8")
        is_ascii = True
    else:
        raw_bytes = data
        if len(raw_bytes) >= 84:
            tri_count = struct.unpack("<I", raw_bytes[80:84])[0]
            is_ascii = (84 + 50 * tri_count != len(raw_bytes))
        else:
            is_ascii = True

    tris: List[Tuple[Tuple[float, float, float], ...]] = []
    if not is_ascii:
        tri_count = struct.unpack("<I", raw_bytes[80:84])[0]
        offset = 84
        for _ in range(tri_count):
            vals = struct.unpack("<12fH", raw_bytes[offset:offset + 50])
            offset += 50
            v0 = (vals[3] * scale_to_m, vals[4] * scale_to_m, vals[5] * scale_to_m)
            v1 = (vals[6] * scale_to_m, vals[7] * scale_to_m, vals[8] * scale_to_m)
            v2 = (vals[9] * scale_to_m, vals[10] * scale_to_m, vals[11] * scale_to_m)
            tris.append((v0, v1, v2))
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
        cur_vs: List[Tuple[float, float, float]] = []
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith("vertex"):
                parts = s.split()
                if len(parts) >= 4:
                    cur_vs.append((float(parts[1]) * scale_to_m, float(parts[2]) * scale_to_m, float(parts[3]) * scale_to_m))
                    if len(cur_vs) == 3:
                        tris.append((cur_vs[0], cur_vs[1], cur_vs[2]))
                        cur_vs.clear()

    vert_lookup: Dict[Tuple[int, int, int], int] = {}
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for tri in tris:
        f_idx = []
        for v in tri:
            key = (round(v[0] * 1e9), round(v[1] * 1e9), round(v[2] * 1e9))
            if key not in vert_lookup:
                vert_lookup[key] = len(verts)
                verts.append(v)
            f_idx.append(vert_lookup[key])
        faces.append((f_idx[0], f_idx[1], f_idx[2]))
    return SolidMesh3D(
        element_id=element_id,
        ifc_type="STL_SOLID",
        name=name,
        discipline=_infer_discipline("STL_SOLID", name),
        vertices=np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3)),
        faces=np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32),
    )


def parse_obj_mesh(obj_text: str, element_id: str = "obj_0", name: str = "obj_part", scale_to_m: float = 1.0) -> SolidMesh3D:
    verts: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []
    for line in obj_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(parts[1]) * scale_to_m, float(parts[2]) * scale_to_m, float(parts[3]) * scale_to_m])
        elif parts[0] == "f" and len(parts) >= 4:
            idxs = []
            for tok in parts[1:]:
                vi = int(tok.split("/")[0])
                idxs.append(vi - 1 if vi > 0 else len(verts) + vi)
            for k in range(1, len(idxs) - 1):
                faces.append((idxs[0], idxs[k], idxs[k + 1]))
    return SolidMesh3D(
        element_id=element_id,
        ifc_type="OBJ_SOLID",
        name=name,
        discipline=_infer_discipline("OBJ_SOLID", name),
        vertices=np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3)),
        faces=np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32),
    )


def _segment_pierces_triangle(p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Optional[np.ndarray]:
    d = p1 - p0
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-12:
        return None
    f = 1.0 / a
    s = p0 - v0
    u = f * float(np.dot(s, h))
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(d, q))
    if v < -1e-9 or (u + v) > 1.0 + 1e-9:
        return None
    t = f * float(np.dot(e2, q))
    if -1e-9 <= t <= 1.0 + 1e-9:
        return p0 + max(0.0, min(1.0, t)) * d
    return None


def _triangles_intersect_3d(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Optional[np.ndarray]:
    amin = np.minimum(np.minimum(a0, a1), a2)
    amax = np.maximum(np.maximum(a0, a1), a2)
    bmin = np.minimum(np.minimum(b0, b1), b2)
    bmax = np.maximum(np.maximum(b0, b1), b2)
    if np.any(amax < bmin - 1e-9) or np.any(bmax < amin - 1e-9):
        return None
    for (p0, p1) in ((a0, a1), (a1, a2), (a2, a0)):
        hit = _segment_pierces_triangle(p0, p1, b0, b1, b2)
        if hit is not None:
            return hit
    for (p0, p1) in ((b0, b1), (b1, b2), (b2, b0)):
        hit = _segment_pierces_triangle(p0, p1, a0, a1, a2)
        if hit is not None:
            return hit
    return None


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _segment_segment_dist_3d(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> Tuple[float, np.ndarray]:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a <= 1e-15 and e <= 1e-15:
        return float(np.linalg.norm(p1 - p2)), 0.5 * (p1 + p2)
    if a <= 1e-15:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-15:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > 1e-15 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return float(np.linalg.norm(c1 - c2)), 0.5 * (c1 + c2)


def _triangle_triangle_distance_3d(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Tuple[float, np.ndarray]:
    hit = _triangles_intersect_3d(a0, a1, a2, b0, b1, b2)
    if hit is not None:
        return 0.0, hit
    best_d = float("inf")
    best_pt = a0
    for p in (a0, a1, a2):
        q = _closest_point_on_triangle(p, b0, b1, b2)
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best_pt = 0.5 * (p + q)
    for p in (b0, b1, b2):
        q = _closest_point_on_triangle(p, a0, a1, a2)
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best_pt = 0.5 * (p + q)
    for (p1, q1) in ((a0, a1), (a1, a2), (a2, a0)):
        for (p2, q2) in ((b0, b1), (b1, b2), (b2, b0)):
            d, mid = _segment_segment_dist_3d(p1, q1, p2, q2)
            if d < best_d:
                best_d = d
                best_pt = mid
    return best_d, best_pt


def _point_inside_mesh(pt: np.ndarray, mesh: SolidMesh3D) -> bool:
    if len(mesh.faces) == 0:
        return False
    if np.any(pt < mesh.aabb_min - 1e-9) or np.any(pt > mesh.aabb_max + 1e-9):
        return False
    ray_dir = np.array([0.931723, 0.271828, 0.240719], dtype=np.float64)
    ray_dir /= np.linalg.norm(ray_dir)
    far = pt + ray_dir * (float(np.linalg.norm(mesh.aabb_max - mesh.aabb_min)) * 4.0 + 10.0)
    hits = 0
    v = mesh.vertices
    for f in mesh.faces:
        v0, v1, v2 = v[f[0]], v[f[1]], v[f[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        d = far - pt
        h = np.cross(d, e2)
        a = float(np.dot(e1, h))
        if abs(a) < 1e-13:
            continue
        inv_a = 1.0 / a
        s = pt - v0
        u = inv_a * float(np.dot(s, h))
        if u < 0.0 or u > 1.0:
            continue
        q = np.cross(s, e1)
        vv = inv_a * float(np.dot(d, q))
        if vv < 0.0 or (u + vv) > 1.0:
            continue
        t = inv_a * float(np.dot(e2, q))
        if 1e-9 < t <= 1.0:
            hits += 1
    return (hits % 2) == 1


def _aabb_overlap_volume(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    ox = max(0.0, float(min(max_a[0], max_b[0]) - max(min_a[0], min_b[0])))
    oy = max(0.0, float(min(max_a[1], max_b[1]) - max(min_a[1], min_b[1])))
    oz = max(0.0, float(min(max_a[2], max_b[2]) - max(min_a[2], min_b[2])))
    return ox * oy * oz


def evaluate_pair_clash(mesh_a: SolidMesh3D, mesh_b: SolidMesh3D, required_clearance_m: float = 0.0) -> ClashResult:
    ov_vol = _aabb_overlap_volume(mesh_a.aabb_min, mesh_a.aabb_max, mesh_b.aabb_min, mesh_b.aabb_max)
    va, fa = mesh_a.vertices, mesh_a.faces
    vb, fb = mesh_b.vertices, mesh_b.faces
    b_tri_verts = vb[fb]
    b_mins = np.min(b_tri_verts, axis=1)
    b_maxs = np.max(b_tri_verts, axis=1)

    best_dist = float("inf")
    best_pt = 0.5 * (mesh_a.aabb_min + mesh_b.aabb_min)

    if ov_vol > 0.0:
        for tri_a in fa:
            a0, a1, a2 = va[tri_a[0]], va[tri_a[1]], va[tri_a[2]]
            amin = np.minimum(np.minimum(a0, a1), a2)
            amax = np.maximum(np.maximum(a0, a1), a2)
            mask = np.all(b_maxs >= amin - 1e-9, axis=1) & np.all(b_mins <= amax + 1e-9, axis=1)
            for j in np.nonzero(mask)[0]:
                b0, b1, b2 = b_tri_verts[j]
                hit = _triangles_intersect_3d(a0, a1, a2, b0, b1, b2)
                if hit is not None:
                    return ClashResult(
                        element_a=mesh_a.element_id,
                        element_b=mesh_b.element_id,
                        status="hard_clash",
                        min_distance_m=0.0,
                        overlap_aabb_volume_m3=ov_vol,
                        witness_point_m=(float(hit[0]), float(hit[1]), float(hit[2])),
                        required_clearance_m=required_clearance_m,
                    )
        if len(va) > 0 and _point_inside_mesh(va[0], mesh_b):
            p = va[0]
            return ClashResult(
                element_a=mesh_a.element_id,
                element_b=mesh_b.element_id,
                status="hard_clash",
                min_distance_m=0.0,
                overlap_aabb_volume_m3=ov_vol,
                witness_point_m=(float(p[0]), float(p[1]), float(p[2])),
                required_clearance_m=required_clearance_m,
            )
        if len(vb) > 0 and _point_inside_mesh(vb[0], mesh_a):
            p = vb[0]
            return ClashResult(
                element_a=mesh_a.element_id,
                element_b=mesh_b.element_id,
                status="hard_clash",
                min_distance_m=0.0,
                overlap_aabb_volume_m3=ov_vol,
                witness_point_m=(float(p[0]), float(p[1]), float(p[2])),
                required_clearance_m=required_clearance_m,
            )

    for tri_a in fa:
        a0, a1, a2 = va[tri_a[0]], va[tri_a[1]], va[tri_a[2]]
        amin = np.minimum(np.minimum(a0, a1), a2)
        amax = np.maximum(np.maximum(a0, a1), a2)
        dx = np.maximum(0.0, np.maximum(amin[0] - b_maxs[:, 0], b_mins[:, 0] - amax[0]))
        dy = np.maximum(0.0, np.maximum(amin[1] - b_maxs[:, 1], b_mins[:, 1] - amax[1]))
        dz = np.maximum(0.0, np.maximum(amin[2] - b_maxs[:, 2], b_mins[:, 2] - amax[2]))
        lb_dists = np.sqrt(dx * dx + dy * dy + dz * dz)
        order = np.argsort(lb_dists)
        for j in order:
            if lb_dists[j] >= best_dist - 1e-12:
                break
            b0, b1, b2 = b_tri_verts[j]
            d, pt = _triangle_triangle_distance_3d(a0, a1, a2, b0, b1, b2)
            if d < best_dist:
                best_dist = d
                best_pt = pt
                if best_dist <= 1e-12:
                    break
        if best_dist <= 1e-12:
            break

    if best_dist <= 1e-9:
        status = "hard_clash"
        best_dist = 0.0
    elif best_dist < required_clearance_m - 1e-9:
        status = "clearance_violation"
    else:
        status = "clear"

    return ClashResult(
        element_a=mesh_a.element_id,
        element_b=mesh_b.element_id,
        status=status,
        min_distance_m=float(best_dist),
        overlap_aabb_volume_m3=ov_vol if status == "hard_clash" else 0.0,
        witness_point_m=(float(best_pt[0]), float(best_pt[1]), float(best_pt[2])),
        required_clearance_m=required_clearance_m,
    )


def detect_ifc_clashes(
    ifc_text: str,
    required_clearance_m: float = 0.0,
    cross_discipline_only: bool = False,
) -> Dict[str, Any]:
    solids = load_ifc_solids(ifc_text)
    results: List[ClashResult] = []
    use_broadphase = len(solids) > 25
    mins = [s.aabb_min for s in solids]
    maxs = [s.aabb_max for s in solids]
    for i in range(len(solids)):
        for j in range(i + 1, len(solids)):
            sa, sb = solids[i], solids[j]
            if cross_discipline_only and sa.discipline == sb.discipline:
                continue
            if use_broadphase:
                dx = max(0.0, float(mins[i][0] - maxs[j][0]), float(mins[j][0] - maxs[i][0]))
                dy = max(0.0, float(mins[i][1] - maxs[j][1]), float(mins[j][1] - maxs[i][1]))
                dz = max(0.0, float(mins[i][2] - maxs[j][2]), float(mins[j][2] - maxs[i][2]))
                aabb_d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if aabb_d > max(required_clearance_m, 0.0) + 1e-6:
                    results.append(ClashResult(
                        element_a=sa.element_id,
                        element_b=sb.element_id,
                        status="clear",
                        min_distance_m=aabb_d,
                        overlap_aabb_volume_m3=0.0,
                        witness_point_m=(0.0, 0.0, 0.0),
                        required_clearance_m=required_clearance_m,
                    ))
                    continue
            res = evaluate_pair_clash(sa, sb, required_clearance_m=required_clearance_m)
            results.append(res)

    hard = [r for r in results if r.status == "hard_clash"]
    violations = [r for r in results if r.status == "clearance_violation"]
    min_clear = min((r.min_distance_m for r in results), default=float("inf"))
    return {
        "solid_count": len(solids),
        "pair_count": len(results),
        "hard_clash_count": len(hard),
        "clearance_violation_count": len(violations),
        "min_clearance_m": float(min_clear) if results else None,
        "total_solid_volume_m3": float(sum(s.volume_m3() for s in solids)),
        "clashes": [r.__dict__ for r in hard + violations],
        "all_pairs": [r.__dict__ for r in results],
    }



# --- Wave-3 Live Router Endpoint & Single-File Preflight Adapter ---
def _wave3_to_jsonable(obj):
    import dataclasses as _dc
    from pathlib import Path as _Path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, _Path):
        return str(obj)
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        return {str(k): _wave3_to_jsonable(v) for k, v in _dc.asdict(obj).items()}
    if hasattr(obj, "_asdict") and callable(obj._asdict):
        return {str(k): _wave3_to_jsonable(v) for k, v in obj._asdict().items()}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _wave3_to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _wave3_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_wave3_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _wave3_to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return _wave3_to_jsonable(obj.item())
    if hasattr(obj, "__dict__"):
        return {str(k): _wave3_to_jsonable(v) for k, v in vars(obj).items() if not str(k).startswith("_")}
    return str(obj)


def run_router_ifc_clash(payload: dict) -> dict:
    import base64 as _b64
    import inspect as _ins
    import os as _os
    import tempfile as _tf
    from pathlib import Path as _Path

    req_clear = float(payload.get("required_clearance_m", payload.get("clearance_m", 0.05)))
    cross_only = bool(payload.get("cross_discipline_only", True))
    tmp_file = None
    try:
        if payload.get("path") and _Path(payload["path"]).is_file():
            ifc_path = _Path(payload["path"])
        else:
            fd, tname = _tf.mkstemp(suffix=".ifc", prefix="ple-router-")
            _os.close(fd)
            tmp_file = _Path(tname)
            if "ifc_b64" in payload and payload["ifc_b64"]:
                tmp_file.write_bytes(_b64.b64decode(payload["ifc_b64"]))
            elif "ifc_text" in payload and payload["ifc_text"]:
                tmp_file.write_text(str(payload["ifc_text"]), encoding="utf-8")
            else:
                raise FileNotFoundError(f"IFC file not found: {payload.get('path')}")
            ifc_path = tmp_file

        preferred = (
            "analyze_ifc_spatial_clash",
            "analyze_ifc_model",
            "analyze_ifc_file",
            "analyze_ifc",
            "evaluate_ifc_spatial_clash",
            "run_ifc_spatial_clash",
            "inspect_ifc_model",
            "clash_report_from_ifc",
        )
        g = globals()
        cands = []
        for nm in preferred:
            if nm in g and callable(g[nm]):
                cands.append((nm, g[nm]))
        for nm, fn in g.items():
            if callable(fn) and not nm.startswith("_") and nm not in ("check_deliverable_preflight", "run_router_ifc_clash", "run_router_preflight"):
                if any(k in nm.lower() for k in ("analyze", "clash", "inspect", "evaluate", "takeoff", "summary")):
                    if (nm, fn) not in cands:
                        cands.append((nm, fn))
        ifc_txt = ifc_path.read_text(encoding="utf-8", errors="replace")
        has_entities = ("#1=" in ifc_txt or "IFCEXTRUDEDAREASOLID" in ifc_txt or "IFCPROJECT" in ifc_txt)
        fallback_out = None
        for nm, fn in cands:
            try:
                sig = _ins.signature(fn)
            except Exception:
                continue
            params = list(sig.parameters.values())
            if not params:
                continue
            p0 = params[0].name.lower()
            if any(w in p0 for w in ("text", "content", "raw", "step_str")):
                arg_list = (ifc_txt,)
            elif any(w in p0 for w in ("path", "file")):
                arg_list = (ifc_path, str(ifc_path))
            else:
                arg_list = (ifc_txt, ifc_path, str(ifc_path))
            kw = {}
            for pk in sig.parameters:
                if pk in ("required_clearance_m", "clearance_m", "min_clearance_m", "clearance"):
                    kw[pk] = req_clear
                elif pk in ("cross_discipline_only", "inter_discipline_only"):
                    kw[pk] = cross_only
            for arg0 in arg_list:
                try:
                    raw_out = fn(arg0, **kw)
                    if raw_out is None:
                        continue
                    out = _wave3_to_jsonable(raw_out)
                    if isinstance(out, list):
                        out = {"items": out, "count": len(out)}
                    if isinstance(out, dict) and len(out) >= 1:
                        out["ok"] = True
                        out.setdefault("tool", "hamon_ifc_spatial_clash")
                        if fallback_out is None:
                            fallback_out = out
                        zero_solids = any(
                            k in out and out[k] == 0
                            for k in ("solid_count", "n_solids", "step_entity_count", "entity_count", "element_count")
                        )
                        if has_entities and zero_solids:
                            continue
                        return out
                except UnresolvedSolidGeometry as exc:
                    return {"ok":False,"tool":"hamon_ifc_spatial_clash","status":"unresolved_solid_geometry",
                            "reason":str(exc),"total_solid_volume_m3":None}
                except UnresolvedOpening as exc:
                    return {"ok":False,"tool":"hamon_ifc_spatial_clash","status":"unresolved_opening",
                            "reason":str(exc),"total_solid_volume_m3":None}
                except Exception:
                    continue
        if fallback_out is not None:
            return fallback_out
        raise RuntimeError("No compatible IFC spatial clash entrypoint matched")
    finally:
        if tmp_file is not None:
            tmp_file.unlink(missing_ok=True)



if __name__ == "__main__" and "--json-stdin" in __import__("sys").argv:
    import json as _cli_json, sys as _cli_sys
    _req = _cli_json.loads(_cli_sys.stdin.read() or "{}")
    if _req.get("mode") == "preflight":
        _resp = run_router_preflight(_req)
    else:
        _resp = run_router_ifc_clash(_req)
    print(_cli_json.dumps(_wave3_to_jsonable(_resp)))
    raise SystemExit(0)

