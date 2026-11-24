"""3D IFC4/IFC2x3 STEP-P21, STL & OBJ solid geometry + exact spatial clash solver.

Provides built-in 3D solid extraction (extruded rectangular, hollow,
circular, pipe, I-beam/wide-flange, arbitrary closed polyline profiles, and
explicit B-Rep/triangulated face sets), optional polyhedral set operations,
hierarchical SE(3) IFCLOCALPLACEMENT
transforms, divergence-theorem volume/area takeoff, and exact 3D triangle-mesh
intersection + minimum surface-to-surface clearance for Physical-Engineering's
Last Exam (PLE).
"""
from __future__ import annotations

import math
from functools import cache
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
    # Each extracted IFC solid contributes material. Boundary shells within
    # one solid can enclose voids; independently supplied solids form a union.
    material_face_counts: Tuple[int, ...] = ()
    certified_union_volume_m3: Optional[float] = None

    @property
    def aabb_min(self) -> np.ndarray:
        return np.min(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)

    @property
    def aabb_max(self) -> np.ndarray:
        return np.max(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)

    def _local_tetrahedra(self, faces):
        # The tetrahedra share a point on the mesh. Using the global origin
        # makes their signed moments cancel at survey-coordinate magnitudes.
        origin = self.vertices[faces[0, 0]]
        a, b, c = (self.vertices[faces[:, i]] - origin for i in range(3))
        six_volumes = np.einsum("ij,ij->i", a, np.cross(b, c))
        return origin, a, b, c, six_volumes

    def _compound_moments(self):
        """Integrate disjoint positive Body items, retaining signed cavities inside each item."""
        if any(count <= 0 for count in self.material_face_counts) or sum(self.material_face_counts) != len(self.faces):
            raise UnresolvedMeshMass("Incomplete face ranges for independent Body items")
        reference = self.vertices[self.faces[0, 0]]
        volumes = []
        moments = []
        bounds = []
        start = 0
        for count in self.material_face_counts:
            faces = self.faces[start:start + count]
            start += count
            vertices, local = np.unique(faces, return_inverse=True)
            points = self.vertices[vertices]
            _, closed, coherent = _closed_surface_components(points, local.reshape(faces.shape))
            if len(closed) != count or not coherent:
                raise UnresolvedMeshMass("A Body item lacks a closed, consistently oriented boundary")
            low, high = points.min(axis=0), points.max(axis=0)
            for other_low, other_high in bounds:
                if np.all(np.minimum(high, other_high) > np.maximum(low, other_low)):
                    raise UnresolvedMeshMass("Body item bounds overlap, so additive volume is unverified")
            bounds.append((low, high))
            origin, a, b, c, signed = self._local_tetrahedra(faces)
            total = np.sum(signed)
            if not np.isfinite(total) or total == 0.0:
                raise UnresolvedMeshMass("A Body item has no finite signed volume")
            # Reverse the whole item's measure, including inner void shells and
            # its first moment, if its source winding or placement was reversed.
            weight = abs(total)
            local_moment = np.sum((a + b + c) * signed[:, None], axis=0) / 4
            moments.append(weight * (origin - reference) + np.sign(total) * local_moment)
            volumes.append(weight)
        weight = math.fsum(volumes)
        centroid = reference + np.sum(moments, axis=0) / weight
        return weight / 6, tuple(float(x) for x in centroid)

    def volume_m3(self) -> float:
        if self.certified_union_volume_m3 is not None:
            return self.certified_union_volume_m3
        if self.metadata.get("mass_issue"):
            raise UnresolvedMeshMass(self.metadata["mass_issue"])
        if len(self.faces) == 0:
            return 0.0
        if len(self.material_face_counts) > 1:
            return self._compound_moments()[0]
        _, _, _, _, six_volumes = self._local_tetrahedra(self.faces)
        return float(abs(np.sum(six_volumes)) / 6.0)

    def surface_area_m2(self) -> float:
        if self.certified_union_volume_m3 is not None:
            raise UnresolvedMeshMass("Only union volume is certified; component surface area overlaps")
        if len(self.faces) == 0:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        return float(np.sum(areas))

    def centroid_m(self) -> Tuple[float, float, float]:
        if self.certified_union_volume_m3 is not None:
            raise UnresolvedMeshMass("Only union volume is certified; centroid is unresolved")
        if self.metadata.get("mass_issue"):
            raise UnresolvedMeshMass(self.metadata["mass_issue"])
        if len(self.faces) == 0:
            c = np.mean(self.vertices, axis=0) if len(self.vertices) else np.zeros(3)
            return (float(c[0]), float(c[1]), float(c[2]))
        if len(self.material_face_counts) > 1:
            return self._compound_moments()[1]
        origin, a, b, c_local, six_volumes = self._local_tetrahedra(self.faces)
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
    min_distance_m: Optional[float]
    overlap_aabb_volume_m3: float
    witness_point_m: Optional[Tuple[float, float, float]]
    required_clearance_m: float = 0.0
    distance_lower_bound_m: Optional[float] = None
    closest_points_m: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    contact_evidence: Optional[str] = None



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
    """Return the metres per project length unit of this IFC model."""
    prefixes = {
        None: 1., "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9,
        "MEGA": 1e6, "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1,
        "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3, "MICRO": 1e-6,
        "NANO": 1e-9, "PICO": 1e-12, "FEMTO": 1e-15, "ATTO": 1e-18,
    }

    def resolve(ref):
        if isinstance(ref, tuple) and ref[0] == "ref" and ref[1] in entities:
            return entities[ref[1]]
        raise ValueError("Unresolved IFC project unit reference")

    def metre_scale(kind, attrs):
        if (kind != "IFCSIUNIT" or len(attrs) != 4 or attrs[1] != "LENGTHUNIT"
                or attrs[3] != "METRE" or attrs[2] not in prefixes):
            raise ValueError("Expected an IFC SI length unit based on metres")
        return prefixes[attrs[2]]

    def assignment_scale(ref):
        kind, args = resolve(ref)
        if kind != "IFCUNITASSIGNMENT" or len(args) != 1 or not isinstance(args[0], list):
            raise ValueError("Expected an IFC project unit assignment")
        units = [resolve(unit) for unit in args[0]]
        lengths = [(kind, attrs) for kind, attrs in units if len(attrs) >= 2 and attrs[1] == "LENGTHUNIT"]
        if len(lengths) != 1:
            raise ValueError("An IFC unit assignment must contain one length unit")
        kind, attrs = lengths[0]
        if kind != "IFCCONVERSIONBASEDUNIT":
            return metre_scale(kind, attrs)
        if len(attrs) != 4:
            raise ValueError("Invalid IFC length conversion")
        measure, factor = resolve(attrs[3])
        if measure not in ("IFCMEASUREWITHUNIT", "IFCLENGTHMEASUREWITHUNIT") or len(factor) != 2:
            raise ValueError("Expected an IFC length conversion measure")
        value = factor[0]
        if (not isinstance(value, tuple) or value[0] not in ("IFCLENGTHMEASURE", "IFCPOSITIVELENGTHMEASURE")
                or len(value[1]) != 1 or type(value[1][0]) not in (int, float)):
            raise ValueError("Expected a typed IFC length conversion factor")
        scale = value[1][0] * metre_scale(*resolve(factor[1]))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid IFC length conversion factor")
        return scale

    projects = [args for kind, args in entities.values() if kind == "IFCPROJECT"]
    if projects:
        scales = {assignment_scale(args[8] if len(args) > 8 else None) for args in projects}
        if len(scales) != 1:
            raise ValueError("IFC projects declare different length scales")
        return scales.pop()

    # Geometry-only fixtures may omit IfcProject. A sole assignment is unambiguous.
    assignments = [("ref", eid) for eid, (kind, _) in entities.items() if kind == "IFCUNITASSIGNMENT"]
    if len(assignments) > 1:
        raise ValueError("Ambiguous IFC unit assignments without a project")
    return assignment_scale(assignments[0]) if assignments else 1.


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


def _placement_ratios(entities, ref, default):
    """Resolve one placement axis before its scale-independent normalization."""
    if ref is None:
        return np.array(default, dtype=float)
    try:
        (coords,) = _required_ifc_entity(entities, ref, "IFCDIRECTION")
        ratios = np.asarray(coords, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnresolvedSolidGeometry("Invalid IFC placement direction") from exc
    if ratios.shape != (len(default),) or not np.isfinite(ratios).all():
        raise UnresolvedSolidGeometry("Invalid IFC placement direction dimension or ratios")
    if not np.any(ratios):
        raise UnresolvedSolidGeometry("IFC placement direction must be nonzero")
    return ratios


def _unit_ratios(ratios):
    scaled = ratios / np.max(np.abs(ratios))
    return scaled / np.linalg.norm(scaled)


def _placement_normal(z, x):
    """Return the positive local Y axis or None for exactly parallel directions."""
    # Every binary64 denominator is a power of two. A common denominator
    # per direction gives integer coordinates for the same two directions.
    def integer_ratios(ratios):
        parts = [float(value).as_integer_ratio() for value in ratios]
        common = max(denominator for _, denominator in parts)
        return [numerator * (common // denominator) for numerator, denominator in parts]

    a = integer_ratios(z)
    b = integer_ratios(x)
    cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
    size = max(map(abs, cross))
    if size == 0:
        return None
    scaled = np.array([v / size for v in cross])
    return scaled / math.hypot(*scaled)


def _axis2placement3d_matrix(entities: Dict[int, Tuple[str, List[Any]]], ref: Any, unit_scale: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if not (isinstance(ref, tuple) and ref[0] == "ref" and ref[1] in entities):
        return mat
    etype, args = entities[ref[1]]
    if etype == "IFCAXIS2PLACEMENT3D":
        loc = _vec3_from_ref(entities, args[0] if len(args) > 0 else None, (0.0, 0.0, 0.0)) * unit_scale
        z_ref = args[1] if len(args) > 1 else None
        x_ref = args[2] if len(args) > 2 else None
        z = _placement_ratios(entities, z_ref, (0., 0., 1.))
        x = _placement_ratios(entities, x_ref, (1., 0., 0.))
        z_axis = _unit_ratios(z)
        y_axis = _placement_normal(z, x)
        if y_axis is None:
            if x_ref is not None and z_ref is not None:
                raise UnresolvedSolidGeometry("Parallel IFC placement axes")
            # Preserve the implicit perpendicular axis when one axis is omitted.
            fallback = [1., 0., 0.] if abs(z_axis[0]) < 0.9 else [0., 1., 0.]
            y_axis = _unit_ratios(np.cross(z_axis, fallback))
        x_axis = np.cross(y_axis, z_axis)
        mat[:3, 0] = x_axis
        mat[:3, 1] = y_axis
        mat[:3, 2] = z_axis
        mat[:3, 3] = loc
    elif etype == "IFCAXIS2PLACEMENT2D":
        loc = _vec3_from_ref(entities, args[0] if len(args) > 0 else None, (0.0, 0.0, 0.0)) * unit_scale
        x_ref = args[1] if len(args) > 1 else None
        x2 = _unit_ratios(_placement_ratios(entities, x_ref, (1., 0.)))
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


def _extrude_polygon_2d(outer_pts: np.ndarray, sweep: np.ndarray, inner_pts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if _signed_area_2d(outer_pts) < 0:
        outer_pts = outer_pts[::-1].copy()

    if inner_pts is not None and len(inner_pts) >= 3:
        if _signed_area_2d(inner_pts) < 0:
            inner_pts = inner_pts[::-1].copy()
        if len(outer_pts) == len(inner_pts):
            k = len(outer_pts)
            b_out = np.column_stack([outer_pts, np.zeros(k)])
            b_in = np.column_stack([inner_pts, np.zeros(k)])
            t_out = b_out + sweep
            t_in = b_in + sweep
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
            if sweep[2] < 0:
                faces = [(a, c, b) for (a, b, c) in faces]
            return verts, np.array(faces, dtype=np.int32)

    k = len(outer_pts)
    cap_tris = _triangulate_simple_polygon_2d(outer_pts)
    bottom = np.column_stack([outer_pts, np.zeros(k)])
    top = bottom + sweep
    verts = np.vstack([bottom, top])
    faces = []
    for (a, b, c) in cap_tris:
        faces.append((a, c, b))
        faces.append((k + a, k + b, k + c))
    for i in range(k):
        nxt = (i + 1) % k
        faces.append((i, nxt, k + nxt))
        faces.append((i, k + nxt, k + i))
    if sweep[2] < 0:
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
                    list_type, largs = entities[lref[1]]
                    if list_type != "IFCCARTESIANPOINTLIST2D" or not largs:
                        raise UnresolvedSolidGeometry("Indexed profile needs an IFC 2D point list")
                    if len(cargs) > 2 and cargs[2] is True:
                        raise UnresolvedSolidGeometry("Self-intersecting IFC indexed profile")
                    points = largs[0]
                    if len(cargs) > 1 and cargs[1] is not None:
                        from hamon_ifc_indexed_curves import ordered_segments
                        try:
                            edges = ordered_segments(points, cargs[1])
                        except (TypeError, ValueError) as exc:
                            raise UnresolvedSolidGeometry(str(exc)) from exc
                        if any(len(edge) == 3 for edge in edges):
                            raise UnresolvedSolidGeometry("Circular IFC profile requires the source BRep route")
                        points = [edge[0] for edge in edges]
                        from shapely import Polygon
                        polygon = Polygon(points) if len(points) >= 3 else None
                        if polygon is None or not polygon.is_valid or polygon.area == 0:
                            raise UnresolvedSolidGeometry("Invalid straight indexed profile boundary")
                    elif len(points) > 1 and points[0] == points[-1]:
                        points = points[:-1]
                    else:
                        raise UnresolvedSolidGeometry("Unsegmented IFC indexed profile is not closed")
                    if len(points) >= 3:
                        return np.asarray(points, dtype=np.float64) * unit_scale, None, pos_mat
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


class UnresolvedMeshMass(ValueError):
    """Clash geometry is available, but the compound net mass is unresolved."""


class UnresolvedSolidGeometry(ValueError):
    """A physical Body was present but could not supply verified solid geometry."""


class _VolumeOnlyUnion(UnresolvedSolidGeometry):
    """A kernel union has qualified volume but no position-welded output surface."""

    def __init__(self, volume_m3):
        super().__init__("Only indexed material-union volume is certified")
        self.volume_m3 = volume_m3


_EMPTY_CLIP = (np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32))


def _covering_interval(values):
    """Strictly bracket the host projections, including the last representable ulp."""
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max(hi - lo, 16 * math.ulp(max(abs(lo), abs(hi))))
    lower, upper = lo - pad, hi + pad
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < lo <= hi < upper):
        raise UnresolvedSolidGeometry("Cannot bracket the finite clipping host")
    return lower, upper


def _finite_prism(frame, ring, bottom, top):
    if not (np.isfinite(ring).all() and math.isfinite(top-bottom) and bottom < top):
        raise UnresolvedSolidGeometry("Invalid clipping prism")
    vertices, faces = _extrude_polygon_2d(ring, np.array([0., 0., top - bottom]))
    vertices[:, 2] += bottom
    vertices = _apply_transform(vertices, frame)
    if not np.isfinite(vertices).all():
        raise UnresolvedSolidGeometry("Nonfinite IFC half-space or footprint")
    return vertices, faces


def _halfspace_difference(entities, kind, args, unit_scale, host):
    """Bound a half-space and its directed polygonal footprint on the finite first operand."""
    bounded = kind == "IFCPOLYGONALBOUNDEDHALFSPACE"
    if len(args) != (4 if bounded else 2) or type(args[1]) is not bool:
        raise UnresolvedSolidGeometry("Invalid IFC half-space agreement or attributes")
    plane, = _required_ifc_entity(entities, args[0], "IFCPLANE")
    frame = _rigid_ifc_frame(entities, plane, unit_scale)
    if bounded:
        footprint_frame = _rigid_ifc_frame(entities, args[2], unit_scale)
        boundary = _closed_polyline(entities, args[3], unit_scale)
    if not len(host[0]):
        return _EMPTY_CLIP
    vertices = host[0]
    local = (vertices - frame[:3, 3]) @ frame[:3, :3]
    if not np.isfinite(local).all():
        raise UnresolvedSolidGeometry("Nonfinite host projections")
    material_z = -local[:, 2] if args[1] else local[:, 2]
    if np.max(material_z) <= 0:
        return host
    if not bounded and np.min(material_z) >= 0:
        return _EMPTY_CLIP
    x0, x1 = _covering_interval(local[:, 0])
    y0, y1 = _covering_interval(local[:, 1])
    z0, z1 = _covering_interval(local[:, 2])
    rectangle = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    cutter = _finite_prism(frame, rectangle, z0 if args[1] else 0., 0. if args[1] else z1)
    if bounded:
        heights = (vertices - footprint_frame[:3, 3]) @ footprint_frame[:3, 2]
        footprint = _finite_prism(footprint_frame, boundary, *_covering_interval(heights))
        cutter = _boolean_meshes([cutter, footprint], "INTERSECTION", allow_empty=True)
        if not len(cutter[0]):
            return host
    return _boolean_meshes([host, cutter], "DIFFERENCE", allow_empty=True)


def _boolean_meshes(meshes, operator="UNION", *, allow_empty=False):
    """Combine closed material, retaining exact rectilinear source planes.

    Input triangles are the geometry being combined. A polygonal approximation
    of a curved profile does not become an analytic curved solid. The shared
    local frame avoids cancellation at survey coordinates. Exact-position
    welding preserves distinct walls and gaps; no approximate repair is used.
    """
    if operator not in {"UNION", "INTERSECTION", "DIFFERENCE"} or len(meshes) < 2:
        raise UnresolvedSolidGeometry("invalid polyhedral Boolean operands")
    if sum(len(f) for _, f in meshes) > 50000:
        raise UnresolvedSolidGeometry("polyhedral Boolean exceeds 50000 source triangles")
    for vertices, faces in meshes:
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.isfinite(vertices).all() or faces.min() < 0
                or faces.max() >= len(vertices)):
            raise UnresolvedSolidGeometry("invalid polyhedral Boolean mesh")
        _, closed, coherent = _closed_surface_components(vertices, faces)
        if len(closed) != len(faces) or not coherent:
            raise UnresolvedSolidGeometry("Boolean operand is not closed and consistently oriented")
    from hamon_rectilinear_boolean import rectilinear_boolean
    try:
        exact = rectilinear_boolean(meshes, operator, allow_empty=allow_empty)
    except ValueError as exc:
        raise UnresolvedSolidGeometry(str(exc)) from exc
    if exact is not None:
        vertices, faces = exact
        if allow_empty and not len(vertices):
            return _EMPTY_CLIP
        _, closed, coherent = _closed_surface_components(vertices, faces)
        if len(closed) != len(faces) or not coherent:
            raise UnresolvedSolidGeometry("rectilinear Boolean has a nonmanifold material contact")
        return vertices, faces
    try:
        import manifold3d
    except ImportError as exc:
        raise UnresolvedSolidGeometry("polyhedral Boolean requires manifold3d") from exc
    operations = {"UNION": manifold3d.OpType.Add,
                  "INTERSECTION": manifold3d.OpType.Intersect,
                  "DIFFERENCE": manifold3d.OpType.Subtract}
    low = np.min([v.min(axis=0) for v, _ in meshes], axis=0)
    high = np.max([v.max(axis=0) for v, _ in meshes], axis=0)
    scale = float(np.max(high - low))
    if not 0 < scale < math.inf:
        raise UnresolvedSolidGeometry("Boolean extent is not finite and positive")
    try:
        operands = []
        for vertices, faces in meshes:
            vertices, inverse = np.unique((vertices - low) / scale, axis=0, return_inverse=True)
            faces = inverse[faces]
            origin = vertices[faces[0, 0]]
            a, b, c = (vertices[faces[:, k]] - origin for k in range(3))
            signed = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())
            if not math.isfinite(signed) or signed == 0:
                raise UnresolvedSolidGeometry("Boolean operand has no finite enclosed volume")
            # Reverse the whole solid, including cavity shells, when a source
            # winding or mapped reflection has reversed its orientation.
            if signed < 0:
                faces = faces[:, ::-1]
            source = manifold3d.Mesh64(np.ascontiguousarray(vertices, dtype=np.float64),
                                       np.ascontiguousarray(faces, dtype=np.uint64))
            solid = manifold3d.Manifold(source)
            if solid.status() != manifold3d.Error.NoError or solid.is_empty():
                raise UnresolvedSolidGeometry("Boolean operand is not a valid oriented manifold")
            operands.append(solid)
        result = manifold3d.Manifold.batch_boolean(operands, operations[operator])
        if allow_empty and result.status() == manifold3d.Error.NoError and result.is_empty():
            return _EMPTY_CLIP
        if result.status() != manifold3d.Error.NoError or result.is_empty():
            raise UnresolvedSolidGeometry("Boolean produced no qualified bounded solid")
        output = result.to_mesh64()
        vertices = np.asarray(output.vert_properties, dtype=np.float64)[:, :3]
        faces = np.asarray(output.tri_verts, dtype=np.int32)
        _, closed, coherent = _closed_surface_components(vertices, faces)
        if not np.isfinite(vertices).all() or len(closed) != len(faces) or not coherent:
            if operator == "UNION":
                from hamon_ifc_indexed_union import certified_indexed_union_volume
                volume = certified_indexed_union_volume(
                    vertices, faces, kernel_volume=float(result.volume()),
                    operand_volumes=[float(part.volume()) for part in operands],
                    world_vertices=vertices * scale + low, scale=scale)
                raise _VolumeOnlyUnion(volume)
            raise UnresolvedSolidGeometry("Boolean result lacks a closed, consistently oriented boundary")
        return vertices * scale + low, faces
    except UnresolvedSolidGeometry:
        raise
    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise UnresolvedSolidGeometry("polyhedral Boolean kernel failed") from exc


def _ear_clip_bounded_face(rings):
    """Triangulate a planar region, bridging holes without adding mesh vertices."""
    # Finite IEEE coordinates share a power-of-two denominator. Exact integer
    # predicates keep collinear surveying vertices and narrow holes distinct.
    coords = [[tuple(map(float, pt)) for pt in ring] for ring in rings]
    scale = max(x.as_integer_ratio()[1] for ring in coords for pt in ring for x in pt)
    def integer(x):
        numerator, denominator = x.as_integer_ratio()
        return 2 * numerator * (scale // denominator)
    polygon = [[(integer(x), integer(y)) for x, y in ring] for ring in coords]
    original = {p: uv for ring, uv_ring in zip(polygon, coords) for p, uv in zip(ring, uv_ring)}

    def edges(ring):
        return zip(ring, ring[1:] + ring[:1])

    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def area(ring):
        return sum(a[0]*b[1] - a[1]*b[0] for a, b in edges(ring))

    def conflict(a, b, c, d):
        # Normal adjacency at a single endpoint is allowed. Collinear
        # segments leaving their common endpoint in the same direction overlap.
        common = {a, b}.intersection((c, d))
        if len(common) == 2:
            return True
        if common:
            p = common.pop()
            u, v = (a if b == p else b), (c if d == p else d)
            return orient(p, u, v) == 0 and (u[0]-p[0])*(v[0]-p[0]) + (u[1]-p[1])*(v[1]-p[1]) > 0
        for i in (0, 1):
            if max(a[i], b[i]) < min(c[i], d[i]) or max(c[i], d[i]) < min(a[i], b[i]):
                return False
        return orient(a, b, c)*orient(a, b, d) <= 0 and orient(c, d, a)*orient(c, d, b) <= 0

    def inside(p, ring):
        winding = 0
        for a, b in edges(ring):
            if a[1] <= p[1] < b[1] and orient(a, b, p) > 0:
                winding += 1
            elif b[1] <= p[1] < a[1] and orient(a, b, p) < 0:
                winding -= 1
        return winding != 0

    def fail(reason):
        raise UnresolvedSolidGeometry("polygonal Body: " + reason)

    if len(original) != sum(map(len, polygon)):
        fail("repeated or overlapping boundary vertex")
    boundary = [edge for ring in polygon for edge in edges(ring)]
    for i, (a, b) in enumerate(boundary):
        if any(conflict(a, b, c, d) for c, d in boundary[:i]):
            fail("intersecting polygon boundaries")
    outer, holes = polygon[0], polygon[1:]
    for i, hole in enumerate(holes):
        if not inside(hole[0], outer) or any(inside(hole[0], other) for j, other in enumerate(holes) if j != i):
            fail("invalid inner ring nesting")
    if area(outer) < 0:
        outer = outer[::-1]
    holes = [h if area(h) < 0 else h[::-1] for h in holes]
    needed_area = area(outer) + sum(map(area, holes))

    chain = list(outer)
    for hole in sorted(holes, key=max, reverse=True):
        h = max(hole)
        # A visible bridge joins a hole to the growing outer boundary. Both
        # bridge copies cancel in the final triangulation's oriented edge sum.
        candidates = sorted(enumerate(chain), key=lambda iv: (iv[1][0]-h[0])**2 + (iv[1][1]-h[1])**2)
        for i, p in candidates:
            prev, following = chain[i-1], chain[(i+1) % len(chain)]
            left = orient(prev, p, h) >= 0
            right = orient(p, following, h) >= 0
            # A bridge endpoint can occur twice after joining earlier holes.
            # Choose the occurrence whose interior wedge contains the new ray.
            fits = (left and right) if orient(prev, p, following) >= 0 else (left or right)
            if not fits:
                continue
            mid = ((p[0]+h[0])//2, (p[1]+h[1])//2)
            if any(conflict(h, p, a, b) for a, b in boundary):
                continue
            if inside(mid, outer) and not any(inside(mid, inner) for inner in holes):
                j = hole.index(h)
                chain[i+1:i+1] = hole[j:] + hole[:j+1] + [p]
                boundary.append((h, p))
                break
        else:
            fail("could not connect inner ring without crossing a boundary")

    result = []
    while len(chain) > 3:
        for i, b in enumerate(chain):
            a, c = chain[i-1], chain[(i+1) % len(chain)]
            if orient(a, b, c) <= 0:
                continue
            if any(orient(a, b, p) >= 0 and orient(b, c, p) >= 0 and orient(c, a, p) >= 0
                   for p in chain if p not in (a, b, c)):
                continue
            if any(conflict(a, c, x, y) for x, y in edges(chain)):
                continue
            result.append((a, b, c))
            chain.pop(i)
            break
        else:
            fail("could not triangulate bounded planar region")
    if orient(*chain) <= 0:
        fail("degenerate final triangle")
    result.append(tuple(chain))
    if sum(orient(*tri) for tri in result) != needed_area:
        fail("incomplete constrained triangulation")
    return [tuple(original[p] for p in tri) for tri in result]


def _bounded_face_triangles(rings):
    try:
        from shapely import Polygon, constrained_delaunay_triangles
        from shapely.errors import GEOSException
    except ImportError:
        return _ear_clip_bounded_face(rings)
    try:
        polygon = Polygon(rings[0], rings[1:])
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
            raise UnresolvedSolidGeometry("polygonal Body: invalid polygon or inner ring nesting")
        triangles = constrained_delaunay_triangles(polygon).geoms
        if any(not polygon.covers(tri) for tri in triangles):
            raise UnresolvedSolidGeometry("polygonal Body: constrained tessellation escaped polygon boundaries")
        if not math.isclose(sum(tri.area for tri in triangles), polygon.area, rel_tol=1e-10, abs_tol=0.):
            raise UnresolvedSolidGeometry("polygonal Body: incomplete constrained triangulation")
        return [list(tri.exterior.coords)[:-1] for tri in triangles]
    except (GEOSException, ValueError) as exc:
        if isinstance(exc, UnresolvedSolidGeometry):
            raise
        raise UnresolvedSolidGeometry("invalid planar polygonal Body") from exc


def _polygonal_solid_mesh(entities, args, unit_scale):
    """Tessellate bounded planar regions; preserve all holes and oriented shell edges."""
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

    face_rings = []
    for ref in args[2]:
        kind, bounds = reference(ref, {"IFCINDEXEDPOLYGONALFACE", "IFCINDEXEDPOLYGONALFACEWITHVOIDS"})
        if not bounds:
            fail("missing exterior ring")
        rings = [indices(bounds[0])]
        if kind == "IFCINDEXEDPOLYGONALFACEWITHVOIDS":
            if len(bounds) < 2 or not isinstance(bounds[1], list) or not bounds[1]:
                fail("missing inner rings")
            rings.extend(indices(loop) for loop in bounds[1])

        face_rings.append(rings)
    return _bounded_solid_mesh(vertices, face_rings)


def _bounded_solid_mesh(vertices, face_rings, label="polygonal Body"):
    """Tessellate complete planar face domains for both IFC mesh encodings."""
    from collections import Counter

    def fail(reason):
        raise UnresolvedSolidGeometry(label + ": " + reason)

    boundary = Counter()
    for rings in face_rings:
        for ring in rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                boundary[(tuple(vertices[a]), tuple(vertices[b]))] += 1
    if any(a == b or count != 1 or boundary[(b, a)] != 1 for (a, b), count in boundary.items()):
        fail("shell is open, non-manifold or inconsistently oriented")

    directed = Counter()
    triangles = []
    for rings in face_rings:
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
        for coords in _bounded_face_triangles(faces_2d):
            if len(coords) != 3 or any(c not in vertex_for for c in coords):
                fail("constrained tessellation escaped polygon boundaries")
            tri = [vertex_for[c] for c in coords]
            if _signed_area_2d(np.asarray(coords)) * normal[omit] < 0:
                tri.reverse()
            triangles.append(tri)
            for a, b in zip(tri, tri[1:] + tri[:1]):
                directed[(tuple(vertices[a]), tuple(vertices[b]))] += 1
    if not triangles or any(count != 1 or directed[(b, a)] != 1 for (a, b), count in directed.items()):
        fail("shell is open, non-manifold or inconsistently oriented")
    # Drop unreferenced coordinates: PnIndex may select only one part of a shared list.
    used, inverse = np.unique(np.asarray(triangles, dtype=np.int32), return_inverse=True)
    return [(vertices[used], inverse.reshape(-1, 3).astype(np.int32))]



def _faceted_solid_mesh(entities, args, unit_scale):
    """Resolve oriented PolyLoops into one outer ring and all holes per face."""
    def fail(reason):
        raise UnresolvedSolidGeometry("faceted BREP: " + reason)

    def reference(ref, accepted):
        if not isinstance(ref, tuple) or len(ref) != 2 or ref[0] != "ref" or ref[1] not in entities:
            fail("missing referenced IFC entity")
        kind, values = entities[ref[1]]
        if kind not in accepted:
            fail("unexpected referenced " + kind)
        return kind, values

    if len(args) != 1:
        fail("expected one closed outer shell")
    _, shell = reference(args[0], {"IFCCLOSEDSHELL"})
    if len(shell) != 1 or not isinstance(shell[0], list) or not shell[0]:
        fail("empty or malformed closed shell")
    vertices = []
    vertex_for = {}
    face_rings = []
    for face_ref in shell[0]:
        _, face = reference(face_ref, {"IFCFACE"})
        if len(face) != 1 or not isinstance(face[0], list) or not face[0]:
            fail("face has no boundary loops")
        outer = None
        inner = []
        for bound_ref in face[0]:
            kind, bound = reference(bound_ref, {"IFCFACEOUTERBOUND", "IFCFACEBOUND"})
            if len(bound) != 2 or type(bound[1]) is not bool:
                fail("face bound lacks an explicit Boolean orientation")
            _, loop = reference(bound[0], {"IFCPOLYLOOP"})
            if len(loop) != 1 or not isinstance(loop[0], list) or len(loop[0]) < 3:
                fail("face bound needs at least three polygon vertices")
            ring = []
            for point_ref in loop[0]:
                _, point = reference(point_ref, {"IFCCARTESIANPOINT"})
                if point_ref not in vertex_for:
                    if (len(point) != 1 or not isinstance(point[0], list) or len(point[0]) != 3
                            or any(type(value) not in (int, float) for value in point[0])):
                        fail("expected three numeric Cartesian coordinates")
                    try:
                        coordinates = np.asarray(point[0], dtype=np.float64) * unit_scale
                    except (IndexError, TypeError, ValueError, OverflowError) as exc:
                        raise UnresolvedSolidGeometry("faceted BREP: malformed point") from exc
                    if not np.isfinite(coordinates).all():
                        fail("non-finite or non-3D point")
                    vertex_for[point_ref] = len(vertices)
                    vertices.append(coordinates)
                ring.append(vertex_for[point_ref])
            if not bound[1]:
                ring.reverse()
            if kind == "IFCFACEOUTERBOUND":
                if outer is not None:
                    fail("face has more than one outer bound")
                outer = ring
            else:
                inner.append(ring)
        if outer is None:
            fail("face has no explicit outer bound")
        face_rings.append([outer, *inner])
    return _bounded_solid_mesh(np.asarray(vertices, dtype=np.float64), face_rings, "faceted BREP")


def _faceted_void_solid_mesh(entities, args, unit_scale):
    """Certify embedded oriented shells before retaining their signed boundaries as one Body."""
    def fail(reason):
        raise UnresolvedSolidGeometry("faceted BREP with voids: " + reason)

    if len(args) != 2 or not isinstance(args[1], list) or not args[1]:
        fail("expected one outer shell and at least one void shell")
    refs = [args[0], *args[1]]
    if any(not isinstance(ref, tuple) or len(ref) != 2 or ref[0] != "ref" for ref in refs):
        fail("invalid shell reference")
    if len(set(refs)) != len(refs):
        fail("shell referenced more than once")
    shells = [_faceted_solid_mesh(entities, [ref], unit_scale)[0] for ref in refs]
    from hamon_shell_relations import certify_void_shells
    try:
        certify_void_shells(shells)
    except ValueError as exc:
        fail(str(exc))

    faces, offset = [], 0
    for vertices, triangles in shells:
        faces.append(triangles + offset)
        offset += len(vertices)
    return [(np.vstack([vertices for vertices, _ in shells]), np.vstack(faces))]


def _advanced_brep_mesh(model, step):
    """Accept a single closed shell only when the source, CAD shape and mesh agree."""
    from collections import Counter
    import ifcopenshell.geom
    import ifcopenshell.ifcopenshell_wrapper as wrapper

    try:
        item = model.by_id(step)
        faces = item.Outer.CfsFaces
        if not 1 <= len(faces) <= 2048 or len({face.id() for face in faces}) != len(faces):
            raise UnresolvedSolidGeometry("invalid advanced BRep face inventory")
        edge_uses = Counter()
        for face in faces:
            if not face.is_a("IfcAdvancedFace") or not face.Bounds:
                raise UnresolvedSolidGeometry("unverified advanced BRep face")
            for bound in face.Bounds:
                loop = bound.Bound
                if not loop.is_a("IfcEdgeLoop") or not loop.EdgeList:
                    raise UnresolvedSolidGeometry("unverified advanced BRep edge loop")
                edge_uses.update(edge.EdgeElement.id() for edge in loop.EdgeList)
        if not edge_uses or any(count != 2 for count in edge_uses.values()):
            raise UnresolvedSolidGeometry("advanced BRep shell does not close on its source edges")

        settings = ifcopenshell.geom.settings()
        settings.set("iterator-output", wrapper.NATIVE)
        # Native iterator output is exposed on product elements. Isolate this
        # exact item from siblings, placements and opening relationships.
        contexts = [ctx for ctx in model.by_type("IfcGeometricRepresentationContext", False)
                    if ctx.CoordinateSpaceDimension == 3]
        if len(contexts) != 1:
            raise UnresolvedSolidGeometry("advanced BRep needs one 3D model context")
        rep = model.create_entity("IfcShapeRepresentation", contexts[0], "Body", "AdvancedBrep", (item,))
        definition = model.create_entity("IfcProductDefinitionShape", Representations=(rep,))
        import ifcopenshell.guid
        proxy = model.create_entity("IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(),
                                    Representation=definition)
        native_element = ifcopenshell.geom.create_shape(settings, proxy)
        brep = native_element.geometry
        if brep.size() != 1:
            raise UnresolvedSolidGeometry("expected one native advanced BRep")
        native = brep.item(0)
        native_count = native.num_faces()
        # OCC may retain coincident but separately indexed source seams. The
        # tessellation below must independently close using exact positions.
        if native_count != len(faces):
            raise UnresolvedSolidGeometry(
                f"native advanced BRep has {native_count}/{len(faces)} faces, "
                f"manifold={native.is_manifold()}, area={native.area().to_double()}, "
                f"signed volume={native.volume().to_double()}"
            )
        exact_volume = native.volume().to_double()
        exact_area = native.area().to_double()
        if not (0 < exact_volume < math.inf and 0 < exact_area < math.inf):
            raise UnresolvedSolidGeometry("advanced BRep lacks positive finite kernel measures")

        settings.set("iterator-output", wrapper.TRIANGULATED)
        # A dimensionless scale relative to V/A keeps mm and metre models comparable.
        settings.set("mesher-linear-deflection", .003 * exact_volume / exact_area)
        settings.set("mesher-angular-deflection", .1)
        mesh_element = ifcopenshell.geom.create_shape(settings, proxy)
        mesh = mesh_element.geometry
        vertices = np.asarray(mesh.verts, dtype=float).reshape(-1, 3)
        triangles = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
        if not (0 < len(triangles) <= 250000 and np.isfinite(vertices).all()):
            raise UnresolvedSolidGeometry("invalid or oversized advanced BRep tessellation")
        from hamon_shell_relations import certify_exterior_shell
        certify_exterior_shell(vertices, triangles)
        a, b, c = (vertices[triangles[:, i]] - vertices[triangles[0, 0]] for i in range(3))
        volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
        area = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)
        if not (math.isclose(volume, exact_volume, rel_tol=.002)
                and math.isclose(area, exact_area, rel_tol=.002)):
            raise UnresolvedSolidGeometry("advanced BRep tessellation disagrees with native material measures")
        return vertices, triangles
    except (AttributeError, ImportError, RuntimeError, ValueError, TypeError) as exc:
        if isinstance(exc, UnresolvedSolidGeometry):
            raise
        raise UnresolvedSolidGeometry(f"advanced BRep conversion failed: {exc}") from exc


def _extract_solid_item(entities: Dict[int, Tuple[str, List[Any]]], iref: Any, unit_scale: float,
                        mass_issues: Optional[List[str]] = None,
                        active_items: frozenset = frozenset(), advanced_mesh=None, *, strict_geometry=False) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not (isinstance(iref, tuple) and iref[0] == "ref" and iref[1] in entities):
        return []
    if iref[1] in active_items or len(active_items) >= 128:
        raise UnresolvedSolidGeometry("cyclic or excessively nested physical representation")
    active_items = active_items | {iref[1]}
    itype, args = entities[iref[1]]
    if itype == "IFCCSGSOLID" and len(args) == 1:
        return _extract_solid_item(entities, args[0], unit_scale, mass_issues, active_items, advanced_mesh, strict_geometry=strict_geometry)
    if itype == "IFCBLOCK" and len(args) == 4:
        try:
            location, _, _ = _required_ifc_entity(entities, args[0], "IFCAXIS2PLACEMENT3D")
            (coordinates,) = _required_ifc_entity(entities, location, "IFCCARTESIANPOINT")
            coordinates = np.asarray(coordinates, dtype=float)
            if coordinates.shape != (3,) or not np.isfinite(coordinates * unit_scale).all():
                raise ValueError("expected a finite 3D block location")
            x, y, z = (float(value) * unit_scale for value in args[1:])
        except (ValueError, TypeError, OverflowError) as exc:
            raise UnresolvedSolidGeometry("invalid IFC block placement or dimensions") from exc
        if not all(0 < value < math.inf for value in (x, y, z)):
            raise UnresolvedSolidGeometry("IFC block dimensions must be finite and positive")
        vertices, faces = _extrude_polygon_2d(np.array([(0., 0.), (x, 0.), (x, y), (0., y)]),
                                             np.array([0., 0., z]))
        return [(_apply_transform(vertices, _axis2placement3d_matrix(entities, args[0], unit_scale)), faces)]
    if itype in ("IFCBOOLEANRESULT", "IFCBOOLEANCLIPPINGRESULT") and len(args) == 3:
        operator = str(args[0]).strip(".").upper()
        if itype == "IFCBOOLEANCLIPPINGRESULT" and operator != "DIFFERENCE":
            raise UnresolvedSolidGeometry("IFC clipping result must use DIFFERENCE")
        operands = []
        issues = []
        operand_ref = args[2]
        operand_kind = entities.get(operand_ref[1], ("",))[0] if isinstance(operand_ref, tuple) and operand_ref[0] == "ref" else ""
        plane_clip = operator == "DIFFERENCE" and operand_kind in ("IFCHALFSPACESOLID", "IFCPOLYGONALBOUNDEDHALFSPACE")
        if plane_clip:
            try:
                _require_prism_or_clipped_prism(entities, args[1])
            except UnresolvedOpening as exc:
                raise UnresolvedSolidGeometry("Half-space clipping requires a qualified finite prism host") from exc
        for reference in args[1:]:
            if operands and operator == "DIFFERENCE" and isinstance(reference, tuple) and reference[0] == "ref":
                second_kind, fields = entities.get(reference[1], (None, []))
                if second_kind in ("IFCHALFSPACESOLID", "IFCPOLYGONALBOUNDEDHALFSPACE"):
                    try:
                        return [_halfspace_difference(entities, second_kind, fields, unit_scale, operands[0])]
                    except (TypeError, ValueError, ArithmeticError, IndexError) as exc:
                        raise UnresolvedSolidGeometry(f"Unresolved IFC plane clipping: {exc}") from exc
            pieces = _extract_solid_item(entities, reference, unit_scale, issues, active_items, advanced_mesh,
                                         strict_geometry=strict_geometry or plane_clip)
            if not pieces or issues:
                raise UnresolvedSolidGeometry("unresolved physical Boolean operand")
            material = [piece for piece in pieces if len(piece[0])]
            operands.append(_boolean_meshes(material) if len(material) > 1 else material[0] if material else _EMPTY_CLIP)
        if any(not len(mesh[0]) for mesh in operands):
            if operator == "UNION":
                return [next((mesh for mesh in operands if len(mesh[0])), _EMPTY_CLIP)]
            if operator == "INTERSECTION":
                return [_EMPTY_CLIP]
            if operator == "DIFFERENCE":
                return [operands[0]]
            raise UnresolvedSolidGeometry("Invalid Boolean operator")
        return [_boolean_meshes(operands, operator)]
    if itype == "IFCEXTRUDEDAREASOLID" and strict_geometry:
        try:
            return [_qualified_polygon_sweep(entities, iref, unit_scale)]
        except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
            raise UnresolvedSolidGeometry(f"Unresolved source extrusion: {exc}") from exc
    if itype == "IFCEXTRUDEDAREASOLID" and len(args) >= 4:
        outer, inner, prof_mat = _profile_polygons_2d(entities, args[0], unit_scale)
        if outer is None:
            return []
        solid_mat = _axis2placement3d_matrix(entities, args[1], unit_scale)
        try:
            (direction_values,) = _required_ifc_entity(entities, args[2], "IFCDIRECTION")
            ratios = np.asarray(direction_values, dtype=float)
            depth = float(args[3]) * unit_scale
        except (TypeError, ValueError, OverflowError) as exc:
            raise UnresolvedSolidGeometry("Invalid IFC extrusion direction or depth") from exc
        if (ratios.shape != (3,) or not np.isfinite(ratios).all() or ratios[2] == 0
                or not 0 < depth < math.inf):
            raise UnresolvedSolidGeometry("Expected a positive depth and a finite direction crossing the profile plane")
        # Normalize in the solid's coordinate system. Direction ratios can
        # have arbitrary scale; the profile's placement rotates the section
        # but must not rotate the extrusion a second time.
        direction = _unit_ratios(ratios)
        sweep = (prof_mat[:3, :3].T @ direction) * depth
        if not np.isfinite(sweep).all() or sweep[2] == 0:
            raise UnresolvedSolidGeometry("IFC sweep is not representable as a finite solid")
        verts, faces = _extrude_polygon_2d(outer, sweep, inner_pts=inner)
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
                            sub_meshes = _extract_solid_item(entities, sub, unit_scale, mass_issues, active_items, advanced_mesh, strict_geometry=strict_geometry)
                            if not sub_meshes:
                                raise UnresolvedSolidGeometry("unresolved physical item in mapped representation")
                            for (vv, ff) in sub_meshes:
                                out_items.append(_EMPTY_CLIP if vv is _EMPTY_CLIP[0] else (_apply_transform(vv, map_mat @ orig_mat), ff))
        return out_items

    if itype == "IFCTRIANGULATEDFACESET" and len(args) >= 4:
        if args[2] is False and mass_issues is not None:
            mass_issues.append("IFC triangulation is explicitly declared open")
        cref = args[0]
        if isinstance(cref, tuple) and cref[0] == "ref" and cref[1] in entities:
            _, cargs = entities[cref[1]]
            if cargs and isinstance(cargs[0], list):
                verts = np.array([[float(c) * unit_scale for c in pt] for pt in cargs[0]], dtype=np.float64)
                faces = np.array([[int(idx) - 1 for idx in tri] for tri in args[3]], dtype=np.int32)
                return [(verts, faces)]

    if itype == "IFCPOLYGONALFACESET":
        return _polygonal_solid_mesh(entities, args, unit_scale)

    if itype == "IFCADVANCEDBREP" and advanced_mesh is not None:
        return [advanced_mesh(iref[1])]
    if itype == "IFCFACETEDBREPWITHVOIDS":
        return _faceted_void_solid_mesh(entities, args, unit_scale)
    if itype == "IFCFACETEDBREP":
        return _faceted_solid_mesh(entities, args, unit_scale)
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


# Physical element occurrences qualified for additive rectangular-Body takeoffs.
MASS_PRODUCT_TYPES = frozenset({
    "IFCBEAM", "IFCBEAMSTANDARDCASE", "IFCBUILDINGELEMENTPROXY", "IFCCOLUMN",
    "IFCCOLUMNSTANDARDCASE", "IFCFOOTING", "IFCMEMBER", "IFCMEMBERSTANDARDCASE",
    "IFCPLATE", "IFCPLATESTANDARDCASE", "IFCSLAB", "IFCSLABSTANDARDCASE",
    "IFCWALL", "IFCWALLSTANDARDCASE",
})


def _product_shapes(entities, *, strict=False):
    """Yield each product owning a shape and its representations."""
    for eid, (etype, args) in sorted(entities.items()):
        if strict and etype in MASS_PRODUCT_TYPES and len(args) < 7:
            raise ValueError(f"Incomplete IFC product on #{eid}")
        if len(args) < 7 or not etype.startswith("IFC"):
            continue
        placement, rep = args[5:7]
        placed = (isinstance(placement, tuple) and placement[0] == "ref"
                  and entities.get(placement[1], ("",))[0].endswith("PLACEMENT"))
        shape = entities.get(rep[1]) if isinstance(rep, tuple) and rep[0] == "ref" else None
        if not shape or shape[0] != "IFCPRODUCTDEFINITIONSHAPE":
            if strict and (etype in MASS_PRODUCT_TYPES or (placed and rep is not None)):
                raise ValueError(f"Unresolved IFC product representation on #{eid}")
            continue
        rargs = shape[1]
        if len(rargs) < 3 or not isinstance(rargs[2], list):
            if strict:
                raise ValueError(f"Invalid IFC product definition shape on #{eid}")
            continue
        yield eid, etype, args, rargs


def _required_ifc_entity(entities, ref, kind):
    if isinstance(ref, tuple) and ref[0] == "ref":
        entity = entities.get(ref[1])
        if entity and entity[0] == kind:
            return entity[1]
    raise ValueError(f"Expected a resolved {kind} reference")


def _ifc_number(value):
    if type(value) not in (int, float):
        raise UnresolvedSolidGeometry("Expected a numeric IFC measure")
    number = float(value)
    if not math.isfinite(number):
        raise UnresolvedSolidGeometry("Nonfinite IFC measure")
    return number


def _ifc_vector(entities, ref, kind, dimension):
    attrs = _required_ifc_entity(entities, ref, kind)
    if len(attrs) != 1 or not isinstance(attrs[0], list) or len(attrs[0]) != dimension:
        raise UnresolvedSolidGeometry(f"Expected {dimension} numeric coordinates on {kind}")
    return np.array([_ifc_number(v) for v in attrs[0]])


def _closed_polyline(entities, ref, unit_scale):
    """Admit the authored simple, explicitly closed straight-line boundary."""
    attrs = _required_ifc_entity(entities, ref, "IFCPOLYLINE")
    if len(attrs) != 1 or not isinstance(attrs[0], list) or not 4 <= len(attrs[0]) <= 2049:
        raise UnresolvedSolidGeometry("Expected a bounded closed polygon with 3 to 2048 corners")
    points = np.array([_ifc_vector(entities, p, "IFCCARTESIANPOINT", 2) for p in attrs[0]]) * unit_scale
    if not np.isfinite(points).all() or not np.array_equal(points[0], points[-1]):
        raise UnresolvedSolidGeometry("IFC polygon lacks a finite, explicit closing edge")
    points = points[:-1]
    if len(np.unique(points, axis=0)) != len(points):
        raise UnresolvedSolidGeometry("Repeated IFC boundary corner")
    from shapely.geometry import Polygon
    polygon = Polygon(points)
    if not polygon.is_valid or not 0 < polygon.area < math.inf:
        raise UnresolvedSolidGeometry("Invalid IFC polygon boundary")
    return points


def _qualified_polygon_sweep(entities, item_ref, unit_scale):
    """Construct a finite extrusion from a fully resolved, straight IFC profile."""
    args = _required_ifc_entity(entities, item_ref, "IFCEXTRUDEDAREASOLID")
    if len(args) != 4 or not isinstance(args[0], tuple) or args[0][0] != "ref":
        raise UnresolvedSolidGeometry("Invalid IFC extrusion")
    profile, fields = entities.get(args[0][1], (None, []))
    if not fields or fields[0] != "AREA":
        raise UnresolvedSolidGeometry("Expected an IFC AREA profile")
    profile_frame = np.eye(4)
    if profile == "IFCARBITRARYCLOSEDPROFILEDEF" and len(fields) == 3:
        ring = _closed_polyline(entities, fields[2], unit_scale)
    elif profile == "IFCRECTANGLEPROFILEDEF" and len(fields) == 5:
        x, y = [_ifc_number(v) * unit_scale * .5 for v in fields[3:]]
        if not (0 < x < math.inf and 0 < y < math.inf):
            raise UnresolvedSolidGeometry("Invalid IFC rectangle dimensions")
        ring = np.array([(-x, -y), (x, -y), (x, y), (-x, y)])
        if fields[2] is not None:
            p, axis = _required_ifc_entity(entities, fields[2], "IFCAXIS2PLACEMENT2D")
            profile_frame[:2, 3] = _ifc_vector(entities, p, "IFCCARTESIANPOINT", 2) * unit_scale
            if axis is not None:
                direction = _ifc_vector(entities, axis, "IFCDIRECTION", 2)
                if not np.any(direction):
                    raise UnresolvedSolidGeometry("Invalid rectangle reference direction")
                d = _unit_ratios(direction)
                profile_frame[:2, :2] = ((d[0], -d[1]), (d[1], d[0]))
    else:
        raise UnresolvedSolidGeometry("Only closed straight polylines and rectangles are qualified for this extrusion")
    solid_frame = np.eye(4) if args[1] is None else _rigid_ifc_frame(entities, args[1], unit_scale)
    direction = _ifc_vector(entities, args[2], "IFCDIRECTION", 3)
    depth = _ifc_number(args[3]) * unit_scale
    if not 0 < depth < math.inf or direction[2] == 0:
        raise UnresolvedSolidGeometry("Invalid sweep depth or direction parallel to the profile")
    sweep = profile_frame[:3, :3].T @ _unit_ratios(direction) * depth
    vertices, faces = _extrude_polygon_2d(ring, sweep)
    vertices = _apply_transform(vertices, solid_frame @ profile_frame)
    if not np.isfinite(vertices).all():
        raise UnresolvedSolidGeometry("Nonfinite transformed sweep")
    source_volume = abs(_signed_area_2d(ring - ring[0]) * sweep[2])
    origin = vertices[faces[0, 0]]
    a, b, c = (vertices[faces[:, k]] - origin for k in range(3))
    mesh_volume = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())) / 6.
    if not (math.isfinite(source_volume) and source_volume > 0 and mesh_volume > 0
            and math.isclose(source_volume, mesh_volume, rel_tol=1e-8, abs_tol=0.)):
        raise UnresolvedSolidGeometry("Transformed mesh does not preserve the finite authored prism volume")
    return vertices, faces


def _rigid_ifc_frame(entities, ref, unit_scale):
    """Resolve a qualified rigid 3D placement for mass properties."""
    args = _required_ifc_entity(entities, ref, "IFCAXIS2PLACEMENT3D")
    if len(args) != 3:
        raise ValueError("Invalid IFC 3D frame attributes")

    def vector(ref, kind):
        attrs = _required_ifc_entity(entities, ref, kind)
        if len(attrs) != 1 or not isinstance(attrs[0], list) or len(attrs[0]) != 3:
            raise ValueError(f"Expected three coordinates for {kind}")
        v = np.array([_ifc_number(value) for value in attrs[0]])
        if not np.isfinite(v).all():
            raise ValueError(f"Nonfinite {kind}")
        if kind == "IFCDIRECTION":
            length = math.hypot(*v)
            if not 0 < length < math.inf:
                raise ValueError("Invalid IFC placement direction")
            v = v / length
        return v

    origin = vector(args[0], "IFCCARTESIANPOINT") * unit_scale
    if (args[1] is None) != (args[2] is None):
        raise ValueError("Qualified IFC 3D frame needs both axes or neither")
    mat = np.eye(4)
    if args[1] is not None:
        z = vector(args[1], "IFCDIRECTION")
        xref = vector(args[2], "IFCDIRECTION")
        y = np.cross(z, xref)
        length = math.hypot(*y)
        if length <= 64 * np.finfo(float).eps:
            raise ValueError("Degenerate or unresolved IFC placement axes")
        y /= length
        mat[:3, :3] = np.column_stack((np.cross(y, z), y, z))
    mat[:3, 3] = origin
    return mat


def _placed_product_frame(entities, placement, unit_scale):
    if placement is None:
        raise ValueError("An IFC product with a Body needs an object placement")
    frame = np.eye(4)
    visited = set()
    while placement is not None:
        args = _required_ifc_entity(entities, placement, "IFCLOCALPLACEMENT")
        if placement[1] in visited or len(args) != 2:
            raise ValueError("Cyclic or malformed IFC product placement")
        visited.add(placement[1])
        placement, local = args
        frame = _rigid_ifc_frame(entities, local, unit_scale) @ frame
    return frame


def _opening_body_items(entities, step, *, mass_takeoff=False):
    """Refuse partial bodies and curves the vertex reader cannot certify."""
    def resolve(ref):
        if not isinstance(ref, tuple) or ref[0] != "ref" or ref[1] not in entities:
            raise UnresolvedOpening("missing source geometry for opening subtraction")
        return entities[ref[1]]
    _, shape = resolve(entities[step][1][6])
    items = []
    body_count = 0
    for ref in shape[2]:
        kind, rep = resolve(ref)
        if kind != "IFCSHAPEREPRESENTATION" or len(rep) < 4:
            raise UnresolvedOpening("unverified opening representation")
        if str(rep[1]).upper() in ("AXIS","BOX","FOOTPRINT"):
            continue
        if str(rep[1]).upper() != "BODY" or not isinstance(rep[3], list):
            raise UnresolvedOpening("unverified opening body")
        body_count += 1
        if mass_takeoff:
            kind, attrs = resolve(rep[0])
            if kind == "IFCGEOMETRICREPRESENTATIONSUBCONTEXT" and len(attrs) > 6:
                kind, attrs = resolve(attrs[6])
            if kind != "IFCGEOMETRICREPRESENTATIONCONTEXT" or len(attrs) < 5 or attrs[2] != 3:
                raise ValueError("Expected a 3D IFC representation context")
            if not np.array_equal(_rigid_ifc_frame(entities, attrs[4], 1.), np.eye(4)):
                raise ValueError("Only an identity IFC world coordinate system is qualified")
        items.extend(rep[3])
    if body_count != 1 or not items:
        raise UnresolvedOpening("Expected one complete physical Body")
    return items


def _require_single_swept_body(entities, step, *, mass_takeoff=False):
    items = _opening_body_items(entities, step, mass_takeoff=mass_takeoff)
    def resolve(ref):
        if not isinstance(ref, tuple) or ref[0] != "ref" or ref[1] not in entities:
            raise UnresolvedOpening("Missing swept geometry")
        return entities[ref[1]]
    if len(items) != 1:
        raise UnresolvedOpening("expected one complete swept opening or host body")
    kind, solid = resolve(items[0])
    if kind != "IFCEXTRUDEDAREASOLID":
        raise UnresolvedOpening("only direct cuboid extrusions are qualified")
    if not solid:
        raise UnresolvedOpening("missing swept profile")
    profile, attrs = resolve(solid[0])
    if profile == "IFCRECTANGLEPROFILEDEF":
        return items[0][1]
    if not mass_takeoff and profile == "IFCARBITRARYCLOSEDPROFILEDEF":
        curve, args = resolve(attrs[2])
        if curve == "IFCPOLYLINE" and len(args[0]) == 5 and args[0][0] == args[0][-1]:
            if all(resolve(ref)[0] == "IFCCARTESIANPOINT" for ref in args[0]):
                return items[0][1]
    raise UnresolvedOpening("unsupported swept opening or host profile")


def _require_prism_or_clipped_prism(entities, item):
    visited = set()
    while isinstance(item, tuple) and item[0] == "ref" and item[1] not in visited and len(visited) < 128:
        visited.add(item[1])
        kind, fields = entities.get(item[1], (None, []))
        if kind == "IFCEXTRUDEDAREASOLID":
            return
        if kind != "IFCBOOLEANCLIPPINGRESULT" or len(fields) != 3 or fields[0] != "DIFFERENCE":
            break
        second = fields[2]
        if not isinstance(second, tuple) or second[0] != "ref" or entities.get(second[1], ("",))[0] not in (
                "IFCHALFSPACESOLID", "IFCPOLYGONALBOUNDEDHALFSPACE"):
            break
        item = fields[1]
    raise UnresolvedOpening("Expected a polygonal prism or a complete planar clipping chain")


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


def _physical_occurrences(solids, entities, unit_scale, related, empty=None):
    empty = set() if empty is None else empty
    by_step = {s.metadata["step_id"]: s for s in solids}
    opening_types = ("IFCOPENINGELEMENT", "IFCOPENINGSTANDARDCASE")
    for host, opening_steps in related.items():
        if (host not in by_step and host not in empty) or any(
                (i not in by_step and i not in empty) or entities.get(i, ("",))[0] not in opening_types
                for i in opening_steps):
            raise UnresolvedOpening("missing or unsupported host/opening body")
        if host in empty:
            continue
        material = by_step[host]
        openings = [by_step[i] for i in sorted(opening_steps) if i not in empty]
        if not openings:
            continue
        if any("mass_issue" in s.metadata or s.certified_union_volume_m3 is not None
               for s in [material, *openings]):
            raise UnresolvedOpening("Unqualified compound host or opening material")
        placement = entities[host][1][5]
        frame = np.eye(4) if placement is None else _placed_product_frame(entities, placement, unit_scale)
        try:
            for component in [material, *openings]:
                local = (component.vertices - frame[:3, 3]) @ frame[:3, :3]
                _box_bounds(local, component.faces)
        except UnresolvedOpening:
            meshes = [(s.vertices, s.faces) for s in [material, *openings]]
            vertices, faces = _boolean_meshes(meshes, "DIFFERENCE", allow_empty=True)
            by_step[host] = None if not len(faces) else SolidMesh3D(
                material.element_id, material.ifc_type, material.name, material.discipline,
                vertices, faces, {**material.metadata, "opening_ids": [o.element_id for o in openings]})
        else:
            by_step[host] = _subtract_opening_boxes(material, openings, frame)
    return [s for s in by_step.values() if s is not None and s.ifc_type not in opening_types]


def load_ifc_solids(ifc_text: str) -> List[SolidMesh3D]:
    entities = parse_ifc_entities(ifc_text)
    unit_scale = _detect_length_unit_scale(entities)
    related = _opening_relationships(entities)
    participants = set(related).union(*related.values())

    @cache
    def advanced_model():
        try:
            import ifcopenshell
            return ifcopenshell.file.from_string(ifc_text)
        except (ImportError, RuntimeError, ValueError) as exc:
            raise UnresolvedSolidGeometry("advanced BRep requires a readable IFC kernel model") from exc

    @cache
    def advanced_mesh(step):
        return _advanced_brep_mesh(advanced_model(), step)

    solids, empty = _gross_ifc_solids(entities, unit_scale, participants, _product_shapes(entities), advanced_mesh)
    return _physical_occurrences(solids, entities, unit_scale, related, empty)


def _gross_ifc_solids(entities, unit_scale, participants, products, advanced_mesh=None, *, strict_geometry=False):
    solids, empty = [], set()
    for eid, etype, args, rargs in products:
        if strict_geometry or eid in participants:
            items = _opening_body_items(entities, eid, mass_takeoff=strict_geometry)
            if eid in participants:
                for item in items:
                    _require_prism_or_clipped_prism(entities, item)
        global_id = str(args[0] or f"#{eid}")
        name = str(args[2] or f"{etype}_{eid}")
        if strict_geometry or (eid in participants and args[5] is not None):
            world_mat = _placed_product_frame(entities, args[5], unit_scale)
        else:
            world_mat = _resolve_local_placement(entities, args[5], unit_scale)
        has_empty = False
        mass_issues = []
        certified_union_volume = None
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
                try:
                    item_meshes = _extract_solid_item(entities, item_ref, unit_scale, mass_issues, advanced_mesh=advanced_mesh,
                                                     strict_geometry=strict_geometry or eid in participants)
                except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                    if eid in participants:
                        raise UnresolvedOpening(f"Unresolved source opening or host #{eid}: {exc}") from exc
                    raise
                if rep_ident == "BODY" and not item_meshes:
                    raise UnresolvedSolidGeometry(f"unsupported or empty physical Body item on #{eid}")
                for (v_local, f_local) in item_meshes:
                    if v_local is _EMPTY_CLIP[0]:
                        has_empty = True
                        continue
                    v_world = _apply_transform(v_local, world_mat)
                    all_verts.append(v_world)
                    all_faces.append(f_local + v_offset)
                    v_offset += len(v_world)

        if len(all_verts) > 1 and not mass_issues:
            bounds = [(v.min(axis=0), v.max(axis=0)) for v in all_verts]
            if any(np.all(np.minimum(a1, b1) > np.maximum(a0, b0))
                   for i, (a0, a1) in enumerate(bounds) for b0, b1 in bounds[i + 1:]):
                pieces = []
                offset = 0
                for vertices, faces in zip(all_verts, all_faces):
                    pieces.append((vertices, faces - offset))
                    offset += len(vertices)
                try:
                    vertices, faces = _boolean_meshes(pieces)
                    all_verts, all_faces = [vertices], [faces]
                except _VolumeOnlyUnion as exc:
                    # Retain original component surfaces for geometric queries.
                    # This exception never certifies an opening or clipping mesh.
                    certified_union_volume = exc.volume_m3
                except UnresolvedSolidGeometry as exc:
                    # Existing surface/containment queries remain useful. No
                    # additive mass is reported for an unqualified union.
                    mass_issues.append(str(exc))
        if all_verts and all_faces:
            solids.append(SolidMesh3D(
                element_id=global_id,
                ifc_type=etype,
                name=name,
                discipline=_infer_discipline(etype, name),
                vertices=np.vstack(all_verts),
                faces=np.vstack(all_faces),
                metadata={"step_id": eid, "unit_scale_m": unit_scale,
                          **({"mass_issue": mass_issues[0]} if mass_issues else {})},
                material_face_counts=tuple(len(faces) for faces in all_faces),
                certified_union_volume_m3=certified_union_volume,
            ))
        elif has_empty:
            empty.add(eid)
    return solids, empty


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


def _segment_segment_dist_3d(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    if a <= 1e-15 and e <= 1e-15:
        return float(np.linalg.norm(p1 - p2)), p1, p2
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
    return float(np.linalg.norm(c1 - c2)), c1, c2


def _triangle_triangle_distance_3d(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    hit = _triangles_intersect_3d(a0, a1, a2, b0, b1, b2)
    if hit is not None:
        return 0.0, hit, hit
    best_d = float("inf")
    best_a, best_b = a0, b0
    for p in (a0, a1, a2):
        q = _closest_point_on_triangle(p, b0, b1, b2)
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best_a, best_b = p, q
    for p in (b0, b1, b2):
        q = _closest_point_on_triangle(p, a0, a1, a2)
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best_a, best_b = q, p
    for (p1, q1) in ((a0, a1), (a1, a2), (a2, a0)):
        for (p2, q2) in ((b0, b1), (b1, b2), (b2, b0)):
            d, point_a, point_b = _segment_segment_dist_3d(p1, q1, p2, q2)
            if d < best_d:
                best_d = d
                best_a, best_b = point_a, point_b
    return best_d, best_a, best_b


def _closed_surface_components(vertices: np.ndarray, faces: np.ndarray):
    """Return one boundary vertex per closed shell and their triangle indices."""
    # Some tessellations repeat vertex indices across faces. Join exact positions
    # only, so distinct sides of a thin wall keep their physical gap.
    _, welded = np.unique(vertices, axis=0, return_inverse=True)
    edge_faces = {}
    neighbors = [[] for _ in faces]
    closed_face = [True] * len(faces)
    coherent_face = [True] * len(faces)
    for i, (a, b, c) in enumerate(welded[faces]):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(u, v), max(u, v))
            edge_faces.setdefault(edge, []).append((i, u < v))
    for (u, v), incident in edge_faces.items():
        if u != v and len(incident) == 2:
            (a, forward_a), (b, forward_b) = incident
            if forward_a == forward_b:
                coherent_face[a] = coherent_face[b] = False
            neighbors[a].append(b)
            neighbors[b].append(a)
        else:
            for i, _ in incident:
                closed_face[i] = False

    seeds = []
    closed_indices = []
    seen = set()
    for first in range(len(faces)):
        if first in seen:
            continue
        pending = [first]
        seen.add(first)
        component = []
        while pending:
            i = pending.pop()
            component.append(i)
            for other in neighbors[i]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        if all(closed_face[i] for i in component):
            seeds.append(vertices[faces[first, 0]])
            closed_indices.extend(component)
    return seeds, faces[closed_indices], all(coherent_face[i] for i in closed_indices)


def _material_boundaries(mesh: SolidMesh3D):
    start = 0
    boundaries = []
    for count in mesh.material_face_counts or (len(mesh.faces),):
        faces = mesh.faces[start:start + count]
        start += count
        # Remap only this solid's used vertices so coincident positive Body
        # items remain separate, including when their entire boundaries match.
        indices, local = np.unique(faces, return_inverse=True)
        seeds, triangles, coherent = _closed_surface_components(mesh.vertices[indices], local.reshape(faces.shape))
        if seeds:
            boundaries.append((seeds, indices[triangles], coherent))
    return boundaries


def _point_inside_mesh(pt: np.ndarray, mesh: SolidMesh3D, faces: np.ndarray, coherent: bool) -> bool:
    if len(faces) == 0:
        return False
    if np.any(pt < mesh.aabb_min - 1e-9) or np.any(pt > mesh.aabb_max + 1e-9):
        return False
    if coherent:
        # The signed solid angle is additive. Outward shells add occupied
        # material; an inward shell subtracts a true nested cavity.
        rays = mesh.vertices[faces] - pt
        lengths = np.linalg.norm(rays, axis=2)
        if np.any(lengths == 0):
            return True
        unit = rays / lengths[:, :, None]
        a, b, c = unit[:, 0], unit[:, 1], unit[:, 2]
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = 1 + np.sum(a * b + b * c + c * a, axis=1)
        # A whole turn in occupied volume subtends 4π steradians.
        return abs(float(np.sum(2 * np.arctan2(numerator, denominator)))) > 2 * math.pi
    # Preserve orientation-independent parity for a closed, inconsistently
    # wound tessellation. Separate IFC items are still tested independently.
    ray_dir = np.array([0.931723, 0.271828, 0.240719], dtype=np.float64)
    ray_dir /= np.linalg.norm(ray_dir)
    far = pt + ray_dir * (float(np.linalg.norm(mesh.aabb_max - mesh.aabb_min)) * 4.0 + 10.0)
    hits = 0
    v = mesh.vertices
    for f in faces:
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
    best_a, best_b = mesh_a.aabb_min, mesh_b.aabb_min

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
                        witness_point_m=tuple(map(float, hit)),
                        closest_points_m=(tuple(map(float, hit)),) * 2,
                        contact_evidence="triangle_boundary_intersection",
                        required_clearance_m=required_clearance_m,
                    )
        a_parts = _material_boundaries(mesh_a)
        b_parts = _material_boundaries(mesh_b)
        for candidate, container, containing_parts in ((a_parts, mesh_b, b_parts),
                                                       (b_parts, mesh_a, a_parts)):
            for seeds, _, _ in candidate:
                for point in seeds:
                    if any(_point_inside_mesh(point, container, faces, coherent)
                           for _, faces, coherent in containing_parts):
                        witness = tuple(map(float, point))
                        return ClashResult(
                            element_a=mesh_a.element_id,
                            element_b=mesh_b.element_id,
                            status="hard_clash",
                            min_distance_m=0.0,
                            overlap_aabb_volume_m3=ov_vol,
                            witness_point_m=witness,
                            closest_points_m=(witness, witness),
                            contact_evidence="containment_witness",
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
            d, point_a, point_b = _triangle_triangle_distance_3d(a0, a1, a2, b0, b1, b2)
            if d < best_dist:
                best_dist = d
                best_a, best_b = point_a, point_b
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
        contact_evidence="surface_distance_within_tolerance" if status == "hard_clash" else None,
        overlap_aabb_volume_m3=ov_vol if status == "hard_clash" else 0.0,
        witness_point_m=tuple(map(float, 0.5 * (best_a + best_b))),
        closest_points_m=(tuple(map(float, best_a)), tuple(map(float, best_b))),
        required_clearance_m=required_clearance_m,
    )


def _single_advanced_brep_takeoff(ifc_text):
    """Measure one placed curved Body when no pairwise clearance is required."""
    import ifcopenshell
    import ifcopenshell.geom
    from ifcopenshell import ifcopenshell_wrapper

    model = ifcopenshell.file.from_string(ifc_text)
    products = [p for p in model.by_type("IfcProduct") if p.Representation]
    if len(products) != 1 or getattr(products[0], "HasOpenings", ()):
        return None
    product = products[0]
    bodies = [r for r in product.Representation.Representations
              if r.RepresentationIdentifier not in ("Axis", "Box", "FootPrint")]
    if len(bodies) != 1:
        return None
    rep = bodies[0]
    visited = set()
    while rep.RepresentationIdentifier == "Body" and len(rep.Items) == 1:
        if rep.id() in visited:
            return None
        visited.add(rep.id())
        item = rep.Items[0]
        if not item.is_a("IfcMappedItem"):
            break
        rep = item.MappingSource.MappedRepresentation
    else:
        return None

    # A missing edge or disconnected boundary must not become a new solid
    # merely because the converter can close a nearby gap while sewing.
    if item.is_a() != "IfcAdvancedBrep" or not item.Outer.is_a("IfcClosedShell"):
        return None
    faces = item.Outer.CfsFaces
    ids = {face.id() for face in faces}
    if not ids or len(ids) != len(faces):
        return None
    edge_faces = {}
    for face in faces:
        if not face.is_a("IfcAdvancedFace") or not face.Bounds:
            return None
        for bound in face.Bounds:
            loop = bound.Bound
            if not loop.is_a("IfcEdgeLoop") or not loop.EdgeList:
                return None
            for edge in loop.EdgeList:
                edge_faces.setdefault(edge.EdgeElement.id(), []).append(face.id())
    neighbors = {fid: set() for fid in ids}
    for uses in edge_faces.values():
        if len(uses) != 2:
            return None
        first, second = uses
        neighbors[first].add(second)
        neighbors[second].add(first)
    reached = set()
    pending = [next(iter(ids))]
    while pending:
        face_id = pending.pop()
        if face_id not in reached:
            reached.add(face_id)
            pending.extend(neighbors[face_id] - reached)
    if reached != ids:
        return None

    settings = ifcopenshell.geom.settings()
    settings.set("iterator-output", ifcopenshell_wrapper.NATIVE)
    settings.set("reorient-shells", True)
    settings.set("use-world-coords", True)
    element = ifcopenshell.geom.create_shape(settings, product)
    geometry = element.geometry
    if geometry.size() != 1:
        return None
    shape = geometry.item(0)
    if not shape.is_manifold() or shape.num_faces() != len(faces):
        return None
    volume = shape.volume().to_double()
    area = shape.area().to_double()
    if not all(math.isfinite(v) and v > 0 for v in (volume, area)):
        return None
    return {
        "solid_count": 1, "pair_count": 0, "hard_clash_count": 0,
        "clearance_violation_count": 0, "min_clearance_m": None,
        "clashes": [], "all_pairs": [], "total_solid_volume_m3": volume,
        "total_surface_area_m2": area, "source_product_step_id": product.id(),
        "source_advanced_brep_step_id": item.id(), "source_face_count": len(faces),
        "volume_method": "ifcopenshell_reoriented_closed_brep",
    }


def detect_ifc_clashes(
    ifc_text: str,
    required_clearance_m: float = 0.0,
    cross_discipline_only: bool = False,
) -> Dict[str, Any]:
    upper = ifc_text.upper()
    if "IFCARCINDEX" in upper or "IFCREVOLVEDAREASOLID" in upper:
        from hamon_ifc_indexed_curves import curved_scene, has_direct_arc
        entities = parse_ifc_entities(ifc_text)
        if has_direct_arc(entities, _product_shapes(entities), include_revolutions=True):
            try:
                return curved_scene(ifc_text, _product_shapes(entities, strict=True),
                                    required_clearance_m, cross_discipline_only, _infer_discipline)
            except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, IndexError, ArithmeticError) as exc:
                raise UnresolvedSolidGeometry(f"Unresolved IFC curved Body: {exc}") from exc
    # A single closed curved product needs no distance mesh. Preserve the
    # native integral and its source identity before trying the mesh pipeline.
    if "IFCADVANCEDBREP" in ifc_text.upper():
        try:
            takeoff = _single_advanced_brep_takeoff(ifc_text)
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
            takeoff = None
        if takeoff is not None:
            return takeoff
    return _clashes_for_solids(load_ifc_solids(ifc_text), required_clearance_m, cross_discipline_only)


def _clashes_for_solids(solids, required_clearance_m, cross_discipline_only, *, include_quantity=True):
    # Bounds can prove clearance and eliminate candidates for the minimum.
    # Only a narrow-phase result supplies an attainable distance and witness.
    use_broadphase = len(solids) > 25
    mins = [s.aabb_min for s in solids]
    maxs = [s.aabb_max for s in solids]
    pairs = []
    for i, sa in enumerate(solids):
        for j in range(i + 1, len(solids)):
            if cross_discipline_only and sa.discipline == solids[j].discipline:
                continue
            bound = 0.0
            if use_broadphase:
                gap = np.maximum(0.0, np.maximum(mins[i] - maxs[j], mins[j] - maxs[i]))
                bound = math.hypot(*gap)
            pairs.append((bound, i, j))

    # Store rows in model order even though nearest bounds are refined first.
    results = [None] * len(pairs)
    min_clear = float("inf")
    for position in sorted(range(len(pairs)), key=lambda k: pairs[k][0]):
        bound, i, j = pairs[position]
        sa, sb = solids[i], solids[j]
        if use_broadphase and bound > max(required_clearance_m, min_clear) + 1e-6:
            result = ClashResult(element_a=sa.element_id, element_b=sb.element_id,
                                 status="clear", min_distance_m=None, witness_point_m=None,
                                 overlap_aabb_volume_m3=0.0, distance_lower_bound_m=bound,
                                 required_clearance_m=required_clearance_m)
        else:
            result = evaluate_pair_clash(sa, sb, required_clearance_m)
            min_clear = min(min_clear, result.min_distance_m)
        results[position] = result

    hard = [r for r in results if r.status == "hard_clash"]
    violations = [r for r in results if r.status == "clearance_violation"]
    quantity = {}
    if include_quantity:
        try:
            quantity["total_solid_volume_m3"] = float(sum(s.volume_m3() for s in solids))
        except UnresolvedMeshMass as exc:
            quantity.update(total_solid_volume_m3=None, volume_status="unresolved", volume_reason=str(exc))
    return {
        "solid_count": len(solids),
        "pair_count": len(results),
        "hard_clash_count": len(hard),
        "clearance_violation_count": len(violations),
        "min_clearance_m": float(min_clear) if results else None,
        **quantity,
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

        if type(payload.get("allow_partial", False)) is not bool:
            raise ValueError("allow_partial must be Boolean")
        if payload.get("allow_partial"):
            from hamon_ifc_partial import partial_clash_report
            return partial_clash_report(ifc_path.read_text(encoding="utf-8"), req_clear, cross_only)

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



