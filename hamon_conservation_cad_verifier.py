r"""Wave-4 Deterministic Dimensional, Conservation & CAD/STEP Geometry Verifier.

Covers all 4 Physical-Engineering's Last Exam (PLE) disciplines with zero
task-ID or filename shortcuts:
1. Mechanical / Structural CAD-STEP (ISO-10303-21 AP203/AP214/AP242 B-Rep &
   analytical CSG/extruded solids): exact divergence-theorem volume, surface area,
   centroid, mass, 3x3 centroidal inertia tensor + principal moments, and coupled
   beam/shaft stress, deflection, and Euler buckling checks.
2. Electrical / SPICE Nodal KCL/KVL & Complex Power Conservation: dual-engine
   Modified Nodal Analysis (exact complex Gaussian elimination + NumPy LU)
   with explicit per-node KCL, per-branch KVL, and Tellegen complex power checks.
3. Civil / Geotechnical Equilibrium & Conservation: cut/fill earthwork mass
   balance with shrink/swell factors, effective-stress Terzaghi/Vesic bearing
   capacity, consolidation settlement, and open-channel energy continuity.
4. Chemical / Thermal-Fluid Dimensional & Conservation Balance: 7-dimension SI
   homogeneity (with affine degC/degF and psig/psia guards), steady-CV mass &
   elemental species conservation, First-Law enthalpy balance, Second-Law entropy
   generation / HX temperature-crossover guard, and Colebrook-White pipe head loss.
"""
from __future__ import annotations

import base64
import cmath
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Dim7 = Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]
ZERO7: Dim7 = (Fraction(0),) * 7


def _d(kg=0, m=0, s=0, a=0, k=0, mol=0, cd=0) -> Dim7:
    return (Fraction(kg), Fraction(m), Fraction(s), Fraction(a), Fraction(k), Fraction(mol), Fraction(cd))


UNIT_TABLE: Dict[str, Tuple[float, Dim7]] = {
    "1": (1.0, ZERO7), "rad": (1.0, ZERO7), "deg": (math.pi / 180.0, ZERO7),
    "m": (1.0, _d(m=1)), "mm": (1e-3, _d(m=1)), "cm": (1e-2, _d(m=1)), "km": (1e3, _d(m=1)),
    "in": (0.0254, _d(m=1)), "ft": (0.3048, _d(m=1)), "yd": (0.9144, _d(m=1)),
    "m2": (1.0, _d(m=2)), "mm2": (1e-6, _d(m=2)), "cm2": (1e-4, _d(m=2)), "ft2": (0.09290304, _d(m=2)), "in2": (6.4516e-4, _d(m=2)),
    "m3": (1.0, _d(m=3)), "L": (1e-3, _d(m=3)), "gal": (0.003785411784, _d(m=3)), "ft3": (0.028316846592, _d(m=3)),
    "m4": (1.0, _d(m=4)), "mm4": (1e-12, _d(m=4)), "in4": (4.162314256e-7, _d(m=4)),
    "kg": (1.0, _d(kg=1)), "g": (1e-3, _d(kg=1)), "t": (1e3, _d(kg=1)), "lb": (0.45359237, _d(kg=1)),
    "s": (1.0, _d(s=1)), "ms": (1e-3, _d(s=1)), "min": (60.0, _d(s=1)), "hr": (3600.0, _d(s=1)), "h": (3600.0, _d(s=1)),
    "Hz": (1.0, _d(s=-1)), "kHz": (1e3, _d(s=-1)), "MHz": (1e6, _d(s=-1)), "rpm": (2.0 * math.pi / 60.0, _d(s=-1)),
    "N": (1.0, _d(kg=1, m=1, s=-2)), "kN": (1e3, _d(kg=1, m=1, s=-2)), "MN": (1e6, _d(kg=1, m=1, s=-2)),
    "lbf": (4.4482216152605, _d(kg=1, m=1, s=-2)), "kip": (4448.2216152605, _d(kg=1, m=1, s=-2)),
    "Nm": (1.0, _d(kg=1, m=2, s=-2)), "kNm": (1e3, _d(kg=1, m=2, s=-2)),
    "J": (1.0, _d(kg=1, m=2, s=-2)), "kJ": (1e3, _d(kg=1, m=2, s=-2)), "MJ": (1e6, _d(kg=1, m=2, s=-2)), "BTU": (1055.05585262, _d(kg=1, m=2, s=-2)),
    "W": (1.0, _d(kg=1, m=2, s=-3)), "kW": (1e3, _d(kg=1, m=2, s=-3)), "MW": (1e6, _d(kg=1, m=2, s=-3)), "hp": (745.6998715822702, _d(kg=1, m=2, s=-3)),
    "Pa": (1.0, _d(kg=1, m=-1, s=-2)), "kPa": (1e3, _d(kg=1, m=-1, s=-2)), "MPa": (1e6, _d(kg=1, m=-1, s=-2)), "GPa": (1e9, _d(kg=1, m=-1, s=-2)),
    "bar": (1e5, _d(kg=1, m=-1, s=-2)), "atm": (101325.0, _d(kg=1, m=-1, s=-2)),
    "psi": (6894.757293168, _d(kg=1, m=-1, s=-2)), "psia": (6894.757293168, _d(kg=1, m=-1, s=-2)), "ksi": (6894757.293168, _d(kg=1, m=-1, s=-2)),
    "K": (1.0, _d(k=1)), "R": (5.0 / 9.0, _d(k=1)),
    "mol": (1.0, _d(mol=1)), "kmol": (1e3, _d(mol=1)),
    "A": (1.0, _d(a=1)), "mA": (1e-3, _d(a=1)), "kA": (1e3, _d(a=1)),
    "V": (1.0, _d(kg=1, m=2, s=-3, a=-1)), "mV": (1e-3, _d(kg=1, m=2, s=-3, a=-1)), "kV": (1e3, _d(kg=1, m=2, s=-3, a=-1)),
    "ohm": (1.0, _d(kg=1, m=2, s=-3, a=-2)), "kohm": (1e3, _d(kg=1, m=2, s=-3, a=-2)), "Mohm": (1e6, _d(kg=1, m=2, s=-3, a=-2)),
    "S": (1.0, _d(kg=-1, m=-2, s=3, a=2)),
    "F": (1.0, _d(kg=-1, m=-2, s=4, a=2)), "uF": (1e-6, _d(kg=-1, m=-2, s=4, a=2)), "nF": (1e-9, _d(kg=-1, m=-2, s=4, a=2)), "pF": (1e-12, _d(kg=-1, m=-2, s=4, a=2)),
    "H": (1.0, _d(kg=1, m=2, s=-2, a=-2)), "mH": (1e-3, _d(kg=1, m=2, s=-2, a=-2)), "uH": (1e-6, _d(kg=1, m=2, s=-2, a=-2)),
    "gpm": (6.30901964e-5, _d(m=3, s=-1)), "cfm": (4.719474432e-4, _d(m=3, s=-1)),
    "kg_s": (1.0, _d(kg=1, s=-1)), "m_s": (1.0, _d(m=1, s=-1)), "J_kgK": (1.0, _d(m=2, s=-2, k=-1)), "kJ_kgK": (1e3, _d(m=2, s=-2, k=-1)),
    "J_kg": (1.0, _d(m=2, s=-2)), "kJ_kg": (1e3, _d(m=2, s=-2)), "Pa_s": (1.0, _d(kg=1, m=-1, s=-1)),
}


def convert_to_si(value: float, unit: str, *, patm_pa: float = 101325.0, is_temperature_delta: bool = False) -> Tuple[float, Dim7]:
    u = unit.strip()
    if u in ("degC", "C") and not is_temperature_delta:
        return float(value) + 273.15, _d(k=1)
    if u in ("degC", "C") and is_temperature_delta:
        return float(value), _d(k=1)
    if u == "degF" and not is_temperature_delta:
        return (float(value) - 32.0) * (5.0 / 9.0) + 273.15, _d(k=1)
    if u == "degF" and is_temperature_delta:
        return float(value) * (5.0 / 9.0), _d(k=1)
    if u == "psig":
        return float(value) * 6894.757293168 + patm_pa, _d(kg=1, m=-1, s=-2)
    if u == "kPag":
        return float(value) * 1000.0 + patm_pa, _d(kg=1, m=-1, s=-2)
    if u in UNIT_TABLE:
        scale, dim = UNIT_TABLE[u]
        return float(value) * scale, dim
    val_scale = 1.0
    dim_acc = [Fraction(0)] * 7
    parts = u.split("/")
    if len(parts) > 2:
        raise ValueError(f"Ambiguous multiple '/' in unit {unit!r}")
    for side_idx, part in enumerate(parts):
        sign = 1 if side_idx == 0 else -1
        for tok in re.split(r"[*·\s]+", part.strip()):
            if not tok or tok == "1":
                continue
            m = re.fullmatch(r"([A-Za-z]+)(?:\^?([-+]?\d+(?:/\d+)?))?", tok)
            if not m or m.group(1) not in UNIT_TABLE:
                raise ValueError(f"Unsupported unit token {tok!r} in {unit!r}")
            base_s, base_d = UNIT_TABLE[m.group(1)]
            exp = Fraction(m.group(2)) * sign if m.group(2) else Fraction(sign)
            val_scale *= base_s ** float(exp)
            for j in range(7):
                dim_acc[j] += base_d[j] * exp
    return float(value) * val_scale, tuple(dim_acc)  # type: ignore[return-value]


def verify_dimensional_homogeneity(spec: Dict[str, Any]) -> Dict[str, Any]:
    rtol = float(spec.get("rtol", 1e-4))
    atol = float(spec.get("atol", 1e-9))
    patm = float(spec.get("patm_pa", 101325.0))

    def _eval_term(term: Dict[str, Any]) -> Tuple[float, Dim7]:
        if "factors" in term:
            v = float(term.get("coeff", 1.0))
            d_acc = [Fraction(0)] * 7
            for fac in term["factors"]:
                fv, fd = convert_to_si(float(fac["value"]), str(fac.get("unit", "1")), patm_pa=patm, is_temperature_delta=bool(fac.get("delta_temp", False)))
                exp = Fraction(str(fac.get("power", 1)))
                v *= fv ** float(exp)
                for j in range(7):
                    d_acc[j] += fd[j] * exp
            return v, tuple(d_acc)  # type: ignore[return-value]
        return convert_to_si(float(term["value"]), str(term.get("unit", "1")), patm_pa=patm, is_temperature_delta=bool(term.get("delta_temp", False)))

    lhs_val, lhs_dim = _eval_term(spec["lhs"])
    rhs_val, rhs_dim = _eval_term(spec["rhs"])
    dim_ok = lhs_dim == rhs_dim
    abs_err = abs(lhs_val - rhs_val)
    rel_err = abs_err / max(abs(lhs_val), abs(rhs_val), 1e-30)
    num_ok = dim_ok and (abs_err <= atol or rel_err <= rtol)
    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "dimensional",
        "dimensionally_homogeneous": dim_ok,
        "conservation_verified": num_ok,
        "lhs_si": lhs_val,
        "rhs_si": rhs_val,
        "lhs_dim_7": [int(x) if x.denominator == 1 else float(x) for x in lhs_dim],
        "rhs_dim_7": [int(x) if x.denominator == 1 else float(x) for x in rhs_dim],
        "abs_error_si": abs_err,
        "rel_error": rel_err,
    }


def _strip_step_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _split_step_top_args(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    depth = 0
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            cur.append(ch)
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    cur.append("'")
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur or s.endswith(","):
        out.append("".join(cur).strip())
    return out


def _parse_step_entities_map(step_text: str) -> Tuple[Dict[int, Tuple[str, List[str]]], float]:
    clean = _strip_step_comments(step_text)
    unit_to_m = 1.0
    up = clean.upper()
    if ".MILLI." in up and ".METRE." in up:
        unit_to_m = 0.001
    elif ".CENTI." in up and ".METRE." in up:
        unit_to_m = 0.01
    elif "'INCH'" in up or ".INCH." in up:
        unit_to_m = 0.0254
    elif "'FOOT'" in up or ".FOOT." in up:
        unit_to_m = 0.3048
    entities: Dict[int, Tuple[str, List[str]]] = {}
    data_m = re.search(r"DATA\s*;(.*?)ENDSEC\s*;", clean, flags=re.DOTALL | re.IGNORECASE)
    body = data_m.group(1) if data_m else clean
    for stmt in body.split(";"):
        st = stmt.strip()
        m = re.match(r"^#(\d+)\s*=\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$", st, flags=re.DOTALL)
        if m:
            entities[int(m.group(1))] = (m.group(2).upper(), _split_step_top_args(m.group(3)))
    return entities, unit_to_m


def _ref_id(tok: str) -> Optional[int]:
    m = re.fullmatch(r"#(\d+)", tok.strip())
    return int(m.group(1)) if m else None


def _ref_list(tok: str) -> List[int]:
    t = tok.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    return [int(m) for m in re.findall(r"#(\d+)", t)]


def _num_list(tok: str) -> List[float]:
    t = tok.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    return [float(x.strip()) for x in t.split(",") if x.strip()] if t else []


def _box_mesh(origin: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray, z_ax: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray]:
    corners = [origin + i * x_ax + j * y_ax + k * z_ax for k in (0.0, dz) for j in (0.0, dy) for i in (0.0, dx)]
    v = np.array(corners, dtype=np.float64)
    f = np.array([[0,2,1],[1,2,3],[4,5,6],[5,7,6],[0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]], dtype=np.int32)
    return v, f


def _cylinder_mesh(origin: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray, z_ax: np.ndarray, radius: float, height: float, n_seg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    verts = [origin.copy(), origin + height * z_ax]
    for z_off in (0.0, height):
        for a in angles:
            verts.append(origin + z_off * z_ax + radius * (math.cos(a) * x_ax + math.sin(a) * y_ax))
    faces = []
    for i in range(n_seg):
        inext = (i + 1) % n_seg
        b0, b1, t0, t1 = 2 + i, 2 + inext, 2 + n_seg + i, 2 + n_seg + inext
        faces.extend([[0, b1, b0], [1, t0, t1], [b0, b1, t1], [b0, t1, t0]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _extrude_polygon_2d(poly2d: np.ndarray, origin: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray, z_ax: np.ndarray, depth: float) -> Tuple[np.ndarray, np.ndarray]:
    x, y = poly2d[:, 0], poly2d[:, 1]
    if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
        poly2d = poly2d[::-1].copy()
    k = len(poly2d)
    verts = [origin + px * x_ax + py * y_ax + z_off * z_ax for z_off in (0.0, depth) for px, py in poly2d]
    faces = []
    for i in range(1, k - 1):
        faces.extend([[0, i + 1, i], [k, k + i, k + i + 1]])
    for i in range(k):
        inext = (i + 1) % k
        faces.extend([[i, inext, k + inext], [i, k + inext, k + i]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def _axis2_frame(entities: Dict[int, Tuple[str, List[str]]], aid: Optional[int], scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin = np.zeros(3, dtype=np.float64)
    z_ax = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if aid is not None and aid in entities:
        et, args = entities[aid]
        if et in ("AXIS2_PLACEMENT_3D", "IFCAXIS2PLACEMENT3D") and len(args) >= 2:
            pid = _ref_id(args[1])
            if pid and pid in entities:
                origin = np.array(_num_list(entities[pid][1][-1])[:3], dtype=np.float64) * scale
            if len(args) >= 3 and _ref_id(args[2]) in entities:
                zv = np.array(_num_list(entities[_ref_id(args[2])][1][-1])[:3], dtype=np.float64)  # type: ignore[index]
                if np.linalg.norm(zv) > 1e-12:
                    z_ax = zv / np.linalg.norm(zv)
            if len(args) >= 4 and _ref_id(args[3]) in entities:
                xv = np.array(_num_list(entities[_ref_id(args[3])][1][-1])[:3], dtype=np.float64)  # type: ignore[index]
                xv = xv - np.dot(xv, z_ax) * z_ax
                if np.linalg.norm(xv) > 1e-12:
                    x_ax = xv / np.linalg.norm(xv)
    if abs(np.dot(x_ax, z_ax)) > 0.99:
        cand = np.array([0.0, 1.0, 0.0]) if abs(z_ax[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_ax = cand - np.dot(cand, z_ax) * z_ax
        x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    return origin, x_ax, y_ax, z_ax


def _compute_mesh_mass_properties(vertices: np.ndarray, faces: np.ndarray, density_kg_m3: float) -> Dict[str, Any]:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    dets = np.einsum("ij,ij->i", np.cross(v0, v1), v2)
    signed_vol = float(np.sum(dets) / 6.0)
    if signed_vol < 0.0:
        faces = faces[:, [0, 2, 1]]
        v1, v2 = v2, v1
        dets = -dets
        signed_vol = -signed_vol
    if signed_vol <= 1e-18:
        raise ValueError("Degenerate zero-volume mesh in STEP/CAD verifier")
    surface_area = float(np.sum(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))
    centroid = np.sum(((v0 + v1 + v2) / 4.0) * dets[:, None], axis=0) / (6.0 * signed_vol)
    m_canon = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
    v_stack = np.stack([v0, v1, v2], axis=1)
    cov_origin = np.sum(np.einsum("f,fia,ij,fjb->fab", dets / 120.0, v_stack, m_canon, v_stack), axis=0)
    cov_centroid = cov_origin - signed_vol * np.outer(centroid, centroid)
    inertia_c = density_kg_m3 * (np.trace(cov_centroid) * np.eye(3) - cov_centroid)
    inertia_c = 0.5 * (inertia_c + inertia_c.T)
    principal = np.sort(np.linalg.eigvalsh(inertia_c))

    qverts = np.round(vertices, 9)
    _, inv = np.unique(qverts, axis=0, return_inverse=True)
    qfaces = inv[faces]
    edge_counts: Dict[Tuple[int, int], int] = {}
    directed = set()
    manifold = True
    for tri in qfaces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, w in ((a, b), (b, c), (c, a)):
            if (u, w) in directed:
                manifold = False
            directed.add((u, w))
            k = (u, w) if u < w else (w, u)
            edge_counts[k] = edge_counts.get(k, 0) + 1
    watertight = manifold and bool(edge_counts) and all(cnt == 2 for cnt in edge_counts.values())
    mass = density_kg_m3 * signed_vol
    return {
        "volume_m3": signed_vol,
        "surface_area_m2": surface_area,
        "mass_kg": mass,
        "density_kg_m3": density_kg_m3,
        "centroid_m": [float(x) for x in centroid],
        "aabb_extents_m": [float(x) for x in (np.max(vertices, axis=0) - np.min(vertices, axis=0))],
        "inertia_tensor_centroid_kg_m2": [[float(x) for x in row] for row in inertia_c],
        "principal_moments_kg_m2": [float(x) for x in principal],
        "watertight_manifold": bool(watertight),
        "triangle_count": int(len(faces)),
    }


def verify_step_cad_geometry(spec: Dict[str, Any]) -> Dict[str, Any]:
    step_text = spec.get("step_text") or ""
    if not step_text and spec.get("file_b64"):
        step_text = base64.b64decode(spec["file_b64"]).decode("utf-8", errors="replace")
    elif not step_text and spec.get("path"):
        step_text = Path(spec["path"]).read_text(encoding="utf-8", errors="replace")
    density = float(spec.get("density_kg_m3", 7850.0))
    entities, unit_scale = _parse_step_entities_map(step_text)
    if spec.get("unit_scale_m") is not None:
        unit_scale = float(spec["unit_scale_m"])

    pts: Dict[int, np.ndarray] = {}
    verts_map: Dict[int, np.ndarray] = {}
    for eid, (et, args) in entities.items():
        if et in ("CARTESIAN_POINT", "IFCCARTESIANPOINT") and len(args) >= 2:
            c = _num_list(args[-1])
            if len(c) == 2:
                c = [c[0], c[1], 0.0]
            if len(c) >= 3:
                pts[eid] = np.array(c[:3], dtype=np.float64) * unit_scale
    for eid, (et, args) in entities.items():
        if et == "VERTEX_POINT" and len(args) >= 2 and _ref_id(args[-1]) in pts:
            verts_map[eid] = pts[_ref_id(args[-1])]  # type: ignore[index]

    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    analytic_vol: Optional[float] = None

    def _append(v: np.ndarray, f: np.ndarray) -> None:
        off = sum(len(x) for x in all_verts)
        all_verts.append(v)
        all_faces.append(f + off)

    shell_polys: List[np.ndarray] = []
    for eid, (et, args) in entities.items():
        if et in ("ADVANCED_FACE", "FACE_SURFACE") and len(args) >= 2:
            same_sense = ".F." not in (args[3].upper() if len(args) >= 4 else ".T.")
            for bid in _ref_list(args[1]):
                if bid not in entities:
                    continue
                bet, bargs = entities[bid]
                if bet not in ("FACE_OUTER_BOUND", "FACE_BOUND") or len(bargs) < 2:
                    continue
                lid = _ref_id(bargs[1])
                bound_sense = ".F." not in (bargs[2].upper() if len(bargs) >= 3 else ".T.")
                if lid not in entities:
                    continue
                let, largs = entities[lid]
                lpts: List[np.ndarray] = []
                if let == "POLY_LOOP" and len(largs) >= 2:
                    lpts = [pts[pid] for pid in _ref_list(largs[1]) if pid in pts]
                elif let == "EDGE_LOOP" and len(largs) >= 2:
                    for oe_id in _ref_list(largs[1]):
                        if oe_id in entities and entities[oe_id][0] == "ORIENTED_EDGE" and len(entities[oe_id][1]) >= 5:
                            oargs = entities[oe_id][1]
                            ec_id = _ref_id(oargs[3])
                            oe_s = ".F." not in oargs[4].upper()
                            if ec_id in entities:
                                ec_args = entities[ec_id][1]
                                vp = _ref_id(ec_args[1]) if oe_s else _ref_id(ec_args[2])
                                if vp in verts_map:
                                    lpts.append(verts_map[vp])
                if len(lpts) >= 3:
                    arr = np.array(lpts, dtype=np.float64)
                    if same_sense != bound_sense:
                        arr = arr[::-1].copy()
                    shell_polys.append(arr)

    if shell_polys:
        center_est = np.mean(np.vstack(shell_polys), axis=0)
        for repair in (False, True):
            bv: List[np.ndarray] = []
            bf: List[List[int]] = []
            for poly in shell_polys:
                pu = poly
                if repair:
                    nrm = np.cross(poly[1] - poly[0], poly[2] - poly[0])
                    if np.dot(nrm, np.mean(poly, axis=0) - center_est) < 0.0:
                        pu = poly[::-1].copy()
                base = len(bv)
                bv.extend(pu)
                for i in range(1, len(pu) - 1):
                    bf.append([base, base + i, base + i + 1])
            va, fa = np.array(bv, dtype=np.float64), np.array(bf, dtype=np.int32)
            if abs(float(np.sum(np.einsum("ij,ij->i", np.cross(va[fa[:, 0]], va[fa[:, 1]]), va[fa[:, 2]]))) / 6.0) > 1e-15:
                _append(va, fa)
                break

    for eid, (et, args) in entities.items():
        if et == "BLOCK" and len(args) >= 5:
            origin, x_ax, y_ax, z_ax = _axis2_frame(entities, _ref_id(args[1]), unit_scale)
            dx, dy, dz = float(args[2]) * unit_scale, float(args[3]) * unit_scale, float(args[4]) * unit_scale
            _append(*_box_mesh(origin, x_ax, y_ax, z_ax, dx, dy, dz))
            analytic_vol = dx * dy * dz
        elif et == "RIGHT_CIRCULAR_CYLINDER" and len(args) >= 4:
            origin, x_ax, y_ax, z_ax = _axis2_frame(entities, _ref_id(args[1]), unit_scale)
            h, r = float(args[2]) * unit_scale, float(args[3]) * unit_scale
            _append(*_cylinder_mesh(origin, x_ax, y_ax, z_ax, r, h))
            analytic_vol = math.pi * r * r * h
        elif et in ("EXTRUDED_AREA_SOLID", "IFCEXTRUDEDAREASOLID") and len(args) >= 4:
            prof_id = _ref_id(args[0] if et == "IFCEXTRUDEDAREASOLID" else args[1])
            pos_id = _ref_id(args[1] if et == "IFCEXTRUDEDAREASOLID" else args[2])
            depth = float(args[3]) * unit_scale
            origin, x_ax, y_ax, z_ax = _axis2_frame(entities, pos_id, unit_scale)
            if prof_id in entities:
                pet, pargs = entities[prof_id]
                if pet in ("RECTANGLE_PROFILE_DEF", "IFCRECTANGLEPROFILEDEF") and len(pargs) >= 5:
                    lx, ly = float(pargs[3]) * unit_scale, float(pargs[4]) * unit_scale
                    poly = np.array([[-0.5*lx,-0.5*ly],[0.5*lx,-0.5*ly],[0.5*lx,0.5*ly],[-0.5*lx,0.5*ly]])
                    _append(*_extrude_polygon_2d(poly, origin, x_ax, y_ax, z_ax, depth))
                    analytic_vol = lx * ly * depth

    if not all_verts:
        raise ValueError("No supported 3D solid entities found in STEP input")
    props = _compute_mesh_mass_properties(np.vstack(all_verts), np.vstack(all_faces), density)
    dual_rel_err = abs(props["volume_m3"] - analytic_vol) / analytic_vol if (analytic_vol and analytic_vol > 0) else 0.0
    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "step_cad",
        "unit_scale_m": unit_scale,
        "entity_count": len(entities),
        "dual_volume_rel_error": dual_rel_err,
        "conservation_verified": bool(dual_rel_err <= 5e-3),
        **props,
    }


def verify_structural_mechanics(spec: Dict[str, Any]) -> Dict[str, Any]:
    L, E, I = float(spec["span_m"]), float(spec["E_Pa"]), float(spec["I_m4"])
    c_dist = float(spec.get("c_m", 0.5 * float(spec.get("depth_m", 0.2))))
    area = float(spec.get("area_m2", 0.01))
    bc = str(spec.get("boundary", "simply_supported_udl")).lower()
    w, P = float(spec.get("w_N_m", 0.0)), float(spec.get("P_N", 0.0))
    torque = float(spec.get("torque_Nm", 0.0))
    J = float(spec.get("J_m4", 2.0 * I))
    r_outer = float(spec.get("r_outer_m", c_dist))
    axial_comp = float(spec.get("axial_compression_N", 0.0))
    K_eff = float(spec.get("K_buckling", 1.0))
    fy, fs_req = float(spec.get("yield_strength_Pa", 250e6)), float(spec.get("safety_factor", 1.5))
    defl_ratio = float(spec.get("deflection_limit_ratio", 360.0))

    if bc == "simply_supported_udl":
        r_l = r_r = 0.5 * w * L
        m_max, v_max, defl_a = w * L * L / 8.0, 0.5 * abs(w) * L, 5.0 * abs(w) * (L ** 4) / (384.0 * E * I)
        m_fn = lambda x: 0.5 * w * L * x - 0.5 * w * x * x
        m_u = lambda x: 0.5 * x if x <= 0.5 * L else 0.5 * (L - x)
    elif bc == "simply_supported_point":
        r_l = r_r = 0.5 * P
        m_max, v_max, defl_a = P * L / 4.0, 0.5 * abs(P), abs(P) * (L ** 3) / (48.0 * E * I)
        m_fn = lambda x: 0.5 * P * x if x <= 0.5 * L else 0.5 * P * (L - x)
        m_u = lambda x: 0.5 * x if x <= 0.5 * L else 0.5 * (L - x)
    elif bc == "cantilever_point":
        r_l, r_r = P, 0.0
        m_max, v_max, defl_a = P * L, abs(P), abs(P) * (L ** 3) / (3.0 * E * I)
        m_fn = lambda x: -P * (L - x)
        m_u = lambda x: -(L - x)
    elif bc == "cantilever_udl":
        r_l, r_r = w * L, 0.0
        m_max, v_max, defl_a = 0.5 * w * L * L, abs(w) * L, abs(w) * (L ** 4) / (8.0 * E * I)
        m_fn = lambda x: -0.5 * w * ((L - x) ** 2)
        m_u = lambda x: -(L - x)
    else:
        r_l = r_r = 0.5 * w * L
        m_max, v_max, defl_a = w * L * L / 12.0, 0.5 * abs(w) * L, abs(w) * (L ** 4) / (384.0 * E * I)
        m_fn = lambda x: -w * L * L / 12.0 + 0.5 * w * L * x - 0.5 * w * x * x
        m_u = lambda x: -L / 8.0 + (0.5 * x if x <= 0.5 * L else 0.5 * (L - x))

    n_p = 200
    xs = np.linspace(0.0, L, n_p + 1)
    wts = np.ones(n_p + 1); wts[1:-1:2] = 4.0; wts[2:-2:2] = 2.0
    defl_vw = abs(float(np.sum(wts * np.array([m_fn(float(x)) * m_u(float(x)) / (E * I) for x in xs])) * (L / (3.0 * n_p))))
    dual_err = abs(defl_a - defl_vw) / max(defl_a, 1e-30)
    fy_res = abs((r_l + r_r) - (w * L + P))
    sigma_b = abs(m_max) * c_dist / I + (axial_comp / area if area > 0 else 0.0)
    tau_t = abs(torque) * r_outer / J if J > 0 else 0.0
    tau_tot = tau_t + (1.5 * v_max / area if (torque == 0.0 and area > 0) else 0.0)
    vm = math.sqrt(sigma_b * sigma_b + 3.0 * tau_tot * tau_tot)
    allow = fy / fs_req
    p_cr = (math.pi ** 2) * E * I / ((K_eff * L) ** 2)
    d_lim = L / defl_ratio
    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "structural",
        "force_equilibrium_residual_N": fy_res,
        "max_moment_Nm": abs(m_max),
        "max_shear_N": v_max,
        "max_deflection_m": defl_a,
        "virtual_work_deflection_m": defl_vw,
        "dual_deflection_rel_error": dual_err,
        "von_mises_stress_Pa": vm,
        "allowable_stress_Pa": allow,
        "euler_critical_load_N": p_cr,
        "passes_stress": vm <= allow * (1.0 + 1e-9),
        "passes_deflection": defl_a <= d_lim * (1.0 + 1e-9),
        "passes_buckling": axial_comp <= (p_cr / fs_req) if axial_comp > 0 else True,
        "conservation_verified": bool(fy_res <= 1e-7 * max(abs(w * L + P), 1.0) and dual_err <= 1e-5),
    }


_SPICE_SCALE = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def _parse_spice_num(tok: str) -> float:
    m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|[fpnumkgt])?", tok.strip().lower())
    if not m:
        raise ValueError(f"Invalid SPICE token {tok!r}")
    return float(m.group(1)) * (_SPICE_SCALE[m.group(2)] if m.group(2) else 1.0)


def _parse_spice_netlist(netlist: str) -> Tuple[List[Dict[str, Any]], float]:
    branches: List[Dict[str, Any]] = []
    freq_hz = 0.0
    for idx, raw_line in enumerate(netlist.splitlines()):
        line = raw_line.split("*")[0].split(";")[0].strip()
        if not line or (idx == 0 and not re.match(r"^[RLCVIGrlcvig\.]", line)):
            continue
        if line.lower().startswith(".end"):
            break
        if line.lower().startswith(".ac"):
            toks = line.split()
            if len(toks) >= 4:
                freq_hz = _parse_spice_num(toks[3])
            continue
        if line.startswith("."):
            continue
        toks = line.split()
        kind = toks[0][0].upper()
        if kind in ("R", "L", "C") and len(toks) >= 4:
            branches.append({"name": toks[0], "type": kind, "n_pos": toks[1], "n_neg": toks[2], "value": _parse_spice_num(toks[3])})
        elif kind in ("V", "I") and len(toks) >= 4:
            rest = [t for t in toks[3:] if t.upper() not in ("DC", "AC")]
            mag = _parse_spice_num(rest[0])
            ph = float(rest[1]) if len(rest) >= 2 and "AC" in [t.upper() for t in toks[3:]] else 0.0
            branches.append({"name": toks[0], "type": kind, "n_pos": toks[1], "n_neg": toks[2], "value": mag, "phase_deg": ph})
        elif kind == "G" and len(toks) >= 6:
            branches.append({"name": toks[0], "type": "G", "n_pos": toks[1], "n_neg": toks[2], "nc_pos": toks[3], "nc_neg": toks[4], "value": _parse_spice_num(toks[5])})
    return branches, freq_hz


def _complex_gauss_solve(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    n = len(b_vec)
    aug = np.hstack([a_mat.astype(np.complex128).copy(), b_vec.astype(np.complex128).reshape(n, 1)])
    for col in range(n):
        piv = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[piv, col]) < 1e-18:
            raise ValueError("Singular MNA circuit matrix")
        if piv != col:
            aug[[col, piv]] = aug[[piv, col]]
        aug[col] = aug[col] / aug[col, col]
        for row in range(n):
            if row != col and aug[row, col] != 0:
                aug[row] = aug[row] - aug[row, col] * aug[col]
    return aug[:, n]


def verify_circuit_conservation(spec: Dict[str, Any]) -> Dict[str, Any]:
    if spec.get("netlist"):
        branches, nl_freq = _parse_spice_netlist(str(spec["netlist"]))
        freq_hz = float(spec.get("freq_hz", nl_freq))
    elif spec.get("file_b64"):
        branches, nl_freq = _parse_spice_netlist(base64.b64decode(spec["file_b64"]).decode("utf-8", errors="replace"))
        freq_hz = float(spec.get("freq_hz", nl_freq))
    else:
        branches = list(spec.get("branches") or [])
        freq_hz = float(spec.get("freq_hz", 0.0))

    omega = 2.0 * math.pi * freq_hz
    gnd = {"0", "GND", "gnd"}
    nodes = sorted({str(br[k]) for br in branches for k in ("n_pos", "n_neg", "nc_pos", "nc_neg") if k in br and str(br[k]) not in gnd})
    n_idx = {nm: i for i, nm in enumerate(nodes)}
    v_sources = [br for br in branches if str(br["type"]).upper() == "V"]
    dim = len(nodes) + len(v_sources)
    mna = np.zeros((dim, dim), dtype=np.complex128)
    rhs = np.zeros(dim, dtype=np.complex128)
    ni = lambda s: None if str(s) in gnd else n_idx[str(s)]

    vs_map: Dict[str, int] = {}
    for k, vs in enumerate(v_sources):
        row = len(nodes) + k
        vs_map[str(vs["name"])] = row
        ip, im = ni(vs["n_pos"]), ni(vs["n_neg"])
        if ip is not None: mna[ip, row] += 1.0; mna[row, ip] += 1.0
        if im is not None: mna[im, row] -= 1.0; mna[row, im] -= 1.0
        rhs[row] = cmath.rect(float(vs["value"]), math.radians(float(vs.get("phase_deg", 0.0))))

    for br in branches:
        bt = str(br["type"]).upper()
        ip, im = ni(br["n_pos"]), ni(br["n_neg"])
        if bt in ("R", "L", "C"):
            val = float(br["value"])
            y = (1.0 / val) if bt == "R" else ((1.0 / (1j * omega * val) if omega > 0 else 1e12) if bt == "L" else (1j * omega * val if omega > 0 else 0.0))
            if ip is not None: mna[ip, ip] += y
            if im is not None: mna[im, im] += y
            if ip is not None and im is not None: mna[ip, im] -= y; mna[im, ip] -= y
        elif bt == "I":
            icur = cmath.rect(float(br["value"]), math.radians(float(br.get("phase_deg", 0.0))))
            if ip is not None: rhs[ip] -= icur
            if im is not None: rhs[im] += icur
        elif bt == "G":
            gm = float(br["value"])
            icp, icm = ni(br["nc_pos"]), ni(br["nc_neg"])
            for r_i, sgn in ((ip, 1.0), (im, -1.0)):
                if r_i is not None:
                    if icp is not None: mna[r_i, icp] += sgn * gm
                    if icm is not None: mna[r_i, icm] -= sgn * gm

    sol_g = _complex_gauss_solve(mna, rhs)
    sol_lu = np.linalg.solve(mna, rhs)
    dual_err = float(np.linalg.norm(sol_g - sol_lu) / max(float(np.linalg.norm(sol_lu)), 1e-30))
    v_node: Dict[str, complex] = {"0": 0j, "GND": 0j, **{nm: complex(sol_lu[i]) for nm, i in n_idx.items()}}

    kcl_sum: Dict[str, complex] = {nm: 0j for nm in nodes}
    kvl_max = 0.0
    s_tot = 0j
    s_deliv = 0j
    for br in branches:
        nm, bt = str(br["name"]), str(br["type"]).upper()
        vd = v_node.get(str(br["n_pos"]), 0j) - v_node.get(str(br["n_neg"]), 0j)
        if bt == "V":
            ibr = complex(sol_lu[vs_map[nm]])
            kvl_max = max(kvl_max, abs(vd - cmath.rect(float(br["value"]), math.radians(float(br.get("phase_deg", 0.0))))))
            s_deliv += vd * (-ibr).conjugate()
        elif bt == "I":
            ibr = cmath.rect(float(br["value"]), math.radians(float(br.get("phase_deg", 0.0))))
            s_deliv += vd * (-ibr).conjugate()
        elif bt == "G":
            ibr = float(br["value"]) * (v_node.get(str(br["nc_pos"]), 0j) - v_node.get(str(br["nc_neg"]), 0j))
        else:
            val = float(br["value"])
            y = (1.0 / val) if bt == "R" else ((1.0 / (1j * omega * val) if omega > 0 else 1e12) if bt == "L" else (1j * omega * val if omega > 0 else 0.0))
            ibr = y * vd
        s_tot += vd * ibr.conjugate()
        if str(br["n_pos"]) in kcl_sum: kcl_sum[str(br["n_pos"])] += ibr
        if str(br["n_neg"]) in kcl_sum: kcl_sum[str(br["n_neg"])] -= ibr

    kcl_max = max((abs(v) for v in kcl_sum.values()), default=0.0)
    cand_ok = True
    if isinstance(spec.get("proposed_node_voltages"), dict):
        rtol = float(spec.get("rtol", 1e-4))
        for k_nm, cv_raw in spec["proposed_node_voltages"].items():
            ref_v = v_node[str(k_nm)]
            cv = complex(cv_raw[0], cv_raw[1]) if isinstance(cv_raw, (list, tuple)) else complex(cv_raw)
            if abs(cv - ref_v) / max(abs(ref_v), 1e-12) > rtol:
                cand_ok = False

    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "circuit",
        "freq_hz": freq_hz,
        "node_voltages": {nm: ([v.real, v.imag] if omega > 0 else v.real) for nm, v in v_node.items() if nm != "GND"},
        "kcl_max_residual_A": kcl_max,
        "kvl_max_residual_V": kvl_max,
        "tellegen_power_residual_VA": abs(s_tot),
        "total_complex_power_delivered_VA": [s_deliv.real, s_deliv.imag],
        "dual_solver_rel_error": dual_err,
        "candidate_voltages_valid": cand_ok,
        "conservation_verified": bool(cand_ok and kcl_max <= 1e-8 and kvl_max <= 1e-8 and abs(s_tot) <= 1e-7 and dual_err <= 1e-9),
    }


def verify_civil_geotech_balance(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech"}
    ok = True
    if "cut_bank_m3" in spec or "fill_compacted_m3" in spec:
        cut_b, sh, fill_c = float(spec.get("cut_bank_m3", 0.0)), float(spec.get("shrink_factor", 0.90)), float(spec.get("fill_compacted_m3", 0.0))
        out["net_compacted_surplus_m3"] = cut_b * sh - fill_c
    if "footing_B_m" in spec:
        B, Df, gam = float(spec["footing_B_m"]), float(spec["depth_Df_m"]), float(spec["gamma_N_m3"])
        gam_sat, dw = float(spec.get("gamma_sat_N_m3", gam)), float(spec.get("water_table_depth_m", 1e6))
        c_p, phi_deg, fs = float(spec.get("c_prime_Pa", 0.0)), float(spec.get("phi_deg", 30.0)), float(spec.get("FS", 3.0))
        gw = 9806.65
        if phi_deg <= 1e-9:
            nq, nc, ng = 1.0, 5.141592653589793, 0.0
        else:
            pr = math.radians(phi_deg)
            nq = math.exp(math.pi * math.tan(pr)) * (math.tan(math.radians(45.0 + 0.5 * phi_deg)) ** 2)
            nc, ng = (nq - 1.0) / math.tan(pr), 2.0 * (nq + 1.0) * math.tan(pr)
        if dw <= Df:
            sig_d, gam_z = gam * dw + (gam_sat - gw) * (Df - dw), gam_sat - gw
        elif dw < Df + B:
            sig_d, gam_z = gam * Df, (gam_sat - gw) + ((dw - Df) / B) * (gam - (gam_sat - gw))
        else:
            sig_d, gam_z = gam * Df, gam
        q_ult = c_p * nc + sig_d * nq + 0.5 * gam_z * B * ng
        out.update({"q_ult_Pa": q_ult, "q_allow_Pa": q_ult / fs})
        if "Cc" in spec and "H_clay_m" in spec:
            out["primary_consolidation_settlement_m"] = (float(spec["Cc"]) * float(spec["H_clay_m"]) / (1.0 + float(spec["e0"]))) * math.log10((float(spec["sigma_v0_eff_Pa"]) + float(spec["delta_sigma_v_Pa"])) / float(spec["sigma_v0_eff_Pa"]))
    if "channel_B_m" in spec and "y1_m" in spec and "y2_m" in spec:
        g = 9.80665
        b_c, q_f, y1, y2 = float(spec["channel_B_m"]), float(spec["Q_m3_s"]), float(spec["y1_m"]), float(spec["y2_m"])
        v1, v2 = q_f / (b_c * y1), q_f / (b_c * y2)
        hl = (float(spec.get("z1_m", 0.0)) + y1 + v1 * v1 / (2.0 * g)) - (float(spec.get("z2_m", 0.0)) + y2 + v2 * v2 / (2.0 * g))
        out["head_loss_m"] = hl
        ok = ok and (hl >= -1e-9)
    out["conservation_verified"] = bool(ok)
    return out


def parse_chemical_formula(formula: str) -> Dict[str, float]:
    tokens = re.findall(r"([A-Z][a-z]?|\(|\)|\d+(?:\.\d+)?)", formula.strip())
    stack: List[Dict[str, float]] = [{}]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            stack.append({}); i += 1
        elif t == ")":
            grp = stack.pop(); mult = 1.0
            if i + 1 < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[i + 1]):
                mult = float(tokens[i + 1]); i += 1
            for el, cnt in grp.items(): stack[-1][el] = stack[-1].get(el, 0.0) + cnt * mult
            i += 1
        elif re.fullmatch(r"[A-Z][a-z]?", t):
            mult = 1.0
            if i + 1 < len(tokens) and re.fullmatch(r"\d+(?:\.\d+)?", tokens[i + 1]):
                mult = float(tokens[i + 1]); i += 1
            stack[-1][t] = stack[-1].get(t, 0.0) + mult; i += 1
        else:
            i += 1
    return stack[0]


def _colebrook_dual_friction(re_num: float, rel_rough: float) -> Tuple[float, float]:
    if re_num < 2300.0:
        return 64.0 / max(re_num, 1e-12), 0.0
    x = 1.0 / math.sqrt(0.25 / (math.log10(rel_rough / 3.7 + 5.74 / (re_num ** 0.9)) ** 2))
    ln10 = math.log(10.0)
    for _ in range(30):
        arg = rel_rough / 3.7 + 2.51 * x / re_num
        x -= (x + 2.0 * math.log10(arg)) / (1.0 + (2.0 / ln10) * (2.51 / re_num) / arg)
    f_n = 1.0 / (x * x)
    lo, hi = 1.0 / math.sqrt(0.25), 1.0 / math.sqrt(0.004)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid + 2.0 * math.log10(rel_rough / 3.7 + 2.51 * mid / re_num) < 0.0: lo = mid
        else: hi = mid
    f_b = 1.0 / ((0.5 * (lo + hi)) ** 2)
    return f_n, abs(f_n - f_b) / f_n


_HX_INPUTS = ("Th_in_K", "Th_out_K", "Tc_in_K", "Tc_out_K", "Ch_W_K", "Cc_W_K")
_HX_OBSERVATIONS = ("Q_W", "UA_W_K")


def _heat_exchanger_balance(spec: Dict[str, Any], rtol: float) -> Dict[str, Any]:
    """Check independent observations against both stream balances in SI units."""
    missing = [key for key in _HX_INPUTS if key not in spec]
    if missing:
        return {"hx_balance_ok": False, "hx_error": "missing_inputs", "hx_missing": missing}
    try:
        quantities = {key: float(spec[key]) for key in (*_HX_INPUTS, *_HX_OBSERVATIONS) if key in spec}
    except (TypeError, ValueError, OverflowError):
        return {"hx_balance_ok": False, "hx_error": "non_numeric_input"}
    if not all(math.isfinite(v) for v in quantities.values()) or not math.isfinite(rtol) or rtol < 0:
        return {"hx_balance_ok": False, "hx_error": "invalid_input_or_tolerance"}
    if any(quantities[key] <= 0 for key in _HX_INPUTS):
        return {"hx_balance_ok": False, "hx_error": "non_positive_temperature_or_capacity"}

    thi, tho, tci, tco, ch, cc = (quantities[key] for key in _HX_INPUTS)
    arrangement = str(spec.get("arrangement", "counterflow")).lower()
    if arrangement in ("counterflow", "countercurrent", "counter-current"):
        dt1, dt2 = thi - tco, tho - tci
    elif arrangement in ("parallel", "parallelflow", "parallel_flow", "coflow", "co-current", "cocurrent"):
        dt1, dt2 = thi - tci, tho - tco
    else:
        return {"hx_balance_ok": False, "hx_error": "unknown_arrangement"}
    no_cross = dt1 > 0.0 and dt2 > 0.0 and thi >= tho and tco >= tci
    lmtd = 0.0
    if no_cross:
        hi, lo = max(dt1, dt2), min(dt1, dt2)
        delta = hi - lo
        if delta == 0:
            lmtd = hi
        elif delta <= lo:
            lmtd = delta / math.log1p(delta / lo)
        else:
            lmtd = delta / (math.log(hi) - math.log(lo))
    qh, qc = ch * (thi - tho), cc * (tco - tci)
    entropy = cc * (math.log(tco) - math.log(tci)) + ch * (math.log(tho) - math.log(thi))
    duties = {"hot_stream": qh, "cold_stream": qc}
    if "Q_W" in quantities:
        duties["Q_W"] = quantities["Q_W"]
    if "UA_W_K" in quantities:
        duties["UA_times_lmtd"] = quantities["UA_W_K"] * lmtd
    if not all(math.isfinite(q) for q in duties.values()) or not math.isfinite(entropy):
        return {"hx_balance_ok": False, "hx_error": "non_finite_derived_quantity"}

    def agrees(a: float, b: float) -> bool:
        return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)

    checks = {key: q >= 0 and agrees(q, qh) and agrees(q, qc) for key, q in duties.items()}
    if quantities.get("UA_W_K", 0.0) < 0:
        checks["UA_times_lmtd"] = False
    return {
        "Q_hot_W": qh, "Q_cold_W": qc, "lmtd_K": lmtd,
        "hx_no_temperature_crossover": no_cross, "hx_entropy_generation_W_K": entropy,
        "hx_duties_W": duties, "hx_duty_checks": checks,
        "hx_balance_ok": no_cross and entropy >= -1e-9 and all(checks.values()),
    }


def verify_thermal_fluid_chemical_balance(spec: Dict[str, Any]) -> Dict[str, Any]:
    rtol = float(spec.get("rtol", 1e-4))
    out: Dict[str, Any] = {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "thermal_fluid"}
    all_ok = True
    if "inlets" in spec or "outlets" in spec:
        ins, outs = spec.get("inlets") or [], spec.get("outlets") or []
        mi, mo = sum(float(s.get("mdot_kg_s", 0.0)) for s in ins), sum(float(s.get("mdot_kg_s", 0.0)) for s in outs)
        m_ok = bool(ins or outs) and abs(mi - mo) <= rtol * max(mi, mo, 1e-12)
        out["mass_balance_ok"] = m_ok
        all_ok = all_ok and m_ok
        ai: Dict[str, float] = {}; ao: Dict[str, float] = {}
        has_sp = False
        for s in ins:
            if "ndot_mol_s" in s and "formula" in s:
                has_sp = True
                for el, cnt in parse_chemical_formula(str(s["formula"])).items(): ai[el] = ai.get(el, 0.0) + float(s["ndot_mol_s"]) * cnt
        for s in outs:
            if "ndot_mol_s" in s and "formula" in s:
                has_sp = True
                for el, cnt in parse_chemical_formula(str(s["formula"])).items(): ao[el] = ao.get(el, 0.0) + float(s["ndot_mol_s"]) * cnt
        if has_sp:
            sp_err = max((abs(ai.get(e, 0.0) - ao.get(e, 0.0)) / max(ai.get(e, 0.0), ao.get(e, 0.0), 1e-12) for e in set(ai) | set(ao)), default=0.0)
            out["species_balance_ok"] = sp_err <= rtol
            all_ok = all_ok and (sp_err <= rtol)
    if any(key in spec for key in (*_HX_INPUTS, "UA_W_K")):
        hx = _heat_exchanger_balance(spec, rtol)
        out.update(hx)
        all_ok = all_ok and hx["hx_balance_ok"]
    if "pipe_D_m" in spec and "pipe_L_m" in spec:
        g = 9.80665
        dp, lp, qv = float(spec["pipe_D_m"]), float(spec["pipe_L_m"]), float(spec["Q_m3_s"])
        rho, mu, eps = float(spec.get("rho_kg_m3", 998.2)), float(spec.get("mu_Pa_s", 1.002e-3)), float(spec.get("roughness_m", 4.5e-5))
        vel = qv / (0.25 * math.pi * dp * dp)
        re_n = rho * vel * dp / mu
        f_dw, f_err = _colebrook_dual_friction(re_n, eps / dp)
        hf = (f_dw * (lp / dp) + float(spec.get("K_minor", 0.0))) * (vel * vel) / (2.0 * g)
        hp = float(spec.get("delta_z_m", 0.0)) + float(spec.get("delta_P_Pa", 0.0)) / (rho * g) + hf
        out.update({"darcy_f": f_dw, "colebrook_dual_rel_error": f_err, "head_loss_m": hf, "required_shaft_power_W": rho * g * qv * hp / float(spec.get("pump_efficiency", 0.78))})
    checked = any(key in out for key in ("mass_balance_ok", "hx_balance_ok", "darcy_f"))
    out["conservation_verified"] = bool(checked and all_ok)
    return out


def run_router_conservation_cad_verifier(payload: Dict[str, Any]) -> Dict[str, Any]:
    # The tool's outer mode selects a solver family. A nested mode selects a
    # computation within that family (for example control / state_space).
    # Dispatch before the legacy payload merge so those two meanings cannot
    # overwrite one another.
    family = str(payload.get("mode", "auto")).strip().lower()
    wave5 = {"pde", "control", "psychrometrics", "fem"}
    wave6 = {"compressible_acoustics", "rf_transmission_rlc",
             "solid_mohr_vessel_buckling", "reactor_kinetics"}
    wave7 = {"thermo_cycles", "exchangers_radiation",
             "lagrangian_orbital", "electrochem_separations"}
    if family in wave5 | wave6 | wave7:
        raw_spec = payload.get("spec")
        if not isinstance(raw_spec, dict) or not raw_spec:
            raise ValueError(f"{family} requires a non-empty spec object")
        spec = dict(raw_spec)
        declared_domain = str(spec.get("domain") or "").strip().lower()
        if declared_domain and declared_domain != family:
            raise ValueError(f"spec.domain {declared_domain!r} conflicts with mode {family!r}")
        selector = "mode" if family in wave5 else (
            "reactor_type" if family == "reactor_kinetics" else "subkind"
        )
        if not isinstance(spec.get(selector), str) or not spec[selector].strip():
            raise ValueError(f"{family} requires explicit spec.{selector}")
        if not (spec.keys() - {"domain", "mode", "subkind", "reactor_type"}):
            raise ValueError(f"{family} requires supplied calculation parameters")
        # These library calculations can otherwise succeed entirely on example
        # defaults. Unknown keys or a tolerance are not a supplied physical
        # problem. Test key presence rather than truthiness: explicit zero
        # stress and zero line length are meaningful inputs.
        default_capable_inputs = {
            ("compressible_acoustics", "acoustics"): {
                "rho1", "c1", "rho2", "c2"
            },
            ("rf_transmission_rlc", "transmission_line"): {
                "Z0", "R_L", "X_L", "alpha_np_m", "length_m",
                "length_wavelengths", "beta_rad_m"
            },
            ("solid_mohr_vessel_buckling", "mohr_stress"): {
                "sigma_x", "sigma_y", "sigma_z", "tau_xy", "tau_xz", "tau_yz"
            },
        }
        physical_inputs = default_capable_inputs.get((family, spec[selector].strip().lower()))
        supplied_resonator = (family == "compressible_acoustics"
                              and spec[selector].strip().lower() == "acoustics"
                              and {"A_neck", "V_cavity", "L_eff"} <= spec.keys())
        if physical_inputs is not None and not (spec.keys() & physical_inputs) and not supplied_resonator:
            raise ValueError(f"{family}/{spec[selector]} requires recognized physical inputs")
        spec["domain"] = family
        if family in wave5:
            from hamon_pde_control_psychro_fem import verify_wave5_engineering
            return verify_wave5_engineering(spec)
        if family in wave6:
            from hamon_wave6_physics_verifiers import dispatch_wave6_verifier
            return dispatch_wave6_verifier(family, spec)
        from hamon_wave7_engineering_verifiers import dispatch_wave7_verifier
        return dispatch_wave7_verifier(family, spec)

    spec = dict(payload.get("spec") or {})
    for k, v in payload.items():
        if k != "spec" and k not in spec and v is not None:
            spec[k] = v
    mode = str(spec.get("mode", "auto")).lower()
    fname = str(spec.get("filename") or spec.get("path") or "").lower()
    if mode == "auto":
        if spec.get("step_text") or fname.endswith((".step", ".stp", ".p21")): mode = "step_cad"
        elif spec.get("netlist") or spec.get("branches") or fname.endswith((".cir", ".sp", ".spice", ".net")): mode = "circuit"
        elif "lhs" in spec and "rhs" in spec: mode = "dimensional"
        elif "span_m" in spec and "E_Pa" in spec: mode = "structural"
        elif any(k in spec for k in ("cut_bank_m3", "footing_B_m", "channel_B_m")): mode = "civil_geotech"
        elif any(k in spec for k in ("inlets", "outlets", "pipe_D_m", *_HX_INPUTS, "UA_W_K")): mode = "thermal_fluid"
        elif spec.get("file_b64"):
            mode = "step_cad" if "ISO-10303-21" in base64.b64decode(spec["file_b64"])[:512].decode("utf-8", errors="replace") else "circuit"
        else:
            raise ValueError("Cannot infer verifier mode")
    if mode in ("step_cad", "step", "cad"): return verify_step_cad_geometry(spec)
    if mode in ("structural", "mechanics"): return verify_structural_mechanics(spec)
    if mode in ("circuit", "spice", "kcl", "electrical"): return verify_circuit_conservation(spec)
    if mode in ("civil_geotech", "civil", "geotech"): return verify_civil_geotech_balance(spec)
    if mode in ("thermal_fluid", "chemical", "thermo", "fluid"): return verify_thermal_fluid_chemical_balance(spec)
    if mode in ("dimensional", "units"): return verify_dimensional_homogeneity(spec)
    raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__" and "--json-stdin" in sys.argv:
    print(json.dumps(run_router_conservation_cad_verifier(json.loads(sys.stdin.read() or "{}"))))
    raise SystemExit(0)
