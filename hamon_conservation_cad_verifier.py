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
   Linear SPICE K cards couple ideal constant inductors through their dotted
   first terminals. Structured K records use name, inductors (names), and value.
   Linear F/H sources sense current flowing into the positive terminal of their
   named independent voltage source (control_source in structured branches).
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
from itertools import combinations
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
    # A slash inside a rational exponent belongs to the unit token itself.
    divisions = [m.start() for m in re.finditer(r"[A-Za-z]+\^?[+-]?\d+/\d+|(/)", u) if m[1]]
    if len(divisions) > 1:
        raise ValueError(f"Ambiguous multiple '/' in unit {unit!r}")
    parts = (u[:divisions[0]], u[divisions[0] + 1:]) if divisions else (u,)
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


def _parse_step_entities_map(step_text: str):
    from hamon_step_units import parse_step_records
    return parse_step_records(step_text, _split_step_top_args)


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


def _position_ifc_profile(poly, entities, reference, scale):
    """Place profile vertices in the swept solid's XY plane."""
    if reference == "$":
        return poly
    kind, args = entities.get(_ref_id(reference), (None, []))
    if kind != "IFCAXIS2PLACEMENT2D" or len(args) != 2:
        raise ValueError("IFC rectangle requires a two-dimensional profile placement")
    kind, loc = entities.get(_ref_id(args[0]), (None, []))
    if kind != "IFCCARTESIANPOINT" or not loc:
        raise ValueError("IFC profile placement requires a two-dimensional location")
    offset = np.array(_num_list(loc[0]), dtype=float) * scale
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("IFC profile location must have two finite coordinates")
    if args[1] == "$":
        return poly + offset
    kind, ref = entities.get(_ref_id(args[1]), (None, []))
    if kind != "IFCDIRECTION" or not ref:
        raise ValueError("IFC profile reference direction must be defined")
    ratios = np.array(_num_list(ref[0]), dtype=float)
    if ratios.shape != (2,) or not np.isfinite(ratios).all() or not np.any(ratios):
        raise ValueError("IFC profile reference direction must be finite and nonzero")
    ratios /= np.max(np.abs(ratios))
    x, y = ratios / np.linalg.norm(ratios)
    return poly @ np.array([[x, y], [-y, x]]) + offset


def _ifc_extrusion_direction(entities, reference):
    kind, args = entities.get(_ref_id(reference), (None, []))
    if kind != "IFCDIRECTION" or not args:
        raise ValueError("IFC extrusion requires a three-dimensional direction")
    ratios = np.array(_num_list(args[0]), dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or ratios[2] == 0:
        raise ValueError("IFC extrusion must have a finite nonzero component normal to the profile")
    # Direction ratios need not be unit length or share any particular scale.
    ratios /= np.max(np.abs(ratios))
    return ratios / np.linalg.norm(ratios)


def _extrude_polygon_2d(poly2d: np.ndarray, origin: np.ndarray, x_ax: np.ndarray, y_ax: np.ndarray, direction: np.ndarray, depth: float) -> Tuple[np.ndarray, np.ndarray]:
    x, y = poly2d[:, 0], poly2d[:, 1]
    if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
        poly2d = poly2d[::-1].copy()
    k = len(poly2d)
    verts = [origin + px * x_ax + py * y_ax + offset * direction for offset in (0.0, depth) for px, py in poly2d]
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
        if et in ("AXIS2_PLACEMENT_3D", "IFCAXIS2PLACEMENT3D"):
            # STEP representation items start with a name; IFC placements
            # start with Location. Both then carry Axis and RefDirection.
            placement = args[1:] if et == "AXIS2_PLACEMENT_3D" else args
            pid = _ref_id(placement[0]) if placement else None
            if pid and pid in entities:
                origin = np.array(_num_list(entities[pid][1][-1])[:3], dtype=np.float64) * scale
            if len(placement) >= 2 and _ref_id(placement[1]) in entities:
                zv = np.array(_num_list(entities[_ref_id(placement[1])][1][-1])[:3], dtype=np.float64)  # type: ignore[index]
                if np.linalg.norm(zv) > 1e-12:
                    z_ax = zv / np.linalg.norm(zv)
            if len(placement) >= 3 and _ref_id(placement[2]) in entities:
                xv = np.array(_num_list(entities[_ref_id(placement[2])][1][-1])[:3], dtype=np.float64)  # type: ignore[index]
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


def _compute_mesh_mass_properties(vertices: np.ndarray, faces: np.ndarray, density_kg_m3: float,
                                  *, require_connected: bool = False,
                                  surface_face_ids: Optional[List[int]] = None,
                                  surface_shell_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    # Form edge differences and signed tetrahedra in the body's coordinate scale.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    reference = low + 0.5 * (high - low)
    local = vertices - reference
    v0, v1, v2 = local[faces[:, 0]], local[faces[:, 1]], local[faces[:, 2]]
    # Selected STEP BReps must close at their actual boundary coordinates.
    # Rounded welding can hide a real parallel cap gap with zero net area.
    qverts = vertices if require_connected else np.round(local, 9)
    unique, inv = np.unique(qverts, axis=0, return_inverse=True)
    qfaces = inv.reshape(-1)[faces]
    edge_counts: Dict[Tuple[int, int], int] = {}
    directed = set()
    manifold = True
    parents = list(range(len(qfaces)))
    edge_faces = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, tri in enumerate(qfaces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a == b or b == c or c == a:
            manifold = False
        for u, w in ((a, b), (b, c), (c, a)):
            if (u, w) in directed:
                manifold = False
            directed.add((u, w))
            k = (u, w) if u < w else (w, u)
            edge_counts[k] = edge_counts.get(k, 0) + 1
            if require_connected:
                if k in edge_faces:
                    parents[root(index)] = root(edge_faces[k])
                else:
                    edge_faces[k] = index
    watertight = manifold and bool(edge_counts) and all(cnt == 2 for cnt in edge_counts.values())
    edge1, edge2 = v1 - v0, v2 - v0
    area_vectors = 0.5 * np.cross(edge1, edge2)
    surface_area = float(np.hypot.reduce(area_vectors, axis=1).sum())
    net_area = np.array([math.fsum(area_vectors[:, i]) for i in range(3)])
    area_residual = math.hypot(*net_area)
    cross_scale = 0.5 * np.sum(np.hypot.reduce(edge1, axis=1) * np.hypot.reduce(edge2, axis=1))
    area_tolerance = float(64 * np.finfo(np.float64).eps * cross_scale)
    diagnostics = {
        "watertight_manifold": bool(watertight),
        "net_oriented_area_residual_m2": area_residual,
        "net_oriented_area_tolerance_m2": area_tolerance,
        "triangle_count": int(len(faces)),
    }
    if require_connected:
        diagnostics["embedded_surface_verified"] = False
    if not watertight or not area_residual <= area_tolerance:
        return {**diagnostics, "reason": "Mesh lacks consistently paired edges or balanced oriented surface area"}

    if require_connected:
        components = len({root(index) for index in range(len(qfaces))})
        diagnostics["connected_component_count"] = components
        shell_ids = [0]*len(qfaces) if surface_shell_ids is None else surface_shell_ids
        if len(shell_ids) != len(qfaces):
            raise ValueError("STEP triangle shell labels are incomplete")
        group_roots = {}
        for index, shell_id in enumerate(shell_ids):
            group_roots.setdefault(shell_id, set()).add(root(index))
        diagnostics["void_shell_count"] = len(group_roots)-1
        if components != len(group_roots) or any(len(roots) != 1 for roots in group_roots.values()):
            return {**diagnostics, "reason": "Declared STEP shell has disconnected surface components"}
        from hamon_step_surface import integrate_closed_surface, verify_surface_embedding, verify_void_shells
        diagnostics.update(verify_surface_embedding(unique, qfaces, surface_face_ids))
        if "reason" in diagnostics:
            return diagnostics
        if len(group_roots) > 1:
            diagnostics.update(verify_void_shells(unique, qfaces, shell_ids))
            if "reason" in diagnostics:
                return diagnostics
        if not math.isfinite(surface_area) or surface_area <= 0:
            return {**diagnostics, "reason": "STEP surface area is outside finite binary64 range"}
        moments = integrate_closed_surface(unique, qfaces, density_kg_m3)
        if "reason" in moments:
            return {**diagnostics, **moments}
        return {**diagnostics, **moments, "surface_area_m2": surface_area}

    dets = np.einsum("ij,ij->i", np.cross(v0, v1), v2)
    signed_vol = float(np.sum(dets) / 6.0)
    if signed_vol < 0.0:
        v1, v2 = v2, v1
        dets = -dets
        signed_vol = -signed_vol
    if signed_vol <= 1e-18:
        raise ValueError("Degenerate zero-volume mesh in STEP/CAD verifier")
    centroid = np.sum(((v0 + v1 + v2) / 4.0) * dets[:, None], axis=0) / (6.0 * signed_vol)
    m_canon = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
    v_stack = np.stack([v0, v1, v2], axis=1)
    cov_origin = np.sum(np.einsum("f,fia,ij,fjb->fab", dets / 120.0, v_stack, m_canon, v_stack), axis=0)
    cov_centroid = cov_origin - signed_vol * np.outer(centroid, centroid)
    inertia_c = density_kg_m3 * (np.trace(cov_centroid) * np.eye(3) - cov_centroid)
    inertia_c = 0.5 * (inertia_c + inertia_c.T)
    principal = np.sort(np.linalg.eigvalsh(inertia_c))

    mass = density_kg_m3 * signed_vol
    return {
        "volume_m3": signed_vol,
        "surface_area_m2": surface_area,
        "mass_kg": mass,
        "density_kg_m3": density_kg_m3,
        "centroid_m": [float(x) for x in (reference + centroid)],
        "aabb_extents_m": [float(x) for x in (np.max(vertices, axis=0) - np.min(vertices, axis=0))],
        "inertia_tensor_centroid_kg_m2": [[float(x) for x in row] for row in inertia_c],
        "principal_moments_kg_m2": [float(x) for x in principal],
        **diagnostics,
    }


def _extrude_rectangular_ring(outer, inner, origin, x_axis, y_axis, direction, depth):
    """Oriented annular caps and both wall loops; the void has no closing face."""
    profile = np.vstack((outer, inner))
    vertices = np.array([origin + x*x_axis + y*y_axis + offset*direction
                         for offset in (0., depth) for x, y in profile])
    faces = []
    for i in range(4):
        n = (i+1) % 4
        a, b, c, d = i, n, n+4, i+4
        for u, v, w in ((a,b,c), (a,c,d)):
            faces.extend(((u,w,v), (u+8,v+8,w+8)))
        faces.extend(((i,n,n+8), (i,n+8,i+8)))
        faces.extend(((n+4,i+4,i+12), (n+4,i+12,n+12)))
    return vertices, np.asarray(faces, dtype=np.int32)


def _rectangular_sweep(entities, et, args, unit_scale, frame=None, magnification=1.):
    prof_id = _ref_id(args[0] if et == "IFCEXTRUDEDAREASOLID" else args[1])
    pos_id = _ref_id(args[1] if et == "IFCEXTRUDEDAREASOLID" else args[2])
    depth = float(args[3]) * unit_scale
    if frame is None:
        frame = _axis2_frame(entities, pos_id, unit_scale)
    origin, x_ax, y_ax, z_ax = frame
    if prof_id in entities:
        pet, pargs = entities[prof_id]
        if pet in ("RECTANGLE_PROFILE_DEF", "IFCRECTANGLEPROFILEDEF", "IFCRECTANGLEHOLLOWPROFILEDEF") and len(pargs) >= 5:
            lx, ly = float(pargs[3]) * unit_scale, float(pargs[4]) * unit_scale
            poly = np.array([[-0.5*lx,-0.5*ly],[0.5*lx,-0.5*ly],[0.5*lx,0.5*ly],[-0.5*lx,0.5*ly]])
            direction, normal = z_ax, 1.0
            inner = None
            area = None
            if pet == "IFCRECTANGLEHOLLOWPROFILEDEF" and et != "IFCEXTRUDEDAREASOLID":
                raise ValueError("IFC hollow profiles require an IFC swept solid")
            if et == "IFCEXTRUDEDAREASOLID":
                if min(lx, ly, depth) <= 0 or not all(map(math.isfinite, (lx, ly, depth))):
                    raise ValueError("Rectangular IFC extrusion dimensions must be finite and positive")
                if pet == "IFCRECTANGLEHOLLOWPROFILEDEF":
                    if len(pargs) != 8 or any(v != "$" and float(v) != 0 for v in pargs[6:8]):
                        raise ValueError("Only sharp-corner rectangular hollow profiles are qualified")
                    wall = float(pargs[5]) * unit_scale
                    if not math.isfinite(wall) or not 0 < wall < min(lx, ly)/2:
                        raise ValueError("Hollow wall thickness must leave a positive rectangular void")
                    ix, iy = (float(Fraction(length)-2*Fraction(wall)) for length in (lx, ly))
                    inner = np.array([[-ix/2,-iy/2],[ix/2,-iy/2],[ix/2,iy/2],[-ix/2,iy/2]])
                    inner = _position_ifc_profile(inner, entities, pargs[2], unit_scale)
                    area = 2*Fraction(wall)*(Fraction(lx)+Fraction(ly)-2*Fraction(wall))
                poly = _position_ifc_profile(poly, entities, pargs[2], unit_scale)
                local = _ifc_extrusion_direction(entities, args[2])
                direction = local[0] * x_ax + local[1] * y_ax + local[2] * z_ax
                normal = local[2]
            if inner is None:
                vertices, faces = _extrude_polygon_2d(poly, origin, x_ax, y_ax, direction, depth)
            else:
                vertices, faces = _extrude_rectangular_ring(poly, inner, origin, x_ax, y_ax, direction, depth)
            # Keep the scale product exact until physical volume is rounded.
            if area is None:
                area = Fraction(lx) * Fraction(ly)
            volume = float(area * Fraction(depth) * Fraction(abs(normal))
                           * Fraction(magnification) ** 3)
            return vertices, faces if normal > 0 else faces[:, ::-1], volume


def _ifc_mass_mapped_body(entities, product, unit_scale):
    """Resolve one complete rectangular occurrence, including nested reusable maps.

    The qualified map has an identity source origin, finite positive uniform
    scale, and either three orthogonal target axes or the default axes. Each
    Body resolves to one rectangular extrusion; map cycles and partial Bodies
    are refused. Reusing a definition does not merge distinct product instances.
    """
    from hamon_ifc_spatial_clash import _required_ifc_entity, _rigid_ifc_frame

    def required(ref, kind):
        return _required_ifc_entity(entities, ref, kind)

    def vector(ref, kind):
        args = required(ref, kind)
        if len(args) != 1 or not isinstance(args[0], list) or len(args[0]) != 3:
            raise ValueError(f"Mapped IFC mass requires three coordinates for {kind}")
        result = np.asarray(args[0], dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Mapped IFC coordinates must be finite")
        if kind == "IFCDIRECTION":
            size = max(abs(result))
            if size == 0:
                raise ValueError("Mapped IFC direction must be nonzero")
            result /= size
            result /= np.linalg.norm(result)
        return result

    def item(ref):
        rep = required(ref, "IFCSHAPEREPRESENTATION")
        if len(rep) != 4 or str(rep[1]).upper() != "BODY" or not isinstance(rep[3], list) or len(rep[3]) != 1:
            raise ValueError("Mapped IFC mass requires one complete item in each Body")
        context = rep[0]
        kind, attrs = entities.get(context[1], (None, [])) if isinstance(context, tuple) else (None, [])
        if kind == "IFCGEOMETRICREPRESENTATIONSUBCONTEXT" and len(attrs) > 6:
            attrs = required(attrs[6], "IFCGEOMETRICREPRESENTATIONCONTEXT")
            kind = "IFCGEOMETRICREPRESENTATIONCONTEXT"
        if kind != "IFCGEOMETRICREPRESENTATIONCONTEXT" or len(attrs) < 5 or attrs[2] != 3:
            raise ValueError("Expected a 3D IFC representation context")
        if not np.array_equal(_rigid_ifc_frame(entities, attrs[4], 1.), np.eye(4)):
            raise ValueError("Only an identity IFC world coordinate system is qualified")
        return rep[3][0]

    shape = required(entities[product][1][6], "IFCPRODUCTDEFINITIONSHAPE")
    if len(shape) < 3 or not isinstance(shape[2], list):
        raise ValueError("Invalid IFC product definition shape")
    bodies = []
    for ref in shape[2]:
        rep = required(ref, "IFCSHAPEREPRESENTATION")
        if len(rep) >= 2 and str(rep[1]).upper() in ("AXIS", "BOX", "FOOTPRINT"):
            continue
        bodies.append(item(ref))
    if len(bodies) != 1:
        raise ValueError("IFC mass requires one complete Body per product")

    current = bodies[0]
    mapping = np.eye(4)
    magnification = 1.
    visited = set()
    while True:
        if not isinstance(current, tuple) or current[0] != "ref" or current[1] not in entities:
            raise ValueError("Unresolved mapped IFC body item")
        if current[1] in visited:
            raise ValueError("Cyclic IFC representation map")
        visited.add(current[1])
        kind, args = entities[current[1]]
        if kind == "IFCEXTRUDEDAREASOLID":
            if len(args) != 4:
                raise ValueError("Incomplete IFC swept Body")
            profile_kind = entities.get(args[0][1], (None,))[0] if isinstance(args[0], tuple) else None
            if profile_kind not in ("IFCRECTANGLEPROFILEDEF", "IFCRECTANGLEHOLLOWPROFILEDEF"):
                raise ValueError("IFC mass requires a rectangular area profile")
            profile = required(args[0], profile_kind)
            count = 8 if profile_kind == "IFCRECTANGLEHOLLOWPROFILEDEF" else 5
            if len(profile) != count or str(profile[0]).upper() != "AREA":
                raise ValueError("IFC mass requires a rectangular area profile")
            return current[1], mapping, magnification
        if kind != "IFCMAPPEDITEM" or len(args) != 2:
            raise ValueError("Only rectangular swept Bodies and their mapped occurrences are qualified")
        source = required(args[0], "IFCREPRESENTATIONMAP")
        if len(source) != 2 or not np.array_equal(_rigid_ifc_frame(entities, source[0], 1.), np.eye(4)):
            raise ValueError("Only an identity IFC MappingOrigin is qualified")
        operator = required(args[1], "IFCCARTESIANTRANSFORMATIONOPERATOR3D")
        if len(operator) != 5:
            raise ValueError("Expected a uniform 3D IFC mapping operator")
        xref, yref, origin, scale, zref = operator
        scale = 1. if scale is None else float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("IFC mapping scale must be finite and positive")
        rotation = np.eye(3)
        if any(ref is not None for ref in (xref, yref, zref)):
            if any(ref is None for ref in (xref, yref, zref)):
                raise ValueError("Qualified IFC mapping needs all three axes or none")
            x, y, z = (vector(ref, "IFCDIRECTION") for ref in (xref, yref, zref))
            rotation = np.column_stack((x, y, z))
            if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=64*np.finfo(float).eps):
                raise ValueError("Qualified IFC mapping axes must be orthogonal")
        transform = np.eye(4)
        transform[:3, :3] = rotation * scale
        transform[:3, 3] = vector(origin, "IFCCARTESIANPOINT") * unit_scale
        mapping = mapping @ transform
        magnification *= scale
        if not np.isfinite(mapping).all() or not math.isfinite(magnification) or magnification == 0:
            raise ValueError("IFC mapping is outside the representable geometry range")
        current = item(source[1])


def _ifc_product_properties(typed, raw, unit_scale, density):
    from hamon_ifc_spatial_clash import (
        MASS_PRODUCT_TYPES, _placed_product_frame, _product_shapes,
        _rigid_ifc_frame,
    )

    owners = list(_product_shapes(typed, strict=True))
    if not owners:
        if any(kind == "IFCPRODUCTDEFINITIONSHAPE" for kind, _ in typed.values()):
            raise ValueError("IFC product definition shape has no resolved owner")
        return None
    if not math.isfinite(density) or density <= 0:
        raise ValueError("IFC product mass requires a positive finite density")
    if not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("IFC product mass requires a positive finite length unit")
    if any(kind == "IFCRELVOIDSELEMENT" for kind, _ in typed.values()):
        raise ValueError("IFC products with opening subtraction are unsupported")

    parts = []
    reference = None
    for eid, kind, args, _ in owners:
        if kind not in MASS_PRODUCT_TYPES:
            raise ValueError(f"Unsupported IFC mass occurrence type {kind} on #{eid}")
        item, mapping, magnification = _ifc_mass_mapped_body(typed, eid, unit_scale)
        if item not in raw or len(typed[item][1]) != 4:
            raise ValueError(f"Unresolved IFC rectangular swept Body on #{eid}")
        world = _placed_product_frame(typed, args[5], unit_scale) @ mapping
        position = typed[item][1][1]
        if position is not None:
            world = world @ _rigid_ifc_frame(typed, position, unit_scale)
        rotation, origin = world[:3, :3], world[:3, 3]
        # Integrate in world-axis directions about the body's origin. This keeps
        # survey-scale translation out of the local mesh and its central moments.
        swept = _rectangular_sweep(raw, *raw[item], unit_scale, (np.zeros(3), *rotation.T), magnification)
        if swept is None:
            raise ValueError(f"Expected a rectangular swept IFC Body on #{eid}")
        vertices, faces, analytic = swept
        if not math.isfinite(analytic) or analytic <= 0:
            raise ValueError("Mapped IFC volume is outside the representable range")
        props = _compute_mesh_mass_properties(vertices, faces, density)
        if "reason" in props:
            return props, None
        if reference is None:
            reference = origin
        offset = origin - reference
        parts.append((props, analytic, offset + props["centroid_m"],
                      offset + vertices.min(axis=0), offset + vertices.max(axis=0)))

    volume = math.fsum(p["volume_m3"] for p, *_ in parts)
    mass = math.fsum(p["mass_kg"] for p, *_ in parts)
    center = np.array([math.fsum(p["mass_kg"] * c[i] for p, _, c, _, _ in parts) / mass
                       for i in range(3)])
    inertia = np.zeros((3, 3))
    for props, _, centroid, _, _ in parts:
        dx, dy, dz = centroid - center
        shift = np.array([[dy*dy + dz*dz, -dx*dy, -dx*dz],
                          [-dx*dy, dx*dx + dz*dz, -dy*dz],
                          [-dx*dz, -dy*dz, dx*dx + dy*dy]])
        inertia += np.asarray(props["inertia_tensor_centroid_kg_m2"]) + props["mass_kg"] * shift
    extents = np.max([p[4] for p in parts], axis=0) - np.min([p[3] for p in parts], axis=0)
    return {
        "geometry_basis": "additive_ifc_product_occurrences", "ifc_product_count": len(parts),
        "volume_m3": volume, "mass_kg": mass, "density_kg_m3": density,
        "centroid_m": (reference + center).tolist(), "aabb_extents_m": extents.tolist(),
        "inertia_tensor_centroid_kg_m2": inertia.tolist(),
        "principal_moments_kg_m2": np.linalg.eigvalsh(inertia).tolist(),
        "surface_area_m2": math.fsum(p["surface_area_m2"] for p, *_ in parts),
        "max_product_volume_rel_error": max(abs(p["volume_m3"] - a) / a for p, a, *_ in parts),
        "watertight_manifold": True,
        "triangle_count": sum(p["triangle_count"] for p, *_ in parts),
        "net_oriented_area_residual_m2": math.fsum(p["net_oriented_area_residual_m2"] for p, *_ in parts),
        "net_oriented_area_tolerance_m2": math.fsum(p["net_oriented_area_tolerance_m2"] for p, *_ in parts),
    }, math.fsum(analytic for _, analytic, *_ in parts)


def _step_face_region(boundaries):
    """Triangulate the whole planar face, retaining its holes and original vertices.

    Separate signed loop fans can integrate volume correctly while counting a
    hole twice in unsigned surface area. A constrained partition represents the
    actual face domain, including concave outer boundaries.
    """
    from shapely import Polygon, constrained_delaunay_triangles, union_all

    if not boundaries:
        raise ValueError("STEP face has no complete boundary")
    rings = [np.asarray(points, dtype=float) for _, points in boundaries]
    if any(p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all()
           for p in rings):
        raise ValueError("STEP face requires finite three-dimensional boundary vertices")
    vertices = np.vstack(rings)
    origin = vertices[0]
    extent = float(np.max(np.abs(vertices - origin)))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("STEP face boundary is degenerate")
    local = [(p - origin) / extent for p in rings]
    normals = []
    for p in local:
        crosses = np.cross(p, np.roll(p, -1, axis=0))
        normals.append(np.array([math.fsum(crosses[:, j]) for j in range(3)]))
    explicit = [i for i, (kind, _) in enumerate(boundaries) if kind == "FACE_OUTER_BOUND"]
    if len(explicit) > 1:
        raise ValueError("STEP face has more than one outer boundary")
    outer = explicit[0] if explicit else max(range(len(rings)), key=lambda i: np.linalg.norm(normals[i]))
    normal_length = float(np.linalg.norm(normals[outer]))
    if not normal_length > 0:
        raise ValueError("STEP face outer boundary has zero oriented area")
    normal = normals[outer] / normal_length
    # Coordinates may already be rounded after a large translation. Account for
    # those ulps without imposing a world-origin absolute tolerance.
    for points, p in zip(rings, local):
        rounding = 8 * ((np.spacing(np.abs(points)) + np.spacing(np.abs(origin))) @ np.abs(normal)) / extent
        if np.any(np.abs(p @ normal) > rounding + 64 * np.finfo(float).eps):
            raise ValueError("STEP face boundaries are non-planar")
    axes = [j for j in range(3) if j != int(np.argmax(np.abs(normal)))]
    projected = [p[:, axes] for p in local]

    def signed_area(p):
        return math.fsum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2

    areas = [signed_area(p) for p in projected]
    if areas[outer] == 0 or any(areas[i] == 0 or (areas[i] > 0) == (areas[outer] > 0)
                              for i in range(len(rings)) if i != outer):
        raise ValueError("STEP face hole winding disagrees with its outer boundary")
    region = Polygon(projected[outer], [p for i, p in enumerate(projected) if i != outer])
    if not region.is_valid or region.is_empty or not region.area > 0:
        raise ValueError("STEP face boundaries do not define a simple region with disjoint interior holes")
    coordinates = {}
    for index, point in enumerate(np.vstack(projected)):
        key = tuple(point)
        if key in coordinates:
            raise ValueError("STEP face boundaries repeat or share a vertex")
        coordinates[key] = index
    triangles = list(constrained_delaunay_triangles(region).geoms)
    if not triangles or not union_all(triangles).equals(region):
        raise ValueError("STEP face triangulation does not cover its boundary region")
    covered_area = math.fsum(triangle.area for triangle in triangles)
    if abs(covered_area - region.area) > 64 * np.finfo(float).eps * len(triangles) * region.area:
        raise ValueError("STEP face triangulation contains overlapping regions")
    faces = []
    for triangle in triangles:
        p = np.asarray(triangle.exterior.coords)[:-1]
        if len(p) != 3 or triangle.interiors or not triangle.area > 0:
            raise ValueError("STEP face triangulation produced a non-triangle")
        try:
            face = [coordinates[tuple(point)] for point in p]
        except KeyError as exc:
            raise ValueError("STEP face triangulation introduced an unqualified vertex") from exc
        if (signed_area(p) > 0) != (areas[outer] > 0):
            face[1], face[2] = face[2], face[1]
        faces.append(face)
    return vertices, np.asarray(faces, dtype=np.int32)


def _step_topology_refs(token, context):
    """Read every selected member; unsupported tokens cannot disappear."""
    if not (token.startswith("(") and token.endswith(")")):
        raise ValueError(f"STEP {context} requires a reference aggregate")
    result = [_ref_id(value) for value in _split_step_top_args(token[1:-1])]
    if not result or any(value is None for value in result) or len(set(result)) != len(result):
        raise ValueError(f"STEP {context} has missing, malformed or repeated references")
    return result


def _step_solid_faces(entities, selector):
    """Resolve one solid definition and its complete oriented shell topology."""
    solid_kinds = {"MANIFOLD_SOLID_BREP", "FACETED_BREP", "BREP_WITH_VOIDS"}
    roots = [eid for eid, (kind, _) in entities.items() if kind in solid_kinds]
    if selector is None:
        construction = {"CSG_SOLID", "BOOLEAN_RESULT", "BOOLEAN_CLIPPING_RESULT", "SOLID_REPLICA"}
        if any(kind in construction for kind, _ in entities.values()):
            raise ValueError("STEP CSG/Boolean construction is unsupported; select a BRep definition explicitly")
        if roots:
            if any(kind.endswith("_SOLID") for kind, _ in entities.values()):
                raise ValueError("STEP has additional solid definitions; select a BRep explicitly")
            for kind, values in entities.values():
                if kind == "SHAPE_REPRESENTATION" or kind.endswith("_SHAPE_REPRESENTATION"):
                    if len(values) != 3:
                        raise ValueError("STEP shape representation is unsupported or incomplete")
                    for item in _step_topology_refs(values[1], "represented items"):
                        item_kind = entities.get(item, (None, []))[0]
                        if item not in roots and item_kind != "AXIS2_PLACEMENT_3D":
                            raise ValueError("STEP representation contains unsupported material; select a BRep explicitly")
    if selector is not None:
        if isinstance(selector, bool) or not isinstance(selector, int) or selector not in roots:
            raise ValueError("STEP solid_entity_id must identify a declared solid definition")
        selected = selector
    elif not roots:
        if any(kind in {"FACE", "FACE_SURFACE", "ADVANCED_FACE", "CLOSED_SHELL", "OPEN_SHELL",
                        "ORIENTED_FACE", "ORIENTED_CLOSED_SHELL"} for kind, _ in entities.values()):
            raise ValueError("STEP face topology has no supported declared solid")
        return None, None
    else:
        if len(roots) != 1:
            raise ValueError("STEP has multiple solid definitions; specify solid_entity_id")
        assembly = {"MAPPED_ITEM", "SOLID_REPLICA", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
                    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"}
        if any(kind in assembly for kind, _ in entities.values()):
            raise ValueError("STEP assembly placement is unsupported; select a solid definition explicitly")
        selected = roots[0]
    kind, args = entities[selected]
    expected_args = 3 if kind == "BREP_WITH_VOIDS" else 2
    if len(args) != expected_args:
        raise ValueError("STEP solid requires complete outer and declared void shell references")

    seen = set()

    def face(eid, forward, path):
        if eid is None or eid not in entities or eid in path:
            raise ValueError("STEP shell references a missing or cyclic face")
        kind, values = entities[eid]
        if kind == "ORIENTED_FACE":
            if len(values) != 4 or values[1] != "*" or values[3] not in (".T.", ".F."):
                raise ValueError("STEP oriented face requires an element and sense")
            return face(_ref_id(values[2]), forward == (values[3] == ".T."), path | {eid})
        expected = 2 if kind == "FACE" else 4
        if kind not in {"FACE", "FACE_SURFACE", "ADVANCED_FACE"} or len(values) != expected:
            raise ValueError("STEP shell includes an unsupported or incomplete face")
        if kind != "FACE" and values[3] not in (".T.", ".F."):
            raise ValueError("STEP face has an invalid surface sense")
        if eid in seen:
            raise ValueError("STEP shell references the same underlying face more than once")
        seen.add(eid)
        return [(eid, forward)]

    def shell(eid, forward, path):
        if eid is None or eid not in entities or eid in path:
            raise ValueError("STEP solid references a missing or cyclic shell")
        kind, values = entities[eid]
        if kind == "ORIENTED_CLOSED_SHELL":
            if len(values) != 4 or values[1] != "*" or values[3] not in (".T.", ".F."):
                raise ValueError("STEP oriented shell requires an element and sense")
            return shell(_ref_id(values[2]), forward == (values[3] == ".T."), path | {eid})
        if kind != "CLOSED_SHELL" or len(values) != 2:
            raise ValueError("STEP solid requires a declared closed shell")
        return [entry for fid in _step_topology_refs(values[1], "shell faces") for entry in face(fid, forward, set())]

    groups = [shell(_ref_id(args[1]), True, set())]
    if kind == "BREP_WITH_VOIDS":
        for void in _step_topology_refs(args[2], "void shells"):
            void_kind, values = entities.get(void, (None, []))
            if (void_kind != "ORIENTED_CLOSED_SHELL" or len(values) != 4
                    or values[1] != "*" or values[3] != ".F."):
                raise ValueError("STEP void shells require a reversed oriented closed shell")
            groups.append(shell(void, True, set()))
    return selected, [(index, eid, forward) for index, group in enumerate(groups) for eid, forward in group]


def verify_step_cad_geometry(spec: Dict[str, Any]) -> Dict[str, Any]:
    step_text = spec.get("step_text") or ""
    if not step_text and spec.get("file_b64"):
        step_text = base64.b64decode(spec["file_b64"]).decode("utf-8", errors="replace")
    elif not step_text and spec.get("path"):
        step_text = Path(spec["path"]).read_text(encoding="utf-8", errors="replace")
    density = float(spec.get("density_kg_m3", 7850.0))
    entities, components = _parse_step_entities_map(step_text)
    unit_scale = 1.0
    unit_metadata = {}
    if spec.get("unit_scale_m") is not None:
        if isinstance(spec["unit_scale_m"], bool):
            raise ValueError("unit_scale_m must be finite and positive")
        unit_scale = float(spec["unit_scale_m"])
        if not math.isfinite(unit_scale) or unit_scale <= 0:
            raise ValueError("unit_scale_m must be finite and positive")
        unit_metadata = {"unit_scale_source": "explicit_override"}

    selected_solid, face_occurrences = _step_solid_faces(entities, spec.get("solid_entity_id"))
    has_ifc = any(kind.startswith("IFC") for kind, _ in entities.values())
    if selected_solid is None and has_ifc:
        if any(kind in {"BLOCK", "RIGHT_CIRCULAR_CYLINDER", "EXTRUDED_AREA_SOLID"}
               for kind, _ in entities.values()):
            raise ValueError("Mixed STEP/IFC primitives require an explicit supported solid definition")
        from hamon_ifc_spatial_clash import _detect_length_unit_scale, parse_ifc_entities
        typed = parse_ifc_entities(step_text)
        if spec.get("unit_scale_m") is None:
            unit_scale = _detect_length_unit_scale(typed)
        occurrences = _ifc_product_properties(typed, entities, unit_scale, density)
        if occurrences is not None:
            return _cad_report(*occurrences, unit_scale, len(entities))

    if spec.get("unit_scale_m") is None and (selected_solid is not None or not has_ifc):
        from hamon_step_units import resolve_step_length_scale
        geometry_ids = ([selected_solid] if selected_solid is not None else
                        [eid for eid, (kind, _) in entities.items()
                         if kind in {"BLOCK", "RIGHT_CIRCULAR_CYLINDER", "EXTRUDED_AREA_SOLID"}])
        unit_scale, unit_metadata = resolve_step_length_scale(components, geometry_ids)

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
    surface_face_ids: List[int] = []
    surface_shell_ids: List[int] = []
    analytic_vol: Optional[float] = None

    def _append(v: np.ndarray, f: np.ndarray) -> None:
        off = sum(len(x) for x in all_verts)
        all_verts.append(v)
        all_faces.append(f + off)

    def _direction(reference):
        kind, values = entities.get(_ref_id(reference), (None, []))
        if kind != "DIRECTION" or len(values) != 2:
            raise ValueError("STEP analytic geometry requires a defined direction")
        ratios = np.asarray(_num_list(values[1]), dtype=float)
        if ratios.shape != (3,) or not np.isfinite(ratios).all() or not np.any(ratios):
            raise ValueError("STEP analytic geometry requires a finite nonzero direction")
        return tuple(Fraction(float(value)) for value in ratios)

    def _cross(a, b):
        return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])

    def _geometry_points(points, anchor, axis, line=False):
        # A supporting line/plane can be anchored arbitrarily far along itself.
        # Exact raw-coordinate predicates avoid cancellation at that anchor;
        # tolerances derive from the finite boundary, not its parameterization.
        points = np.asarray(points)
        scale = float(np.max(np.abs(points - points[0])))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("STEP analytic geometry is degenerate")
        normalized = np.array([float(value / max(map(abs, axis))) for value in axis])
        normalized /= np.linalg.norm(normalized)
        ulps = np.spacing(np.abs(points))
        rounding = np.linalg.norm(ulps, axis=1) if line else ulps @ np.abs(normalized)
        tolerances = 8 * rounding + 64 * np.finfo(float).eps * scale
        exact = [tuple(Fraction(float(value)) for value in point) for point in points]
        anchor = tuple(Fraction(float(value)) for value in anchor)
        norm_sq = sum(value*value for value in axis)
        for point, tolerance in zip(exact, tolerances):
            delta = tuple(value - origin for value, origin in zip(point, anchor))
            if line:
                residual_sq = sum(value*value for value in _cross(delta, axis))
            else:
                residual_sq = sum(value*direction for value, direction in zip(delta, axis))**2
            if residual_sq > Fraction(float(tolerance))**2 * norm_sq:
                kind = "line" if line else "plane"
                raise ValueError(f"STEP boundary vertices disagree with the declared {kind}")
        return exact

    for shell_id, eid, forward in face_occurrences or ():
        et, args = entities[eid]
        if et in ("FACE", "ADVANCED_FACE", "FACE_SURFACE") and len(args) >= 2:
            boundaries = []
            if len(args) >= 3 and args[2] != "$":
                if entities.get(_ref_id(args[2]), (None, []))[0] != "PLANE":
                    raise ValueError("STEP face requires a planar surface; curved faces cannot use boundary chords")
            for bid in _step_topology_refs(args[1], "face bounds"):
                if bid not in entities:
                    raise ValueError("STEP face references a missing boundary")
                bet, bargs = entities[bid]
                if bet not in ("FACE_OUTER_BOUND", "FACE_BOUND") or len(bargs) != 3 or bargs[2] not in (".T.", ".F."):
                    raise ValueError("STEP face requires an oriented boundary")
                lid = _ref_id(bargs[1])
                bound_sense = bargs[2] == ".T."
                if lid not in entities:
                    raise ValueError("STEP face references a missing loop")
                let, largs = entities[lid]
                lpts: List[np.ndarray] = []
                if let == "POLY_LOOP" and len(largs) == 2:
                    pids = _step_topology_refs(largs[1], "polygon loop")
                    if any(pid not in pts for pid in pids):
                        raise ValueError("STEP polygon loop references a missing point")
                    lpts = [pts[pid] for pid in pids]
                elif let == "EDGE_LOOP" and len(largs) == 2:
                    if et == "FACE" or len(args) < 3 or args[2] == "$":
                        raise ValueError("STEP edge-loop face requires a declared supporting plane")
                    endpoints = []
                    for oe_id in _step_topology_refs(largs[1], "edge loop"):
                        oe_kind, oargs = entities.get(oe_id, (None, []))
                        if oe_kind != "ORIENTED_EDGE" or len(oargs) != 5 or oargs[4] not in (".T.", ".F."):
                            raise ValueError("STEP edge loop requires complete oriented edges")
                        ec_kind, ec_args = entities.get(_ref_id(oargs[3]), (None, []))
                        if ec_kind != "EDGE_CURVE" or len(ec_args) != 5 or ec_args[4] not in (".T.", ".F."):
                            raise ValueError("STEP edge loop requires edge curves")
                        curve_kind, curve_args = entities.get(_ref_id(ec_args[3]), (None, []))
                        if curve_kind != "LINE" or len(curve_args) != 3:
                            raise ValueError("STEP edge loop requires straight lines; curved edges cannot use endpoint chords")
                        start, end = _ref_id(ec_args[1]), _ref_id(ec_args[2])
                        if start not in verts_map or end not in verts_map:
                            raise ValueError("STEP edge loop references a missing vertex")
                        vector_kind, vector = entities.get(_ref_id(curve_args[2]), (None, []))
                        anchor = pts.get(_ref_id(curve_args[1]))
                        if vector_kind != "VECTOR" or len(vector) != 3 or anchor is None:
                            raise ValueError("STEP line requires a defined point and vector")
                        magnitude = float(vector[2])
                        if not math.isfinite(magnitude) or magnitude <= 0:
                            raise ValueError("STEP line requires a positive finite vector magnitude")
                        axis = _direction(vector[1])
                        ends = _geometry_points([verts_map[start], verts_map[end]], anchor, axis, line=True)
                        orientation = sum((b-a)*direction for a, b, direction in zip(ends[0], ends[1], axis))
                        if orientation == 0 or (orientation > 0) != (ec_args[4] == ".T."):
                            raise ValueError("STEP edge direction disagrees with the declared curve sense")
                        if oargs[4] == ".F.":
                            start, end = end, start
                        endpoints.append((start, end))
                        lpts.append(verts_map[start])
                    if not endpoints or any(a[1] != b[0] for a, b in zip(endpoints, endpoints[1:] + endpoints[:1])):
                        raise ValueError("STEP edge loop is not a connected cycle")
                else:
                    raise ValueError("STEP face boundary loop type is unsupported")
                arr = np.array(lpts, dtype=np.float64)
                # FACE_SURFACE.same_sense relates surface and face normals.
                # FACE_BOUND.orientation determines the topological loop winding.
                if not bound_sense:
                    arr = arr[::-1].copy()
                boundaries.append((bet, arr))
            face_vertices, face_triangles = _step_face_region(boundaries)
            if len(args) >= 3 and args[2] != "$":
                surface = entities[_ref_id(args[2])][1]
                if len(surface) != 2 or len(args) != 4 or args[3] not in (".T.", ".F."):
                    raise ValueError("STEP planar face requires a surface placement and sense")
                placement_kind, placement = entities.get(_ref_id(surface[1]), (None, []))
                if placement_kind != "AXIS2_PLACEMENT_3D" or len(placement) != 4:
                    raise ValueError("STEP plane requires a three-dimensional placement")
                anchor = pts.get(_ref_id(placement[1]))
                if anchor is None:
                    raise ValueError("STEP plane references a missing location")
                axis = (Fraction(0), Fraction(0), Fraction(1)) if placement[2] == "$" else _direction(placement[2])
                exact = _geometry_points(face_vertices, anchor, axis)
                a, b, c = (exact[index] for index in face_triangles[0])
                cross = _cross(tuple(y-x for x, y in zip(a, b)), tuple(y-x for x, y in zip(a, c)))
                orientation = sum(value*direction for value, direction in zip(cross, axis))
                if orientation == 0 or (orientation > 0) != (args[3] == ".T."):
                    raise ValueError("STEP face winding disagrees with the declared surface sense")
            if not forward:
                face_triangles = face_triangles[:, [0, 2, 1]]
            _append(face_vertices, face_triangles)
            surface_face_ids.extend([eid] * len(face_triangles))
            surface_shell_ids.extend([shell_id] * len(face_triangles))

    analytic_entities = entities.items() if face_occurrences is None else ()
    for eid, (et, args) in analytic_entities:
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
            swept = _rectangular_sweep(entities, et, args, unit_scale)
            if swept is not None:
                vertices, faces, analytic_vol = swept
                _append(vertices, faces)

    if not all_verts:
        raise ValueError("No supported 3D solid entities found in STEP input")
    props = _compute_mesh_mass_properties(np.vstack(all_verts), np.vstack(all_faces), density,
                                          require_connected=face_occurrences is not None,
                                          surface_face_ids=surface_face_ids,
                                          surface_shell_ids=surface_shell_ids)
    report = _cad_report(props, analytic_vol, unit_scale, len(entities))
    report.update(unit_metadata)
    if selected_solid is not None:
        report["selected_solid_entity_id"] = selected_solid
    return report


def _cad_report(props, analytic_vol, unit_scale, entity_count):
    if "reason" in props:
        return {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "step_cad",
                "unit_scale_m": unit_scale, "entity_count": entity_count,
                "conservation_verified": False, **props}
    dual_rel_err = abs(props["volume_m3"] - analytic_vol) / analytic_vol if (analytic_vol and analytic_vol > 0) else 0.0
    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "step_cad",
        "unit_scale_m": unit_scale,
        "entity_count": entity_count,
        "dual_volume_rel_error": dual_rel_err,
        "conservation_verified": bool(max(dual_rel_err, props.get("max_product_volume_rel_error", 0.)) <= 5e-3),
        **props,
    }


def _thin_tube_properties(spec):
    """Uniform ideal thin-wall ring, using the wall median for stress recovery."""
    section = spec["section"]
    if not isinstance(section, dict) or section.get("shape") != "thin_walled_circular_tube":
        raise ValueError("unsupported section shape")
    radius = float(section["mean_radius_m"])
    thickness = float(section["wall_thickness_m"])
    if not (math.isfinite(radius) and 0 < thickness <= 0.05 * radius):
        raise ValueError("require positive finite mean radius and 0 < thickness/radius <= 0.05")
    area = 2 * math.pi * radius * thickness
    inertia = math.pi * radius ** 3 * thickness
    outer = radius + thickness / 2
    expected = {"I_m4": inertia, "J_m4": 2 * inertia, "area_m2": area,
                "c_m": outer, "depth_m": 2 * outer, "r_outer_m": outer}
    if any(key in spec and not math.isclose(float(spec[key]), value, rel_tol=1e-6)
           for key, value in expected.items()):
        raise ValueError("section properties disagree with the uniform thin-wall ring")
    return radius, area, inertia


def _beam_column_response(boundary, span, modulus, inertia, compression, point, distributed):
    """Stable small-slope beam-column under constant centroidal dead compression.

    Let theta=y', M=EI*theta', and S be the global transverse resultant.
    The material-section shear is -M'=S+N*theta. With a uniform lateral load,
    EI*theta''+N*theta=-S0+q*x on the full cantilever or symmetric half-span.
    """
    from fractions import Fraction
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    cantilever = boundary.startswith("cantilever")
    fixed_pair = boundary == "fixed_fixed_udl"
    length = Fraction(span) if cantilever else Fraction(span)/2
    rigidity = Fraction(modulus)*Fraction(inertia)
    load_ratio = Fraction(compression)*length**2/rigidity
    ctx = mp.clone()
    # The finite N->0 solution cancels inverse powers of the dimensionless
    # compression. Preserve those digits rather than dropping that load.
    missing = max(0, load_ratio.denominator.bit_length()-load_ratio.numerator.bit_length())
    ctx.prec = 180 + 2*missing
    def num(q):
        return ctx.mpf(q.numerator)/q.denominator
    # Prove the sign of the first tangent-stiffness eigenvalue against the
    # actual ideal support, independently of the optional design K. Carry the
    # resolved spectral gap's precision into the near-critical trig functions.
    intervals = MPIntervalContext()
    divisor = 1 if fixed_pair else 2
    for bits in (160, 320, 640, 1280):
        intervals.prec = bits
        gap = (intervals.pi/divisor)**2 - intervals.mpf(load_ratio.numerator)/load_ratio.denominator
        if gap.a > 0 or gap.b < 0:
            break
    else:
        raise ValueError("unresolved in-plane Euler stability boundary")
    stable = gap.a > 0
    if stable:
        _, exponent = ctx.frexp(ctx.mpf(gap.a))
        ctx.prec += max(0, -exponent)
    ell, ei = num(length), num(rigidity)
    n, q, force = map(ctx.mpf, (compression, abs(distributed), abs(point)))
    phase = ctx.sqrt(num(load_ratio))
    critical_phase = ctx.pi/divisor
    stability = {"in_plane_euler_critical_load_N": float(ei*(critical_phase/ell)**2)}
    if not stable:
        return {**stability, "stable": False}

    k = phase/ell
    end_force = force if cantilever else force/2
    root_force = end_force+q*ell
    # theta(x) = A sin(kx) + B cos(kx) - S0/N + q*x/N.
    if cantilever or fixed_pair:
        b = root_force/n                       # theta(0)=0
        a = ((b*ctx.sin(phase)-q/(n*k))/ctx.cos(phase) if cantilever  # M(ell)=0
             else (end_force/n-b*ctx.cos(phase))/ctx.sin(phase))      # theta(ell)=0
    else:
        a = -q/(n*k)                           # M(0)=0
        b = (end_force/n-a*ctx.sin(phase))/ctx.cos(phase)             # theta(ell)=0

    def displacement(x):
        angle = k*x
        return (2*a*ctx.sin(angle/2)**2+b*ctx.sin(angle))/k - root_force*x/n + q*x*x/(2*n)

    def moment(x):
        angle = k*x
        return ei*(k*(a*ctx.cos(angle)-b*ctx.sin(angle))+q/n)

    def shear(x):
        angle = k*x
        return n*(a*ctx.sin(angle)+b*ctx.cos(angle))

    def checked_float(value):
        result = float(value)
        if not math.isfinite(result) or (value and not result):
            raise ValueError("beam-column response exceeds representable range")
        return result

    # Both moment and material shear are sinusoidal plus a constant.
    # Splitting at every stationary point bounds their magnitude in each
    # interval by its endpoint magnitudes, including for tube stress recovery.
    xs = [ell*i/128 for i in range(129)]
    for angle in (ctx.atan2(a,b), ctx.atan2(-b,a)):
        turn = angle % ctx.pi
        if 0 < turn < phase:
            xs.append(turn/k)
    xs = sorted(xs)
    moments = [float(abs(moment(x))) for x in xs]
    shears = [float(abs(shear(x))) for x in xs]
    for peak in (max(moments), max(shears)):
        if not math.isfinite(peak) or peak == 0:
            raise ValueError("beam-column section demand exceeds representable range")
    deflection = displacement(ell)

    # Kinematic unit-load identity using the first-order signed virtual
    # moment for a unit force at the required tip or midpoint.
    roots, weights = np.polynomial.legendre.leggauss(20)
    virtual = ctx.zero
    for root, weight in zip(roots,weights):
        x = ell*(1+ctx.mpf(float(root)))/2
        unit = ell-x if cantilever else (ell/4-x/2 if fixed_pair else -x/2)
        virtual += ctx.mpf(float(weight))*moment(x)*unit
    virtual *= ell/(2*ei) if cantilever else ell/ei
    # Section-cut moment equilibrium includes the moment of the axial force.
    root_moment = (force*ell+q*ell**2/2+n*deflection if cantilever else
                   moment(ctx.zero) if fixed_pair else ctx.zero)
    moment_scale = max(map(abs,(moment(x) for x in xs)))
    equilibrium = max(abs(moment(x)-(root_moment-root_force*x+q*x*x/2-n*displacement(x)))
                      for x in (ctx.zero,ell/3,ell/2,ell))/moment_scale
    return {**stability, "stable": True, "deflection": checked_float(deflection),
            "virtual": checked_float(virtual), "moments": moments, "shears": shears,
            "transverse_resultant": checked_float(root_force), "equilibrium": float(equilibrium)}


def _tube_stress_bounds(boundary, moment, shear, torque, axial, radius, area, inertia,
                        profile=None):
    """Enclose peak von Mises over each span cell and circumferential cell.

    Opposite quadrants permit both membrane terms to have reinforcing signs.
    The flexural flow 2V cos(theta)/A and uniform torsional flow TR/(2I)
    act in the same local wall tangent. Bending varies as sin(theta).
    """
    angles = np.linspace(0, math.pi / 2, 257)
    sin_theta, cos_theta = np.sin(angles), np.cos(angles)
    axial_stress = abs(axial) / area
    torsion_stress = abs(torque) * radius / (2 * inertia)
    bending = abs(moment) * radius / inertia
    transverse = 2 * shear / area
    def stress(m, v, sn, cs):
        return np.hypot(axial_stress + m * sn, math.sqrt(3) * (torsion_stress + v * cs))

    if profile is not None:
        m = np.asarray(profile["moments"])[:,None] * radius/inertia
        v = np.asarray(profile["shears"])[:,None] * 2/area
        sample = stress(m, v, sin_theta, cos_theta)
        enclosure = stress(np.maximum(m[:-1],m[1:]), np.maximum(v[:-1],v[1:]),
                           sin_theta[1:], cos_theta[:-1])
    elif boundary == "simply_supported_udl":
        # u ranges from midspan to support. |M|=Mmax*(1-u²), |V|=Vmax*u.
        u = np.linspace(0, 1, 129)[:, None]
        m = bending * (1 - u * u)
        v = transverse * u
        sample = stress(m, v, sin_theta, cos_theta)
        enclosure = stress(m[:-1], v[1:], sin_theta[1:], cos_theta[:-1])
    else:
        # For the other supported load cases both maxima occur together
        # at a built-in end or at the one-sided limit of the central point load.
        sample = stress(bending, transverse, sin_theta, cos_theta)
        enclosure = stress(bending, transverse, sin_theta[1:], cos_theta[:-1])
    return float(np.max(sample)), float(np.max(enclosure))


def verify_structural_mechanics(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Ideal beam response in SI units under one specified load case.

    For mixed shear, pass section={shape: thin_walled_circular_tube,
    mean_radius_m: R, wall_thickness_m: t}. The uniform thin-wall membrane
    approximation uses constant internal torque, shear through the shear center
    and centroidal axial compression. Local load/support concentrations,
    distortional behavior and shell buckling are outside this section model.
    An unresolved stress assessment leaves passes_stress null. Equilibrium and
    virtual-work consistency are reported separately by conservation_verified.
    Euler buckling uses the declared ideal end restraints in the bending plane.
    An explicit K_buckling supplies the effective-length factor for that check.
    Combined lateral load and compression use the declared in-plane restraints
    for the small-slope beam-column equilibrium. Its section shear includes the
    axial force projected by the local rotation; transverse support reactions
    follow global force equilibrium.
    """
    def unsupported(message):
        return {"ok": False, "mode": "structural", "tool": "hamon_conservation_cad_verifier",
                "error": message, "conservation_verified": False}

    if spec.get("calculation") == "diaphragm_membrane":
        from hamon_diaphragm import solve_diaphragm
        try:
            result = solve_diaphragm(spec)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            return unsupported(str(exc))
        return {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "structural",
                "conservation_verified": result["dual_verified"], **result}
    bc = str(spec.get("boundary", "simply_supported_udl")).lower()
    end_factors = {"simply_supported_udl": 1., "simply_supported_point": 1.,
                   "cantilever_point": 2., "cantilever_udl": 2., "fixed_fixed_udl": .5}
    if bc not in end_factors:
        return unsupported(f"unsupported structural boundary: {bc}")
    tube = None
    if "section" in spec:
        try:
            tube = _thin_tube_properties(spec)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return unsupported(str(exc))
    L, E = float(spec["span_m"]), float(spec["E_Pa"])
    I = tube[2] if tube else float(spec["I_m4"])
    c_dist = float(spec.get("c_m", 0.5 * float(spec.get("depth_m", 0.2))))
    area = tube[1] if tube else float(spec.get("area_m2", 0.01))
    w, P = float(spec.get("w_N_m", 0.0)), float(spec.get("P_N", 0.0))
    torque = float(spec.get("torque_Nm", 0.0))
    J = float(spec.get("J_m4", 2.0 * I))
    r_outer = float(spec.get("r_outer_m", c_dist))
    axial_comp = float(spec.get("axial_compression_N", 0.0))
    K_eff = float(spec.get("K_buckling", end_factors[bc]))
    fy, fs_req = float(spec.get("yield_strength_Pa", 250e6)), float(spec.get("safety_factor", 1.5))
    defl_ratio = float(spec.get("deflection_limit_ratio", 360.0))

    if not all(math.isfinite(x) and x > 0 for x in (L, E, I, area, c_dist, J, r_outer, K_eff, fy, fs_req, defl_ratio)):
        return unsupported("section, material, span and limit parameters must be finite and positive")
    if not all(map(math.isfinite, (P, w, torque, axial_comp))) or axial_comp < 0:
        return unsupported("loads must be finite, with nonnegative axial compression")
    if (bc.endswith("_udl") and P != 0) or (bc.endswith("_point") and w != 0):
        return unsupported("applied loads do not match the selected boundary/load case")

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

    p_cr = (math.pi ** 2) * E * I / ((K_eff * L) ** 2)
    profile = None
    response = {}
    if axial_comp and (P or w):
        try:
            profile = _beam_column_response(bc, L, E, I, axial_comp, P, w)
        except (ValueError, OverflowError) as exc:
            return unsupported(str(exc))
        response = {"transverse_model": "elastic_small_slope_beam_column",
                    "in_plane_euler_critical_load_N": profile["in_plane_euler_critical_load_N"]}
        if not profile["stable"]:
            return {**unsupported("no stable combined-load response at or above the in-plane Euler load"),
                    **response, "euler_critical_load_N": p_cr, "passes_buckling": False,
                    "buckling_effective_length_factor": K_eff,
                    "buckling_effective_length_source": "K_buckling" if "K_buckling" in spec else "boundary",
                    "passes_deflection": None, "passes_stress": None}
        m_max, v_max = max(profile["moments"]), max(profile["shears"])
        defl_a, defl_vw = profile["deflection"], profile["virtual"]
        dual_err = abs(defl_a-defl_vw)/max(defl_a,defl_vw)
        response.update(max_transverse_resultant_N=profile["transverse_resultant"],
                        section_moment_equilibrium_rel_error=profile["equilibrium"])
    else:
        n_p = 200
        xs = np.linspace(0.0, L, n_p + 1)
        wts = np.ones(n_p + 1); wts[1:-1:2] = 4.0; wts[2:-2:2] = 2.0
        defl_vw = abs(float(np.sum(wts * np.array([m_fn(float(x)) * m_u(float(x)) / (E * I) for x in xs])) * (L / (3.0 * n_p))))
        dual_err = abs(defl_a - defl_vw) / max(defl_a, 1e-30)
    fy_res = abs((r_l + r_r) - (w * L + P))
    allow = fy / fs_req
    limit = allow * (1.0 + 1e-9)
    stress_bounds = None
    if tube:
        lower, upper = _tube_stress_bounds(bc, m_max, v_max, torque, axial_comp, *tube, profile=profile)
        stress_bounds = {"lower": lower, "upper": upper}
        vm = upper
        passes_stress = True if upper <= limit else False if lower > limit else None
        assessment = "ideal_thin_wall_circular_tube_upper_bound"
    elif torque != 0 and v_max != 0:
        vm = passes_stress = None
        assessment = "indeterminate_combined_shear_requires_supported_section"
    else:
        sigma_b = abs(m_max) * c_dist / I + axial_comp / area
        shear_stress = abs(torque) * r_outer / J if torque else 1.5 * v_max / area
        vm = math.hypot(sigma_b, math.sqrt(3) * shear_stress)
        passes_stress = vm <= limit
        assessment = "legacy_scalar_section_screen"
    d_lim = L / defl_ratio
    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "structural",
        **response,
        "force_equilibrium_residual_N": fy_res,
        "max_moment_Nm": abs(m_max),
        "max_shear_N": v_max,
        "max_deflection_m": defl_a,
        "virtual_work_deflection_m": defl_vw,
        "dual_deflection_rel_error": dual_err,
        "von_mises_stress_Pa": vm,
        "allowable_stress_Pa": allow,
        "euler_critical_load_N": p_cr,
        "buckling_effective_length_factor": K_eff,
        "buckling_effective_length_source": "K_buckling" if "K_buckling" in spec else "boundary",
        "passes_stress": passes_stress,
        "stress_assessment": assessment,
        "von_mises_bounds_Pa": stress_bounds,
        "passes_deflection": defl_a <= d_lim * (1.0 + 1e-9),
        "passes_buckling": axial_comp <= (p_cr / fs_req) if axial_comp > 0 else True,
        "conservation_verified": bool(fy_res <= 1e-7 * max(abs(w * L + P), 1.0) and dual_err <= 1e-5
                                      and (profile is None or profile["equilibrium"] <= 1e-9)),
    }


_SPICE_HIERARCHY = re.compile(r"(?im)^\s*\.(?:subckt|ends)\b|^[ \t]*X\S*\s")
_SPICE_SCALE = {"a": 1e-18, "mil": 25.4e-6, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def _parse_spice_num(tok: str) -> float:
    m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|mil|[afpnumkgt])?", tok.strip().lower())
    if not m:
        raise ValueError(f"Invalid SPICE token {tok!r}")
    return float(m.group(1)) * (_SPICE_SCALE[m.group(2)] if m.group(2) else 1.0)


def _independent_source_value(tokens: List[str], frequency: float) -> Tuple[float, float]:
    """Read independent DC operating-point and AC small-signal excitations."""
    clauses: Dict[str, List[float]] = {}
    current = "DC"
    for token in tokens:
        keyword = token.upper()
        if keyword in ("DC", "AC"):
            if keyword in clauses:
                raise ValueError(f"Duplicate {keyword} clause in independent source")
            current = keyword
            clauses[current] = []
            continue
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[A-Za-z]*", token):
            raise ValueError(f"Unsupported independent source value {token!r}")
        value = _parse_spice_num(token)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite independent source value {token!r}")
        clauses.setdefault(current, []).append(value)

    dc = clauses.get("DC", [0.0])
    ac = clauses.get("AC", [0.0])
    if len(dc) != 1 or len(ac) > 2:
        raise ValueError("Invalid number of values in independent source DC or AC clause")
    if frequency > 0:
        return (ac[0] if ac else 1.0), (ac[1] if len(ac) == 2 else 0.0)
    return dc[0], 0.0


_LINEAR_GAIN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?(?:meg|mil|[afpnumkgt])?", re.I)
_BEHAVIORAL_E = re.compile(r"^(?:poly(?:\(|$)|(?:value|vol|table|laplace|freq)(?:=|$))", re.I)


def _parse_spice_netlist(netlist: str, frequency_override: Optional[float] = None) -> Tuple[List[Dict[str, Any]], float]:
    if _SPICE_HIERARCHY.search(netlist):
        from hamon_spice_hierarchy import elaborate_subcircuits
        netlist = elaborate_subcircuits(netlist, _LINEAR_GAIN)
    branches: List[Dict[str, Any]] = []
    source_clauses = []
    active_lines = []
    freq_hz = 0.0
    for idx, raw_line in enumerate(netlist.splitlines()):
        line = raw_line.split("*")[0].split(";")[0].split("$")[0].strip()
        if not line:
            continue
        if idx == 0 and not re.match(r"^[RLCVIGrlcvig\.]", line):
            first = line.split()
            # Keep ordinary titles beginning with E. Like existing R/V cards,
            # accept an unambiguous numeric E card in the titleless extension.
            is_e_card = first[0][0].upper() == "E" and (
                (len(first) == 6 and _LINEAR_GAIN.fullmatch(first[5]))
                or (len(first) >= 4 and _BEHAVIORAL_E.match(first[3]))
            )
            is_k_card = (first[0][0].upper() == "K" and len(first) >= 3
                         and all(name.upper().startswith("L") for name in first[1:3]))
            is_current_control = first[0][0].upper() in ("F", "H") and (
                (len(first) == 5 and _LINEAR_GAIN.fullmatch(first[4]))
                or (len(first) >= 4 and first[3].lower().startswith("poly("))
            )
            is_d_card = (first[0][0].upper() == "D" and len(first) >= 4 and
                         (re.match(r"D\d", first[0], re.I) or re.search(
                             r"(?im)^\s*\.model\s+" + re.escape(first[3]) + r"\s+D\b", netlist)))
            if not (is_e_card or is_k_card or is_current_control or is_d_card):
                continue
        active_lines.append(raw_line.split(";", 1)[0].split("$", 1)[0].strip())
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
            branch = {"name": toks[0], "type": kind, "n_pos": toks[1], "n_neg": toks[2],
                      "value": _parse_spice_num(toks[3])}
            if kind == "L" and len(toks) != 4:
                branch["winding_options"] = toks[4:]
            branches.append(branch)
        elif kind in ("V", "I") and (len(toks) >= 4 or (idx > 0 and len(toks) == 3)):
            branch = {"name": toks[0], "type": kind, "n_pos": toks[1], "n_neg": toks[2]}
            branches.append(branch)
            source_clauses.append((branch, toks[3:]))
        elif kind == "D":
            if len(toks) not in {4, 5}:
                raise ValueError("Expected diode name, anode, cathode, model and optional area")
            branches.append({"name": toks[0], "type": "D", "n_pos": toks[1], "n_neg": toks[2],
                             "model": toks[3], "area": toks[4] if len(toks) == 5 else "1"})
        elif kind == "K":
            if len(toks) < 4 or not _LINEAR_GAIN.fullmatch(toks[-1]):
                raise ValueError("Expected K name, winding names, and a coupling coefficient")
            branches.append({"name": toks[0], "type": "K", "inductors": toks[1:-1],
                             "value": _parse_spice_num(toks[-1])})
        elif kind in ("F", "H"):
            if len(toks) != 5 or not _LINEAR_GAIN.fullmatch(toks[4]):
                raise ValueError(f"Expected a linear {kind} source: name out+ out- sense_source gain")
            branches.append({"name": toks[0], "type": kind, "n_pos": toks[1], "n_neg": toks[2],
                             "control_source": toks[3], "value": _parse_spice_num(toks[4])})
        elif kind == "E":
            if len(toks) != 6 or _BEHAVIORAL_E.match(toks[3]) or not _LINEAR_GAIN.fullmatch(toks[5]):
                raise ValueError("Expected a linear E source: name out+ out- ctrl+ ctrl- gain")
            branches.append({"name": toks[0], "type": "E", "n_pos": toks[1], "n_neg": toks[2],
                             "nc_pos": toks[3], "nc_neg": toks[4], "value": _parse_spice_num(toks[5])})
        elif kind == "G" and len(toks) >= 6:
            branches.append({"name": toks[0], "type": "G", "n_pos": toks[1], "n_neg": toks[2], "nc_pos": toks[3], "nc_neg": toks[4], "value": _parse_spice_num(toks[5])})
    if frequency_override is not None:
        freq_hz = float(frequency_override)
    has_diodes = any(br["type"] == "D" for br in branches)
    if has_diodes:
        from hamon_spice_diode import configure_junctions
        configure_junctions(branches, active_lines, _parse_spice_num)
    for branch, tokens in source_clauses:
        branch["value"], branch["phase_deg"] = _independent_source_value(tokens, freq_hz)
        if has_diodes:
            branch["dc_value"] = _independent_source_value(tokens, 0.)[0]
    return branches, freq_hz


def _mutual_inductances(branches):
    """Resolve K records to winding indices and check the normalized magnetic energy form."""
    cards = [br for br in branches if str(br["type"]).upper() == "K"]
    if not cards:
        return {}
    indices = {}
    for index, branch in enumerate(branches):
        indices.setdefault(str(branch["name"]).casefold(), []).append(index)

    coefficients = {}
    mutual = {}
    for card in cards:
        if len(indices[str(card["name"]).casefold()]) != 1:
            raise ValueError("Ambiguous mutual coupling name")
        names = card.get("inductors")
        k = float(card["value"])
        if not (0 < k <= 1) or not isinstance(names, (list, tuple)) or len(names) < 2:
            raise ValueError("Coupling requires at least two windings and 0 < k <= 1")
        windings = []
        for name in names:
            hits = indices.get(str(name).casefold(), [])
            if len(hits) != 1 or str(branches[hits[0]]["type"]).upper() != "L":
                raise ValueError(f"Unknown or ambiguous coupled inductor {name!r}")
            index = hits[0]
            if branches[index].get("winding_options"):
                raise ValueError("Coupled winding options are unsupported")
            inductance = float(branches[index]["value"])
            if not math.isfinite(inductance) or inductance <= 0:
                raise ValueError("Coupled windings need positive finite inductance")
            windings.append(index)
            coefficients.setdefault(index, {index: Fraction(1)})
            mutual.setdefault(index, {})
        for a, b in combinations(windings, 2):
            if b in coefficients[a]:
                raise ValueError("Self or duplicate mutual inductance")
            coefficients[a][b] = coefficients[b][a] = Fraction(k)
            mutual[a][b] = mutual[b][a] = k

    # L = diag(sqrt(Li)) C diag(sqrt(Li)). Check C by exact symmetric
    # elimination on the input coefficients, including singular ideal coupling.
    for index, row in coefficients.items():
        pivot = row[index]
        neighbors = [j for j in row if j != index and row[j]]
        if pivot < 0 or (pivot == 0 and neighbors):
            raise ValueError("Coupling matrix permits negative magnetic energy")
        for pos, a in enumerate(neighbors):
            for b in neighbors[pos:]:
                reduced = coefficients[a].get(b, 0) - row[a] * row[b] / pivot
                coefficients[a][b] = coefficients[b][a] = reduced
        for j in row:
            if j != index:
                del coefficients[j][index]
    return mutual


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


def _sum_phasors(values):
    return complex(math.fsum(v.real for v in values), math.fsum(v.imag for v in values))


def _circuit_source(br, cardinal_phases):
    magnitude = float(br["value"])
    angle = float(br.get("phase_deg", 0.0))
    if cardinal_phases:
        quadrant = {0: 1, 90: 1j, -90: -1j, 180: -1, -180: -1}.get(math.remainder(angle, 360))
        if quadrant is not None:
            return complex(magnitude * quadrant)
    return cmath.rect(magnitude, math.radians(angle))


def _current_control_rows(branches, current_row):
    controlled = [i for i, br in enumerate(branches) if str(br["type"]).upper() in ("F", "H")]
    if not controlled:
        return {}
    names = {}
    for index, br in enumerate(branches):
        names.setdefault(str(br["name"]).casefold(), []).append(index)
    controls = {}
    for index in controlled:
        br = branches[index]
        sense = br.get("control_source")
        hits = names.get(sense.casefold(), []) if isinstance(sense, str) and sense else []
        if len(hits) != 1 or str(branches[hits[0]]["type"]).upper() != "V":
            raise ValueError(f"Expected one independent voltage source for current control {sense!r}")
        if not math.isfinite(float(br["value"])):
            raise ValueError("Current-controlled source gain must be finite")
        controls[index] = current_row[hits[0]]
    return controls


def _circuit_matrix(branches, omega, n_idx, current_row, coupling, controls, exact=False):
    """Stamp the same circuit topology in floating or exact coefficient arithmetic."""
    dim = len(n_idx) + len(current_row)
    if exact:
        import sympy as sp
        mna, rhs = sp.zeros(dim), sp.zeros(dim, 1)
        number, jomega = sp.Rational, sp.I * sp.Rational(omega)
    else:
        mna = np.zeros((dim, dim), dtype=np.complex128)
        rhs = np.zeros(dim, dtype=np.complex128)
        number, jomega = float, 1j * omega

    cardinal_phases = bool(coupling) or any(br["type"] == "D" for br in branches)
    injections = [[] for _ in n_idx]

    def source(br):
        v = _circuit_source(br, cardinal_phases)
        return number(v.real) + sp.I * number(v.imag) if exact else v

    mutual = {}
    admittances = {}
    for index, br in enumerate(branches):
        bt = str(br["type"]).upper()
        if bt == "K":
            continue
        ip, im = (n_idx.get(str(br[key])) for key in ("n_pos", "n_neg"))
        val = number(float(br["value"]))
        if index in current_row:
            row = current_row[index]
            if not math.isfinite(float(val)):
                raise ValueError("Nonfinite voltage or winding constraint")
            # In AC a coupled winding enforces v/(j omega) = sum_j Lij ij.
            vf = 1/jomega if index in coupling and omega else 1
            if ip is not None: mna[ip, row] += 1; mna[row, ip] += vf
            if im is not None: mna[im, row] -= 1; mna[row, im] -= vf
            if bt == "V":
                rhs[row] = source(br)
            elif bt == "E":
                cp, cm = (n_idx.get(str(br[key])) for key in ("nc_pos", "nc_neg"))
                if cp is not None: mna[row, cp] -= val
                if cm is not None: mna[row, cm] += val
            elif bt == "H":
                mna[row, controls[index]] -= val
            elif index in coupling and omega:
                mna[row, row] -= val
                mutual[index] = {}
                for other, k in coupling[index].items():
                    lb = number(float(branches[other]["value"]))
                    if exact:
                        mean = sp.sqrt(val * lb)
                    else:
                        ma, ea = math.frexp(val)
                        mb, eb = math.frexp(lb)
                        exponent, odd = divmod(ea + eb, 2)
                        mean = math.ldexp(math.sqrt(math.ldexp(ma * mb, odd)), exponent)
                    inductance = number(k) * mean
                    mutual[index][other] = inductance
                    mna[row, current_row[other]] -= inductance
        elif bt in ("R", "L", "C", "D"):
            if bt == "D":
                y = number(br.get("incremental_S", 0.))
            else:
                y = 1/val if bt == "R" else (1/(jomega*val) if bt == "L" else jomega*val)
            admittances[index] = y
            if ip is not None: mna[ip, ip] += y
            if im is not None: mna[im, im] += y
            if ip is not None and im is not None: mna[ip, im] -= y; mna[im, ip] -= y
        elif bt == "I":
            current = source(br)
            if ip is not None: injections[ip].append(-current)
            if im is not None: injections[im].append(current)
        elif bt == "F":
            sense = controls[index]
            if ip is not None: mna[ip, sense] += val
            if im is not None: mna[im, sense] -= val
        elif bt == "G":
            cp, cm = (n_idx.get(str(br[key])) for key in ("nc_pos", "nc_neg"))
            for row, sign in ((ip, 1), (im, -1)):
                if row is not None:
                    if cp is not None: mna[row, cp] += sign * val
                    if cm is not None: mna[row, cm] -= sign * val
    for node, currents in enumerate(injections):
        rhs[node] = sum(currents) if exact else _sum_phasors(currents)
    return mna, rhs, admittances, mutual


def verify_circuit_conservation(spec: Dict[str, Any]) -> Dict[str, Any]:
    if spec.get("calculation") == "symmetrical_fault":
        from hamon_ac_power_flow import verify_symmetrical_fault
        return {**verify_symmetrical_fault(spec), "mode": "circuit", "tool": "hamon_conservation_cad_verifier"}
    if spec.get("calculation") == "ac_power_flow":
        from hamon_ac_power_flow import verify_ac_power_flow
        return {**verify_ac_power_flow(spec), "mode": "circuit", "tool": "hamon_conservation_cad_verifier"}
    netlist = ""
    if spec.get("netlist") or spec.get("file_b64"):
        netlist = str(spec["netlist"]) if spec.get("netlist") else base64.b64decode(spec["file_b64"]).decode("utf-8", errors="replace")
        branches, freq_hz = _parse_spice_netlist(netlist, spec.get("freq_hz"))
    else:
        branches = list(spec.get("branches") or [])
        freq_hz = float(spec.get("freq_hz", 0.0))

    if not math.isfinite(freq_hz) or freq_hz < 0:
        raise ValueError("Circuit frequency must be finite and nonnegative")
    omega = 2.0 * math.pi * freq_hz
    gnd = {"0", "GND", "gnd"}
    nodes = sorted({str(br[k]) for br in branches for k in ("n_pos", "n_neg", "nc_pos", "nc_neg") if k in br and str(br[k]) not in gnd})
    n_idx = {nm: i for i, nm in enumerate(nodes)}
    # An ideal inductor has zero terminal voltage at DC. Its current, like
    # the current of a voltage source, is an independent MNA unknown.
    constraint_types = {"V", "E", "H", "L"} if freq_hz == 0 else {"V", "E", "H"}
    coupling = _mutual_inductances(branches)
    constraints = [i for i, br in enumerate(branches)
                   if str(br["type"]).upper() in constraint_types or i in coupling]
    current_row = {i: len(nodes) + offset for offset, i in enumerate(constraints)}
    controls = _current_control_rows(branches, current_row)
    diode_states = {}
    if any(str(br["type"]).upper() == "D" for br in branches):
        if not netlist:
            raise ValueError("Diode circuits require a SPICE deck with explicit model definitions")
        from hamon_spice_diode import junction_law, solve_bias
        diode_states, bias_solution, bias_discrepancy = solve_bias(
            branches, n_idx, _circuit_matrix, _complex_gauss_solve)
        for br in branches:
            if br["type"] == "D":
                state = diode_states[br["name"]]
                br["incremental_S"] = state["conductance_S"]
    inputs = (branches, omega, n_idx, current_row, coupling, controls)
    mna, rhs, admittances, mutual = _circuit_matrix(*inputs)
    exact_ports = {}
    if diode_states and freq_hz == 0:
        # Publish the nonlinear fixed point. Re-solving its Norton tangent
        # can lose the current difference that established the junction bias.
        sol_lu, dual_err = bias_solution, bias_discrepancy
    else:
        if diode_states:
            scales = np.max(np.abs(mna), axis=1)
            if np.any(scales == 0):
                raise ValueError("Undetermined small-signal diode circuit")
            mna, rhs = mna/scales[:, None], rhs/scales
        if diode_states and np.linalg.cond(mna) > 1e10:
            from hamon_spice_diode import precise_linear
            sol_lu, exact_state, dual_err = precise_linear(_circuit_matrix, inputs)
            for index, br in enumerate(branches):
                vp = exact_state[n_idx[br["n_pos"]]] if br["n_pos"] in n_idx else 0
                vm = exact_state[n_idx[br["n_neg"]]] if br["n_neg"] in n_idx else 0
                exact_ports[index] = complex(vp - vm)
        else:
            sol_g = _complex_gauss_solve(mna, rhs)
            sol_lu = np.linalg.solve(mna, rhs)
            dual_err = float(np.linalg.norm(sol_g - sol_lu) / max(float(np.linalg.norm(sol_lu)), 1e-30))
    # A small winding voltage may be a genuine residual or an exact magnetic
    # null. Recover the original complete network before rounding either one.
    for index, neighbors in mutual.items():
        terms = [float(branches[index]["value"]) * sol_lu[current_row[index]]]
        terms.extend(m * sol_lu[current_row[j]] for j, m in neighbors.items())
        scale = sum(map(abs, terms))
        if scale and abs(sum(terms)) <= 1e-8 * scale:
            exact_mna, exact_rhs, _, _ = _circuit_matrix(*inputs, exact=True)
            reduced, pivots = exact_mna.row_join(exact_rhs).to_DM(extension=True).rref()
            if tuple(pivots) != tuple(range(len(rhs))):
                raise ValueError("Singular coupled circuit")
            exact_solution = reduced.to_Matrix()[:, -1]
            refined = np.array([complex(v.evalf(50)) for v in exact_solution])
            dual_err = max(dual_err, float(np.linalg.norm(refined-sol_lu) /
                                            max(float(np.linalg.norm(refined)), 1e-30)))
            sol_lu = refined
            break
    v_node: Dict[str, complex] = {"0": 0j, "GND": 0j, **{nm: complex(sol_lu[i]) for nm, i in n_idx.items()}}

    terminal_currents = {nm: [] for nm in nodes}
    cardinal_phases = bool(coupling or diode_states)
    kvl_max = 0.0
    powers = []
    delivered = []
    branch_currents = []
    for index, br in enumerate(branches):
        nm, bt = str(br["name"]), str(br["type"]).upper()
        if bt == "K":
            continue  # K relates existing windings; it has no additional terminal current.
        vd = exact_ports.get(index, v_node.get(str(br["n_pos"]), 0j) - v_node.get(str(br["n_neg"]), 0j))
        if index in current_row:
            ibr = complex(sol_lu[current_row[index]])
            expected_voltage = 0j
            if bt == "V":
                expected_voltage = _circuit_source(br, cardinal_phases)
            elif bt == "E":
                control = v_node.get(str(br["nc_pos"]), 0j) - v_node.get(str(br["nc_neg"]), 0j)
                expected_voltage = float(br["value"]) * control
            elif bt == "H":
                expected_voltage = float(br["value"]) * sol_lu[controls[index]]
            elif index in mutual and freq_hz > 0:
                flux = float(br["value"]) * ibr
                for winding, inductance in mutual[index].items():
                    flux += inductance * sol_lu[current_row[winding]]
                expected_voltage = 1j * omega * flux
            kvl_max = max(kvl_max, abs(vd - expected_voltage))
        elif bt == "I":
            ibr = _circuit_source(br, cardinal_phases)
        elif bt == "F":
            ibr = float(br["value"]) * sol_lu[controls[index]]
        elif bt == "G":
            ibr = float(br["value"]) * (v_node.get(str(br["nc_pos"]), 0j) - v_node.get(str(br["nc_neg"]), 0j))
        elif bt == "D" and not omega:
            ibr = complex(junction_law(br, vd.real)[0])
        else:
            ibr = admittances[index] * vd
        branch_currents.append({
            "collection": "branches", "index": index, "name": nm, "type": bt,
            "n1": str(br["n_pos"]), "n2": str(br["n_neg"]),
            "current_A": [float(ibr.real), float(ibr.imag)] if omega > 0 else float(ibr.real),
        })
        absorbed = vd * ibr.conjugate()
        powers.append(absorbed)
        if bt in {"V", "I", "E", "F", "G", "H"}:
            delivered.append(-absorbed)
        if str(br["n_pos"]) in terminal_currents: terminal_currents[str(br["n_pos"])].append(ibr)
        if str(br["n_neg"]) in terminal_currents: terminal_currents[str(br["n_neg"])].append(-ibr)

    s_tot, s_deliv = _sum_phasors(powers), _sum_phasors(delivered)
    kcl_max = max((abs(_sum_phasors(v)) for v in terminal_currents.values()), default=0.0)
    cand_ok = True
    if isinstance(spec.get("proposed_node_voltages"), dict):
        rtol = float(spec.get("rtol", 1e-4))
        if diode_states or _SPICE_HIERARCHY.search(netlist):
            claimed_nodes = {name.casefold(): voltage for name, voltage in v_node.items()}
        else:
            claimed_nodes = v_node
        for k_nm, cv_raw in spec["proposed_node_voltages"].items():
            key = str(k_nm) if claimed_nodes is v_node else str(k_nm).casefold()
            ref_v = claimed_nodes[key]
            cv = complex(cv_raw[0], cv_raw[1]) if isinstance(cv_raw, (list, tuple)) else complex(cv_raw)
            if abs(cv - ref_v) / max(abs(ref_v), 1e-12) > rtol:
                cand_ok = False

    return {
        "ok": True,
        "tool": "hamon_conservation_cad_verifier",
        "mode": "circuit",
        "freq_hz": freq_hz,
        "branch_currents": branch_currents,
        **({"diode_operating_point": diode_states} if diode_states else {}),
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
    if spec.get("calculation") == "rigid_strip_contact":
        from hamon_bearing_capacity import verify_rigid_strip_contact
        return verify_rigid_strip_contact(spec)
    if spec.get("calculation") == "compaction_acceptance":
        from hamon_dual_verifier import verify_compaction_lot
        result = verify_compaction_lot(spec)
        return {**result, "ok": result["dual_verified"],
                "conservation_verified": result["dual_verified"] and result["lot_accepted"]}
    if spec.get("calculation") == "at_rest_plane_stress":
        from hamon_rankine import verify_at_rest_plane_stress
        return verify_at_rest_plane_stress(spec)
    if spec.get("calculation") == "rankine_layers":
        from hamon_rankine import verify_rankine_layers
        return verify_rankine_layers(spec)
    if spec.get("calculation") == "consolidation_history_1d":
        from hamon_consolidation import verify_consolidation_history
        return verify_consolidation_history(spec)
    if spec.get("calculation") == "consolidation_1d":
        from hamon_consolidation import verify_consolidation_1d
        return verify_consolidation_1d(spec)
    out: Dict[str, Any] = {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech"}
    if spec.get("calculation") not in (None, "bearing_capacity"):
        out.update(ok=False, conservation_verified=False, reason="Unsupported civil/geotechnical calculation")
        return out
    evaluated = False
    ok = True
    if "cut_bank_m3" in spec or "fill_compacted_m3" in spec:
        evaluated = True
        cut_b, sh, fill_c = float(spec.get("cut_bank_m3", 0.0)), float(spec.get("shrink_factor", 0.90)), float(spec.get("fill_compacted_m3", 0.0))
        out["net_compacted_surplus_m3"] = cut_b * sh - fill_c
    if "footing_B_m" in spec or spec.get("calculation") == "bearing_capacity":
        evaluated = True
        from hamon_bearing_capacity import verify_bearing_capacity
        bearing = verify_bearing_capacity(spec)
        out.update(bearing)
        ok = ok and bearing["conservation_verified"]
        if not bearing["ok"]:
            return out
        if "Cc" in spec and "H_clay_m" in spec:
            from hamon_consolidation import consolidation_load, primary_settlement
            try:
                initial = float(spec["sigma_v0_eff_Pa"])
                pc = float(spec.get("sigma_preconsolidation_Pa", initial))
                cr = float(spec["Cr"]) if "Cr" in spec or pc > initial else 0.0
                load, stress_check = consolidation_load(spec, coerce_scalar=True)
                out.update(stress_check)
                ok = ok and stress_check.get("declared_stress_consistent", True)
                out.update(primary_settlement(float(spec["H_clay_m"]), float(spec["e0"]),
                                              float(spec["Cc"]), initial, load, pc, cr))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                out.update(ok=False, reason=str(exc))
                ok = False
    if "channel_B_m" in spec and "y1_m" in spec and "y2_m" in spec:
        evaluated = True
        g = 9.80665
        b_c, q_f, y1, y2 = float(spec["channel_B_m"]), float(spec["Q_m3_s"]), float(spec["y1_m"]), float(spec["y2_m"])
        v1, v2 = q_f / (b_c * y1), q_f / (b_c * y2)
        hl = (float(spec.get("z1_m", 0.0)) + y1 + v1 * v1 / (2.0 * g)) - (float(spec.get("z2_m", 0.0)) + y2 + v2 * v2 / (2.0 * g))
        out["head_loss_m"] = hl
        ok = ok and (hl >= -1e-9)
    if not evaluated:
        out.update(ok=False, conservation_verified=False,
                   reason="No recognized civil/geotechnical calculation was evaluated")
        return out
    out["conservation_verified"] = bool(ok)
    return out


def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """Count a neutral formula with grouped subscripts and explicit hydrate dots.

    A period between digits is a decimal point. Use a middle dot for an
    adduct, for example CaSO4·2H2O. A component coefficient scales every
    atom in that component. Molar flow supplies the whole-formula amount.
    """
    token_pattern = r"[A-Z][a-z]?|[0-9]+(?:\.[0-9]+)?|[()]"
    totals: Dict[str, float] = {}
    for component_index, part in enumerate(formula.strip().split("·")):
        tokens = re.findall(token_pattern, part)
        if not tokens or "".join(tokens) != re.sub(r"\s+", "", part):
            raise ValueError("Empty hydrate component or unsupported formula character")
        i = 0

        def count():
            nonlocal i
            if i == len(tokens) or not tokens[i][0].isdigit():
                return 1.0
            value = float(tokens[i])
            i += 1
            if not math.isfinite(value) or value < 0:
                raise ValueError("Formula counts must be finite and nonnegative")
            return value

        multiplier = count() if component_index else 1.0
        if multiplier == 0:
            raise ValueError("A hydrate component coefficient must be positive")
        groups: List[Dict[str, float]] = [{}]
        while i < len(tokens):
            token = tokens[i]
            i += 1
            if token == "(":
                groups.append({})
                continue
            if token == ")":
                if len(groups) == 1 or not groups[-1]:
                    raise ValueError("Unmatched or empty group in chemical formula")
                atoms = groups.pop()
            elif token[0].isupper():
                atoms = {token: 1.0}
            else:
                raise ValueError("Unexpected coefficient in chemical formula")
            factor = count()
            for element, amount in atoms.items():
                groups[-1][element] = groups[-1].get(element, 0.0) + factor * amount
        if len(groups) != 1 or not any(groups[0].values()):
            raise ValueError("Incomplete group or hydrate component in chemical formula")
        for element, amount in groups[0].items():
            totals[element] = totals.get(element, 0.0) + multiplier * amount
            if not math.isfinite(totals[element]):
                raise ValueError("Chemical formula atom count exceeds numerical range")
    return totals


def _colebrook_dual_friction(re_num: float, rel_rough: float) -> Tuple[float, float]:
    if re_num < 2300.0:
        return 64.0 / re_num, 0.0
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


def _scalar_pipe_energy(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Steady pump duty for signed 1→2 flow and endpoint differences (2 minus 1).

    Head loss is a magnitude. Excess hydraulic power requires dissipation or
    separate recovery equipment; a pump efficiency does not model a turbine.
    """
    g = 9.80665
    diameter, length, flow = (float(spec[k]) for k in ("pipe_D_m", "pipe_L_m", "Q_m3_s"))
    rho = float(spec.get("rho_kg_m3", 998.2))
    mu = float(spec.get("mu_Pa_s", 1.002e-3))
    roughness = float(spec.get("roughness_m", 4.5e-5))
    minor = float(spec.get("K_minor", 0.0))
    efficiency = float(spec.get("pump_efficiency", 0.78))
    dz = float(spec.get("delta_z_m", 0.0))
    dp = float(spec.get("delta_P_Pa", 0.0))
    rtol = float(spec.get("rtol", 1e-4))
    if not all(math.isfinite(x) for x in (diameter, length, flow, rho, mu, roughness, minor, efficiency, dz, dp, rtol)):
        raise ValueError("nonfinite scalar pipe input")
    if min(diameter, rho, mu) <= 0 or min(length, roughness, minor, rtol) < 0 or not 0 < efficiency <= 1:
        raise ValueError("invalid scalar pipe geometry, fluid properties, loss coefficient, tolerance or efficiency")

    friction = error = None
    head_loss = reynolds = hydraulic = 0.0
    if flow != 0:
        speed = abs(flow) / (0.25 * math.pi * diameter * diameter)
        reynolds = rho * speed * diameter / mu
        if not math.isfinite(reynolds) or reynolds <= 0:
            raise ValueError("unrepresentable Reynolds number for nonzero flow")
        friction, error = _colebrook_dual_friction(reynolds, roughness / diameter)
        head_loss = (friction * (length / diameter) + minor) * (speed * speed) / (2.0 * g)
        pump_head = math.fsum((dz, dp / (rho * g), math.copysign(head_loss, flow)))
        hydraulic = rho * g * flow * pump_head
        if not all(math.isfinite(x) for x in (friction, error, head_loss, hydraulic)) or friction <= 0:
            raise ValueError("unrepresentable scalar pipe result")

    shaft = max(hydraulic, 0.0) / efficiency
    if not math.isfinite(shaft):
        raise ValueError("unrepresentable shaft power")
    return {"darcy_f": friction, "colebrook_dual_rel_error": error, "reynolds_number": reynolds,
            "head_loss_m": head_loss, "required_hydraulic_power_W": hydraulic,
            "required_shaft_power_W": shaft, "excess_hydraulic_power_W": max(-hydraulic, 0.0),
            "pipe_balance_ok": error is None or error <= rtol}


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
    if spec.get("calculation") == "pump_operating_point":
        from hamon_pipe_network import verify_pump_operating_point
        return verify_pump_operating_point(spec)
    if spec.get("calculation") == "pump_duty":
        from hamon_pipe_network import verify_pump_duty
        return verify_pump_duty(spec)
    if spec.get("calculation") == "euler_rotor":
        from hamon_turbomachinery import verify_turbomachinery
        return {**verify_turbomachinery(spec), "mode": "thermal_fluid", "tool": "hamon_conservation_cad_verifier"}
    if spec.get("calculation") == "pipe_network" or ("nodes" in spec and "pipes" in spec):
        from hamon_pipe_network import verify_pipe_network
        return {**verify_pipe_network(spec), "mode": "thermal_fluid", "tool": "hamon_conservation_cad_verifier"}
    rtol = float(spec.get("rtol", 1e-4))
    out: Dict[str, Any] = {"ok": True, "tool": "hamon_conservation_cad_verifier", "mode": "thermal_fluid"}
    all_ok = True
    if "inlets" in spec or "outlets" in spec:
        ins, outs = spec.get("inlets") or [], spec.get("outlets") or []
        streams = ins + outs
        has_species = any("ndot_mol_s" in s or "formula" in s for s in streams)
        if not has_species or any("mdot_kg_s" in s for s in streams):
            mass_ok = bool(streams) and all("mdot_kg_s" in s for s in streams)
            if mass_ok:
                mi, mo = (sum(float(s["mdot_kg_s"]) for s in side) for side in (ins, outs))
                mass_ok = math.isfinite(mi) and math.isfinite(mo) and abs(mi - mo) <= rtol * max(mi, mo, 1e-12)
            out["mass_balance_ok"] = mass_ok
            all_ok = mass_ok
        if has_species:
            joint = any("mdot_kg_s" in stream for stream in streams)
            if joint:
                from hamon_species_stream_mass import check_stream_mass
                out["stream_mass_mole_ok"] = False
                out["stream_molar_mass_default"] = "CIAAW_2024_abridged_normal_material"
                out["declared_molar_mass_streams"] = [
                    {"side": name, "index": index}
                    for name, side in (("inlets", ins), ("outlets", outs))
                    for index, stream in enumerate(side) if "molar_mass_g_mol" in stream
                ]
            species_ok = all("ndot_mol_s" in s and s.get("formula") for s in streams)
            if species_ok:
                ai, ao = {}, {}
                errors = []
                for side_name, side, totals in (("inlets", ins, ai), ("outlets", outs, ao)):
                    for index, stream in enumerate(side):
                        try:
                            elements = parse_chemical_formula(str(stream["formula"]))
                        except ValueError as exc:
                            species_ok = False
                            out.setdefault("species_balance_error", str(exc))
                            continue
                        molar_rate = float(stream["ndot_mol_s"])
                        if joint:
                            if "mdot_kg_s" in stream:
                                reason = check_stream_mass(elements, float(stream["mdot_kg_s"]), molar_rate, rtol, stream)
                            else:
                                reason = "missing mass rate"
                            if reason:
                                errors.append({"side": side_name, "index": index, "reason": reason})
                        for element, count in elements.items():
                            totals[element] = totals.get(element, 0.0) + molar_rate * count
                if joint:
                    out["stream_mass_mole_ok"] = species_ok and not errors
                    if errors:
                        out["stream_mass_mole_errors"] = errors
                    all_ok = all_ok and out["stream_mass_mole_ok"]
                species_ok = species_ok and all(
                    math.isfinite(ai.get(e, 0.0)) and math.isfinite(ao.get(e, 0.0))
                    and abs(ai.get(e, 0.0) - ao.get(e, 0.0)) / max(ai.get(e, 0.0), ao.get(e, 0.0), 1e-12) <= rtol
                    for e in ai.keys() | ao.keys()
                )
            out["species_balance_ok"] = species_ok
            all_ok = all_ok and species_ok
    if any(key in spec for key in (*_HX_INPUTS, "UA_W_K")):
        hx = _heat_exchanger_balance(spec, rtol)
        out.update(hx)
        all_ok = all_ok and hx["hx_balance_ok"]
    if "pipe_D_m" in spec or "pipe_L_m" in spec:
        try:
            pipe = _scalar_pipe_energy(spec)
            out.update(pipe)
            all_ok = all_ok and pipe["pipe_balance_ok"]
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            out.update(ok=False, pipe_balance_ok=False, pipe_error=str(exc))
            all_ok = False
    checked = any(key in out for key in ("mass_balance_ok", "species_balance_ok", "hx_balance_ok", "darcy_f"))
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
        elif spec.get("calculation") in ("euler_rotor", "pump_duty", "pump_operating_point"): mode = "thermal_fluid"
        elif spec.get("calculation") == "pipe_network" or ("nodes" in spec and "pipes" in spec): mode = "thermal_fluid"
        elif spec.get("calculation") in ("ac_power_flow", "symmetrical_fault"): mode = "circuit"
        elif spec.get("netlist") or spec.get("branches") or fname.endswith((".cir", ".sp", ".spice", ".net")): mode = "circuit"
        elif "lhs" in spec and "rhs" in spec: mode = "dimensional"
        elif spec.get("calculation") == "diaphragm_membrane" or ("span_m" in spec and "E_Pa" in spec): mode = "structural"
        elif spec.get("calculation") in ("consolidation_1d", "consolidation_history_1d", "rankine_layers", "at_rest_plane_stress", "bearing_capacity", "rigid_strip_contact", "compaction_acceptance") or any(k in spec for k in ("cut_bank_m3", "footing_B_m", "channel_B_m")): mode = "civil_geotech"
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
