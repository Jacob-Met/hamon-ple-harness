"""Small-displacement, orthotropic rectangular deck membranes with axial bars and wall springs.

All quantities are SI. Nodes are numbered from zero; omit cells within openings.
The panel rigidities are force per unit width, with zero Poisson coupling and
material axes along global X and Y. Bars and springs carry signed axial force.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


def _positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("stiffnesses, lengths and capacities must be finite and positive")
    return value


@lru_cache(maxsize=256)
def _panel(width, height, dx, dy, shear):
    """Return the integrated stiffness and strain operators for one rectangular Q4."""
    signs_x = np.array([-1., 1., 1., -1.])
    signs_y = np.array([-1., -1., 1., 1.])

    def strain(r, s):
        nx = signs_x * (1 + s * signs_y) / (2 * width)
        ny = signs_y * (1 + r * signs_x) / (2 * height)
        b = np.zeros((3, 8))
        b[0, ::2], b[1, 1::2] = nx, ny
        b[2, ::2], b[2, 1::2] = ny, nx
        return b

    g = 1 / math.sqrt(3)
    gauss = tuple(strain(r, s) for r in (-g, g) for s in (-g, g))
    weights = np.array([dx, dy, shear]) * (width * height / 4)
    stiffness = sum(b.T @ (weights[:, None] * b) for b in gauss)
    return stiffness, gauss, weights, strain(0, 0)[2] * shear


def _check_rectangles(rectangles):
    """A positive area may be integrated once; a shared side must be conforming."""
    active = []
    for left, right, bottom, top in sorted(rectangles):
        active = [r for r in active if r[1] >= left]
        for x0, x1, y0, y1 in active:
            overlap_x = min(x1, right) - left
            overlap_y = min(y1, top) - max(y0, bottom)
            if overlap_x > 0 and overlap_y > 0:
                raise ValueError("overlapping membrane panels")
            if overlap_x == 0 and overlap_y > 0 and (y0, y1) != (bottom, top):
                raise ValueError("nonconforming vertical edge with a hanging node")
            if overlap_y == 0 and overlap_x > 0 and (x0, x1) != (left, right):
                raise ValueError("nonconforming horizontal edge with a hanging node")
        active.append((left, right, bottom, top))


def solve_diaphragm(spec):
    xy = np.asarray(spec["nodes_m"], dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) < 4:
        raise ValueError("nodes_m must contain finite [x,y] coordinates")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("coincident diaphragm nodes")
    ndof = 2 * len(xy)
    force = np.zeros(ndof)
    rows, cols, entries = [], [], []

    def node(value):
        if type(value) is not int or not 0 <= value < len(xy):
            raise ValueError("invalid zero-based node index")
        return value

    def scatter(dofs, local):
        n = len(dofs)
        rows.extend(np.repeat(dofs, n))
        cols.extend(np.tile(dofs, n))
        entries.extend(local.ravel())

    panels, bars, springs = [], [], []
    rectangles = []
    for panel in spec["panels"]:
        ids = tuple(node(i) for i in panel["nodes"])
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("each panel needs four distinct nodes and a unique cell")
        p = xy[list(ids)]
        width, height = p[2] - p[0]
        expected = [[0, 0], [width, 0], [width, height], [0, height]]
        if min(width, height) <= 0 or not np.array_equal(p - p[0], expected):
            raise ValueError("panel nodes must be rectangular in SW,SE,NE,NW order")
        rectangles.append((p[0, 0], p[2, 0], p[0, 1], p[2, 1]))
        dx, dy, shear = (_positive(panel[k]) for k in ("Dx_N_m", "Dy_N_m", "G_N_m"))
        local, gauss, weights, recover = _panel(width, height, dx, dy, shear)
        dofs = np.array([[2*i, 2*i+1] for i in ids]).ravel()
        scatter(dofs, local)
        panels.append((dofs, gauss, weights, recover))
    _check_rectangles(rectangles)
    if not panels:
        raise ValueError("at least one membrane panel is required")

    for bar in spec.get("bars", []):
        a, b = (node(i) for i in bar["nodes"])
        vector = xy[b] - xy[a]
        length = _positive(np.linalg.norm(vector))
        axis = np.r_[-vector, vector] / length
        k = _positive(bar["EA_N"]) / length
        dofs = np.array([2*a, 2*a+1, 2*b, 2*b+1])
        scatter(dofs, k * np.outer(axis, axis))
        bars.append((dofs, axis, k))
    for spring in spec["springs"]:
        axis = {"x": 0, "y": 1}[spring["axis"]]
        dof = 2 * node(spring["node"]) + axis
        k = _positive(spring["k_N_m"])
        scatter([dof], np.array([[k]]))
        springs.append((dof, k))
    for load in spec["loads"]:
        dof = 2 * node(load["node"])
        f = np.asarray(load["force_N"], dtype=float)
        if f.shape != (2,) or not np.isfinite(f).all():
            raise ValueError("force_N must be a finite [Fx,Fy] vector")
        force[dof:dof+2] += f

    stiffness = coo_matrix((entries, (rows, cols)), shape=(ndof, ndof)).tocsc()
    diagonal = stiffness.diagonal()
    if not np.isfinite(diagonal).all() or np.min(diagonal) <= 0:
        raise ValueError("unconnected or invalid diaphragm degrees of freedom")
    scale = 1 / np.sqrt(diagonal)
    scaled = (diags(scale) @ stiffness @ diags(scale)).tocsc()
    try:
        factor = splu(scaled)
    except RuntimeError as exc:
        raise ValueError("unsupported or singular diaphragm") from exc
    pivots = np.abs(factor.U.diagonal())
    if pivots.min() <= 1e-12 * pivots.max():
        raise ValueError("unsupported or ill-conditioned diaphragm")
    u = scale * factor.solve(scale * force)

    if not np.isfinite(u).all():
        raise ValueError("nonfinite diaphragm displacement")

    reactions = np.zeros(ndof)
    spring_forces, bar_forces, panel_shears, double_energy = [], [], [], []
    for dof, k in springs:
        value = k * u[dof]
        reactions[dof] += value
        spring_forces.append(value)
        double_energy.append(value * u[dof])
    for dofs, axis, k in bars:
        extension = axis @ u[dofs]
        bar_forces.append(k * extension)
        double_energy.append(k * extension**2)
    for dofs, gauss, weights, recover in panels:
        local_u = u[dofs]
        panel_shears.append(recover @ local_u)
        double_energy.extend(weights @ (b @ local_u)**2 for b in gauss)

    # Ground forces below are forces transmitted TO the supports. Taking moments
    # about a point in the model avoids cancellation under large world offsets.
    arms = xy - xy[0]
    balance = (force - reactions).reshape((-1, 2))
    force_error = np.sum(balance, axis=0)
    moment_error = float(arms[:, 0] @ balance[:, 1] - arms[:, 1] @ balance[:, 0])
    load_scale = max(np.linalg.norm(force, 1) + np.linalg.norm(reactions, 1), 1e-30)
    span = np.max(np.ptp(xy, axis=0))
    work = float(force @ u)
    energy = math.fsum(double_energy)
    errors = [np.linalg.norm(stiffness @ u - force, np.inf) / load_scale,
              np.linalg.norm(force_error, 1) / load_scale,
              abs(moment_error) / (span * load_scale),
              abs(work - energy) / max(abs(work), abs(energy), 1e-30)]
    if not np.isfinite(errors).all() or max(errors) > 1e-7:
        raise ValueError("diaphragm failed local/global equilibrium or work-energy balance")
    member_checks = []
    for i, (bar, tension) in enumerate(zip(spec.get("bars", []), bar_forces)):
        if any(k in bar for k in ("allowable_tension_N", "allowable_compression_N", "demand_factor")):
            tension_limit = _positive(bar["allowable_tension_N"])
            compression_limit = _positive(bar["allowable_compression_N"])
            demand = abs(tension) * _positive(bar.get("demand_factor", 1.0))
            ratio = demand / (tension_limit if tension >= 0 else compression_limit)
            member_checks.append({"bar": i, "ratio": float(ratio), "passes": bool(ratio <= 1)})
    return {"bar_capacity_checks": member_checks,
            "displacements_m": u.reshape((-1, 2)).tolist(),
            "panel_center_shear_N_m": list(map(float, panel_shears)),
            "bar_tension_N": list(map(float, bar_forces)),
            "spring_transmitted_N": list(map(float, spring_forces)),
            "force_residual_N": force_error.tolist(), "moment_residual_Nm": moment_error,
            "strain_energy_J": energy / 2, "external_work_J": work / 2,
            "dual_rel_discrepancy": float(max(errors)), "dual_verified": True}
