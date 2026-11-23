"""Small-displacement, orthotropic rectangular deck membranes with axial bars and wall springs.

All quantities are SI. Nodes are numbered from zero; omit cells within openings.
The panel rigidities are force per unit width, with zero Poisson coupling and
material axes along global X and Y. Bars and springs carry signed axial force.
"""
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import bmat, coo_matrix, diags, eye
from hamon_diaphragm_modes import diaphragm_coordinates
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


def _recover_stations(requests, cells, xy, shears):
    """Recover center values of cells incident on the requested exterior wall segment."""
    if not requests:
        return []
    edge_owners = {}
    for cell, item in enumerate(cells):
        ids = item["nodes"]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edge_owners.setdefault(tuple(sorted((a, b))), []).append(cell)
    exterior = [(xy[a], xy[b], owners[0]) for (a, b), owners in edge_owners.items()
                if len(owners) == 1]
    results = []
    for station in requests:
        ids = station["edge_nodes"]
        if len(ids) != 2 or any(type(i) is not int or not 0 <= i < len(xy) for i in ids):
            raise ValueError("station requires two valid wall endpoint indices")
        a, b = xy[ids[0]], xy[ids[1]]
        tangent = b - a
        if np.count_nonzero(tangent) != 1:
            raise ValueError("station wall must follow a horizontal or vertical exterior edge")
        ax = int(tangent[1] != 0)
        length = abs(tangent[ax])
        direction = np.sign(tangent[ax])
        fraction = float(station["fraction"])
        if not 0 <= fraction <= 1:
            raise ValueError("station fraction must be within [0,1]")
        faces = []
        for p, q, cell in exterior:
            if p[1-ax] != a[1-ax] or q[1-ax] != a[1-ax]:
                continue
            start, end = sorted(((p[ax]-a[ax])*direction, (q[ax]-a[ax])*direction))
            if start >= 0 and end <= length:
                faces.append((start, end, cell))
        faces.sort()
        if (not faces or faces[0][0] != 0 or faces[-1][1] != length
                or any(left[1] != right[0] for left, right in zip(faces, faces[1:]))):
            raise ValueError("station wall does not follow a continuous set of exterior faces")
        distances = []
        for _, _, cell in faces:
            sw, _, ne, _ = cells[cell]["nodes"]
            rel_center = ((xy[sw] - a) + (xy[ne] - a)) / 2
            distances.append(np.linalg.norm(rel_center - fraction * tangent))
        nearest = min(distances)
        tolerance = 32 * np.finfo(float).eps * max(length, nearest)
        tied = [face[2] for face, d in zip(faces, distances) if d <= nearest + tolerance]
        signed = [float(shears[i]) for i in tied]
        rule = station.get("tie_rule")
        if rule == "first_from_start" or (rule is None and len(tied) == 1):
            magnitude = abs(signed[0])
        elif rule == "max_abs":
            magnitude = max(map(abs, signed))
        elif rule is None:
            magnitude = None
        else:
            raise ValueError("tie_rule must be first_from_start or max_abs")
        results.append({"panels": tied, "signed_shear_N_m": signed, "magnitude_N_m": magnitude})
    return results


def solve_diaphragm(spec):
    precision = spec.get("displacement_precision", "binary64")
    if precision == "compensated":
        from hamon_diaphragm_compensated import check_compensated_size, solve_compensated
        check_compensated_size(spec)
    elif precision != "binary64":
        raise ValueError("displacement_precision must be binary64 or compensated")
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
    panel_nodes = []
    for panel in spec["panels"]:
        ids = tuple(node(i) for i in panel["nodes"])
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("each panel needs four distinct nodes and a unique cell")
        panel_nodes.append(ids)
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
        bars.append((dofs, axis, k))
    for spring in spec["springs"]:
        axis = {"x": 0, "y": 1}[spring["axis"]]
        dof = 2 * node(spring["node"]) + axis
        k = _positive(spring["k_N_m"])
        springs.append((dof, k))
    for load in spec["loads"]:
        dof = 2 * node(load["node"])
        f = np.asarray(load["force_N"], dtype=float)
        if f.shape != (2,) or not np.isfinite(f).all():
            raise ValueError("force_N must be a finite [Fx,Fy] vector")
        force[dof:dof+2] += f

    membrane = coo_matrix((entries, (rows, cols)), shape=(ndof, ndof)).tocsc()
    ground = np.zeros(ndof)
    for dof, k in springs:
        ground[dof] += k
    if not np.isfinite(ground).all():
        raise ValueError("nonfinite assembled spring stiffness")
    basis = diaphragm_coordinates(xy, panel_nodes, bars, ground)
    if precision == "compensated":
        result = solve_compensated(spec, xy, springs, _bar_checks)
        result["stations"] = _recover_stations(spec.get("stations", []), spec["panels"],
                                              xy, result["panel_center_shear_N_m"])
        return result
    for use_nodal in (False, True):
        if use_nodal:
            # Mixed restraints on slender panels can lose spring displacement
            # when reconstructed from rigid and elastic coordinates. In nodal
            # coordinates every spring acts on its own displacement directly.
            active = np.ones(ndof, dtype=bool)
            basis = eye(ndof, format="csc"), active, active, np.ones(len(bars), dtype=bool)
        result = _solve_in_basis(spec, xy, force, panels, bars, springs, membrane, ground, basis, use_nodal)
        if result is not None:
            return result
    raise ValueError("diaphragm failed local/global equilibrium or work-energy balance")


def _factor(scaled):
    try:
        factor = splu(scaled)
    except RuntimeError as exc:
        raise ValueError("unsupported or singular diaphragm") from exc
    pivots = np.abs(factor.U.diagonal())
    if pivots.min() <= 1e-12 * pivots.max():
        raise ValueError("unsupported or ill-conditioned diaphragm")
    return factor


def _mixed_bar_solution(stiffness, rhs, bar_map, axial_k, bars):
    # A connector force is its own unknown. This avoids both domination
    # of B.T K B by the stiffest link and cancellation in force recovery.
    # Bars sharing both endpoints have identical signed extension, including
    # when their endpoints are reversed. Solve once for their total force,
    # then divide it according to each member's axial stiffness.
    endpoints = np.array([sorted((dofs[0], dofs[2])) for dofs, _, _ in bars])
    _, first, group = np.unique(endpoints, axis=0, return_index=True, return_inverse=True)
    group = group.ravel()
    group_k = np.bincount(group, weights=axial_k)
    group_map = bar_map[first]
    scaled = bmat([[stiffness, group_map.T], [group_map, -diags(1 / group_k)]], format="csc")
    scale = np.ones(scaled.shape[0])
    # Zero displacement diagonals are possible on a patch supported solely
    # by connectors. Equilibrate entire rows/columns of this symmetric system.
    for _ in range(12):
        norm = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        if not np.isfinite(norm).all() or np.min(norm) <= 0:
            raise ValueError("unconnected or invalid diaphragm degrees of freedom")
        step = 1 / np.sqrt(norm)
        scale *= step
        scaling = diags(step)
        scaled = (scaling @ scaled @ scaling).tocsc()
        if np.max(np.abs(np.log(norm))) < .05:
            break
    solution = scale * _factor(scaled).solve(scale * np.r_[rhs, np.zeros(len(group_k))])
    total_tension = solution[len(rhs):]
    return solution[:len(rhs)], total_tension[group] * (axial_k / group_k[group])


def _equilibrium_estimates(stiffness, rhs, bar_map, axial_k, cross_patch, use_nodal,
                           force, ground, panels, bars):
    if cross_patch:
        try:
            mixed = _mixed_bar_solution(stiffness, rhs, bar_map, axial_k, bars)
        except ValueError:
            # Redundant nearly rigid connectors may have unresolved self-stress
            # in the mixed system. Retain their stiffness ratios by condensation.
            pass
        else:
            yield mixed
    if bars:
        stiffness = (stiffness + bar_map.T @ diags(axial_k) @ bar_map).tocsc()
    diagonal = stiffness.diagonal()
    if not np.isfinite(diagonal).all() or np.min(diagonal) <= 0:
        raise ValueError("unconnected or invalid diaphragm degrees of freedom")
    scale = 1 / np.sqrt(diagonal)
    scaled = (diags(scale) @ stiffness @ diags(scale)).tocsc()
    factor = _factor(scaled)
    coordinates = scale * factor.solve(scale * rhs)
    yield coordinates, None
    if not use_nodal:
        return
    # Use element strains to form the correction load independently of the
    # assembled matrix, whose small stiffness terms can round away beside a
    # stiff ground spring. Reuse its sparse factors for at most two corrections.
    x = coordinates.astype(np.longdouble)
    for _ in range(2):
        residual = force - ground * x
        for dofs, quadrature, weights, _ in panels:
            local = x[dofs]
            for strain in quadrature:
                residual[dofs] -= strain.T @ (weights * (strain @ local))
        for dofs, axis, stiffness in bars:
            residual[dofs] -= axis * (stiffness * (axis @ x[dofs]))
        x = x + scale * factor.solve(scale * residual.astype(float))
        yield x, None


def _solve_in_basis(spec, xy, force, panels, bars, springs, membrane, ground, basis, use_nodal):
    ndof = len(force)
    transform, elastic, bar_free, elastic_only_bars = basis
    projection = diags(elastic.astype(float))
    stiffness = (projection @ membrane @ projection + transform.T @ diags(ground) @ transform).tocsc()
    bar_map, axial_k = None, None
    if bars:
        b_rows, b_cols, b_data = [], [], []
        for i, (dofs, axis, _) in enumerate(bars):
            b_rows.extend([i]*4)
            b_cols.extend(dofs)
            b_data.extend(axis)
        bar_axes = coo_matrix((b_data, (b_rows, b_cols)), shape=(len(bars), ndof)).tocsr()
        # An in-patch bar also annihilates *relative* patch motions.
        # Cross-patch bars retain exactly the straining patch coordinates.
        relative = diags((~elastic & ~bar_free).astype(float))
        connector_axes = diags((~elastic_only_bars).astype(float)) @ bar_axes
        bar_map = bar_axes @ projection + connector_axes @ transform @ relative
        axial_k = np.array([k for _, _, k in bars])
    estimates = _equilibrium_estimates(stiffness, transform.T @ force, bar_map, axial_k,
                                      np.any(~elastic_only_bars), use_nodal, force, ground, panels, bars)
    for coordinates, mixed_tensions in estimates:
        deformation = elastic * coordinates
        u = transform @ coordinates

        if not np.isfinite(u).all():
            raise ValueError("nonfinite diaphragm displacement")

        reactions = np.zeros(ndof)
        spring_forces, bar_forces, panel_shears, double_energy = [], [], [], []
        for dof, k in springs:
            value = k * u[dof]
            reactions[dof] += value
            spring_forces.append(value)
            double_energy.append(value * u[dof])
        constitutive_error = 0.
        internal_force = membrane @ deformation
        if bars:
            extensions = bar_map @ coordinates
            if mixed_tensions is None:
                bar_forces = axial_k * extensions
            else:
                bar_forces = mixed_tensions
                # Compare independent force and kinematic extensions at the
                # scale of the endpoint motions, including rigid translation.
                constitutive_extension = bar_forces / axial_k
                magnitude = (abs(bar_axes) @ abs(u) + abs(bar_map) @ abs(coordinates)
                             + abs(constitutive_extension))
                residual = abs(extensions - constitutive_extension)
                constitutive_error = float(np.max(residual / np.maximum(magnitude, 1e-30)))
                extensions = constitutive_extension
            internal_force += bar_axes.T @ bar_forces
            double_energy.extend(bar_forces * extensions)
        for dofs, gauss, weights, recover in panels:
            local_u = deformation[dofs]
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
        errors = [constitutive_error, np.linalg.norm(internal_force + reactions - force, np.inf) / load_scale,
                  np.linalg.norm(force_error, 1) / load_scale,
                  abs(moment_error) / (span * load_scale),
                  abs(work - energy) / max(abs(work), abs(energy), 1e-30)]
        if np.isfinite(errors).all() and max(errors) <= 1e-7:
            break
    else:
        return None
    member_checks = _bar_checks(spec, bar_forces)
    stations = _recover_stations(spec.get("stations", []), spec["panels"], xy, panel_shears)
    return {"stations": stations, "bar_capacity_checks": member_checks,
            "displacements_m": u.astype(float, copy=False).reshape((-1, 2)).tolist(),
            "panel_center_shear_N_m": list(map(float, panel_shears)),
            "bar_tension_N": list(map(float, bar_forces)),
            "spring_transmitted_N": list(map(float, spring_forces)),
            "force_residual_N": force_error.tolist(), "moment_residual_Nm": moment_error,
            "strain_energy_J": energy / 2, "external_work_J": work / 2,
            "dual_rel_discrepancy": float(max(errors)), "dual_verified": True}


def _bar_checks(spec, bar_forces, *, convert=float, output=float):
    member_checks = []
    for i, (bar, tension) in enumerate(zip(spec.get("bars", []), bar_forces)):
        if any(k in bar for k in ("allowable_tension_N", "allowable_compression_N", "demand_factor")):
            tension_limit = convert(_positive(bar["allowable_tension_N"]))
            compression_limit = convert(_positive(bar["allowable_compression_N"]))
            factor = convert(_positive(bar.get("demand_factor", 1.0)))
            limit = tension_limit if tension >= 0 else compression_limit
            demand = abs(tension) * factor
            passes = demand <= limit
            # A binary64 product may round down to the allowable. Resolve this
            # tie using the same force published in bar_tension_N.
            if convert is float and demand == limit:
                passes = abs(Fraction(tension)) * Fraction(factor) <= limit
            member_checks.append({"bar": i, "ratio": output(demand / limit), "passes": bool(passes)})
    return member_checks
