"""Saturated small-strain 1D consolidation with connected pressure and Darcy flux."""
import math

import numpy as np
from scipy.linalg import eigh_tridiagonal

from hamon_consolidation import _area_integral, _number, _rectangles
from hamon_footing_settlement import _depth_intervals, _depth_stress, _gauss_integral, _precise_corner


def _properties(spec):
    rows = spec["hydraulic_layers"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError("Specify one to four contiguous hydraulic layers")
    layers = []
    for row in rows:
        keys = ("top_depth_m", "bottom_depth_m", "mv_per_Pa", "cv_m2_s")
        if not isinstance(row, dict) or set(row) != set(keys):
            raise ValueError("Each layer needs only its depth interval, constant mv and cv")
        a, b, mv, cv = (_number(row, k) for k in keys)
        if a < 0 or b <= a or mv <= 0 or cv <= 0 or (layers and a != layers[-1][1]):
            raise ValueError("Require contiguous ordered layers with positive thickness, mv and cv")
        if not 0 < cv * mv < math.inf:
            raise ValueError("Unrepresentable hydraulic conductivity divided by fluid unit weight")
        layers.append((a, b, mv, cv))
    return np.array(layers)


def _loading(spec, layers):
    if "delta_sigma_v_Pa" in spec:
        if "rectangular_loads" in spec or "settlement_column" in spec:
            raise ValueError("Specify one uniform surcharge or one rectangular loading system")
        load = _number(spec, "delta_sigma_v_Pa")
        if load <= 0:
            raise ValueError("A positive instantaneous load step is required")
        return lambda depth: load, (layers[:, 1] - layers[:, 0]) * load, load
    column = spec["settlement_column"]
    if set(column) != {"x_m", "y_m"} or not 1 <= len(spec["rectangular_loads"]) <= 4:
        raise ValueError("Specify one settlement column and one to four rectangular loads")
    for row in spec["rectangular_loads"]:
        if set(row) != {"center_x_m", "center_y_m", "width_m", "length_m", "net_pressure_Pa"}:
            raise ValueError("Unexpected or missing rectangular load field")
    rects = _rectangles(spec["rectangular_loads"], _number(column, "x_m"), _number(column, "y_m"))
    peak = math.fsum(r[-1] for r in rects)
    if peak <= 0:
        raise ValueError("A positive instantaneous load step is required")

    def load(depth):
        return _depth_stress(rects, depth, _precise_corner)

    # The area-kernel integral is independent of the grid's corner-stress samples.
    abscissa, weights = np.polynomial.legendre.leggauss(12)
    nodes = list(zip((abscissa + 1)/2, weights/2))
    integrals = []
    for a, b, _, _ in layers:
        integrals.append(math.fsum(_gauss_integral(lambda z: _depth_stress(rects, z, _area_integral),
                                                   nodes, left, right)
                                   for left, right in _depth_intervals(a, b, rects)))
    return load, np.array(integrals), peak


def _grid(layers, load, drained, elapsed, n, vertex):
    """Cell-centred finite volumes or linear finite elements with lumped storage."""
    thickness = (layers[:, 1] - layers[:, 0]) / n
    dx = np.repeat(thickness, n)
    storage = dx * np.repeat(layers[:, 2], n)
    mobility = np.repeat(layers[:, 2] * layers[:, 3], n)
    # Generate each layer locally; retain the supplied interface ordinates.
    z = np.concatenate([np.linspace(a, b, n + 1)[:-1] for a, b, _, _ in layers] + [layers[-1:, 1]])
    if vertex:
        mass = (np.pad(storage, (0, 1)) + np.pad(storage, (1, 0))) / 2
        conductance = mobility / dx
        diagonal = np.pad(conductance, (0, 1)) + np.pad(conductance, (1, 0))
        first, stop = int(drained[0]), len(mass) - int(drained[1])
        points = z
    else:
        mass = storage
        # Both half-cell resistances lie in series at every face, including material interfaces.
        conductance = 2 / (dx[:-1] / mobility[:-1] + dx[1:] / mobility[1:])
        diagonal = np.pad(conductance, (0, 1)) + np.pad(conductance, (1, 0))
        if drained[0]:
            diagonal[0] += 2 * mobility[0] / dx[0]
        if drained[1]:
            diagonal[-1] += 2 * mobility[-1] / dx[-1]
        first, stop = 0, len(mass)
        points = .5 * z[:-1] + .5 * z[1:]
    initial = np.array([load(p) for p in points])
    active = slice(first, stop)
    root = np.sqrt(mass[active])
    diag = diagonal[active] / mass[active]
    off = -conductance[first:stop-1] / (root[:-1] * root[1:])
    eigen, vectors = eigh_tridiagonal(diag, off, lapack_driver="stev")
    if not any(drained):
        # A closed column has a constant pressure mode exactly. Restore its zero rate.
        if abs(eigen[0]) > 128 * len(root) * np.finfo(float).eps * eigen[-1]:
            raise ValueError("Closed-column zero diffusion mode was unresolved")
        eigen[0] = 0.
    slowest = eigen[0] if any(drained) else eigen[1]
    if slowest <= 0 or np.finfo(float).eps * eigen[-1] > slowest * 1e-5:
        raise ValueError("Diffusion spectrum unresolved")
    pressure = np.zeros(len(mass))
    pressure[active] = (vectors @ (np.exp(-eigen * elapsed) * (vectors.T @ (root * initial[active])))) / root
    average = (pressure[:-1] + pressure[1:]) * .5 if vertex else pressure
    retained = (storage * average).reshape(len(layers), n).sum(axis=1)
    faces = []
    for j in range(n, len(dx), n):
        if vertex:
            left = mobility[j-1] * (pressure[j-1] - pressure[j]) / dx[j-1]
            right = mobility[j] * (pressure[j] - pressure[j+1]) / dx[j]
            # Correct element-centre fluxes for storage in the two half elements.
            flux = (storage[j]*left + storage[j-1]*right) / (storage[j]+storage[j-1])
            value = pressure[j]
        else:
            left, right = dx[j-1] / (2*mobility[j-1]), dx[j] / (2*mobility[j])
            flux = (pressure[j-1] - pressure[j]) / (left + right)
            value = pressure[j-1] - flux*left
        faces.append((value, flux))
    initial_mean = (initial[:-1]+initial[1:])*.5 if vertex else initial
    initial_water = (storage*initial_mean).reshape(len(layers), n).sum(axis=1)
    return retained, np.array(faces).reshape(-1, 2), initial_water



def verify_layered_transient(spec):
    result = dict(ok=False, conservation_verified=False, mode="civil_geotech",
                  tool="hamon_conservation_cad_verifier", calculation="layered_consolidation_transient")
    try:
        allowed = {"mode", "calculation", "hydraulic_layers", "top_boundary", "bottom_boundary", "time_s",
                   "delta_sigma_v_Pa", "rectangular_loads", "settlement_column", "proposed_settlement_m"}
        if spec.get("mode", "auto") not in ("civil_geotech", "civil", "geotech", "auto"):
            raise ValueError("Unrecognized geotechnical calculation mode")
        if spec.keys() - allowed:
            raise ValueError("Additional loads, compression laws or time histories are unsupported")
        layers = _properties(spec)
        boundaries = (spec["top_boundary"], spec["bottom_boundary"])
        if any(x not in ("drained", "impermeable") for x in boundaries):
            raise ValueError("Specify drained or impermeable at both column ends")
        drained = tuple(x == "drained" for x in boundaries)
        elapsed = _number(spec, "time_s")
        if elapsed < 0:
            raise ValueError("Elapsed time must be nonnegative")
        load, integral, peak = _loading(spec, layers)
        full = integral * layers[:, 2]
        if max(layers[:, 2]) * peak >= 1 or not all(np.isfinite(full)) or min(full) <= 0:
            raise ValueError("Excessive strain or unresolved initial loading")
        storage = layers[:, 2] * (layers[:, 1] - layers[:, 0])
        interfaces = []
        residual = 0.
        retained = None
        if elapsed == 0:
            retained = full
            interfaces = [{"depth_m": float(z), "excess_pressure_Pa": float(load(z)),
                           "downward_darcy_flux_m_s": None} for z in layers[:-1, 1]]
        else:
            coarse = None
            for count in (24, 48, 96, 192):
                if count * len(layers) > 576:
                    break
                cell, cell_faces, cell_initial = _grid(layers, load, drained, elapsed, count, False)
                node, node_faces, node_initial = _grid(layers, load, drained, elapsed, count, True)
                current = (cell, node, cell_faces, node_faces, cell_initial, node_initial)
                if coarse is not None:
                    # Extrapolate the two distinct spatial operators separately.
                    cell_ex, node_ex, fv_faces, fe_faces, fv_initial, fe_initial = [
                        fine + (fine-old)/3 for fine, old in zip(current, coarse)]
                    motion = math.fsum(full-cell_ex)
                    h = layers[:, 1] - layers[:, 0]
                    phase = [math.sqrt(layers[i, 3]*elapsed) * count / h[i]
                             for i, is_drained in ((0, drained[0]), (-1, drained[1])) if is_drained]
                    residual = max(max(abs(cell_ex-node_ex)/full), max(abs(fv_initial-full)/full),
                                   max(abs(fe_initial-full)/full))
                    if any(drained):
                        # Small settlements must be resolved relative to their own scale.
                        residual = max(residual, abs(math.fsum(cell_ex-node_ex))/max(abs(motion), np.finfo(float).tiny))
                    if len(fv_faces):
                        mobility = layers[:, 2] * layers[:, 3]
                        flux_scale = peak / (h[:-1]/mobility[:-1] + h[1:]/mobility[1:])
                        residual = max(residual, max(abs(fv_faces[:, 0]-fe_faces[:, 0]))/peak,
                                       max(abs(fv_faces[:, 1]-fe_faces[:, 1])/flux_scale))
                    if (not phase or min(phase) >= 1) and residual < 1e-6:
                        retained = cell_ex
                        interfaces = [{"depth_m": float(z), "excess_pressure_Pa": float(p),
                                       "downward_darcy_flux_m_s": float(q)}
                                      for z, (p, q) in zip(layers[:-1, 1], fv_faces)]
                        break
                coarse = current
            else:
                raise ValueError("Finite-volume and finite-element responses did not converge")
            if retained is None:
                raise ValueError("Refinement budget exhausted before resolving the response")
        settlement = math.fsum(full-retained)
        target = math.fsum(full)
        if not any(drained):
            if abs(settlement) > target*1e-6:
                raise ValueError("Closed-column total water balance failed")
            settlement = 0.
        if not 0 <= settlement <= target*(1+1e-6) or not all(np.isfinite(retained)):
            raise ValueError("Transient response outside the physical range")
        accepted = True
        if "proposed_settlement_m" in spec:
            proposed = _number(spec, "proposed_settlement_m")
            accepted = proposed == 0 if settlement == 0 else math.isclose(proposed, settlement, rel_tol=1e-6)
        result.update(ok=True, conservation_verified=bool(accepted), settlement_at_time_m=settlement,
                      fully_drained_reference_m=target, layers_mean_excess_pressure_Pa=(retained/storage).tolist(),
                      dual_solver_rel_residual=float(residual), interfaces=interfaces,
                      model="instantaneous vertical total-stress step, saturated linear storage and vertical Darcy flow")
    except (KeyError, TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
    return result
