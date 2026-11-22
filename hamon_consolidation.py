"""Terzaghi primary consolidation for one layer and a uniform step in effective load."""
import math



def _number(spec, key):
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _average_fourier(tv):
    if tv < 1.0 / 144:
        # The first reflected image is exponentially suppressed by exp(-1/Tv).
        return 2.0 * math.sqrt(tv / math.pi)
    remainder = 0.0
    for n in range(200):
        k = math.pi * (n + 0.5)
        term = 2.0 * math.exp(-(k*k)*tv) / (k*k)
        remainder += term
        if term < 1e-16:
            break
    return 1.0 - remainder


def _average_cn(tv, cells):
    import numpy as np
    from scipy.linalg import solve_banded

    if tv == 0.0:
        return 0.0
    # Work on one drainage path: zero excess pressure at x=0 and zero flux
    # at x=1. At early times 12 diffusion lengths suffice to double precision.
    if tv < 1.0 / 144:
        length, end_time = 12.0 * math.sqrt(tv), 1.0 / 144
    else:
        length, end_time = 1.0, min(tv, 20.0)
    # Beyond Tv=20 the surviving lowest diffusion mode is below 4e-22.
    steps = cells
    r = end_time * cells**2 / (2.0 * steps)
    diagonal = np.full(cells, -2.0)
    diagonal[0], diagonal[-1] = -3.0, -1.0
    implicit = np.zeros((3, cells))
    implicit[0, 1:] = implicit[2, :-1] = -r
    implicit[1] = 1.0 - r * diagonal
    state = np.ones(cells)
    # Two half steps of backward Euler damp the initially discontinuous
    # drainage boundary before the centered Crank–Nicolson time steps.
    for _ in range(2):
        state = solve_banded((1, 1), implicit, state, check_finite=False)
    for _ in range(1, steps):
        rhs = (1.0 + r * diagonal) * state
        rhs[1:] += r * state[:-1]
        rhs[:-1] += r * state[1:]
        state = solve_banded((1, 1), implicit, rhs, check_finite=False)
    return length * float(np.mean(1.0 - state))


def primary_settlement(h, e0, cc, initial, load, pc, cr):
    """One-dimensional compression with fixed solid volume and nonnegative void space."""
    if not all(math.isfinite(x) for x in (h, e0, cc, initial, load, pc, cr)):
        raise ValueError("Layer properties and effective stresses must be finite")
    if h <= 0 or initial <= 0 or cc < 0 or min(e0, load) < 0:
        raise ValueError("Invalid layer, void ratio, compression index, or stress increment")
    if pc < initial or not 0 <= cr <= cc:
        raise ValueError("Require preconsolidation stress >= initial effective stress and 0 <= Cr <= Cc")
    if cc == 0:
        return settlement_from_void_loss(h, e0, 0.0)
    recompression = min(load, pc - initial)
    virgin = load - recompression
    loss = (cr * math.log1p(recompression / initial)
            + cc * math.log1p(virgin / (initial + recompression))) / math.log(10.0)
    return settlement_from_void_loss(h, e0, loss)


def settlement_from_void_loss(h, e0, loss):
    """Convert a signed change in void ratio to settlement at constant solid volume."""
    if not math.isfinite(h) or not math.isfinite(e0) or h <= 0 or e0 < 0:
        raise ValueError("Require finite positive layer height and nonnegative void ratio")
    # e is the ratio of void to solid volume. No compression history can
    # consume more than the initial voids (USACE EM 1110-2-1906, App. VIII).
    if not math.isfinite(loss):
        raise ValueError("Compression curve exceeds finite numerical range")
    if loss > e0 and not math.isclose(loss, e0, rel_tol=8 * math.ulp(1.0)):
        raise ValueError("Compression curve predicts a negative final void ratio")
    loss = min(loss, e0)
    return {"primary_consolidation_settlement_m": h * (loss / (1.0 + e0)),
            "initial_void_height_m": h * (e0 / (1.0 + e0)),
            "final_void_ratio": e0 - loss}


def _corner(x, y, z):
    """Signed influence of a corner rectangle (USACE EM 1110-1-1905, Fig. 6-8)."""
    r = math.hypot(x, y, z)
    rx, ry = math.hypot(x, z), math.hypot(y, z)
    angle = math.atan2((x / r) * y, z)
    correction = ((x / rx) * (z / rx) * (y / r)
                  + (y / ry) * (z / ry) * (x / r))
    return (angle + correction) / (2 * math.pi)


def _area_integral(x0, x1, y0, y1, z):
    """Independently integrate the point-load kernel over the actual footprint."""
    from scipy.integrate import quad

    # Far from a small footprint, taking differences of near-identical
    # angles loses significant digits. Integrate on a unit square there.
    width, length = x1 - x0, y1 - y0
    nearest_x = max(x0, -x1, 0.0)
    nearest_y = max(y0, -y1, 0.0)
    if math.hypot(nearest_x, nearest_y, z) >= max(width, length):
        def row(u):
            x = x0 + u * width

            def kernel(v):
                r = math.hypot(x, y0 + v * length, z)
                return (width / r) * (length / r) * (z / r)**3

            return quad(kernel, 0, 1)[0]

        return 3 * quad(row, 0, 1)[0] / (2 * math.pi)

    # x = z*tan(theta), y = hypot(z,x)*tan(phi). This bounds both
    # integration intervals even for a shallow point inside a wide footprint.
    def strip(theta):
        c = math.cos(theta)
        phi0, phi1 = math.atan2(y0 * c, z), math.atan2(y1 * c, z)
        inner, _ = quad(lambda phi: math.cos(phi)**3, phi0, phi1, epsabs=1e-13)
        return c*c * inner

    area, _ = quad(strip, math.atan2(x0, z), math.atan2(x1, z), epsabs=1e-13)
    return 3 * area / (2 * math.pi)


def rectangular_load_stress(spec, claim_key):
    """Derive the increase in vertical stress from uniformly distributed net pressures."""
    point = spec["stress_point"]
    loads = spec["rectangular_loads"]
    if not isinstance(point, dict) or not isinstance(loads, list) or not loads:
        raise ValueError("Provide stress_point and a nonempty rectangular_loads list")
    x, y, z = (_number(point, k) for k in ("x_m", "y_m", "depth_m"))
    if z <= 0:
        raise ValueError("Stress evaluation depth below the common loading plane must be positive")
    closed, integrated = [], []
    for load in loads:
        if not isinstance(load, dict):
            raise ValueError("Each rectangular load must be an object")
        cx, cy, width, length, pressure = (
            _number(load, k) for k in ("center_x_m", "center_y_m", "width_m", "length_m", "net_pressure_Pa")
        )
        if width <= 0 or length <= 0 or pressure < 0:
            raise ValueError("Require positive rectangle dimensions and nonnegative net pressure")
        x0, x1 = (cx - x) - width / 2, (cx - x) + width / 2
        y0, y1 = (cy - y) - length / 2, (cy - y) + length / 2
        if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or not (x0 < x1 and y0 < y1):
            raise ValueError("Footprint offsets exceed finite coordinate resolution")
        influence = math.fsum((_corner(x1, y1, z), -_corner(x0, y1, z),
                               -_corner(x1, y0, z), _corner(x0, y0, z)))
        check = _area_integral(x0, x1, y0, y1, z)
        if not 0 <= check <= 1 + 1e-12 or not math.isclose(influence, check, abs_tol=2e-12, rel_tol=1e-8):
            raise ValueError("Boussinesq area integral disagrees with the corner influence solution")
        # The corner subtraction can lose significant digits far outside the
        # footprint. The positive point-load integral remains the load estimate.
        closed.append(pressure * influence)
        integrated.append(pressure * check)
    value = math.fsum(integrated)
    closed_value = math.fsum(closed)
    if not math.isfinite(value) or not math.isfinite(closed_value):
        raise ValueError("Superposed stress exceeds finite numerical range")
    matched = claim_key not in spec or math.isclose(
        _number(spec, claim_key), value, rel_tol=1e-6, abs_tol=1e-6
    )
    return value, {"stress_model": "Boussinesq homogeneous isotropic elastic half-space",
                   "derived_delta_sigma_v_Pa": value, "corner_delta_sigma_v_Pa": closed_value,
                   "declared_stress_consistent": matched}



def consolidation_load(spec, key="delta_sigma_v_Pa", *, coerce_scalar=False):
    """Resolve geometry, or preserve the invoking verifier's scalar input contract."""
    if "rectangular_loads" in spec or "stress_point" in spec:
        return rectangular_load_stress(spec, key)
    return (float(spec[key]) if coerce_scalar else _number(spec, key)), {}


def verify_consolidation_1d(spec):
    """Check a settlement-time calculation in SI units, including compression history."""
    result = {"ok": False, "conservation_verified": False, "mode": "civil_geotech",
              "tool": "hamon_conservation_cad_verifier", "calculation": "consolidation_1d"}
    try:
        h = _number(spec, "H_clay_m")
        e0 = _number(spec, "e0")
        cc = _number(spec, "Cc")
        initial = _number(spec, "sigma_v0_eff_Pa")
        load, stress_check = consolidation_load(spec)
        cv = _number(spec, "cv_m2_s")
        elapsed = _number(spec, "time_s")
        drainage = spec["drainage"]
        if cv <= 0 or elapsed < 0 or cc <= 0:
            raise ValueError("Require positive Cv and Cc and nonnegative elapsed time")
        if drainage not in ("single", "double"):
            raise ValueError("drainage must be single or double")
        pc = _number(spec, "sigma_preconsolidation_Pa") if "sigma_preconsolidation_Pa" in spec else initial
        cr = _number(spec, "Cr") if "Cr" in spec or pc > initial else 0.0
        compression = primary_settlement(h, e0, cc, initial, load, pc, cr)
        path = h if drainage == "single" else h / 2.0
        tv = (cv / path) * (elapsed / path)
        if not math.isfinite(tv):
            raise ValueError("Time factor exceeds finite numerical range")
        degree = _average_fourier(tv)
        coarse = _average_cn(tv, 128)
        fine = _average_cn(tv, 256)
        cn = fine + (fine - coarse) / 3.0
        ultimate = compression["primary_consolidation_settlement_m"]
        settlement = degree * ultimate
        if not all(math.isfinite(v) for v in (degree, cn, ultimate, settlement)):
            raise ValueError("Nonfinite consolidation response")
        residual = abs(degree - cn)
        valid = 0 <= degree <= 1 and abs(max(0.0, min(1.0, cn)) - cn) < 1e-6 and residual < 1e-6
        for key, predicted, upper in (("proposed_degree_consolidation", degree, 1.0),
                                      ("proposed_settlement_m", settlement, compression["initial_void_height_m"])):
            if key in spec:
                proposed = _number(spec, key)
                exceeds_upper = proposed > upper and not math.isclose(proposed, upper, rel_tol=8 * math.ulp(1.0))
                if proposed < 0 or exceeds_upper or abs(proposed - predicted) > 1e-6:
                    valid = False
        result.update(ok=True, conservation_verified=valid and stress_check.get("declared_stress_consistent", True),
                      **stress_check, drainage_path_m=path, time_factor=tv,
                      degree_consolidation=degree, degree_consolidation_cn=cn,
                      dual_solver_abs_residual=residual, **compression,
                      settlement_at_time_m=settlement)
        return result
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
        return result
