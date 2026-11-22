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


def verify_consolidation_1d(spec):
    """Check a settlement-time calculation in SI units, including compression history."""
    result = {"ok": False, "conservation_verified": False, "mode": "civil_geotech",
              "tool": "hamon_conservation_cad_verifier", "calculation": "consolidation_1d"}
    try:
        h = _number(spec, "H_clay_m")
        e0 = _number(spec, "e0")
        cc = _number(spec, "Cc")
        initial = _number(spec, "sigma_v0_eff_Pa")
        load = _number(spec, "delta_sigma_v_Pa")
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
        result.update(ok=True, conservation_verified=valid, drainage_path_m=path, time_factor=tv,
                      degree_consolidation=degree, degree_consolidation_cn=cn,
                      dual_solver_abs_residual=residual, **compression,
                      settlement_at_time_m=settlement)
        return result
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
        return result
