"""Terzaghi primary consolidation for one homogeneous layer."""
import math
import sys



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
    """Signed influence of a corner rectangle (USACE EM 1110-1-1905 (31 July 2025), Fig. 6-8)."""
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
        # Influence is homogeneous of degree zero. Binary rescaling preserves
        # coordinate differences while keeping all Euclidean norms representable.
        exponent = math.frexp(max(abs(x0), abs(x1), abs(y0), abs(y1), z))[1]
        x0, x1, y0, y1, depth = (
            math.ldexp(value, -exponent) for value in (x0, x1, y0, y1, z)
        )
        if depth == 0 or not (x0 < x1 and y0 < y1):
            raise ValueError("Footprint and depth span unresolvable coordinate scales")
        influence = math.fsum((_corner(x1, y1, depth), -_corner(x0, y1, depth),
                               -_corner(x1, y0, depth), _corner(x0, y0, depth)))
        check = _area_integral(x0, x1, y0, y1, depth)
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



def _history_segments(history, elapsed, scalar=float):
    """Clip the right-continuous total load curve to the observation time."""
    if not isinstance(history, list) or not history:
        raise ValueError("Provide a nonempty load_history")
    rows = []
    for row in history:
        if not isinstance(row, dict) or set(row) != {"time_s", "surcharge_Pa"}:
            raise ValueError("Each load_history row requires time_s and surcharge_Pa")
        time, load = _number(row, "time_s"), _number(row, "surcharge_Pa")
        if min(time, load) < 0 or (rows and time < rows[-1][0]):
            raise ValueError("Require nonnegative surcharges in nondecreasing time order")
        if len(rows) >= 2 and time == rows[-2][0]:
            raise ValueError("At most two ordinates may describe a jump at one time")
        rows.append((scalar(time), scalar(load)))
    if rows[0][0] != 0:
        raise ValueError("Load history must begin at zero time")
    segments = []
    start, current = 0, 0
    for end, target in rows:
        if end > elapsed:
            if start < elapsed:
                fraction = (elapsed-start)/(end-start)
                partial = current + (target-current)*fraction
                segments.append((start, elapsed, current, partial))
            break
        segments.append((start, end, current, target))
        start, current = end, target
    else:
        if start < elapsed:
            segments.append((start, elapsed, current, current))
    return segments


def _early_interval_response(young, span):
    # U(T)=2 sqrt(T/pi) up to exponentially small reflected images. Integrate
    # each linear shape function without subtracting nearly equal roots.
    old = young+span
    a, b = math.sqrt(young), math.sqrt(old)
    root_span = span/(a+b)
    total = (2/math.sqrt(math.pi))*root_span
    old_weight = total*((a+root_span/3)/(a+b))
    new_weight = total*((b-root_span/3)/(a+b))
    mean_u = (4/(3*math.sqrt(math.pi)))*(old+a*b+young)/(a+b)
    return old_weight, new_weight, 1-mean_u


def _interval_response(young, span):
    """Return effective-stress weights of the old/new load ordinates and the mean pressure step response."""
    # Effective stress is the convolution of the actual nonnegative load with
    # U'(T). Combining positive load ordinates retains short surcharge pulses
    # even when their start and end have the same rounded age.
    cutoff = 1/144
    old = young+span
    if old == 0:
        return 0., 0., 1.
    if old <= cutoff:
        return _early_interval_response(young, span)
    if young < cutoff:
        recent = cutoff-young
        fraction = recent/span
        a, b, rp = _early_interval_response(young, recent)
        c, d, op = _interval_response(cutoff, old-cutoff)
        return c+fraction*(a+d), b+(1-fraction)*(a+d), fraction*rp+(1-fraction)*op
    older, newer, pressure = 0., 0., 0.
    for n in range(200):
        k = math.pi*(n+.5)
        rate = k*k
        coefficient = (2/rate)*math.exp(-rate*young)
        x = rate*span
        mean = -math.expm1(-x)/x if x else 1.
        if x < 1e-5:
            old_weight, new_weight = x*(.5-x/3), x*(.5-x/6)
        else:
            old_weight, new_weight = mean-math.exp(-x), 1-mean
        older += coefficient*old_weight
        newer += coefficient*new_weight
        pressure += coefficient*mean
        if coefficient < 1e-16:
            break
    return older, newer, pressure


def _resolved_history_pressure(history, elapsed, height, cv, drainage, scale, factors=()):
    """Resolve a signed response, applying component factors before rounding."""
    from mpmath import mp

    ctx = mp.clone()
    ctx.prec = 120
    # At this precision even an error proportional to the entire surcharge
    # is below the binary64 subnormal range.
    max_bits = max(2 * ctx.prec, math.frexp(scale)[1]
                   + sum(math.frexp(value)[1] for value in factors) + 1160)
    previous = None

    def retention(young, span):
        old = young + span
        if old == 0:
            return ctx.one
        # Reflected diffusion images are smaller than working precision here.
        cutoff = ctx.one / (ctx.prec + 24)
        if old <= cutoff:
            a, b = ctx.sqrt(young), ctx.sqrt(old)
            mean_u = 4 * (old + a*b + young) / (3 * ctx.sqrt(ctx.pi) * (a+b))
            return 1 - mean_u

        # At age zero a Fourier sum converges slowly. Instead integrate R
        # from zero using the identity integral_0^inf R = 1/3.
        integral = young < cutoff / 2
        age = old if integral else young
        total = ctx.zero
        n = 0
        while True:
            rate = (ctx.pi * (n + ctx.mpf("0.5")))**2
            term = 2 * ctx.exp(-rate * age) / rate
            if integral:
                term /= rate
            elif span:
                term *= -ctx.expm1(-rate * span) / (rate * span)
            total += term
            if term <= ctx.eps * total:
                break
            n += 1
        if integral:
            early = young - 4 * young * ctx.sqrt(young) / (3 * ctx.sqrt(ctx.pi))
            return (ctx.one/3 - total - early) / span
        return total

    while True:
        time = ctx.mpf(elapsed)
        path = ctx.mpf(height) / (2 if drainage == "double" else 1)
        factor = ctx.mpf(cv) / path**2
        terms = []
        variation = ctx.zero
        for start, end, first, last in _history_segments(history, time, ctx.mpf):
            change = last - first
            if change:
                terms.append(change * retention((time-end)*factor, (end-start)*factor))
                variation += abs(change)
        multiplier = ctx.fprod(ctx.mpf(value) for value in factors)
        pressure = ctx.fsum(terms) * multiplier
        variation *= abs(multiplier)
        rounded = float(pressure)
        if not math.isfinite(rounded):
            raise ValueError("Nonfinite signed unloading pressure")
        # Resolve the returned component, even when the pressure is much
        # smaller than the individual load changes. Allow for accumulation
        # and verify that precision refinement preserves the rounded value.
        uncertainty = (ctx.prec + 24)**2 * ctx.eps * variation
        if rounded == previous and uncertainty < ctx.mpf(math.ulp(rounded)) / 16:
            return rounded
        if ctx.prec >= max_bits + 64:
            raise ValueError("Signed unloading pressure did not converge")
        previous = rounded
        ctx.prec = min(2 * ctx.prec, max_bits if ctx.prec < max_bits else max_bits + 64)


def _diffusion_history(segments, factor, cells):
    """Integrate the forced finite-volume equations, including undrained load jumps."""
    import numpy as np
    from scipy.linalg import eigh_tridiagonal

    # Grading resolves the drained surface at early times. Face fluxes use
    # center-to-center distances, a zero surface pressure and a sealed base.
    faces = np.linspace(0., 1., cells+1)**3
    volume = np.diff(faces)
    center = (faces[:-1]+faces[1:])/2
    conductance = np.r_[1/center[0], 1/np.diff(center), 0.]
    diag = (conductance[:-1]+conductance[1:])/volume
    off = -conductance[1:-1]/np.sqrt(volume[:-1]*volume[1:])
    rates, modes = eigh_tridiagonal(diag, off, lapack_driver="stemr")
    weights = (np.sqrt(volume) @ modes)**2
    effective, pressure = np.zeros(cells), np.zeros(cells)
    for start, end, initial, final in segments:
        change = final-initial
        dt = (end-start)*factor
        if dt == 0:
            pressure = -effective if final == 0 else pressure+change
            continue
        with np.errstate(over="ignore"):
            x = rates*dt
        decay = np.exp(-x)
        dissipated = -np.expm1(-x)
        retention = np.divide(dissipated, x, out=np.ones_like(x), where=x != 0)
        ramp_settled = 1-retention
        small = x < 1e-5
        ramp_settled[small] = x[small]*(.5-x[small]/6)
        pressure = pressure*decay + change*retention
        old_load_weight = retention-decay
        old_load_weight[small] = x[small]*(.5-x[small]/3)
        effective = effective*decay + initial*old_load_weight + final*ramp_settled
        if final == 0:
            pressure = -effective
    return float(weights @ effective), float(weights @ pressure)


def verify_consolidation_history(spec):
    """Terzaghi diffusion with constant reversible mv and uniform nonnegative surcharge."""
    result = dict(ok=False, conservation_verified=False, mode="civil_geotech",
                  tool="hamon_conservation_cad_verifier", calculation="consolidation_history_1d")
    try:
        h, cv, mv, elapsed = (_number(spec, key) for key in
                              ("H_clay_m", "cv_m2_s", "mv_per_Pa", "time_s"))
        if min(h, cv, mv) <= 0 or elapsed < 0 or spec["drainage"] not in ("single", "double"):
            raise ValueError("Require positive H, Cv and mv, nonnegative time and single or double drainage")
        if {"Cc", "Cr", "e0", "sigma_preconsolidation_Pa", "delta_sigma_v_Pa",
            "rectangular_loads", "stress_point", "proposed_degree_consolidation"} & spec.keys():
            raise ValueError("Use a single load history and constant mv; staged loads have no single degree of consolidation")
        path = h if spec["drainage"] == "single" else h/2
        factor = (cv/path)/path
        if not 0 < factor < math.inf or not math.isfinite(elapsed*factor):
            raise ValueError("Time factor outside finite numerical range")
        segments = _history_segments(spec["load_history"], elapsed)
        scale = max(max(initial, final) for _, _, initial, final in segments)
        effective_terms, pressure_terms = [], []
        subnormal_retention = False
        for start, end, initial, final in segments:
            old, new, retained = _interval_response((elapsed-end)*factor, (end-start)*factor)
            effective_terms.append(initial*old + final*new)
            pressure_terms.append((final-initial)*retained)
            # A unit-load kernel can lose significand bits or round to zero
            # before its load multiplier produces a representable pressure.
            subnormal_retention |= final != initial and retained < sys.float_info.min
        effective, pressure = math.fsum(effective_terms), math.fsum(pressure_terms)
        resolved_settlement = None
        current = segments[-1][3]
        # Use the positive effective-stress integral to recover near-cancelling
        # negative pore-pressure tails after construction loads are removed.
        impulse_scale = math.fsum(map(abs, pressure_terms))
        difference_scale = abs(current) + abs(effective)
        if difference_scale < impulse_scale:
            pressure = current-effective
        cancellation = current and abs(pressure) < math.sqrt(math.ulp(1.0)) * min(impulse_scale, difference_scale)
        if subnormal_retention or cancellation:
            pressure = _resolved_history_pressure(spec["load_history"], elapsed, h, cv, spec["drainage"], scale)
            if current == 0:
                effective = -pressure
                # Settlement can remain representable even when the pressure
                # or strain rounds to zero. Scale the original response first.
                resolved_settlement = _resolved_history_pressure(
                    spec["load_history"], elapsed, h, cv, spec["drainage"], scale, (-mv, h))
        coarse = _diffusion_history(segments, factor, 96)
        fine = _diffusion_history(segments, factor, 192)
        fv_eff, fv_pressure = (b+(b-a)/3 for a,b in zip(coarse, fine))
        strain = mv*effective
        if not 0 <= strain < 1:
            raise ValueError("Predicted compression exhausts or exceeds the layer thickness")
        settlement = strain*h if resolved_settlement is None else resolved_settlement
        if not all(math.isfinite(x) for x in (effective, pressure, settlement, fv_eff, fv_pressure)):
            raise ValueError("Nonfinite consolidation history response")
        residual = max(abs(effective-fv_eff), abs(pressure-fv_pressure), abs(effective+pressure-current))
        valid = residual <= 2e-6*scale
        if "proposed_settlement_m" in spec:
            proposed = _number(spec, "proposed_settlement_m")
            valid = (valid and 0 <= proposed < h and (settlement != 0 or proposed == 0)
                     and math.isclose(proposed, settlement, rel_tol=1e-6, abs_tol=1e-12))
        if "proposed_mean_excess_pore_pressure_Pa" in spec:
            proposed = _number(spec, "proposed_mean_excess_pore_pressure_Pa")
            same_direction = (proposed > 0) == (pressure > 0) and (proposed < 0) == (pressure < 0)
            valid = valid and same_direction and math.isclose(proposed, pressure, rel_tol=1e-6, abs_tol=1e-6)
        result.update(ok=True, conservation_verified=valid, drainage_path_m=path,
                      surcharge_at_time_Pa=current, mean_effective_increment_Pa=effective,
                      mean_excess_pore_pressure_Pa=pressure, settlement_at_time_m=settlement,
                      finite_volume_mean_effective_increment_Pa=fv_eff,
                      finite_volume_mean_excess_pore_pressure_Pa=fv_pressure,
                      dual_solver_abs_residual_Pa=residual,
                      dual_solver_rel_residual=residual / scale if scale > 0 else 0.0)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result


def verify_consolidation_1d(spec):
    """Check a settlement-time calculation in SI units, including compression history."""
    result = {"ok": False, "conservation_verified": False, "mode": "civil_geotech",
              "tool": "hamon_conservation_cad_verifier", "calculation": "consolidation_1d"}
    try:
        if "load_history" in spec or "mv_per_Pa" in spec:
            raise ValueError("Use calculation=consolidation_history_1d for staged loads with constant mv")
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
