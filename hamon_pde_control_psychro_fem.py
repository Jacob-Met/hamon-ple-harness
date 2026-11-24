"""Wave-5 Deterministic Engineering Verifiers: PDE, Control, Psychrometrics & 1D/2D FEM.

Self-contained zero-external-dependency (Python stdlib math/cmath) symbolic and
numerical cross-verification suite for Physical Engineering's Last Exam (PLE):
  1. PDE (`verify_pde_system`):
     - 1D convective extended-surface fin (adiabatic, convective, prescribed-tip, infinite)
       exact boundary solution cross-checked with uniform finite-volume heat balances.
     - 1D transient heat diffusion exact Fourier eigenmode series cross-checked against
       second-order unconditionally stable Crank-Nicolson solver.
     - 2D steady Laplace equation on rectangular domain exact harmonic eigenseries
       cross-checked against Gauss-Seidel SOR finite-difference solver.
  2. Control (`verify_control_system`):
     - Continuous-time LTI transfer-function half-plane pole counts from exact shifted
       Hurwitz minors, separate numerical pole approximations,
       exact Bode gain/phase crossover frequencies, Gain Margin (dB), Phase Margin (deg),
       exact Nyquist winding for regular strictly proper loops, unity-feedback closed-loop
       stability, and static DC gain.
     - State-space exact Faddeev-LeVerrier characteristic polynomial, Cayley-Hamilton
       residual, numerical pole diagnostics and algebraic controllability/observability ranks of the supplied
       finite binary64 coefficients. Continuous Lyapunov or discrete Stein equations
       certify stability (Q=I) and solve for user-specified Q. Discrete time also
       checks the strict Schur criterion on the exact characteristic polynomial.
  3. Psychrometrics (`verify_psychrometric_process`):
     - ASHRAE Fundamentals (Hyland-Wexler 1983) saturation vapor pressure over ice & liquid,
       moist-air state inversion (from RH, wet-bulb, dew-point, or humidity ratio),
       2-stream adiabatic moist-air mixing conservation, and cooling/dehumidification coil
       mass & energy balance (condensate rate, total kW load, sensible heat ratio SHR).
     - Cooling coils support load_split="ashrae_dew_point" for latent heat at the inlet
       dew point; liquid drain subcooling then contributes to sensible heat.
     - Adiabatic mixing accepts condensate_model="equilibrium_ice" to resolve ice down
       to -60 C and coexisting ice and liquid at 0 C. The default is liquid condensate.
  4. Structural FEM (`verify_fem_structure`):
     - Isotropic CST plane-stress and plane-strain meshes with exact equilibrium and work checks.
     - Arbitrary 2D pin-jointed determinate & statically indeterminate truss direct-stiffness
       solver with thermal strain support, kinematic mechanism detection, nodal reaction
       equilibrium, and Clapeyron work-strain-energy theorem verification.
     - 1D Euler-Bernoulli beam with exact span compliance, moment/shear equilibrium,
       affine distributed loading and prescribed support motions.
       Each element accepts a uniform q_N_m or a linear profile given by q_start_N_m and q_end_N_m.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import re
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0]) if b else 0
    k_len = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        out_i = out[i]
        for k in range(k_len):
            aik = row_a[k]
            if aik == 0.0:
                continue
            row_b = b[k]
            for j in range(m):
                out_i[j] += aik * row_b[j]
    return out


def _primitive_integer_vector(values: Sequence[int]) -> List[int]:
    """Remove only a nonzero scalar, preserving the represented direction."""
    divisor = math.gcd(*values)
    if not divisor:
        return list(values)
    if next(value for value in values if value) < 0:
        divisor = -divisor
    return [value // divisor for value in values]


def _algebraic_reachable_rank(a: Sequence[Sequence[float]],
                              ports: Sequence[Sequence[float]]) -> int:
    """Exact reachable dimension for the finite binary64 values supplied.

    A common dyadic denominator turns A into an integer matrix M = D*A.
    Scaling A or individual input columns by nonzero scalars leaves the
    reachable span unchanged. A primitive integer basis is closed under M;
    every independent residual is queued once. At termination the span
    contains the ports, is invariant under A, and consists of reachable
    vectors, so it equals the Kalman span. No float Krylov products or
    numerical rank threshold enter this calculation.

    At most n basis vectors are retained. Integer bit lengths depend on the
    supplied coefficients; this is algebraic rank, not a conditioning test.
    """
    n = len(a)
    ratios = [[value.as_integer_ratio() for value in row] for row in a]
    denominator = max(d for row in ratios for _, d in row)
    operator = [[num * (denominator // den) for num, den in row] for row in ratios]
    common = math.gcd(*(value for row in operator for value in row))
    if common > 1:
        operator = [[value // common for value in row] for row in operator]
    basis: List[Tuple[int, List[int]]] = []
    pending: List[List[int]] = []

    def insert(candidate: List[int]) -> None:
        candidate = _primitive_integer_vector(candidate)
        for pivot, known in basis:
            coefficient = candidate[pivot]
            if coefficient:
                divisor = math.gcd(known[pivot], coefficient)
                left = known[pivot] // divisor
                right = coefficient // divisor
                candidate = _primitive_integer_vector(
                    [left * x - right * y for x, y in zip(candidate, known)]
                )
        if not any(candidate):
            return
        pivot = next(i for i, value in enumerate(candidate) if value)
        basis.append((pivot, candidate))
        basis.sort(key=lambda entry: entry[0])
        pending.append(candidate)

    for column in zip(*ports):
        ratios_column = [value.as_integer_ratio() for value in column]
        denominator_column = max(d for _, d in ratios_column)
        insert([num * (denominator_column // den) for num, den in ratios_column])
        if len(basis) == n:
            return n
    index = 0
    while index < len(pending):
        vector = pending[index]
        index += 1
        insert([sum(coefficient * value for coefficient, value in zip(row, vector))
                for row in operator])
        if len(basis) == n:
            return n
    return len(basis)


def _thomas_solve(lower: Sequence[float], diag: Sequence[float], upper: Sequence[float], rhs: Sequence[float]) -> List[float]:
    n = len(diag)
    c_p = [0.0] * n
    d_p = [0.0] * n
    c_p[0] = upper[0] / diag[0] if n > 1 else 0.0
    d_p[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_p[i - 1]
        c_p[i] = (upper[i] / denom) if i < n - 1 else 0.0
        d_p[i] = (rhs[i] - lower[i] * d_p[i - 1]) / denom
    x = [0.0] * n
    x[n - 1] = d_p[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_p[i] - c_p[i] * x[i + 1]
    return x


def _claim_measure(value):
    """Render a nonnegative exact error or tolerance without losing its range."""
    try:
        return _diagnostic_float(value), "finite"
    except OverflowError:
        return None, "below_float_range" if value < 1 else "above_float_range"


def _check_claimed(result: Dict[str, Any], spec: Dict[str, Any], *, exact_booleans=False) -> None:
    if "claimed_field" not in spec and "claimed_value" not in spec:
        return
    field = spec.get("claimed_field")
    if not isinstance(field, str) or not field.strip() or "claimed_value" not in spec:
        result.update(passed=False, reason="A scalar claim requires a nonempty claimed_field and a claimed_value")
        return
    if field not in result or not isinstance(result[field], (int, float)):
        result.update(passed=False, reason=f"Claimed field {field} is unknown or has no numeric scalar value")
        return
    try:
        claimed = float(spec["claimed_value"])
        actual = float(result[field])
        rtol = float(spec.get("rtol", 1e-2))
        atol = float(spec.get("atol", 1e-4))
    except (TypeError, ValueError, OverflowError):
        result.update(passed=False, reason=f"Claimed {field}, its actual value and tolerances must be finite numbers")
        return
    if not all(math.isfinite(v) for v in (actual, claimed, rtol, atol)) or rtol < 0 or atol < 0:
        result.update(passed=False, reason=f"Claimed {field} requires finite values and finite nonnegative tolerances")
        return
    if exact_booleans and isinstance(result[field], bool):
        value = spec["claimed_value"]
        if type(value) not in (bool, int, float) or value not in (0, 1):
            result.update(passed=False, reason=f"Claimed {field} must be true, false, 1 or 0")
            return
        atol = rtol = 0.0

    actual_exact = Fraction(actual)
    error = abs(actual_exact - Fraction(claimed))
    limit = max(Fraction(atol), Fraction(rtol) * max(abs(actual_exact), Fraction(1e-12)))
    error_display, error_status = _claim_measure(error)
    limit_display, limit_status = _claim_measure(limit)
    within = error <= limit
    result["claim_check"] = {
        "field": field, "claimed": claimed, "actual": actual,
        "abs_error": error_display, "abs_error_exact": _fraction_text(error),
        "abs_error_float_status": error_status,
        "tolerance_limit": limit_display, "tolerance_limit_exact": _fraction_text(limit),
        "tolerance_limit_float_status": limit_status,
        "within_tolerance": within,
    }
    if not within:
        result.update(passed=False, reason=f"Claimed {field} exceeds its tolerance in the exact comparison")


def _heat_sine_phase(numerator, denominator):
    """Evaluate sin(pi*numerator/denominator) after exact quadrant reduction."""
    phase = numerator % (2 * denominator)
    if phase == 0 or phase == denominator:
        return 0.0
    sign = -1.0 if phase > denominator else 1.0
    half_cycle = phase % denominator
    reduced = min(half_cycle, denominator - half_cycle)
    if 2 * reduced == denominator:
        return sign
    return sign * math.sin(math.pi * (reduced / denominator))


def _heat_spatial_factor(numerator, denominator):
    """Keep a subnormal nonzero reduced phase available for amplitude scaling."""
    value = _heat_sine_phase(numerator, denominator)
    # For a subnormal reduced coordinate r, |sin(pi*r)| <= pi*r < 4*min.
    if abs(value) >= 4 * sys.float_info.min:
        return value
    phase = numerator % (2 * denominator)
    half_cycle = phase % denominator
    reduced = min(half_cycle, denominator - half_cycle)
    if reduced == 0 or reduced / denominator >= sys.float_info.min:
        return value
    # Relative to pi*r the small-angle error is bounded by (pi*r)^2/6.
    signed = -reduced if phase > denominator else reduced
    return Fraction(math.pi) * Fraction(signed, denominator)


def _heat_spatial_product(amplitude, decay, factor):
    if isinstance(factor, Fraction):
        return float(Fraction(amplitude) * Fraction(decay) * factor)
    return amplitude * decay * factor


def _heat_positive_normal(value):
    return sys.float_info.min <= value <= sys.float_info.max


def _heat_ordinary_time(alpha, length, time):
    if time == 0:
        return 0.0
    product = alpha * time
    try:
        square = length ** 2
    except OverflowError:
        return None
    if _heat_positive_normal(product) and _heat_positive_normal(square):
        scaled_time = product / square
        if _heat_positive_normal(scaled_time):
            return scaled_time
    return None


def _heat_continuous_decay(n, scaled_time, represented_exponent):
    if n == 0 or scaled_time == 0:
        return 1.0
    if scaled_time is not None:
        try:
            square = (n * math.pi) ** 2
            exponent = scaled_time * square
            if _heat_positive_normal(square) and _heat_positive_normal(exponent):
                return math.exp(-exponent)
        except OverflowError:
            pass
    try:
        exponent = float(represented_exponent(n))
    except OverflowError:
        return 0.0
    return math.exp(-exponent)


def _heat_grid_rate(n, cells, scaled_time, represented_exponent):
    if n == 0:
        return 0.0, 0.0
    if scaled_time is not None:
        try:
            angle = math.pi * n / cells
            if _heat_positive_normal(abs(angle)):
                if scaled_time == 0:
                    return angle, 0.0
                square = (2 * cells * math.sin(angle / 2)) ** 2
                rate = scaled_time * square
                if _heat_positive_normal(square) and _heat_positive_normal(rate):
                    return angle, rate
        except OverflowError:
            pass
    angle = math.pi * (n / cells)
    if scaled_time == 0:
        return angle, 0.0
    # K = S*pi^2*sinc(y)^2, y=|n|*pi/(2C). Below this threshold the
    # sinc correction is at most y²/6, less than half a spacing below one.
    y = abs(angle) / 2
    sinc = 1.0 if y <= 2.0 ** -27 else math.sin(y) / y
    rate = float(represented_exponent(n) * Fraction(sinc) ** 2)
    return angle, rate


def _heat_cn_decay(rate, steps):
    if rate == 0:
        return 1.0
    try:
        ratio = rate / (2 * steps)
        if _heat_positive_normal(ratio):
            return math.exp(steps * (math.log1p(-ratio) - math.log1p(ratio)))
    except OverflowError:
        # The integer 2*steps can exceed binary64 even when rate is finite.
        ratio = float(Fraction(rate) / (2 * steps))
    # The same log amplification is -rate*atanh(ratio)/ratio. Its relative
    # correction to -rate is at most ratio²/(3*(1-ratio²)).
    correction = 1.0 if ratio <= 2.0 ** -27 else math.atanh(ratio) / ratio
    return math.exp(-rate * correction)


def _transient_heat_modes(alpha, length, time, position, modes):
    """Diagonalize the finite-difference CN step on the rod's Dirichlet sine basis."""
    harmonics = []
    for mode in modes:
        raw_mode = mode["n"]
        n = int(raw_mode)
        amplitude = float(mode["amplitude"])
        if isinstance(raw_mode, bool) or float(raw_mode) != n or not math.isfinite(amplitude):
            return {"passed": False, "reason": "Heat modes require integer harmonics and finite amplitudes"}
        harmonics.append((n, amplitude))
    tolerance = max(1e-4, 3e-3 * max(abs(amplitude) for _, amplitude in harmonics))
    cells = max(120, 2 * max(abs(n) for n, _ in harmonics))
    minimum_steps = 120
    scaled_time = _heat_ordinary_time(alpha, length, time)
    exponents = {}

    def represented_exponent(n):
        # Keep the dimensional products and n² together for exceptional modes.
        key = abs(n)
        if key not in exponents:
            scale = Fraction(alpha) * Fraction(time) * key ** 2 / Fraction(length) ** 2
            exponents[key] = scale * Fraction(math.pi) ** 2
        return exponents[key]

    exact_decay = [_heat_continuous_decay(n, scaled_time, represented_exponent) for n, _ in harmonics]
    for _ in range(16):
        # These are the eigenvalues of the centered second difference, not those
        # of the continuous Laplacian. Each CN time step applies its Cayley factor.
        angles, rates = zip(*(_heat_grid_rate(n, cells, scaled_time, represented_exponent)
                              for n, _ in harmonics))
        steps = max(minimum_steps, math.ceil(max(rates)))
        numerical_decay = [_heat_cn_decay(rate, steps) for rate in rates]
        # Linear interpolation on a cell of width dx incurs at most
        # dx² max|u''|/8. Add the modal damping differences, even at nodes
        # where different harmonics cancel. The sum bounds the whole rod.
        error_bound = math.fsum(
            abs(amplitude) * (abs(numeric - exact) + exact * angle ** 2 / 8)
            for (_, amplitude), numeric, exact, angle in
            zip(harmonics, numerical_decay, exact_decay, angles)
        )
        if error_bound <= tolerance:
            break
        cells *= 2
        minimum_steps *= 2
    else:
        return {"passed": False, "reason": "Could not certify the requested heat modes"}

    if position == 0 or position == length:
        u_numeric = u_exact = 0.0
    else:
        coordinate = Fraction(position) / Fraction(length)
        p, q = coordinate.numerator, coordinate.denominator
        left, remainder = divmod(p * cells, q)
        left_weight = (q - remainder) / q
        right_weight = remainder / q

        def interpolated_mode(n):
            left_value = _heat_spatial_factor(n * left, cells)
            if remainder == 0:
                return left_value
            right_value = _heat_spatial_factor(n * (left + 1), cells)
            normal = sys.float_info.min
            if (isinstance(left_value, float) and isinstance(right_value, float)
                    and min(left_weight, right_weight) >= normal):
                left_term = left_weight * left_value
                right_term = right_weight * right_value
                sample = left_term + right_term
                if ((left_value == 0 or abs(left_term) >= normal)
                        and (right_value == 0 or abs(right_term) >= normal)
                        and (sample == 0 or abs(sample) >= normal)):
                    return sample
            # A tiny weight or unit-amplitude sample can still contribute a
            # normal temperature. Carry both signed terms to the modal product.
            return ((q - remainder) * Fraction(left_value)
                    + remainder * Fraction(right_value)) / q

        u_numeric = math.fsum(_heat_spatial_product(amplitude, decay, interpolated_mode(n))
                              for (n, amplitude), decay in zip(harmonics, numerical_decay))
        u_exact = math.fsum(_heat_spatial_product(amplitude, decay, _heat_spatial_factor(n * p, q))
                            for (n, amplitude), decay in zip(harmonics, exact_decay))
    query_error = abs(u_numeric - u_exact)
    return {
        "passed": max(error_bound, query_error) <= tolerance,
        "u_exact": u_exact,
        "u_crank_nicolson": u_numeric,
        "linf_error": max(error_bound, query_error),
        "error_measure": "whole-domain upper bound",
        "query_abs_error": query_error,
        "spatial_cells": cells,
        "time_steps": steps if time else 0,
    }


def _passive_fin_fvm(length, m_len, axial, lateral, tip_loss, base):
    """Condense the finite-volume heat balances of identical uniform fin cells."""
    # At least 512 cells and at least 32 per decay length keep the leading
    # base-flux error (m*dx)^2/8 below the 1e-3 verification tolerance.
    doublings = max(9, math.frexp(m_len)[1] + 5)
    dx = math.ldexp(length, -doublings)
    coupling = axial / dx
    shunt = lateral * dx / 2.0
    # For either end i, Q_i = shunt*T_i + coupling*(T_i - T_j).
    # Eliminate the middle temperature of two identical sections. Keeping
    # the shunt separately avoids subtracting near-equal terms for short fins.
    for _ in range(doublings):
        fraction = coupling / (coupling + shunt)
        coupling *= 0.5 * fraction
        shunt *= 1.0 + fraction
    # Balance the last node against its tip-to-ambient thermal conductance.
    total = shunt + coupling + tip_loss
    tip = base * (coupling / total)
    flux = base * (shunt + coupling * ((shunt + tip_loss) / total))
    return flux, tip


def _passive_fin_ratio(base_excess, conductance, reference):
    if base_excess == 0.0:
        return None, "undefined_zero_base_drive"
    ratio = conductance / reference
    try:
        return _diagnostic_float(ratio), "finite"
    except OverflowError:
        return None, "below_float_range" if ratio < 1 else "above_float_range"


def _constant_top_rectangle(width, height, x, y, top):
    """Harmonic measure of the hot edge, checked by a corner-lifted five-point SOR."""
    # At the two incompatible corner points retain the sidewall convention.
    if x == 0.0 or x == width or y == 0.0:
        return 0.0, 0.0
    if y == height:
        return top, top
    near = min(x, width - x)
    depth = height - y

    if height >= width:
        # Odd sine modes of a constant edge summed on a semi-infinite strip.
        # Alternating Dirichlet images at the bottom recover the finite strip.
        sine = math.sin(math.pi * (near / width))
        def strip(z):
            p = math.pi * z
            return (2.0 / math.pi) * math.atan2(2.0 * math.exp(-p) * sine, -math.expm1(-2.0 * p))

        aspect, d = height / width, depth / width
        if y <= height * 0.5:
            # Here the odd-mode series decays by at least exp(-pi/2)
            # per mode. Retain y itself even if height-y rounds to height.
            exact = 0.0
            for n in range(1, 33, 2):
                k = n * math.pi
                ratio = math.exp(-k * d) * -math.expm1(-2 * k * (y / width)) / -math.expm1(-2 * k * aspect)
                exact += 4.0 * math.sin(k * (near / width)) * ratio / k
        else:
            exact = strip(d) - strip(aspect + y / width)
            for k in range(1, 8):
                exact += strip(2 * k * aspect + d) - strip(2 * k * aspect + aspect + y / width)
    else:
        # Subtract the two cold-side corrections from the linear y/height
        # profile. This orientation converges even on an arbitrarily wide plate.
        z, d = y / height, depth / height
        left, aspect = near / height, width / height
        p = math.pi * left
        r = math.exp(-p)
        # Write z minus the nearest-side angle as one angle; group the
        # remaining images in opposite-sign pairs that vanish at x=0.
        exact = (2.0 / math.pi) * math.atan2(
            -math.expm1(-p) * math.sin(0.5 * math.pi * z),
            (1.0 + r) * math.sin(0.5 * math.pi * d),
        )
        sine, cosine = math.sin(math.pi * min(z, d)), math.cos(math.pi * z)
        for k in range(1, 16):
            # (width-near)/height stays well defined when aspect and left overflow.
            a = (width - near) / height if k == 1 else (k - 1) * aspect + (width - near) / height
            ra = math.exp(-math.pi * a)
            rb = ra * math.exp(-2.0 * p)
            pair = (2.0 / math.pi) * math.atan2(
                ra * -math.expm1(-2.0 * p) * sine, 1.0 + (ra + rb) * cosine + ra * rb)
            exact += pair if k % 2 == 0 else -pair

    # In short-side units, boundary influence beyond 16 widths is O(exp(-16*pi)).
    # Keep the nearest hot corners and cold wall. On a wide plate fold the
    # query by reflection; beyond the cutoff both side corrections have decayed.
    scale = min(width, height)
    w, h = min(width / scale, 32.0), min(height / scale, 16.0)
    qx, qd = min(near / scale, w * 0.5), min(depth / scale, h)
    nx, ny = min(320, math.ceil(28 * w)), min(256, math.ceil(28 * h))
    dx, dy = w / nx, h / ny

    def wedge(x_pos, z_depth):
        # Sum the two half-plane angle solutions. Their coefficient is the
        # 90-degree corner jump, independent of the strip-image solution above.
        short = min(x_pos, w - x_pos)
        if short == 0.0 and z_depth == 0.0:
            return 0.0
        return (2.0 / math.pi) * (math.atan2(short, z_depth) - math.atan2(z_depth, w - short))

    grid = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for j in range(ny + 1):
        grid[0][j] = grid[nx][j] = -wedge(0.0, j * dy)
    for i in range(1, nx):
        grid[i][ny] = -wedge(i * dx, h)
    wx = 0.5 * dy**2 / (dx**2 + dy**2)
    wy = 0.5 - wx
    for _ in range(1500):
        largest = 0.0
        for i in range(1, nx):
            row, prev, nxt = grid[i], grid[i - 1], grid[i + 1]
            for j in range(1, ny):
                update = 1.8 * (wx * (prev[j] + nxt[j]) + wy * (row[j-1] + row[j+1]) - row[j])
                row[j] += update
                largest = max(largest, abs(update))
        if largest < 1e-10:
            break
    else:
        raise ValueError("2D Laplace constant-edge SOR did not converge")
    cold_fraction = 1.0
    if height / scale <= 16.0 and y / scale < dy:
        # Interpolate the full field from its zero trace, keeping the input
        # distance to the bottom instead of subtracting two nearly equal y's.
        cold_fraction = (y / scale) / dy
        qd = h - dy
    fx, fy = qx / dx, qd / dy
    i, j = min(int(fx), nx - 1), min(int(fy), ny - 1)
    tx, ty = fx - i, fy - j
    right = (1 - ty) * grid[i+1][j] + ty * grid[i+1][j+1]
    if i == 0:
        # Use the exact cold sidewall and the angle-difference identity to
        # retain temperatures proportional to very small x.
        wedge_delta = (2.0 / math.pi) * (
            math.atan2(qx, qd) - math.atan2(qd*qx, w*(w-qx) + qd*qd))
        numerical = wedge_delta + tx * (right + wedge(0.0, qd))
        # Recover the smooth residual's one-sided wall gradient using the
        # two interior columns and its known sidewall value.
        second = (1 - ty) * grid[2][j] + ty * grid[2][j+1]
        numerical += 0.5 * tx * (tx - 1.0) * (second - 2.0 * right - wedge(0.0, qd))
        if cold_fraction < 1.0:
            # Both adjacent traces vanish. Approach the bilinear x*y limit
            # continuously as the query tends to their intersection.
            bilinear = tx * (wedge(dx, qd) + right)
            numerical = cold_fraction * numerical + (1 - cold_fraction) * bilinear
    else:
        left = (1 - ty) * grid[i][j] + ty * grid[i][j+1]
        numerical = wedge(qx, qd) + (1 - tx)*left + tx*right
    return top * exact, top * (cold_fraction * numerical)


def _sine_laplace_rectangle(width, height, x, y, top):
    """Continuum harmonic and independent finite-difference SOR on its horizontal grid mode."""
    from bisect import bisect_right

    aspect = height / width
    if not 0.0 < aspect < math.inf:
        raise ValueError("Unrepresentable 2D Laplace aspect ratio")
    depth = (height - y) / width
    x_fraction = x / width
    exact_mode = math.exp(-math.pi * depth) * (
        -math.expm1(-2.0 * math.pi * (y / width)) / -math.expm1(-2.0 * math.pi * aspect)
    )
    sine = 0.0 if x in (0.0, width) else math.sin(math.pi * x_fraction)

    # Scale lengths by width. The hot boundary layer keeps the same mesh
    # spacing as the lateral grid regardless of the full plate height.
    nx = 40
    step = min(1.0 / nx, aspect / 20.0) or aspect
    nodes = [0.0]
    while nodes[-1] < aspect:
        z = nodes[-1]
        if z >= 2.0:
            step *= 1.25
        nodes.append(aspect if aspect - z <= 1.5 * step else z + step)

    # On a uniform x grid the sampled sine is an eigenvector of the exact
    # second-difference operator. Its amplitudes satisfy the same five-point
    # equations as the 2D mesh, with unequal adjacent vertical spacings.
    eigenvalue = (2.0 * nx * math.sin(math.pi / (2.0 * nx))) ** 2
    stencil = []
    for left, center, right in zip(nodes, nodes[1:], nodes[2:]):
        h0, h1 = center - left, right - center
        ratio = h0 / h1
        scale = 1.0 / (1.0 + 0.5 * eigenvalue * h0 * h1)
        lower = scale / (1.0 + ratio)
        stencil.append((lower, scale - lower))
    amplitudes = [1.0] + [0.0] * (len(nodes) - 1)
    for _ in range(4000):
        largest = 0.0
        for j, (lower, upper) in enumerate(stencil, 1):
            residual = lower * amplitudes[j - 1] + upper * amplitudes[j + 1] - amplitudes[j]
            update = 1.72 * residual
            amplitudes[j] += update
            largest = max(largest, abs(update))
        if largest < 1e-11:
            break
    else:
        raise ValueError("2D Laplace sine-mode SOR did not converge")

    j = min(bisect_right(nodes, depth) - 1, len(nodes) - 2)
    fy = (depth - nodes[j]) / (nodes[j + 1] - nodes[j])
    vertical = (1.0 - fy) * amplitudes[j] + fy * amplitudes[j + 1]
    fx = nx * x_fraction
    i = min(int(fx), nx - 1)
    left = math.sin(math.pi * i / nx)
    right = 0.0 if i + 1 == nx else math.sin(math.pi * (i + 1) / nx)
    horizontal = (1.0 - (fx - i)) * left + (fx - i) * right
    return top * (sine * exact_mode), top * (horizontal * vertical)


def verify_pde_system(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "convective_fin_1d")).strip().lower()

    if mode == "convective_fin_1d":
        k = float(spec["k_w_mk"])
        ac = float(spec["area_m2"])
        perim = float(spec["perimeter_m"])
        h = float(spec["h_w_m2k"])
        length = float(spec["length_m"])
        t_b = float(spec["t_base_c"])
        t_inf = float(spec["t_inf_c"])
        tip_bc = str(spec.get("tip_bc", "adiabatic")).strip().lower()
        x_eval = float(spec.get("x_eval_m", length))

        if (not all(map(math.isfinite, (k, ac, perim, h, length, t_b, t_inf, x_eval)))
                or min(k, ac, perim, h, length) <= 0.0 or not (0.0 <= x_eval <= length + 1e-12)):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Non-positive fin geometry/property or out-of-range x_eval_m"}

        theta_b = t_b - t_inf
        axial, lateral = k * ac, h * perim
        if not (0 < axial < math.inf and 0 < lateral < math.inf):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Unrepresentable fin conductance"}
        m = math.sqrt(lateral) / math.sqrt(axial)
        m_len = m * length
        flux_scale = math.sqrt(lateral) * math.sqrt(axial)
        cap_m = flux_scale * theta_b
        if not (0 < m_len < math.inf and 0 < flux_scale < math.inf and math.isfinite(cap_m)):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Unrepresentable fin heat scale"}

        if tip_bc in ("adiabatic", "convective"):
            h_tip = float(spec.get("h_tip_w_m2k", h)) if tip_bc == "convective" else 0.0
            if not math.isfinite(h_tip) or h_tip < 0.0:
                return {"passed": False, "domain": "pde", "mode": mode, "reason": "Invalid h_tip_w_m2k"}
            # Divide both hyperbolic solutions by exp(mL), and scale the
            # Robin coefficients together so an almost-isothermal tip is safe.
            tip_loss = h_tip * ac
            if not math.isfinite(tip_loss):
                return {"passed": False, "domain": "pde", "mode": mode, "reason": "Unrepresentable tip conductance"}
            boundary_scale = max(flux_scale, tip_loss)
            a, b = flux_scale / boundary_scale, tip_loss / boundary_scale
            x_m = m * min(x_eval, length)
            remaining = m * max(length - x_eval, 0.0)

            def boundary(z):
                return a * (1.0 + math.exp(-2.0 * z)) + b * -math.expm1(-2.0 * z)

            denom = boundary(m_len)
            decay_x, decay_tip = math.exp(-x_m / 2), math.exp(-m_len / 2)
            theta_x = (theta_b * decay_x) * (decay_x * (boundary(remaining) / denom))
            theta_tip = (theta_b * decay_tip) * (decay_tip * (2.0 * a / denom))
            numerator = a * -math.expm1(-2.0 * m_len) + b * (1.0 + math.exp(-2.0 * m_len))
            # Retain the accepted shape factors before applying the base drive.
            # Only the ratio calculation needs the wider intermediate range.
            conductance = Fraction(flux_scale) * Fraction(numerator) / Fraction(denom)
            q_fin = cap_m * (numerator / denom)
        elif tip_bc == "prescribed_temp":
            theta_L = float(spec["t_tip_c"]) - t_inf
            # Two prescribed temperatures drive the base flux independently.
            # Scaled exponentials keep both temperature weights finite at long lengths.
            if not all(map(math.isfinite, (theta_b, theta_L, m_len))) or m_len <= 0.0:
                return {"passed": False, "domain": "pde", "mode": mode,
                        "reason": "Non-finite fin temperatures or conduction scale"}
            x_m = m * min(x_eval, length)
            remaining = m * max(length - x_eval, 0.0)
            denominator = -math.expm1(-2.0 * m_len)
            base_weight = math.exp(-x_m) * -math.expm1(-2.0 * remaining) / denominator
            tip_weight = math.exp(-remaining) * -math.expm1(-2.0 * x_m) / denominator
            theta_x = theta_b * base_weight + theta_L * tip_weight
            theta_tip = theta_L
            # coth(z) - csch(z) = tanh(z/2). The separated difference avoids
            # cancellation for equal endpoint temperatures on a weakly cooled fin.
            transfer = (k * ac / length) * (2.0 * (m_len * math.exp(-m_len) / denominator))
            surface = k * ac * m * math.tanh(m_len / 2.0)
            q_fin = transfer * (theta_b - theta_L) + surface * theta_b
        elif tip_bc == "infinite":
            decay_x, decay_tip = math.exp(-m * x_eval / 2), math.exp(-m_len / 2)
            theta_x = (theta_b * decay_x) * decay_x
            theta_tip = (theta_b * decay_tip) * decay_tip
            conductance = Fraction(flux_scale)
            q_fin = cap_m
            tip_loss = flux_scale
        else:
            return {"passed": False, "domain": "pde", "mode": mode, "reason": f"Unsupported tip_bc: {tip_bc}"}

        if tip_bc == "prescribed_temp":
            # The grid exponent is 2N asinh(z/(2N)) for z=mL. The leading
            # remote flux error is (z**3/24 + z**2/8)/N**2. Budget half
            # the 1e-3 check for it; a bounded mesh still has to pass.
            capped_z = min(m_len, 1000.0)
            cells = capped_z * math.sqrt((capped_z / 24.0 + 1.0 / 8.0) / 5e-4)
            n_nodes = max(401, 1 + math.ceil(min(65536.0, cells)))
            dx = length / (n_nodes - 1)
            lower = [0.0] * n_nodes
            diag = [1.0] * n_nodes
            upper = [0.0] * n_nodes
            rhs = [0.0] * n_nodes
            diag[0] = 1.0
            rhs[0] = theta_b
            cond = k * ac / dx
            sink = h * perim * dx
            for i in range(1, n_nodes - 1):
                lower[i] = -cond
                diag[i] = 2.0 * cond + sink
                upper[i] = -cond
                rhs[i] = 0.0
            diag[-1] = 1.0
            rhs[-1] = theta_tip
            theta_fvm = _thomas_solve(lower, diag, upper, rhs)
            if m_len < 1.0:
                # Linear test-function balance of the same finite-volume equations.
                # This avoids subtracting adjacent nearly isothermal cells.
                cells = n_nodes - 1
                weighted = math.fsum((1.0 - i / cells) * theta_fvm[i] for i in range(1, cells))
                q_fvm = k * ac / length * (theta_b - theta_tip) + sink * (0.5 * theta_b + weighted)
            else:
                q_fvm = cond * (theta_fvm[0] - theta_fvm[1]) + 0.5 * sink * theta_fvm[0]

            tip_fvm = theta_fvm[-1]
        else:
            q_fvm, tip_fvm = _passive_fin_fvm(length, m_len, axial, lateral, tip_loss, theta_b)

        q_max = h * perim * length * theta_b + (tip_loss * theta_b if tip_bc == "convective" else 0.0)
        rel_q_err = abs(q_fvm - q_fin) / max(abs(q_fin), 1e-9)
        scaled_q_err = rel_q_err
        if tip_bc == "prescribed_temp":
            # Near a zero net base flux, compare the independent FVM result to
            # the scale of the opposing boundary contributions, in watts.
            scale = max(abs(q_fin), abs(transfer * (theta_b - theta_L)), abs(surface * theta_b), 1e-9)
            scaled_q_err = abs(q_fvm - q_fin) / scale
            eta_fin = q_fin / (h * perim * length * theta_b) if theta_b else None
            eps_fin = q_fin / (h * ac * theta_b) if theta_b else None
        else:
            eta_fin, eta_status = _passive_fin_ratio(
                theta_b, conductance, Fraction(h) * Fraction(perim) * Fraction(length))
            eps_fin, eps_status = _passive_fin_ratio(theta_b, conductance, Fraction(h) * Fraction(ac))
        tip_err = abs(tip_fvm - theta_tip)
        passed = (scaled_q_err < 1e-3) and (tip_err < max(1e-3, 1e-3 * abs(theta_b)))

        res: Dict[str, Any] = {
            "passed": passed,
            "domain": "pde",
            "mode": mode,
            "m_inv_m": m,
            "q_fin_w": q_fin,
            "q_fvm_w": q_fvm,
            "t_eval_c": t_inf + theta_x,
            "t_tip_c": t_inf + theta_tip,
            "fin_efficiency": eta_fin,
            "fin_effectiveness": eps_fin,
            "fvm_rel_error": rel_q_err,
            "q_max_ref_w": q_max,
        }
        if tip_bc == "prescribed_temp":
            res["fvm_scaled_flux_residual"] = scaled_q_err
            if (not theta_b and "claimed_value" in spec
                    and spec.get("claimed_field") in ("fin_efficiency", "fin_effectiveness")):
                res.update(passed=False, reason="Fin normalization is undefined at zero base excess temperature")
                return res
        else:
            res["fin_normalization"] = {
                "method": "passive_unit_drive",
                "efficiency_reference": "lateral_h_P_L",
                "effectiveness_reference": "bare_base_h_A",
                "length_reference": "provided_length_m",
                "zero_drive_policy": "undefined_heat_rate_ratio",
                "efficiency_float_status": eta_status,
                "effectiveness_float_status": eps_status,
            }
            field = spec.get("claimed_field")
            if "claimed_value" in spec and field in ("fin_efficiency", "fin_effectiveness") and res[field] is None:
                reason = ("undefined at zero base excess temperature" if not theta_b else "outside the float range")
                res.update(passed=False, reason=f"Cannot verify claimed {field}: {reason}")
                return res
        _check_claimed(res, spec)
        return res

    if mode == "heat_1d_transient":
        alpha = float(spec["alpha_m2_s"])
        length = float(spec["length_m"])
        t_eval = float(spec["t_eval_s"])
        x_eval = float(spec.get("x_eval_m", 0.5 * length))
        modes = spec.get("modes") or [{"n": 1, "amplitude": 100.0}]
        if (not all(map(math.isfinite, (alpha, length, t_eval, x_eval))) or alpha <= 0.0
                or length <= 0.0 or t_eval < 0.0 or not 0.0 <= x_eval <= length):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Invalid 1D transient heat parameters"}

        res = {"domain": "pde", "mode": mode}
        try:
            res.update(_transient_heat_modes(alpha, length, t_eval, x_eval, modes))
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            return {**res, "passed": False, "reason": "Invalid heat modes or unrepresentable diffusion scale"}
        if "u_exact" in res:
            _check_claimed(res, spec)
        return res

    if mode == "laplace_2d_rect":
        a_len = float(spec["width_m"])
        b_len = float(spec["height_m"])
        t_top = float(spec.get("t_top", 100.0))
        top_profile = str(spec.get("top_profile", "sine")).strip().lower()
        xq = float(spec.get("x_eval_m", 0.5 * a_len))
        yq = float(spec.get("y_eval_m", 0.5 * b_len))
        if not all(map(math.isfinite, (a_len, b_len, t_top, xq, yq))) or a_len <= 0.0 or b_len <= 0.0 or not (0.0 <= xq <= a_len) or not (0.0 <= yq <= b_len):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Invalid 2D Laplace domain or query point"}

        if top_profile == "sine":
            solver = _sine_laplace_rectangle
        elif top_profile in ("constant", "uniform"):
            solver = _constant_top_rectangle
        else:
            return {"passed": False, "domain": "pde", "mode": mode,
                    "reason": "Unsupported top_profile. Use sine, constant, or uniform."}
        try:
            t_exact, t_fdm = solver(a_len, b_len, xq, yq, t_top)
        except ValueError as exc:
            return {"passed": False, "domain": "pde", "mode": mode, "reason": str(exc)}
        err = abs(t_fdm - t_exact)
        passed = err <= max(1e-3, 1.5e-2 * abs(t_top))
        res = {
            "passed": passed,
            "domain": "pde",
            "mode": mode,
            "t_exact": t_exact,
            "t_fdm_sor": t_fdm,
            "abs_error": err,
        }
        _check_claimed(res, spec)
        return res

    return {"passed": False, "domain": "pde", "mode": mode, "reason": f"Unknown PDE mode: {mode}"}


def _poly_eval_complex(coeffs: Sequence[float], z: complex) -> complex:
    acc = 0j
    for c in coeffs:
        acc = acc * z + c
    return acc


def _poly_roots(coeffs: Sequence[float]) -> List[complex]:
    idx = 0
    while idx < len(coeffs) and abs(coeffs[idx]) == 0.0:
        idx += 1
    p = [float(c) / float(coeffs[idx]) for c in coeffs[idx:]]
    deg = len(p) - 1
    if deg <= 0:
        return []
    if deg == 1:
        return [complex(-p[1], 0.0)]
    if deg == 2:
        disc = cmath.sqrt(p[1] * p[1] - 4.0 * p[2])
        return [(-p[1] + disc) / 2.0, (-p[1] - disc) / 2.0]

    radius = 1.0 + max(abs(c) for c in p[1:])
    roots = [
        (0.4 + 0.9 * (k / deg)) * radius * cmath.exp(1j * (2.0 * math.pi * k / deg + 0.37))
        for k in range(deg)
    ]
    for _ in range(400):
        max_step = 0.0
        for i in range(deg):
            zi = roots[i]
            val = _poly_eval_complex(p, zi)
            denom = 1.0 + 0j
            for j in range(deg):
                if i != j:
                    diff = zi - roots[j]
                    denom *= diff if abs(diff) > 1e-18 else (1e-18 + 0j)
            delta = val / denom
            roots[i] = zi - delta
            if abs(delta) > max_step:
                max_step = abs(delta)
        if max_step < 1e-13:
            break
    return roots


def _routh_table(coeffs: Sequence[float]) -> Tuple[List[List[float]], int, bool]:
    n = len(coeffs) - 1
    cols = (n + 2) // 2
    row0 = [float(coeffs[2 * j]) if 2 * j <= n else 0.0 for j in range(cols)]
    row1 = [float(coeffs[2 * j + 1]) if 2 * j + 1 <= n else 0.0 for j in range(cols)]
    table = [row0, row1]
    zero_row_detected = False
    eps = 1e-12
    for i in range(2, n + 1):
        prev2 = table[i - 2]
        prev1 = table[i - 1]
        if all(abs(v) <= 1e-14 for v in prev1):
            zero_row_detected = True
            order_aux = n - i + 2
            prev1 = [prev2[j] * max(0, order_aux - 2 * j) for j in range(cols)]
            if all(abs(v) <= 1e-14 for v in prev1):
                prev1[0] = eps
            table[i - 1] = prev1
        piv = prev1[0]
        if abs(piv) <= 1e-14:
            piv = eps if piv >= 0 else -eps
            table[i - 1][0] = piv
        new_row = [0.0] * cols
        for j in range(cols - 1):
            new_row[j] = (piv * prev2[j + 1] - prev2[0] * prev1[j + 1]) / piv
        table.append(new_row)

    first_col = [r[0] for r in table]
    sign_changes = 0
    prev_s = 1 if first_col[0] >= 0 else -1
    for val in first_col[1:]:
        if abs(val) <= 1e-13:
            continue
        cur_s = 1 if val > 0 else -1
        if cur_s != prev_s:
            sign_changes += 1
            prev_s = cur_s
    return table, sign_changes, zero_row_detected


def _trim_poly(p):
    while p and not p[-1]:
        p.pop()
    return p


def _poly_sum(a, b, sign=1):
    return _trim_poly([x + sign*y for x, y in zip_longest(a, b, fillvalue=0)])


def _poly_product(a, b):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return _trim_poly(out)


def _poly_divrem(a, b):
    remainder = list(a)
    quotient = [0] * max(0, len(a) - len(b) + 1)
    while len(remainder) >= len(b):
        offset = len(remainder) - len(b)
        q = Fraction(remainder[-1], b[-1])
        quotient[offset] = q
        for i, c in enumerate(b):
            remainder[offset+i] -= q*c
        _trim_poly(remainder)
    return quotient, remainder


def _primitive_poly(p):
    """Reduce rational coefficients to signed primitive integers without changing signs."""
    if not p:
        return []
    denominator = math.lcm(*(c.denominator for c in p))
    integers = [int(c*denominator) for c in p]
    divisor = math.gcd(*integers)
    return [v//divisor for v in integers]


def _sturm_chain(p):
    chain = [_primitive_poly(p)]
    derivative = _primitive_poly([i*c for i, c in enumerate(p) if i])
    if derivative:
        chain.append(derivative)
    while len(chain[-1]) > 1:
        _, rem = _poly_divrem(chain[-2], chain[-1])
        if not rem:
            break
        chain.append(_primitive_poly([-c for c in rem]))
    return chain


def _poly_value(p, x):
    out = 0
    for c in reversed(p):
        out = out*x + c
    return out


def _poly_gcd(a, b):
    a, b = _primitive_poly(a), _primitive_poly(b)
    while b:
        a, b = b, _primitive_poly(_poly_divrem(a, b)[1])
    return a


def _resolved_nonzero(poly, lo, hi):
    """Bound nonzero polynomial values even immediately next to a pole or zero."""
    low = high = 0
    for c in reversed(poly):
        bounds = (low*lo, low*hi, high*lo, high*hi)
        low, high = min(bounds) + c, max(bounds) + c
    return (low > 0 or high < 0) and (high-low) * (1 << 45) <= min(abs(low), abs(high))


def _regular_roots(poly, exclude=(), nonzero_witnesses=(), precision_bits=56):
    """Isolate distinct positive roots in rational intervals; remove exact pole/zero factors."""
    p = _primitive_poly(poly)
    while p and not p[0]:
        p.pop(0)
    if len(p) < 2:
        return []
    gcd = _sturm_chain(p)[-1]
    if len(gcd) > 1:
        p = _primitive_poly(_poly_divrem(p, gcd)[0])
    if exclude:
        common = _poly_gcd(p, exclude)
        if len(common) > 1:
            p = _primitive_poly(_poly_divrem(p, common)[0])
    if len(p) < 2:
        return []
    if len(p) == 2:
        x = Fraction(-p[0], p[1])
        return [(x, x)] if x > 0 else []
    chain = _sturm_chain(p)

    def variations(x):
        signs = [_poly_value(q, x) for q in chain]
        nonzero = [v > 0 for v in signs if v]
        return sum(a != b for a, b in zip(nonzero, nonzero[1:]))

    # Cauchy's strict upper bound encloses every finite root, independently of units.
    upper = Fraction(2 + max(abs(v) for v in p[:-1]) // abs(p[-1]))
    pending = [(Fraction(0), upper, variations(0), variations(upper))]
    roots = []
    while pending:
        lo, hi, left, right = pending.pop()
        count = left - right
        if count == 0:
            continue
        if count == 1:
            if not _poly_value(p, hi):
                roots.append((hi, hi))
                continue
            if ((hi-lo) * (1 << precision_bits) <= hi and _poly_value(p, lo)
                    and all(_resolved_nonzero(q, lo, hi) for q in nonzero_witnesses)):
                roots.append((lo, hi))
                continue
        mid = (lo+hi)/2
        middle = variations(mid)
        pending.append((mid, hi, middle, right))
        pending.append((lo, mid, left, middle))
    return roots


def _jw_parts(coeffs):
    """Return A(x), B(x) for P(jw) = A(w²) + jw B(w²)."""
    cs = [Fraction(str(c)) for c in reversed(coeffs)]
    return tuple(_trim_poly([(-1)**i * c for i, c in enumerate(cs[k::2])]) for k in (0, 1))


def _ordered_gain_roots(unity, imag, real, mag_d):
    """Merge real-axis and off-axis crossings in certified squared-frequency order."""
    axis = _poly_gcd(unity, imag)
    off_axis_exclude = _poly_product(mag_d, axis)
    bits = 56
    while True:
        on = _regular_roots(axis, mag_d, (real,), precision_bits=bits)
        off = _regular_roots(unity, off_axis_exclude, (imag, mag_d), precision_bits=bits)
        events = sorted([(i, True) for i in on] + [(i, False) for i in off])
        # The real-axis factors were removed from the off-axis polynomial. All
        # these roots are distinct, so disjoint isolation intervals exist.
        if all(a[0][1] < b[0][0] for a, b in zip(events, events[1:])):
            return events
        bits *= 2


def _bode_phase_margin(real, imag, squared_frequency, frequency):
    """Phase margin of real + j*sqrt(x)*imag, including unrenderable frequencies."""
    scale = max(abs(real), abs(imag))
    if frequency is not None:
        re = float(real / scale)
        scaled_imag = float(imag / scale)
        im = scaled_imag * frequency
        # Preserve the established evaluation when both components are well
        # scaled. Subnormal intermediate values can lose their relative size.
        if ((not real or abs(re) >= sys.float_info.min)
                and (not imag or abs(im) >= sys.float_info.min
                     and abs(scaled_imag) >= sys.float_info.min)
                and math.isfinite(im)):
            return math.degrees(math.atan2(im, re)) % 360 - 180

    real_sq, imag_sq = real * real, squared_frequency * imag * imag
    if real_sq >= imag_sq:
        re, im = 1.0, float(_scaled_sqrt(imag_sq / real_sq))
    else:
        re, im = float(_scaled_sqrt(real_sq / imag_sq)), 1.0
    angle = math.degrees(math.atan2(-im if imag < 0 else im, -re if real < 0 else re))
    # A very small negative angle can round to -0.0. Its side of the phase
    # wrap still comes from the sign of the exact imaginary component.
    return angle + 180 if imag < 0 else angle - 180


def _bode_margins(num: Sequence[float], den: Sequence[float]) -> Dict[str, Any]:
    """Find positive-frequency margins from the real rational transfer function.

    The four scalar fields select the smallest absolute margin (degrees or dB)
    among isolated crossings. Open intervals describe non-isolated crossover
    sets separately; their right endpoint is None when unbounded. Denominator
    zeros are excluded at the original input frequencies, even when removable.
    """
    nr, ni = _jw_parts(num)
    dr, di = _jw_parts(den)

    def dot(ar, ai, br, bi):
        return _poly_sum(_poly_product(ar, br), [0] + _poly_product(ai, bi))

    mag_n = dot(nr, ni, nr, ni)
    mag_d = dot(dr, di, dr, di)
    if not mag_d:
        raise ValueError("The transfer denominator must be nonzero")
    real = dot(nr, ni, dr, di)
    imag = _poly_sum(_poly_product(ni, dr), _poly_product(nr, di), -1)
    unity = _poly_sum(mag_n, mag_d, -1)
    undefined = _poly_product(mag_n, mag_d)
    denominator_zeros = _regular_roots(mag_d)
    common = _poly_gcd(mag_d, mag_n)
    reduced_den, _ = _poly_divrem(mag_d, common)
    poles = _regular_roots(reduced_den)
    removable = _regular_roots(common, reduced_den)

    def center(interval):
        return (interval[0] + interval[1])/2

    def frequency(interval):
        x = center(interval)
        exponent = (x.numerator.bit_length() - x.denominator.bit_length()) // 2
        scaled = x / (Fraction(2) ** (2*exponent))
        try:
            value = math.ldexp(math.sqrt(float(scaled)), exponent)
        except OverflowError:
            value = None
        if value is None:
            status = "above_float_range"
        elif value == 0.0:
            value, status = None, "below_float_range"
        else:
            status = "finite"
        return value, {"status": status, "squared_interval": [_fraction_text(v) for v in interval]}

    def open_intervals(boundaries, include):
        runs, evidence = [], []
        previous = None
        for following in [*boundaries, None]:
            lo = previous[1] if previous else Fraction(0)
            x = (lo + following[0])/2 if following else lo+1
            if include(x):
                left = (frequency(previous) if previous else
                        (0.0, {"status": "zero", "squared_interval": ["0", "0"]}))
                right = (frequency(following) if following else
                         (None, {"status": "unbounded", "squared_interval": None}))
                runs.append([left[0], right[0]])
                evidence.append([left[1], right[1]])
            previous = following
        return runs, evidence

    gain, phase = [], []
    gain_evidence, phase_evidence = [], []
    if unity:
        # On the real axis, exact unit gain is +1 or -1. Evaluating a nearby
        # midpoint can choose the opposite side of the phase wrap at +1.
        for interval, on_axis in _ordered_gain_roots(unity, imag, real, mag_d):
            x = center(interval)
            w, metadata = frequency(interval)
            re = _poly_value(real, x)
            if on_axis:
                margin = -180.0 if re > 0 else 0.0
            else:
                margin = _bode_phase_margin(re, _poly_value(imag, x), x, w)
            gain.append({"w_rad_s": w, "phase_margin_deg": margin})
            gain_evidence.append(metadata)
    for interval in _regular_roots(imag, undefined, (real, mag_d)):
        x = center(interval)
        re = _poly_value(real, x)
        if re < 0:
            # At a real-axis crossover G = Re(N conj(D)) / |D|².
            ratio = Fraction(_poly_value(mag_d, x), -re)
            if Fraction(1, 2) <= ratio <= 2:
                db = 20 * math.log1p(float(ratio-1)) / math.log(10)
            else:
                db = 20 * (math.log10(ratio.numerator) - math.log10(ratio.denominator))
            w, metadata = frequency(interval)
            phase.append({"w_rad_s": w, "gain_margin_db": db})
            phase_evidence.append(metadata)

    # Constant unity magnitude and real-axis continua have no unique crossover frequency.
    gain_intervals, gain_endpoints = (open_intervals(denominator_zeros, lambda _: True)
                                     if not unity else ([], []))
    phase_intervals, phase_endpoints = (open_intervals(_regular_roots(undefined), lambda x: _poly_value(real, x) < 0)
                                       if not imag and mag_n else ([], []))
    selected_gain = min(range(len(gain)), key=lambda i: abs(gain[i]["phase_margin_deg"]), default=None)
    selected_phase = min(range(len(phase)), key=lambda i: abs(phase[i]["gain_margin_db"]), default=None)
    gc = gain[selected_gain] if selected_gain is not None else {}
    pc = phase[selected_phase] if selected_phase is not None else {}
    pole_frequencies = [frequency(i) for i in poles]
    hole_frequencies = [frequency(i) for i in removable]
    return {"w_gc_rad_s": gc.get("w_rad_s"), "phase_margin_deg": gc.get("phase_margin_deg"),
            "w_pc_rad_s": pc.get("w_rad_s"), "gain_margin_db": pc.get("gain_margin_db"),
            "gain_crossovers": gain, "phase_crossovers": phase,
            "unity_gain_intervals_rad_s": gain_intervals,
            "negative_real_intervals_rad_s": phase_intervals,
            "imaginary_axis_poles_rad_s": [w for w, _ in pole_frequencies],
            "removable_holes_rad_s": [w for w, _ in hole_frequencies],
            "bode_frequency_evidence": {
                "coordinate": "omega_squared_rad2_per_s2",
                "rendering_basis": "isolating_interval_midpoint",
                "selected_gain_index": selected_gain,
                "selected_phase_index": selected_phase,
                "gain_crossovers": gain_evidence,
                "phase_crossovers": phase_evidence,
                "unity_gain_intervals_rad_s": gain_endpoints,
                "negative_real_intervals_rad_s": phase_endpoints,
                "imaginary_axis_poles_rad_s": [meta for _, meta in pole_frequencies],
                "removable_holes_rad_s": [meta for _, meta in hole_frequencies]}}


def _fraction_text(value):
    # Decimal(int) is exact and avoids Python's limit on integer-to-text digits.
    numerator = str(Decimal(value.numerator))
    return numerator if value.denominator == 1 else numerator + "/" + str(Decimal(value.denominator))


def _state_characteristic(a):
    """Integer Faddeev–LeVerrier on D*A, with coefficients and residual in A's units."""
    n = len(a)
    denominator = max(value.denominator for row in a for value in row)
    operator = [[v.numerator * (denominator // v.denominator) for v in row] for row in a]
    accumulator = [[int(i == j) for j in range(n)] for i in range(n)]
    coeffs = [Fraction(1)]
    power = 1
    for k in range(1, n + 1):
        columns = list(zip(*accumulator))
        accumulator = [[sum(x * y for x, y in zip(row, col)) for col in columns]
                       for row in operator]
        coefficient, remainder = divmod(-sum(accumulator[i][i] for i in range(n)), k)
        if remainder:
            raise ArithmeticError("Nonintegral characteristic coefficient for an integer matrix")
        for i in range(n):
            accumulator[i][i] += coefficient
        power *= denominator
        coeffs.append(Fraction(coefficient, power))
    # At this step the accumulator is the full monic polynomial of D*A evaluated
    # at D*A. Dividing each entry by D**n recovers the matrix residual for A.
    residual = Fraction(max(abs(x) for row in accumulator for x in row), power)
    return coeffs, residual


def _diagnostic_float(value):
    rendered = float(value)
    if not math.isfinite(rendered) or (value and rendered == 0.0):
        raise OverflowError("Nonzero or nonfinite diagnostic is outside binary64 range")
    return rendered


def _binary_exponent(value):
    """Floor(log2(value)) for a positive Fraction, without converting to float."""
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent if value >= Fraction(2) ** exponent else exponent - 1


def _scaled_sqrt(value):
    """Positive square root without rounding a large or tiny radicand to binary64."""
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Fraction(numerator, denominator)
    scale = Fraction(2) ** (_binary_exponent(value) // 2)
    return Fraction(math.sqrt(float(value / (scale * scale)))) * scale


def _state_pole_diagnostics(coeffs, matrix_scale):
    """Render finite pole approximations; only the zero multiplicity is certified."""
    end = len(coeffs)
    while end > 1 and not coeffs[end - 1]:
        end -= 1
    zeros = len(coeffs) - end
    reduced = coeffs[:end]
    degree = end - 1
    poles = [[0.0, 0.0] for _ in range(zeros)]
    try:
        if degree == 1:
            poles.append([_diagnostic_float(-reduced[1]), 0.0])
        elif degree == 2:
            _, b, c = reduced
            center = -b / 2
            radius_squared = center * center - c
            if radius_squared < 0:
                real = _diagnostic_float(center)
                imag = _diagnostic_float(_scaled_sqrt(-radius_squared))
                poles.extend([[real, -imag], [real, imag]])
            elif not radius_squared:
                root = _diagnostic_float(center)
                poles.extend([[root, 0.0], [root, 0.0]])
            else:
                radius = _scaled_sqrt(radius_squared)
                # Choose the sum of like-signed magnitudes for one root. Recover the
                # other via their product before rendering either to binary64.
                dominant = center - radius if b > 0 else center + radius
                poles.extend([[_diagnostic_float(dominant), 0.0],
                              [_diagnostic_float(c / dominant), 0.0]])
        elif degree > 2:
            # The roots of p(s*y)/s**degree are dimensionless. Restore the
            # physical scale only after the existing numerical root iteration.
            scale = Fraction(2) ** _binary_exponent(matrix_scale)
            normalized = [_diagnostic_float(v / scale ** i) for i, v in enumerate(reduced)]
            for root in _poly_roots(normalized):
                real = _diagnostic_float(Fraction(root.real) * scale)
                imag = _diagnostic_float(Fraction(root.imag) * scale)
                if real == imag == 0.0:
                    # All exact zero factors were removed before this approximation.
                    return None, "unavailable_numerical_diagnostic", zeros
                poles.append([real, imag])
    except (ArithmeticError, ValueError):
        status = "unavailable_numerical_diagnostic" if degree > 2 else "outside_float_range"
        return None, status, zeros
    return sorted(poles), "finite_approximation", zeros


def _schur_stable_exact(coefficients):
    """Strict Schur recursion for a real monic polynomial in descending powers."""
    denominator = max(value.denominator for value in coefficients)
    row = _primitive_integer_vector([int(value * denominator) for value in coefficients])
    while len(row) > 1:
        leading, constant = row[0], row[-1]
        if abs(constant) >= abs(leading):
            return False
        # Divide (leading * poly - constant * reversed_poly) by z.
        row = _primitive_integer_vector([leading * v - constant * w
                                         for v, w in zip(row[:-1], reversed(row))])
    return True


def _symmetric_lyapunov_solutions(a, forcings, *, discrete=False):
    """Solve the continuous Lyapunov or discrete Stein equation for symmetric P."""
    n = len(a)
    coords = [(i, j) for i in range(n) for j in range(i, n)]
    width = len(coords)
    indices = {(i, j): col for col, (i, j) in enumerate(coords)}
    indices.update({(j, i): col for col, (i, j) in enumerate(coords)})
    rows = []
    for i, j in coords:
        row = [Fraction(0)] * width
        if discrete:
            # Aᵀ P A - P = -Q. Symmetric unknowns share a column.
            row[indices[i, j]] -= 1
            for k in range(n):
                for l in range(n):
                    row[indices[k, l]] += a[k][i] * a[l][j]
        else:
            for k in range(n):
                row[indices[k, j]] += a[k][i]
                row[indices[i, k]] += a[k][j]
        row.extend(-q[i][j] for q in forcings)
        # The input coefficients are dyadic. Multiplying the entire equation,
        # including both right-hand sides, clears every denominator exactly.
        denominator = max(value.denominator for value in row)
        rows.append(_primitive_integer_vector(
            [value.numerator * (denominator // value.denominator) for value in row]
        ))

    rank = 0
    for col in range(width):
        pivot = next((i for i in range(rank, width) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_row = rows[rank]
        for i in range(rank + 1, width):
            row = rows[i]
            if row[col]:
                divisor = math.gcd(pivot_row[col], row[col])
                left, right = pivot_row[col] // divisor, row[col] // divisor
                rows[i] = _primitive_integer_vector(
                    [left * x - right * y for x, y in zip(row, pivot_row)]
                )
        rank += 1

    solutions = []
    for rhs in range(width, width + len(forcings)):
        if rank < width:
            # The bottom rows have zero coefficients. Their right-hand sides
            # distinguish inconsistency from infinitely many symmetric solutions.
            status = "inconsistent" if any(row[rhs] for row in rows[rank:]) else "nonunique"
            solutions.append((status, None))
            continue
        values = [Fraction(0)] * width
        for i in reversed(range(width)):
            rest = sum(rows[i][j] * values[j] for j in range(i + 1, width))
            values[i] = (Fraction(rows[i][rhs]) - rest) / rows[i][i]
        p = [[values[indices[i, j]] for j in range(n)] for i in range(n)]
        solutions.append(("unique", p))
    return solutions


def _positive_definite_exact(matrix):
    """A symmetric rational matrix is positive definite iff all LDLᵀ pivots are positive."""
    work = [row.copy() for row in matrix]
    for k in range(len(work)):
        pivot = work[k][k]
        if pivot <= 0:
            return False
        for i in range(k + 1, len(work)):
            factor = work[i][k] / pivot
            for j in range(k + 1, i + 1):
                work[i][j] -= factor * work[j][k]
    return True


def _lyapunov_certificate(a, q, solution, *, discrete=False):
    status, p = solution
    if p is None:
        return {"solution_status": status, "P_exact": None, "positive_definite": None,
                "residual_exact": None}
    # Check the full original matrix equation independently of the triangular solve.
    n = len(a)
    if discrete:
        pa = [[sum(p[i][k] * a[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        zero_residual = all(
            q[i][j] + sum(a[k][i] * pa[k][j] for k in range(n)) == p[i][j]
            for i in range(n) for j in range(n)
        )
    else:
        zero_residual = all(
            q[i][j] + sum(a[k][i] * p[k][j] + p[i][k] * a[k][j] for k in range(n)) == 0
            for i in range(n) for j in range(n)
        )
    return {"solution_status": status, "P_exact": [[_fraction_text(v) for v in row] for row in p],
            "positive_definite": _positive_definite_exact(p), "residual_exact": zero_residual}


def _lyapunov_float_matrix(p):
    if p is None:
        return None, "not_unique_solution"
    try:
        matrix = [[float(v) for v in row] for row in p]
    except OverflowError:
        return None, "outside_float_range"
    for exact_row, float_row in zip(p, matrix):
        if any(not math.isfinite(value) or (exact and value == 0.0)
               for exact, value in zip(exact_row, float_row)):
            return None, "outside_float_range"
    return matrix, "finite"


def _transfer_root_certificate(monic):
    """Count roots on both sides of the imaginary axis, including multiplicities."""
    degree = len(monic) - 1
    certificate = {
        "method": "infinitesimal_shift_hurwitz",
        "status": "classified" if degree > 0 else "constant_nonzero",
        "degree": degree,
        "normalized_coefficients": [_fraction_text(v) for v in monic],
        "minor_leading_terms": [],
    }
    if not monic:
        return {**certificate, "status": "identically_zero", "degree": None,
                "first_column_signs_plus": None, "first_column_signs_minus": None,
                "rhp_poles": None, "jw_axis_poles": None, "lhp_poles": None,
                "hurwitz_stable": None, "reason": "Identically zero characteristic polynomial"}

    # Coefficients of s**(degree-k) in p(s+t), each in ascending powers of t.
    shifted = [[monic[k-u] * math.comb(degree-k+u, u) for u in range(k+1)]
               for k in range(degree+1)]
    matrix = [[shifted[2*c-r+1] if 0 <= 2*c-r+1 <= degree else []
               for c in range(degree)] for r in range(degree)]
    previous = [1]
    previous_plus = previous_minus = 1
    plus, minus = [1], [1]
    for k in range(degree):
        # Bareiss pivots are the leading Hurwitz minors. Each is a nonzero
        # polynomial since p(s+t) is strictly Hurwitz for sufficiently large t.
        pivot = matrix[k][k]
        power, coefficient = next((i, v) for i, v in enumerate(pivot) if v)
        certificate["minor_leading_terms"].append(
            {"power": power, "coefficient": _fraction_text(coefficient)})
        delta_plus = 1 if coefficient > 0 else -1
        delta_minus = -delta_plus if power % 2 else delta_plus
        # The kth Routh first-column entry is Delta_k / Delta_(k-1).
        plus.append(delta_plus * previous_plus)
        minus.append(delta_minus * previous_minus)
        previous_plus, previous_minus = delta_plus, delta_minus
        for r in range(k+1, degree):
            for c in range(k+1, degree):
                numerator = _poly_sum(_poly_product(pivot, matrix[r][c]),
                                      _poly_product(matrix[r][k], matrix[k][c]), -1)
                if previous != [1]:
                    numerator, remainder = _poly_divrem(numerator, previous)
                    if remainder:
                        raise ArithmeticError("Inexact division of a shifted Hurwitz minor")
                    _trim_poly(numerator)
                matrix[r][c] = numerator
        previous = pivot

    # A positive shift moves axis roots left. A negative shift moves them
    # right. Off-axis roots stay in their original half-plane near t=0.
    right = sum(a != b for a, b in zip(plus, plus[1:]))
    right_and_axis = sum(a != b for a, b in zip(minus, minus[1:]))
    return {**certificate, "first_column_signs_plus": plus, "first_column_signs_minus": minus,
            "rhp_poles": right, "jw_axis_poles": right_and_axis - right,
            "lhp_poles": degree - right_and_axis, "hurwitz_stable": right_and_axis == 0}


def _transfer_pole_diagnostics(monic):
    if not monic:
        return None, "undefined_characteristic"
    exponent = max((_binary_exponent(abs(v)) // i for i, v in enumerate(monic[1:], 1) if v),
                   default=0)
    poles, status, _ = _state_pole_diagnostics(monic, Fraction(2) ** exponent)
    return poles, status


def _nyquist_winding(num, den, closed):
    """Count clockwise encirclements of -1 for a regular strictly proper loop.

    Traverse s=jw from -infinity to +infinity and close through the right
    half-plane. The large arc maps to 1 for F=1+N/D. Crossings of the negative
    real ray by F thus give the complete winding without a frequency grid.
    """
    def unavailable(reason):
        return {"status": "unavailable", "reason": reason,
                "clockwise_encirclements": None, "negative_ray_events": None}

    if len(_trim_poly(list(reversed(num)))) >= len(den):
        return unavailable("requires_strictly_proper_loop")

    cr, ci = _jw_parts(closed)
    dr, di = _jw_parts(den)
    # Work with the supplied realization. Common imaginary-axis factors are
    # not silently cancelled, and an indented contour is outside this count.
    if not den[-1] or _regular_roots(_poly_gcd(dr, di)):
        return unavailable("open_loop_pole_on_contour")
    if not closed[-1] or _regular_roots(_poly_gcd(cr, ci)):
        return unavailable("nyquist_locus_hits_minus_one")

    # Numerators of Re(F) and Im(F)/w in x=w². Their common denominator
    # |D(jw)|² is strictly positive by the contour check above.
    real = _poly_sum(_poly_product(cr, dr), [0] + _poly_product(ci, di))
    imag = _poly_sum(_poly_product(ci, dr), _poly_product(cr, di), -1)
    events = []
    if imag:
        # Im(F) is odd. When its linear term vanishes, its first nonzero
        # coefficient still determines the two sides of the origin.
        if real[0] < 0:
            right = next(c for c in imag if c)
            events.append({"omega_squared_interval": ["0", "0"],
                           "clockwise_contribution": 1 if right > 0 else -1})

        roots = _regular_roots(imag, nonzero_witnesses=(real,))
        for i, (lo, hi) in enumerate(roots):
            if _poly_value(real, (lo + hi) / 2) > 0:
                continue
            # Signs in the adjacent root-free gaps also classify tangencies.
            # Conjugation reverses the frequency AND the imaginary sign, so
            # crossings at +w and -w make the same signed contribution.
            previous = roots[i - 1][1] if i else Fraction(0)
            following = roots[i + 1][0] if i + 1 < len(roots) else hi + 2
            left = _poly_value(imag, (previous + lo) / 2)
            right = _poly_value(imag, (hi + following) / 2)
            contribution = 2 * (int(right > 0) - int(left > 0))
            events.append({"omega_squared_interval": [_fraction_text(lo), _fraction_text(hi)],
                           "clockwise_contribution": contribution})

    return {"status": "certified", "clockwise_encirclements": sum(
                event["clockwise_contribution"] for event in events),
            "negative_ray_events": events}


def verify_control_system(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "transfer_function" if "den" in spec else "state_space")).strip().lower()
    timebase = spec.get("timebase", "continuous")
    if timebase not in ("continuous", "discrete"):
        return {"passed": False, "domain": "control", "mode": mode,
                "reason": "timebase must be continuous or discrete"}
    discrete = timebase == "discrete"
    if discrete and mode != "state_space":
        return {"passed": False, "domain": "control", "mode": mode,
                "reason": "Discrete-time control requires a state_space transition matrix"}
    if "sample_time_s" in spec:
        sample_time = spec["sample_time_s"]
        if (not discrete or type(sample_time) not in (float, int)
                or not 0 < sample_time < math.inf):
            return {"passed": False, "domain": "control", "mode": mode,
                    "reason": "sample_time_s requires timebase=discrete and a finite positive period"}

    if mode == "transfer_function":
        try:
            raw_num, raw_den = spec.get("num", [1.0]), spec["den"]
            if any(not isinstance(row, (list, tuple)) or any(isinstance(c, bool) for c in row)
                   for row in (raw_num, raw_den)):
                raise ValueError
            num = [float(c) for c in raw_num]
            den = [float(c) for c in raw_den]
            if not all(math.isfinite(c) for row in (num, den) for c in row):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"passed": False, "domain": "control", "mode": mode,
                    "reason": "Transfer coefficients must be finite numeric arrays"}
        while len(den) > 1 and abs(den[0]) == 0.0:
            den.pop(0)
        if len(den) < 2:
            return {"passed": False, "domain": "control", "mode": mode, "reason": "Denominator degree must be >= 1"}
        if den[0] < 0.0:
            den = [-c for c in den]
            num = [-c for c in num]

        den_exact = [Fraction(c) for c in den]
        num_exact = [Fraction(c) for c in num]
        # Add in the original coefficient units before making either polynomial
        # monic. Leading cancellations lower the closed-loop degree exactly.
        closed = list(reversed(_poly_sum(list(reversed(den_exact)), list(reversed(num_exact)))))
        open_monic = [c / den_exact[0] for c in den_exact]
        closed_monic = [c / closed[0] for c in closed]
        open_certificate = _transfer_root_certificate(open_monic)
        closed_certificate = _transfer_root_certificate(closed_monic)
        poles, pole_status = _transfer_pole_diagnostics(open_monic)
        cl_poles, cl_pole_status = _transfer_pole_diagnostics(closed_monic)

        nyquist = _nyquist_winding(num_exact, den_exact, closed)
        winding = nyquist["clockwise_encirclements"]
        nyquist_closed_rhp = open_certificate["rhp_poles"] + winding if winding is not None else None
        margins = _bode_margins(spec.get("num", [1.0]), spec["den"])
        # Take the origin limit in the same exact binary64 coefficient units
        # as the transfer certificates, before rounding the scalar for display.
        d_order = next(i for i, c in enumerate(reversed(den_exact)) if c)
        n_order = next((i for i, c in enumerate(reversed(num_exact)) if c), None)
        dc_status = "finite_limit" if d_order else "finite"
        if n_order is None or n_order > d_order:
            dc_exact = Fraction(0)
        elif n_order == d_order:
            dc_exact = num_exact[-1 - n_order] / den_exact[-1 - d_order]
        else:
            dc_exact, dc_status = None, "pole_at_zero"
        dc_gain = None
        if dc_exact is not None:
            try:
                dc_gain = _diagnostic_float(dc_exact)
            except OverflowError:
                dc_status = "outside_float_range"

        res: Dict[str, Any] = {
            "passed": bool(closed_monic),
            "domain": "control",
            "mode": mode,
            "nyquist_certificate": nyquist,
            "nyquist_clockwise_encirclements": winding,
            "nyquist_closed_loop_rhp_poles": nyquist_closed_rhp,
            "open_loop_root_certificate": open_certificate,
            "closed_loop_root_certificate": closed_certificate,
            "hurwitz_stable": open_certificate["hurwitz_stable"],
            "closed_loop_hurwitz_stable": closed_certificate["hurwitz_stable"],
            "rhp_poles_routh": open_certificate["rhp_poles"],
            "rhp_poles_eigen": sum(re > 0 for re, _ in poles) if poles is not None else None,
            "rhp_poles_eigen_is_approximate": True,
            "jw_axis_poles": open_certificate["jw_axis_poles"],
            "dc_gain": dc_gain,
            "dc_gain_exact": _fraction_text(dc_exact) if dc_exact is not None else None,
            "dc_gain_status": dc_status,
            "poles_real_imag": poles,
            "poles_status": pole_status,
            "closed_loop_poles_real_imag": cl_poles,
            "closed_loop_poles_status": cl_pole_status,
            **margins,
        }
        if nyquist_closed_rhp is not None and nyquist_closed_rhp != closed_certificate["rhp_poles"]:
            res.update(passed=False, reason="Nyquist winding and characteristic pole counts disagree")
        if not closed_monic:
            res["reason"] = closed_certificate["reason"]
        field = spec.get("claimed_field")
        if "claimed_value" in spec and field == "dc_gain_exact":
            res["passed"] = False
            res["reason"] = "dc_gain_exact is rational evidence. Use dc_gain for a numeric scalar claim."
        elif "claimed_value" in spec and isinstance(field, str) and field in res and res[field] is None:
            res["passed"] = False
            res["reason"] = f"Cannot verify claimed {field}: the result is undefined or unavailable"
        elif "claimed_value" in spec and field in ("nyquist_clockwise_encirclements", "nyquist_closed_loop_rhp_poles"):
            # A winding or pole count is integral. Relative tolerances and a
            # float conversion could otherwise accept an adjacent/fractional claim.
            value = spec["claimed_value"]
            try:
                count = Decimal(str(value)) if type(value) in (str, int, float) else None
                matches = count is not None and count.is_finite() and count == res[field]
            except (ArithmeticError, ValueError):
                matches = False
            res["claim_check"] = {"field": field, "actual": res[field],
                                  "comparison": "exact_integer", "within_tolerance": matches}
            if not matches:
                res.update(passed=False, reason=f"Claimed {field} must equal the exact integer count")
        else:
            _check_claimed(res, spec)
        return res

    if mode == "state_space":
        a_mat = [[float(v) for v in row] for row in spec["A"]]
        n = len(a_mat)
        if (n == 0 or any(len(r) != n for r in a_mat)
                or any(not math.isfinite(v) for row in a_mat for v in row)):
            return {"passed": False, "domain": "control", "mode": mode, "reason": "Matrix A must be finite, non-empty and square"}

        a_exact = [[Fraction(v) for v in row] for row in a_mat]
        coefficients, ch_residual = _state_characteristic(a_exact)
        try:
            char_poly = [_diagnostic_float(v) for v in coefficients]
            char_status = "finite"
        except OverflowError:
            char_poly, char_status = None, "outside_float_range"
        poles, pole_status, zero_multiplicity = _state_pole_diagnostics(
            coefficients, max(abs(v) for row in a_exact for v in row)
        )

        controllable: Optional[bool] = None
        ctrl_rank: Optional[int] = None
        if "B" in spec:
            b_mat = [[float(v) for v in (row if isinstance(row, list) else [row])] for row in spec["B"]]
            if (len(b_mat) != n or any(len(row) != len(b_mat[0]) for row in b_mat)
                    or any(not math.isfinite(v) for row in b_mat for v in row)):
                return {"passed": False, "domain": "control", "mode": mode,
                        "reason": "Matrix B must be finite and rectangular with one row per state"}
            ctrl_rank = _algebraic_reachable_rank(a_mat, b_mat)
            controllable = ctrl_rank == n

        observable: Optional[bool] = None
        obs_rank: Optional[int] = None
        if "C" in spec:
            c_raw = spec["C"]
            c_mat = [[float(v) for v in c_raw]] if (c_raw and not isinstance(c_raw[0], list)) else [[float(v) for v in r] for r in c_raw]
            if (any(len(row) != n for row in c_mat)
                    or any(not math.isfinite(v) for row in c_mat for v in row)):
                return {"passed": False, "domain": "control", "mode": mode,
                        "reason": "Matrix C must be finite with one column per state"}
            transpose_a = [list(column) for column in zip(*a_mat)]
            transpose_c = [[row[i] for row in c_mat] for i in range(n)]
            obs_rank = _algebraic_reachable_rank(transpose_a, transpose_c)
            observable = obs_rank == n

        identity = [[int(i == j) for j in range(n)] for i in range(n)]
        q = identity
        if "Q" in spec:
            raw = spec["Q"]
            if (not isinstance(raw, (list, tuple)) or len(raw) != n
                    or any(not isinstance(row, (list, tuple)) or len(row) != n for row in raw)):
                return {"passed": False, "domain": "control", "mode": mode,
                        "reason": "Matrix Q must be square with one row and column per state"}
            try:
                q = [[Fraction(float(v)) for v in row] for row in raw]
            except (TypeError, ValueError, OverflowError):
                return {"passed": False, "domain": "control", "mode": mode,
                        "reason": "Matrix Q must contain finite numeric coefficients"}
            if any(q[i][j] != q[j][i] for i in range(n) for j in range(i)):
                return {"passed": False, "domain": "control", "mode": mode,
                        "reason": "Matrix Q must be symmetric"}

        forcings = [identity] if q == identity else [identity, q]
        solutions = _symmetric_lyapunov_solutions(a_exact, forcings, discrete=discrete)
        stability = _lyapunov_certificate(a_exact, identity, solutions[0], discrete=discrete)
        requested = (stability if len(solutions) == 1 else
                     _lyapunov_certificate(a_exact, q, solutions[1], discrete=discrete))
        stable = stability["positive_definite"] is True and stability["residual_exact"] is True
        p_mat, float_status = _lyapunov_float_matrix(solutions[-1][1])

        passed = (ch_residual == 0
                  and stability["residual_exact"] is not False and requested["residual_exact"] is not False)
        res = {
            "passed": passed,
            "domain": "control",
            "mode": mode,
            "char_poly": char_poly,
            "char_poly_exact": [_fraction_text(v) for v in coefficients],
            "char_poly_float_status": char_status,
            "cayley_hamilton_residual": float(ch_residual),
            "cayley_hamilton_residual_exact": _fraction_text(ch_residual),
            "cayley_hamilton_residual_basis": "char_poly_exact",
            ("schur_stable" if discrete else "hurwitz_stable"): stable,
            "controllable": controllable,
            "controllability_rank": ctrl_rank,
            "observable": observable,
            "observability_rank": obs_rank,
            "stability_certificate": stability,
            "lyapunov_solution_status": requested["solution_status"],
            "lyapunov_pd": requested["positive_definite"],
            "lyapunov_P_exact": requested["P_exact"],
            "lyapunov_residual_exact": requested["residual_exact"],
            "lyapunov_float_status": float_status,
            "lyapunov_P": p_mat,
            "eigenvalues_real_imag": poles,
            "eigenvalues_status": pole_status,
            "zero_eigenvalue_multiplicity": zero_multiplicity,
        }
        if discrete:
            spectral = _schur_stable_exact(coefficients)
            res.update(timebase="discrete", sample_time_s=spec.get("sample_time_s"),
                       lyapunov_equation="A^T P A - P = -Q",
                       schur_recursion_stable=spectral)
            if spectral != stable:
                res.update(passed=False, reason="Exact Schur and Stein stability checks disagree")
                return res
            if spec.get("claimed_field") == "sample_time_s":
                res.update(passed=False, reason="sample_time_s is supplied metadata. Claim a computed result.")
                return res
        if ("claimed_value" in spec and spec.get("claimed_field") == "lyapunov_pd"
                and res["lyapunov_pd"] is None):
            res.update(passed=False, reason="No unique symmetric P exists for this definiteness claim")
        else:
            _check_claimed(res, spec, exact_booleans=discrete)
        return res

    return {"passed": False, "domain": "control", "mode": mode, "reason": f"Unknown control mode: {mode}"}


_ICE_SATURATION_COEFFICIENTS = (-5674.5359, 6.3925247, -.0096778430, 6.2215701e-7,
                                2.0747825e-9, -9.4840240e-13, 4.1635019)
_WATER_SATURATION_COEFFICIENTS = (-5800.2206, 1.3914993, -.048640239, 4.1764768e-5,
                                  -1.4452093e-8, 6.5459673)
_DECIMAL_ICE = tuple(Decimal(str(c)) for c in _ICE_SATURATION_COEFFICIENTS)
_DECIMAL_WATER = tuple(Decimal(str(c)) for c in _WATER_SATURATION_COEFFICIENTS)


def _pws_ashrae_pa(t_c, *, ice_below_c=0.0):
    precise = isinstance(t_c, Decimal)
    tk = t_c + (Decimal("273.15") if precise else 273.15)
    if tk <= 0:
        raise ValueError("Temperature below absolute zero")
    logarithm = tk.ln() if precise else math.log(tk)
    if t_c < ice_below_c:
        a, b, c, d, e, f, g = _DECIMAL_ICE if precise else _ICE_SATURATION_COEFFICIENTS
        ln_pws = a/tk + b + c*tk + d*tk**2 + e*tk**3 + f*tk**4 + g*logarithm
    else:
        a, b, c, d, e, f = _DECIMAL_WATER if precise else _WATER_SATURATION_COEFFICIENTS
        ln_pws = a/tk + b + c*tk + d*tk**2 + e*tk**3 + f*logarithm
    return ln_pws.exp() if precise else math.exp(ln_pws)


def _bisect_dew_point(pw_pa: float, lo: float, hi: float, *, ice_below_c=0.) -> float:
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _pws_ashrae_pa(mid, ice_below_c=ice_below_c) < pw_pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _dew_point_from_pw(pw_pa: float, *, ice_below_c=0.) -> Tuple[Optional[float], Optional[str]]:
    """Invert the supported ice or liquid branch of the selected saturation correlation."""
    if not math.isfinite(pw_pa) or pw_pa < 0:
        return None, "invalid_vapor_pressure"
    if pw_pa == 0:
        return None, "dry_no_finite_point"

    # ASHRAE 2017 SI eqs. 5 and 6 have finite, separate temperature ranges.
    # The ice branch ends immediately below the selected switch.
    ice_lo, ice_hi = -100.0, math.nextafter(ice_below_c, -math.inf)
    water_lo, water_hi = ice_below_c, 200.0
    p_ice_lo, p_ice_hi = _pws_ashrae_pa(ice_lo, ice_below_c=ice_below_c), _pws_ashrae_pa(ice_hi, ice_below_c=ice_below_c)
    p_water_lo, p_water_hi = _pws_ashrae_pa(water_lo, ice_below_c=ice_below_c), _pws_ashrae_pa(water_hi, ice_below_c=ice_below_c)
    if pw_pa < p_ice_lo:
        return None, "below_correlation_range"
    if pw_pa > p_water_hi:
        return None, "above_correlation_range"
    in_ice = p_ice_lo <= pw_pa <= p_ice_hi
    in_water = p_water_lo <= pw_pa <= p_water_hi
    if in_ice and in_water:
        return None, "phase_overlap"
    if not (in_ice or in_water):
        return None, "phase_gap"

    if pw_pa == p_ice_lo:
        return ice_lo, None
    if pw_pa == p_water_lo:
        return water_lo, None
    if pw_pa == p_water_hi:
        return water_hi, None

    lo, hi = (ice_lo, ice_hi) if in_ice else (water_lo, water_hi)
    if _pws_ashrae_pa(-80.0, ice_below_c=ice_below_c) <= pw_pa <= _pws_ashrae_pa(150.0, ice_below_c=ice_below_c):
        ordinary = _bisect_dew_point(pw_pa, -80.0, 150.0, ice_below_c=ice_below_c)
        if lo <= ordinary <= hi:
            return ordinary, None
    if in_ice and pw_pa == p_ice_hi:
        return ice_hi, None
    return _bisect_dew_point(pw_pa, lo, hi, ice_below_c=ice_below_c), None


def _signed_w_from_twb(t_db_c: float, t_wb_c: float, p_atm_pa: float, *,
                       ice_below_c: float = 0.0, checked: bool = False) -> float:
    if checked and (not all(map(math.isfinite, (t_db_c, t_wb_c, p_atm_pa))) or p_atm_pa <= 0):
        raise ValueError("Wet-bulb balance requires finite temperatures and finite positive pressure")
    pws_wb = _pws_ashrae_pa(t_wb_c, ice_below_c=ice_below_c)
    if checked and not 0 < pws_wb < p_atm_pa:
        raise ValueError("Wet-bulb saturation pressure must lie strictly between zero and total pressure")
    ws_wb = 0.621945 * pws_wb / (p_atm_pa - pws_wb)
    if t_wb_c >= 0.0:
        num = (2501.0 - 2.326 * t_wb_c) * ws_wb - 1.006 * (t_db_c - t_wb_c)
        den = 2501.0 + 1.86 * t_db_c - 4.186 * t_wb_c
    else:
        num = (2830.0 - 0.24 * t_wb_c) * ws_wb - 1.006 * (t_db_c - t_wb_c)
        den = 2830.0 + 1.86 * t_db_c - 2.1 * t_wb_c
    if checked and not (math.isfinite(den) and den > 0):
        raise ValueError("Wet-bulb enthalpy balance requires a finite positive denominator")
    signed = num / den
    if checked and not math.isfinite(signed):
        raise ValueError("Wet-bulb humidity balance must be finite")
    return signed


def _w_from_twb(t_db_c: float, t_wb_c: float, p_atm_pa: float, *, ice_below_c: float = 0.0):
    """Validate a submitted wet bulb, allowing one same-phase input ULP at dry air."""
    signed = _signed_w_from_twb(t_db_c, t_wb_c, p_atm_pa, ice_below_c=ice_below_c, checked=True)
    if signed > 0:
        return signed, None
    if signed == 0:
        return 0.0, None

    upper = math.nextafter(t_wb_c, math.inf)
    same_phase = (t_wb_c < ice_below_c) == (upper < ice_below_c) and (t_wb_c >= 0) == (upper >= 0)
    if (math.isfinite(upper) and upper > t_wb_c and upper <= t_db_c + 1e-5 and same_phase
            and _signed_w_from_twb(t_db_c, upper, p_atm_pa, ice_below_c=ice_below_c, checked=True) >= 0):
        return 0.0, [t_wb_c, upper]
    raise ValueError("Wet-bulb temperature requires a negative humidity ratio")


def _wet_bulb_from_state(t_db_c: float, w: float, p_atm_pa: float, *, ice_below_c=0.) -> float:
    lo, hi = -60.0, t_db_c
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        # Clipping negative trial ratios would erase the crossing at exactly dry air.
        if _signed_w_from_twb(t_db_c, mid, p_atm_pa, ice_below_c=ice_below_c) < w:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _psychro_state(spec: Dict[str, Any], p_atm_pa: float = 101325.0,
                   *, ice_below_c: float = 0.0) -> Dict[str, Any]:
    t_db = float(spec["t_db_c"])
    pws = _pws_ashrae_pa(t_db, ice_below_c=ice_below_c)
    if pws >= p_atm_pa:
        return {"passed": False, "reason": "Saturation vapor pressure exceeds atmospheric pressure"}

    dry_bracket = None
    if "rh" in spec:
        rh = float(spec["rh"])
        if rh < -1e-9 or rh > 1.0 + 1e-7:
            return {"passed": False, "reason": f"Unphysical relative humidity rh={rh}"}
        rh = min(max(rh, 0.0), 1.0)
        pw = rh * pws
        w = 0.621945 * pw / (p_atm_pa - pw)
    elif "w_kg_per_kgda" in spec:
        w = float(spec["w_kg_per_kgda"])
        if w < 0.0:
            return {"passed": False, "reason": "Negative humidity ratio"}
        pw = p_atm_pa * w / (0.621945 + w)
        rh = pw / pws
        if rh > 1.0 + 1e-4:
            return {"passed": False, "reason": f"Supersaturated humidity ratio w={w} (rh={rh:.4f} > 1)"}
        rh = min(rh, 1.0)
    elif "t_wb_c" in spec:
        t_wb = float(spec["t_wb_c"])
        if t_wb > t_db + 1e-5:
            return {"passed": False, "reason": "Wet-bulb exceeds dry-bulb temperature"}
        try:
            w, dry_bracket = _w_from_twb(t_db, t_wb, p_atm_pa, ice_below_c=ice_below_c)
        except (ValueError, ArithmeticError) as exc:
            return {"passed": False, "reason": f"Infeasible wet-bulb input: {exc}"}
        pw = p_atm_pa * w / (0.621945 + w)
        rh = min(max(pw / pws, 0.0), 1.0)
    elif "t_dp_c" in spec:
        t_dp = float(spec["t_dp_c"])
        if t_dp > t_db + 1e-5:
            return {"passed": False, "reason": "Dew-point exceeds dry-bulb temperature"}
        pw = _pws_ashrae_pa(t_dp, ice_below_c=ice_below_c)
        rh = min(max(pw / pws, 0.0), 1.0)
        w = 0.621945 * pw / (p_atm_pa - pw)
    else:
        return {"passed": False, "reason": "Must specify one of rh, w_kg_per_kgda, t_wb_c, or t_dp_c"}

    h_kj_kgda = 1.006 * t_db + w * (2501.0 + 1.86 * t_db)
    v_m3_kgda = 287.042 * (t_db + 273.15) * (1.0 + 1.607858 * w) / p_atm_pa
    t_dp_calc, t_dp_status = _dew_point_from_pw(pw, ice_below_c=ice_below_c)
    t_wb_calc = _wet_bulb_from_state(t_db, w, p_atm_pa, ice_below_c=ice_below_c)
    ws = 0.621945 * pws / (p_atm_pa - pws)
    deg_sat = w / ws if ws > 0 else 0.0

    state = {
        "passed": True,
        "t_db_c": t_db,
        "rh": rh,
        "pw_pa": pw,
        "pws_pa": pws,
        "w_kg_per_kgda": w,
        "h_kj_per_kgda": h_kj_kgda,
        "v_m3_per_kgda": v_m3_kgda,
        "t_dp_c": t_dp_calc,
        "t_wb_c": t_wb_calc,
        "degree_of_saturation": deg_sat,
    }
    if t_dp_status is not None:
        state["t_dp_status"] = t_dp_status
    if dry_bracket is not None:
        state["wet_bulb_dry_bracket_c"] = dry_bracket
    return state


_FUSION_ENTHALPY_KJ_KG = 333.5


def _ice_enthalpy_kj_kg(temp_c: float) -> float:
    """Ice Ih relative to liquid water at 0 C, approximate from -60 to 0 C."""
    # Integral of a linear ice heat capacity, based on ASHRAE Fundamentals
    # (2025), Psychrometrics, Table 3. Fusion at 0 C is 333.5 kJ/kg.
    return -_FUSION_ENTHALPY_KJ_KG + temp_c * (2.097 + 0.00365 * temp_c)


def _mix_air_streams(spec, pressure, model, states):
    """Dry-air mass, sensible energy and water balances at a common pressure."""
    D = Decimal
    def supplied(value):
        return D.from_float(float(value))

    equilibrium = model == "equilibrium_ice"
    switch = D(".01") if equilibrium else D(0)
    p = supplied(pressure)
    vapor_ratio = D(".621945")
    cp_air, cp_vapor, latent, fusion = D("1.006"), D("1.86"), D(2501), D("333.5")

    def saturation(temp, ice_limit=False):
        pw = _pws_ashrae_pa(temp, ice_below_c=(temp + 1 if ice_limit else switch))
        return vapor_ratio * pw / (p-pw)

    def inlet(row, state):
        temp = supplied(row["t_db_c"])
        if "rh" in row:
            rh = min(1, max(0, supplied(row["rh"])))
            pw = rh * _pws_ashrae_pa(temp, ice_below_c=switch)
            water = vapor_ratio * pw / (p-pw)
        elif "w_kg_per_kgda" in row:
            water = supplied(row["w_kg_per_kgda"])
        elif "t_wb_c" in row:
            wet = supplied(row["t_wb_c"])
            ws = saturation(wet)
            if wet >= 0:
                water = ((latent-D("2.326")*wet)*ws-cp_air*(temp-wet)) / (latent+cp_vapor*temp-D("4.186")*wet)
            else:
                water = ((2830-D(".24")*wet)*ws-cp_air*(temp-wet)) / (2830+cp_vapor*temp-D("2.1")*wet)
            if state["w_kg_per_kgda"] == 0:
                water = D(0)  # The admitted dry-wick boundary.
        else:
            water = saturation(supplied(row["t_dp_c"]))
        if water < 0:
            raise ValueError("Inlet humidity ratio must be nonnegative")
        if "m_da_kg_s" in row:
            mass = supplied(row["m_da_kg_s"])
        else:
            volume = D("287.042")*(temp+D("273.15"))*(1+D("1.607858")*water)/p
            mass = supplied(row["v_dot_m3_s"])/volume
        if not mass.is_finite() or mass <= 0:
            raise ValueError("Non-positive or nonfinite dry-air mass flow")
        return temp, water, mass

    t1, w1, m1 = inlet(spec["stream1"], states[0])
    t2, w2, m2 = inlet(spec["stream2"], states[1])
    mass = m1 + m2
    w = (m1*w1 + m2*w2)/mass if w1 != w2 else w1
    c1, c2 = cp_air + cp_vapor*w1, cp_air + cp_vapor*w2
    sensible = (m1*c1*t1 + m2*c2*t2)/mass
    # Solving for the all-vapor dry bulb directly preserves an isothermal
    # inlet and avoids subtracting latent energy from the total enthalpy.
    t = (m1*c1*t1 + m2*c2*t2)/(m1*c1 + m2*c2) if t1 != t2 else t1
    total_h = sensible + latent*w
    fog = w > saturation(t)
    vapor, ice, liquid = w, D(0), D(0)

    def balance(temp, frozen, ice_limit=False):
        vapor = saturation(temp, ice_limit)
        condensed = w-vapor
        h_cond = (-fusion+temp*(D("2.097")+D(".00365")*temp) if frozen else D("4.186")*temp)
        # Subtract a common 2501 kJ per kg of total water from both sides.
        residual = (cp_air+cp_vapor*vapor)*temp + condensed*(h_cond-latent) - sensible
        return residual, vapor

    if fog:
        hi = max(t1, t2)
        lo = max(D(0), min(t, hi))
        liquid_gap, vapor_zero = balance(D(0), False)
        frozen = hi < 0 or (t < 0 and liquid_gap > 0)
        if frozen and not equilibrium:
            raise ValueError("Fog equilibrium requires an ice or mixed-phase model")
        ice_gap = liquid_gap - fusion*(w-vapor_zero)
        if frozen and hi >= 0 and w >= vapor_zero and ice_gap <= 0 <= liquid_gap:
            t, vapor = D(0), vapor_zero
            # Each phase is determined from its own melting-interval ordinate.
            ice, liquid = liquid_gap/fusion, -ice_gap/fusion
        else:
            if frozen:
                lo, hi = max(D(-60), min(t, hi)), min(D(0), hi)
            if lo > hi or balance(lo, frozen)[0] > 0 or balance(hi, frozen)[0] < 0:
                raise ValueError("Inlets do not bracket a supported condensate equilibrium")
            if (lo < switch <= hi and balance(switch, frozen, ice_limit=True)[0] <= 0
                    < balance(switch, frozen)[0]):
                raise ValueError("Energy balance falls in the saturation-correlation phase gap")
            low_gap, low_vapor = balance(lo, frozen)
            high_gap, high_vapor = balance(hi, frozen)
            # Condensate can be representable even when its temperature rise
            # is below one dry-bulb ulp. Resolve the water difference as well.
            for step in range(4096):
                if (float(lo) == float(hi)
                        and float(w-low_vapor) == float(w-high_vapor)):
                    break
                mid = (lo+hi)/2
                if step % 2 == 0 and low_gap < 0 < high_gap:
                    trial = lo + (hi-lo)*(-low_gap)/(high_gap-low_gap)
                    if lo < trial < hi:
                        mid = trial
                if mid == lo or mid == hi:
                    break
                gap, mid_vapor = balance(mid, frozen)
                if gap == 0:
                    lo = hi = mid
                    break
                if gap < 0:
                    lo, low_gap, low_vapor = mid, gap, mid_vapor
                else:
                    hi, high_gap, high_vapor = mid, gap, mid_vapor
            t = (lo+hi)/2
            vapor = min(w, saturation(t))
            if frozen:
                ice = w-vapor
            else:
                liquid = w-vapor
    rh = (D(1) if fog else min(D(1), p*w/(vapor_ratio+w)/_pws_ashrae_pa(t, ice_below_c=switch)))
    if any(value != 0 and float(value) == 0 for value in (t, ice, liquid)):
        raise ValueError("Nonzero mixed temperature or phase fraction is below the finite output range")
    res = {"m_da_total_kg_s": float(mass), "t_db_mix_c": float(t), "w_mix_kg_per_kgda": float(w),
           "w_vapor_kg_per_kgda": float(vapor), "w_condensate_kg_per_kgda": float(ice+liquid),
           "h_mix_kj_per_kgda": float(total_h), "rh_mix": float(rh), "fog_saturated": fog}
    if not all(math.isfinite(value) for value in res.values()):
        raise ValueError("Mixed stream exceeds the finite output range")
    if equilibrium:
        res.update(condensate_model=model, w_ice_kg_per_kgda=float(ice), w_liquid_kg_per_kgda=float(liquid))
    return res


def _explicit_coil_water(spec):
    """Resolve a declared water product and the state measurements determining its flow."""
    def number(v, name):
        if type(v) not in (int, float) or not math.isfinite(v):
            raise ValueError(f"{name} requires a finite number")
        return v

    phase = spec.get("condensate_phase")
    if phase not in ("ice", "liquid", "mixed") or "condensate_model" in spec:
        raise ValueError("supply one condensate_phase: ice, liquid or mixed")
    if ("m_da_kg_s" in spec) == ("v_dot_in_m3_s" in spec):
        raise ValueError("supply exactly one air mass flow or inlet volume flow")
    flow_key = "m_da_kg_s" if "m_da_kg_s" in spec else "v_dot_in_m3_s"
    pressure = number(spec.get("p_atm_pa", 101325.), "p_atm_pa")
    if number(spec[flow_key], flow_key) <= 0 or pressure <= 0:
        raise ValueError("air flow and pressure must be positive")
    moisture = {"rh", "w_kg_per_kgda", "t_wb_c", "t_dp_c"}
    for key in ("inlet", "outlet"):
        state = spec.get(key)
        if (not isinstance(state, dict) or "t_db_c" not in state
                or len(state.keys() & moisture) != 1 or state.keys() - moisture - {"t_db_c"}):
            raise ValueError(f"{key} requires dry bulb and exactly one humidity measurement")
        for name, v in state.items():
            number(v, name)
        if "rh" in state and not 0 <= state["rh"] <= 1:
            raise ValueError("relative humidity must be in [0,1]")
        if any(state[k] > state["t_db_c"] for k in ("t_wb_c", "t_dp_c") if k in state):
            raise ValueError("wet bulb and dew point cannot exceed dry bulb")
    t = number(spec.get("t_condensate_c", spec["outlet"]["t_db_c"]), "t_condensate_c")
    if phase == "mixed":
        f = number(spec.get("ice_mass_fraction"), "ice_mass_fraction")
        if t != 0 or not 0 <= f <= 1:
            raise ValueError("mixed products require 0 C and a supplied ice mass fraction in [0,1]")
    else:
        if "ice_mass_fraction" in spec:
            raise ValueError("ice_mass_fraction requires a mixed product")
        f = float(phase == "ice")
    if phase == "ice" and not -60 <= t <= 0:
        raise ValueError("ice enthalpy requires -60 to 0 C")
    if f < 1 and (not 0 <= t <= 200 or _pws_ashrae_pa(t) > pressure):
        raise ValueError("liquid product temperature must be between freezing and boiling")
    if phase != "liquid" and spec.get("load_split") == "ashrae_dew_point":
        raise ValueError("ashrae_dew_point requires a liquid drain")
    h = (f*_ice_enthalpy_kj_kg(t) if f else 0.) + (1-f)*4.186*t
    return phase, f, t, h


def verify_psychrometric_process(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "state")).strip().lower()
    water = None
    if mode == "cooling_coil" and spec.keys() & {"condensate_phase", "ice_mass_fraction", "condensate_model"}:
        try:
            water = _explicit_coil_water(spec)
        except (ValueError, ArithmeticError) as exc:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": str(exc)}
    p_atm = float(spec.get("p_atm_pa", 101325.0))

    if mode == "state":
        st = _psychro_state(spec, p_atm)
        st["domain"] = "psychrometrics"
        st["mode"] = mode
        if st.get("passed"):
            _check_claimed(st, spec)
        return st

    if mode == "adiabatic_mixing":
        model = spec.get("condensate_model", "liquid")
        if model not in ("liquid", "equilibrium_ice"):
            return {"passed": False, "domain": "psychrometrics", "mode": mode,
                    "reason": "condensate_model must be liquid or equilibrium_ice"}
        equilibrium = model == "equilibrium_ice"
        # The vapor-pressure correlations meet at the triple point. Use the same
        # switch for both inlet and outlet states of the equilibrium model.
        ice_below = 0.01 if equilibrium else 0.0
        s1 = _psychro_state(spec["stream1"], p_atm, ice_below_c=ice_below)
        s2 = _psychro_state(spec["stream2"], p_atm, ice_below_c=ice_below)
        if not s1.get("passed") or not s2.get("passed"):
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Invalid inlet stream state"}
        try:
            with localcontext() as ctx:
                flow_orders = []
                for row, state in ((spec["stream1"], s1), (spec["stream2"], s2)):
                    if "m_da_kg_s" in row:
                        order = Decimal.from_float(float(row["m_da_kg_s"])).adjusted()
                    else:
                        order = (Decimal.from_float(float(row["v_dot_m3_s"])).adjusted()
                                 - Decimal.from_float(state["v_m3_per_kgda"]).adjusted())
                    flow_orders.append(order)
                ctx.prec = 80 + abs(flow_orders[0]-flow_orders[1])
                res = _mix_air_streams(spec, p_atm, model, (s1, s2))
        except (KeyError, ValueError, ArithmeticError) as exc:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": str(exc)}
        res.update(passed=True, domain="psychrometrics", mode=mode)
        _check_claimed(res, spec)
        return res

    if mode == "cooling_coil":
        try:
            switch = .01 if water else 0.
            s_in = _psychro_state(spec["inlet"], p_atm, ice_below_c=switch)
            s_out = _psychro_state(spec["outlet"], p_atm, ice_below_c=switch)
        except (ValueError, ArithmeticError) as exc:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": str(exc)}
        if not s_in.get("passed") or not s_out.get("passed"):
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": s_in.get("reason") or s_out.get("reason")}
        split = spec.get("load_split", "moist_air_midpoint")
        if split not in ("moist_air_midpoint", "ashrae_dew_point"):
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Unknown cooling-coil load_split"}
        m_da = (float(spec["m_da_kg_s"]) if "m_da_kg_s" in spec
                else float(spec["v_dot_in_m3_s"]) / s_in["v_m3_per_kgda"])
        if not math.isfinite(m_da) or m_da <= 0.0:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Dry-air mass flow must be finite and positive"}
        if s_out["w_kg_per_kgda"] > s_in["w_kg_per_kgda"] + 1e-7 or s_out["t_db_c"] > s_in["t_db_c"] + 1e-7:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Cooling coil cannot increase humidity ratio or dry-bulb"}

        m_cond_kg_s = m_da * max(0.0, s_in["w_kg_per_kgda"] - s_out["w_kg_per_kgda"])
        t_cond = float(spec.get("t_condensate_c", s_out["t_db_c"]))
        if water:
            phase, ice_fraction, t_cond, water_h = water
            if s_out["w_kg_per_kgda"] > s_in["w_kg_per_kgda"] or s_out["t_db_c"] > s_in["t_db_c"]:
                return {"passed": False, "domain": "psychrometrics", "mode": mode,
                        "reason": "A cooling/dehumidifying coil cannot add moisture or heat"}
        else:
            phase, ice_fraction, water_h = "liquid", 0., 4.186*t_cond
        split_fields = {}
        if split == "ashrae_dew_point":
            # ASHRAE HVAC Systems and Equipment (2024), ch. 23, Eq. 38:
            # condensation occurs at the entering-air dew point. Liquid cooling
            # from that point to the drain belongs to the sensible load.
            wi, wo = s_in["w_kg_per_kgda"], s_out["w_kg_per_kgda"]
            ti, to = s_in["t_db_c"], s_out["t_db_c"]
            if not all(map(math.isfinite, (wi, wo, ti, to, t_cond))) or wo > wi or to > ti:
                return {"passed": False, "domain": "psychrometrics", "mode": mode,
                        "reason": "Dew-point split requires finite cooling and nonincreasing humidity"}
            dew = s_in["t_dp_c"] if wi > wo else None
            if wi > wo and (dew is None or dew < 0.0 or not 0.0 <= t_cond <= 200.0
                            or _pws_ashrae_pa(t_cond) > p_atm):
                return {"passed": False, "domain": "psychrometrics", "mode": mode,
                        "reason": "Condensing dew-point split requires a liquid dew point and liquid drain"}
            # Evaluate per kg dry air first so that the intensive SHR does not
            # change when the dry-air flow is rescaled.
            latent = (wi - wo) * (2501.0 + (1.86 - 4.186) * dew) if dew is not None else 0.0
            drain = (wi - wo) * 4.186 * (dew - t_cond) if dew is not None else 0.0
            removed_vapor = (wi - wo) * 1.86 * (ti - dew) if dew is not None else 0.0
            sensible = math.fsum(((1.006 + 1.86 * wo) * (ti - to), removed_vapor, drain))
            total = sensible + latent
            q_sens_kw, q_lat_kw, q_total_kw = m_da * sensible, m_da * latent, m_da * total
            if not all(map(math.isfinite, (q_sens_kw, q_lat_kw, q_total_kw, m_cond_kg_s * 3600.0))):
                return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Unrepresentable cooling-coil rate"}
            shr = sensible / total if total > 0.0 else 1.0
            split_fields = {"load_split": split, "condensation_t_dp_c": dew, "condensate_sensible_kw": m_da * drain}
        elif water:
            wi, wo = s_in["w_kg_per_kgda"], s_out["w_kg_per_kgda"]
            ti, to = s_in["t_db_c"], s_out["t_db_c"]
            sensible = (1.006 + 1.86*(wi+wo)/2)*(ti-to)
            latent = (wi-wo)*(2501. + 1.86*(ti+to)/2 - water_h)
            total = math.fsum((sensible, latent))
            q_sens_kw, q_lat_kw, q_total_kw = m_da*sensible, m_da*latent, m_da*total
            shr = sensible/total if total > 0 else 1.
            if not all(map(math.isfinite, (q_total_kw, q_sens_kw, q_lat_kw, m_cond_kg_s*3600.))):
                return {"passed": False, "domain": "psychrometrics", "mode": mode,
                        "reason": "Unrepresentable cooling-coil rate"}
        else:
            q_total_kw = m_da * (s_in["h_kj_per_kgda"] - s_out["h_kj_per_kgda"]) - m_cond_kg_s * water_h
            cp_moist = 1.006 + 1.86 * 0.5 * (s_in["w_kg_per_kgda"] + s_out["w_kg_per_kgda"])
            q_sens_kw = m_da * cp_moist * (s_in["t_db_c"] - s_out["t_db_c"])
            q_lat_kw = q_total_kw - q_sens_kw
            shr = q_sens_kw / q_total_kw if q_total_kw > 1e-12 else 1.0

        res = {
            "passed": q_total_kw >= -1e-9,
            "domain": "psychrometrics",
            "mode": mode,
            "m_da_kg_s": m_da,
            "condensate_kg_s": m_cond_kg_s,
            "condensate_kg_hr": m_cond_kg_s * 3600.0,
            "condensate_phase": phase, "phase_source": "supplied" if water else "assumed_liquid",
            "ice_mass_fraction": ice_fraction, "water_product_temperature_c": t_cond,
            "water_temperature_source": "supplied" if "t_condensate_c" in spec else "outlet_dry_bulb",
            "water_product_enthalpy_kj_kg": water_h,
            "deposited_ice_kg_s": m_cond_kg_s*ice_fraction,
            "drained_liquid_kg_s": m_cond_kg_s*(1-ice_fraction),
            "inlet_w_kg_per_kgda": s_in["w_kg_per_kgda"],
            "outlet_w_kg_per_kgda": s_out["w_kg_per_kgda"],
            "q_total_kw": q_total_kw,
            "q_sensible_kw": q_sens_kw,
            "q_latent_kw": q_lat_kw,
            "sensible_heat_ratio": shr,
            **split_fields,
            "inlet_h_kj_per_kgda": s_in["h_kj_per_kgda"],
            "outlet_h_kj_per_kgda": s_out["h_kj_per_kgda"],
        }
        _check_claimed(res, spec)
        return res

    return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": f"Unknown psychrometric mode: {mode}"}


def _truss_node_index(value, node_count):
    index = int(value)
    if (isinstance(value, bool) or (not isinstance(value, str) and index != value)
            or not 0 <= index < node_count):
        raise ValueError("Invalid truss node index")
    return index


class _TrussMember:
    """Raw compatibility row and material factors of one represented member."""

    def __init__(self, element, nodes, *, minimum_length=1e-12):
        i = _truss_node_index(element["i"], len(nodes))
        j = _truss_node_index(element["j"], len(nodes))
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        dx, dy = xj - xi, yj - yi
        # Retain the established numerical length. The compatibility factors
        # use differences of accepted coordinates, never rounded unit vectors.
        length = math.hypot(float(xj) - float(xi), float(yj) - float(yi))
        modulus = Fraction(float(element["E_pa"]))
        self.area = Fraction(float(element["A_m2"]))
        if not math.isfinite(length) or length <= minimum_length or modulus <= 0 or self.area <= 0:
            raise ValueError("Degenerate truss member geometry or stiffness")
        self.length = Fraction(length)
        self.ea = modulus * self.area
        self.weight = self.ea / self.length**3
        self.row = [(dof, value) for dof, value in
                    ((2*i, -dx), (2*i+1, -dy), (2*j, dx), (2*j+1, dy)) if value]
        alpha = Fraction(float(element.get("alpha_per_K", 0.0)))
        delta_t = Fraction(float(element.get("dT_K", 0.0)))
        self.thermal_extension = alpha * delta_t * self.length


def _truss_free_basis(rows, free_count):
    """Exact echelon basis of free compatibility rows; no stiffness weights enter."""
    basis = {}
    for entries in rows:
        row = dict(entries)
        while row:
            pivot = min(row)
            coefficient = row[pivot]
            if pivot not in basis:
                basis[pivot] = {j: v / coefficient for j, v in row.items()}
                break
            for j, value in basis[pivot].items():
                remaining = row.get(j, 0) - coefficient * value
                if remaining:
                    row[j] = remaining
                else:
                    row.pop(j, None)
        if len(basis) == free_count:
            break
    return basis


def _truss_free_rank(rows, free_count):
    return len(_truss_free_basis(rows, free_count))


def _truss_solve(members, applied, prescribed, free):
    positions = {d: i for i, d in enumerate(free)}
    rows = [[(positions[d], v) for d, v in member.row if d in positions] for member in members]
    n = len(free)
    if _truss_free_rank(rows, n) < n:
        return None
    gram = [[Fraction(0) for _ in free] for _ in free]
    rhs = [applied[d] for d in free]
    for member, row in zip(members, rows):
        fixed_projection = sum(v * prescribed[d] for d, v in member.row if d in prescribed)
        natural_projection = member.thermal_extension * member.length
        for i, vi in row:
            weighted = member.weight * vi
            rhs[i] += weighted * (natural_projection - fixed_projection)
            for j, vj in row:
                gram[i][j] += weighted * vj

    # The exact full-rank Gram is positive definite, so its leading pivots
    # are positive. No numerical threshold or row regularization is needed.
    for k in range(n):
        for i in range(k+1, n):
            if gram[i][k]:
                factor = gram[i][k] / gram[k][k]
                rhs[i] -= factor * rhs[k]
                for j in range(k+1, n):
                    gram[i][j] -= factor * gram[k][j]
    solution = [Fraction(0)] * n
    for i in reversed(range(n)):
        remaining = rhs[i] - sum(gram[i][j] * solution[j] for j in range(i+1, n))
        solution[i] = remaining / gram[i][i]
    return solution


class _TrussOutputRangeError(ArithmeticError):
    pass


def _truss_render(field, value, index=()):
    if isinstance(value, (list, tuple)):
        return [_truss_render(field, v, index + (i,)) for i, v in enumerate(value)]
    try:
        return _diagnostic_float(value)
    except OverflowError:
        raise _TrussOutputRangeError(field, list(index)) from None


def _truss_response(spec):
    identity = {"domain": "fem", "mode": "truss_2d"}
    try:
        nodes = [(Fraction(float(point[0])), Fraction(float(point[1]))) for point in spec["nodes"]]
        members = [_TrussMember(element, nodes) for element in spec["elements"]]
        applied = [Fraction(0)] * (2 * len(nodes))
        for load in spec.get("loads", []):
            i = _truss_node_index(load["node"], len(nodes))
            applied[2*i] += Fraction(float(load.get("fx_N", 0.0)))
            applied[2*i+1] += Fraction(float(load.get("fy_N", 0.0)))
        prescribed = {}
        for support in spec.get("supports", []):
            i = _truss_node_index(support["node"], len(nodes))
            for axis, (flag, field) in enumerate((("fix_x", "ux_m"), ("fix_y", "uy_m"))):
                if support.get(flag, False):
                    dof = 2*i + axis
                    value = Fraction(float(support.get(field, 0.0)))
                    if dof in prescribed and prescribed[dof] != value:
                        raise ValueError(f"Conflicting prescribed {field} at truss node {i}")
                    prescribed[dof] = value
    except (IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {**identity, "passed": False, "reason": f"Invalid truss input: {exc}"}

    free = [d for d in range(len(applied)) if d not in prescribed]
    solution = _truss_solve(members, applied, prescribed, free)
    if solution is None:
        return {**identity, "passed": False,
                "reason": "Singular stiffness matrix (kinematic mechanism or insufficient supports)"}
    u = [prescribed.get(d, 0) for d in range(len(applied))]
    for d, value in zip(free, solution):
        u[d] = value

    internal = [Fraction(0)] * len(applied)
    forces, stresses = [], []
    energy = thermal_work = Fraction(0)
    for member in members:
        extension = sum(v * u[d] for d, v in member.row) / member.length
        force = member.ea * (extension - member.thermal_extension) / member.length
        forces.append(force)
        stresses.append(force / member.area)
        energy += force**2 * member.length / (2 * member.ea)
        thermal_work += force * member.thermal_extension
        for d, v in member.row:
            internal[d] += v * force / member.length

    reactions = [total - applied_force for total, applied_force in zip(internal, applied)]
    fx, fy = sum(internal[::2]), sum(internal[1::2])
    moment = sum(x * internal[2*i+1] - y * internal[2*i] for i, (x, y) in enumerate(nodes))
    free_equilibrium = all(reactions[d] == 0 for d in free)
    work = sum(force * value for force, value in zip(applied, u)) * Fraction(1, 2)
    member_work = sum(total * value for total, value in zip(internal, u)) - thermal_work
    passed = free_equilibrium and fx == fy == moment == 0 and member_work == 2 * energy

    fields = {
        "displacements_xy_m": list(zip(u[::2], u[1::2])),
        "axial_forces_N": forces,
        "axial_stresses_Pa": stresses,
        "reactions_xy_N": list(zip(reactions[::2], reactions[1::2])),
        "strain_energy_J": energy,
        "external_work_J": work,
        "equilibrium_residual_N": max(abs(fx), abs(fy)),
    }
    try:
        rendered = {key: _truss_render(key, value) for key, value in fields.items()}
    except _TrussOutputRangeError as exc:
        field, index = exc.args
        return {**identity, "passed": False, "response_status": "outside_float_range",
                "range_field": field, "range_index": index,
                "reason": f"Truss output {field} at {index} is outside the nonzero binary64 range"}
    result = {**identity, "passed": passed, **rendered,
              "equilibrium_residual_basis": "represented_member_factors_before_float_rounding"}
    _check_claimed(result, spec)
    return result


def _beam_system(rows, size):
    """Eliminate local compatibility and equilibrium equations in node order."""
    pivots = {}
    for row, rhs in rows:
        row = {index: Fraction(coef) for index, coef in row.items() if coef}
        while row:
            column = min(row)
            coefficient = row.pop(column)
            if column not in pivots:
                pivots[column] = ({j: value / coefficient for j, value in row.items()}, rhs / coefficient)
                break
            other, value = pivots[column]
            rhs -= coefficient * value
            for j, multiplier in other.items():
                term = row.get(j, 0) - coefficient * multiplier
                if term:
                    row[j] = term
                else:
                    row.pop(j, None)
        else:
            if rhs:
                return None
    if len(pivots) != size:
        return None
    states = [Fraction(0)] * size
    for i in reversed(range(size)):
        row, value = pivots[i]
        states[i] = value - sum(coefficient * states[j] for j, coefficient in row.items())
    return states


def _beam_response(spec):
    identity = {"domain": "fem", "mode": "euler_bernoulli_beam_1d"}
    try:
        # The exact equations operate on the represented finite input values.
        def number(value):
            return Fraction(float(value))

        positions = [number(x) for x in spec["x_nodes_m"]]
        n = len(positions)
        if n < 2 or any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError("x_nodes_m must be strictly increasing with >= 2 nodes")
        default_ei = spec.get("EI_Nm2", 1.0)
        default_load = spec.get("q_uniform_N_m", 0.0)
        records = spec.get("elements") or [{} for _ in range(n - 1)]
        if len(records) != n - 1:
            raise ValueError("Provide one element record per beam span")
        loads = [[Fraction(0), Fraction(0)] for _ in positions]
        fixed = {}

        def node(record):
            value = record["node"]
            index = int(value)
            if (
                isinstance(value, bool)
                or (not isinstance(value, str) and index != value)
                or not 0 <= index < n
            ):
                raise ValueError("Invalid beam node index")
            return index

        for load in spec.get("nodal_loads", []):
            at = node(load)
            loads[at][0] += number(load.get("fy_N", 0))
            loads[at][1] += number(load.get("mz_Nm", 0))
        for support in spec.get("supports", []):
            at = node(support)
            for axis, (flag, key) in enumerate((("fix_w", "w_m"), ("fix_theta", "theta_rad"))):
                if support.get(flag):
                    value = number(support.get(key, 0))
                    if (at, axis) in fixed and fixed[at, axis] != value:
                        raise ValueError("Conflicting beam support values")
                    fixed[at, axis] = value

        # Each span owns its left moment M = EI*w'' and shear V = M'.
        # Its adjacent nodes share deflection and slope, even across an EI jump.
        rows, spans = [], []
        end_force = ({}, Fraction(0))
        end_moment = ({}, Fraction(0))
        for i in range(n):
            k = 4 * i
            force, force_offset = end_force
            moment, moment_offset = end_moment
            if i < n - 1:
                force[k + 3] = 1
                moment[k + 2] = -1
            for axis, (terms, offset) in enumerate(((force, force_offset), (moment, moment_offset))):
                if (i, axis) in fixed:
                    rows.append(({k + axis: 1}, fixed[i, axis]))
                else:
                    rows.append((terms, loads[i][axis] - offset))
            if i == n - 1:
                break

            length = positions[i + 1] - positions[i]
            record = records[i]
            ei = number(record.get("EI_Nm2", default_ei))
            if ei <= 0:
                raise ValueError("Non-positive flexural rigidity EI_Nm2")
            if "q_start_N_m" in record or "q_end_N_m" in record:
                if not {"q_start_N_m", "q_end_N_m"} <= record.keys() or "q_N_m" in record:
                    raise ValueError("Give both endpoint loads or one uniform element load")
                q0, q1 = number(record["q_start_N_m"]), number(record["q_end_N_m"])
            else:
                q0 = q1 = number(record.get("q_N_m", default_load))
            spans.append((length, ei, q0, q1))
            # Integrate EI*w'''' = q0 + (q1-q0)*x/L once through the span.
            rows.append(
                (
                    {
                        k + 4: 1,
                        k: -1,
                        k + 1: -length,
                        k + 2: -(length**2) / (2 * ei),
                        k + 3: -(length**3) / (6 * ei),
                    },
                    length**4 * (4 * q0 + q1) / (120 * ei),
                )
            )
            rows.append(
                (
                    {k + 5: 1, k + 1: -1, k + 2: -length / ei, k + 3: -(length**2) / (2 * ei)},
                    length**3 * (3 * q0 + q1) / (24 * ei),
                )
            )
            end_force = ({k + 3: -1}, -length * (q0 + q1) / 2)
            end_moment = ({k + 2: 1, k + 3: length}, length**2 * (2 * q0 + q1) / 6)

        u = _beam_system(rows, 4 * n - 2)
        if u is None:
            raise ValueError("Free beam has an unrestrained rigid motion or incompatible constraints")

        reactions = [[-force, -moment] for force, moment in loads]
        midspans = []
        strain = external = Fraction(0)
        for i, (length, ei, q0, q1) in enumerate(spans):
            w, theta, m, v = u[4 * i : 4 * i + 4]
            # Coefficients in t = x/L. Integrate curvature energy independently
            # of external work, including distributed loads between nodes.
            bending = [m, v * length, q0 * length**2 / 2, (q1 - q0) * length**2 / 6]
            displacement = [w, theta * length] + [
                value * length**2 / (ei * (j + 1) * (j + 2)) for j, value in enumerate(bending)
            ]
            midspans.append(sum(value / 2**j for j, value in enumerate(displacement)))
            strain += (
                length
                / (2 * ei)
                * sum(a * b / (j + k + 1) for j, a in enumerate(bending) for k, b in enumerate(bending))
            )
            external += (
                length
                / 2
                * sum(value * (q0 / (j + 1) + (q1 - q0) / (j + 2)) for j, value in enumerate(displacement))
            )
            reactions[i][0] += v
            reactions[i][1] -= m
            reactions[i + 1][0] -= v + length * (q0 + q1) / 2
            reactions[i + 1][1] += sum(bending)
        for i, (force, moment) in enumerate(loads):
            external += (force * u[4 * i] + moment * u[4 * i + 1]) / 2
        restraint = sum((reactions[i][axis] * value / 2 for (i, axis), value in fixed.items()), Fraction(0))

        free_equilibrium = all(value == 0 for i, pair in enumerate(reactions)
                               for axis, value in enumerate(pair) if (i, axis) not in fixed)

        def render(value):
            result = float(value)
            if value and result == 0:
                raise OverflowError("Nonzero beam result underflows binary64")
            return result

        result = {
            **identity,
            "passed": free_equilibrium and strain == external + restraint,
            "deflections_m": [render(u[4 * i]) for i in range(n)],
            "rotations_rad": [render(u[4 * i + 1]) for i in range(n)],
            "element_midspan_deflections_m": list(map(render, midspans)),
            "tip_deflection_m": render(u[-2]),
            "reactions_V_M": [[render(f), render(m)] for f, m in reactions],
            "strain_energy_J": render(strain),
            "external_work_J": render(external),
            "support_work_J": render(restraint),
        }
        _check_claimed(result, spec)
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {**identity, "passed": False, "reason": str(exc)}


def _cst_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite JSON number")
    return Fraction(value)


def _cst_index(value, count):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError("mesh index outside supplied nodes")
    return value


def _cst_cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def _cst_mesh(nodes, elements):
    """Require a conforming mesh: interiors disjoint, shared boundaries match by node."""
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate mesh coordinates")
    triangles = []
    for row in elements:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("each element must have exactly three node indices")
        a, b, c = (_cst_index(i, len(nodes)) for i in row)
        det = _cst_cross(nodes[a], nodes[b], nodes[c])
        if not det:
            raise ValueError("degenerate triangle")
        triangles.append((a, b, c) if det > 0 else (a, c, b))
    if set(i for t in triangles for i in t) != set(range(len(nodes))):
        raise ValueError("unreferenced mesh node")
    for i, first in enumerate(triangles):
        for second in triangles[:i]:
            separated = False
            for face, other in ((first, second), (second, first)):
                for j in range(3):
                    a, b = nodes[face[j]], nodes[face[(j+1) % 3]]
                    sides = [_cst_cross(a, b, nodes[k]) for k in other]
                    if max(sides) <= 0:
                        separated = True
                    for k, side in zip(other, sides):
                        if k not in face and side == 0:
                            x, y = nodes[k]
                            if ((x-a[0])*(x-b[0]) + (y-a[1])*(y-b[1])) < 0:
                                raise ValueError("nonconforming node on triangle edge")
            if not separated:
                raise ValueError("overlapping triangle interiors")
    return triangles


def _q4_mesh(nodes, elements):
    quads = []
    for row in elements:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("each Q4 element must have four consecutive boundary nodes")
        ids = tuple(_cst_index(i, len(nodes)) for i in row)
        turns = [_cst_cross(nodes[ids[j-1]], nodes[ids[j]], nodes[ids[(j+1) % 4]])
                 for j in range(4)]
        if all(value < 0 for value in turns):
            ids = (ids[0], ids[3], ids[2], ids[1])
        elif not all(value > 0 for value in turns):
            raise ValueError("folded or degenerate bilinear Q4 mapping")
        quads.append(ids)
    # Convex planar quads are exactly the union of two triangles. The
    # triangulation is used only for overlap and hanging-node checks.
    _cst_mesh(nodes, [tri for a,b,c,d in quads for tri in ((a,b,c), (a,c,d))])
    return quads


def _q4_gradient_numerators(points, xi, eta):
    """Return det J and det J times each physical shape gradient, affine in xi,eta."""
    signs = ((-1,-1), (1,-1), (1,1), (-1,1))
    dxi = [Fraction(sx*(1 + sy*eta), 4) for sx,sy in signs]
    deta = [Fraction(sy*(1 + sx*xi), 4) for sx,sy in signs]
    x_xi, y_xi = (sum(p[k]*d for p,d in zip(points,dxi)) for k in (0,1))
    x_eta, y_eta = (sum(p[k]*d for p,d in zip(points,deta)) for k in (0,1))
    det = x_xi*y_eta - y_xi*x_eta
    columns = []
    for dx, de in zip(dxi, deta):
        gx, gy = y_eta*dx - y_xi*de, x_xi*de - x_eta*dx
        columns.extend(((gx, 0, gy), (0, gy, gx)))
    return det, columns


def _q4_pairs(nodes, quad, thickness):
    """Pair opposite 2x2 Gauss points and sum their conjugates over the rationals."""
    points = [nodes[i] for i in quad]
    det, mids = _q4_gradient_numerators(points, 0, 0)
    dofs = [j for i in quad for j in (2*i, 2*i+1)]
    center = [(j, tuple(v/det for v in mid)) for j, mid in zip(dofs, mids)]
    pairs = []
    for sign in (-1, 1):
        corner_det, corners = _q4_gradient_numerators(points, 1, sign)
        slope_det = corner_det - det
        # At (r,sign*r) and (-r,-sign*r), r^2=1/3. After dividing by
        # det +/- slope_det*r, the three coefficients integrate 1,r,r^2.
        denom = det*det - slope_det*slope_det/3
        w0 = 2*thickness*det/denom
        w1 = -2*thickness*slope_det/(3*denom)
        cols = [(j, mid, tuple(x-y for x,y in zip(corner, mid)))
                for j,mid,corner in zip(dofs,mids,corners)]
        pairs.append(((w0, w1, w0/3), cols))
    return center, pairs


def _q4_pair_work(weights, a, slope_a, stress_b, slope_stress_b):
    w0, w1, w2 = weights
    return sum((w0*x+w1*y)*s + (w1*x+w2*y)*t
               for x,y,s,t in zip(a,slope_a,stress_b,slope_stress_b))


def _plane_mesh_response(spec):
    """Small homogeneous, isotropic, small-strain meshes with nodal forces in SI."""
    mode = spec["mode"].strip().lower()
    identity = {"domain": "fem", "mode": mode}
    try:
        allowed = {"domain", "mode", "kinematics", "nodes", "elements", "loads", "supports",
                   "E_Pa", "nu", "thickness_m", "claimed_field", "claimed_value", "rtol", "atol"}
        if set(spec) - allowed:
            raise ValueError("unsupported plane elasticity input: " + ", ".join(sorted(set(spec)-allowed)))
        kin = spec["kinematics"]
        e, nu, t = (_cst_number(spec[k]) for k in ("E_Pa", "nu", "thickness_m"))
        if kin not in ("plane_stress", "plane_strain") or e <= 0 or not -1 < nu < Fraction(1, 2) or t <= 0:
            raise ValueError("invalid isotropic material, thickness or plane assumption")
        if not 3 <= len(spec["nodes"]) <= 64 or not 1 <= len(spec["elements"]) <= 128:
            raise ValueError("requires 3 to 64 nodes and 1 to 128 elements")
        nodes = []
        for point in spec["nodes"]:
            if len(point) != 2:
                raise ValueError("each node requires exactly two coordinates in metres")
            nodes.append(tuple(_cst_number(x) for x in point))
        if mode == "q4_2d":
            triangles, quads = [], _q4_mesh(nodes, spec["elements"])
        else:
            triangles, quads = _cst_mesh(nodes, spec["elements"]), []
        loads = [Fraction(0)] * (2 * len(nodes))
        prescribed = {}
        for row in spec.get("loads", []):
            if set(row) - {"node", "fx_N", "fy_N"} or not ({"fx_N", "fy_N"} & set(row)):
                raise ValueError("nodal load requires fx_N or fy_N")
            node = _cst_index(row["node"], len(nodes))
            for j, field in enumerate(("fx_N", "fy_N")):
                loads[2*node+j] += _cst_number(row.get(field, 0))
        for row in spec.get("supports", []):
            if set(row) - {"node", "ux_m", "uy_m"} or not ({"ux_m", "uy_m"} & set(row)):
                raise ValueError("prescribe ux_m or uy_m; omit a free component")
            node = _cst_index(row["node"], len(nodes))
            for j, field in enumerate(("ux_m", "uy_m")):
                if field in row:
                    dof, value = 2*node+j, _cst_number(row[field])
                    if dof in prescribed and prescribed[dof] != value:
                        raise ValueError("conflicting prescribed displacement")
                    prescribed[dof] = value
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        return {**identity, "passed": False, "reason": str(exc)}

    shear = e/(2*(1+nu))
    coupling = e*nu/(1-nu*nu) if kin == "plane_stress" else e*nu/((1+nu)*(1-2*nu))
    diagonal = 2*shear + coupling
    def stress(strain):
        ex, ey, gamma = strain
        return (diagonal*ex + coupling*ey, coupling*ex + diagonal*ey, shear*gamma)

    size = len(loads)
    stiffness = [{} for _ in loads]
    blocks = []
    for tri in triangles:
        p = [nodes[i] for i in tri]
        det = _cst_cross(*p)
        volume = t*det/2
        columns = []
        for j, i in enumerate(tri):
            a, b = p[(j+1) % 3], p[(j+2) % 3]
            dx, dy = (a[1]-b[1])/det, (b[0]-a[0])/det
            columns.extend(((2*i, (dx, 0, dy)), (2*i+1, (0, dy, dx))))
        for j, bj in columns:
            db = stress(bj)
            for i, bi in columns:
                kij = volume * sum(x*y for x,y in zip(bi, db))
                stiffness[i][j] = stiffness[i].get(j, 0) + kij
        blocks.append((volume, columns))
    quad_blocks = []
    for quad in quads:
        center, pairs = _q4_pairs(nodes, quad, t)
        for weights, columns in pairs:
            for j, mid_j, slope_j in columns:
                d_mid, d_slope = stress(mid_j), stress(slope_j)
                for i, mid_i, slope_i in columns:
                    kij = _q4_pair_work(weights, mid_i, slope_i, d_mid, d_slope)
                    stiffness[i][j] = stiffness[i].get(j, 0) + kij
        quad_blocks.append((center, pairs))

    equations = [({i: 1}, prescribed[i]) if i in prescribed else (stiffness[i], loads[i])
                 for i in range(size)]
    u = _beam_system(equations, size)
    if u is None:
        return {**identity, "passed": False, "reason": "singular constrained plane elasticity system"}

    internal = [Fraction(0)] * size
    strains, stresses, out_of_plane = [], [], []
    energy = Fraction(0)
    for volume, columns in blocks:
        strain = [sum(u[j]*b[k] for j, b in columns) for k in range(3)]
        sigma = stress(strain)
        for j, b in columns:
            internal[j] += volume * sum(x*y for x, y in zip(b, sigma))
        energy += volume * sum(x*y for x, y in zip(strain, sigma))/2
        strains.append(strain)
        stresses.append(sigma)
        out_of_plane.append(0 if kin == "plane_stress" else coupling*(strain[0]+strain[1]))
    for center, pairs in quad_blocks:
        center_strain = [sum(u[j]*b[k] for j,b in center) for k in range(3)]
        strains.append(center_strain)
        stresses.append(stress(center_strain))
        out_of_plane.append(0 if kin == "plane_stress" else coupling*(center_strain[0]+center_strain[1]))
        for weights, columns in pairs:
            mid = [sum(u[j]*b[k] for j,b,_ in columns) for k in range(3)]
            slope = [sum(u[j]*b[k] for j,_,b in columns) for k in range(3)]
            d_mid, d_slope = stress(mid), stress(slope)
            energy += _q4_pair_work(weights, mid, slope, d_mid, d_slope)/2
            for j, b, db in columns:
                internal[j] += _q4_pair_work(weights, b, db, d_mid, d_slope)

    reactions = [f-p for f,p in zip(internal, loads)]
    force_work = sum(f*d for f,d in zip(loads,u))/2
    support_work = sum(reactions[i]*d for i,d in prescribed.items())/2
    moment = sum(x*internal[2*i+1]-y*internal[2*i] for i,(x,y) in enumerate(nodes))
    verified = (all(reactions[i] == 0 for i in range(size) if i not in prescribed)
                and sum(internal[::2]) == sum(internal[1::2]) == moment == 0
                and energy == force_work+support_work)
    fields = {"displacements_xy_m": list(zip(u[::2], u[1::2])),
              "reactions_xy_N": list(zip(reactions[::2], reactions[1::2])),
              "element_strain_xx_yy_gamma_xy": strains, "element_stress_xx_yy_xy_Pa": stresses,
              "element_sigma_zz_Pa": out_of_plane, "strain_energy_J": energy,
              "nodal_load_work_J": force_work, "prescribed_support_work_J": support_work}
    try:
        rendered = {key: _truss_render(key, value) for key, value in fields.items()}
    except _TrussOutputRangeError as exc:
        field, index = exc.args
        return {**identity, "passed": False, "reason": "response outside binary64 range",
                "range_field": field, "range_index": index}
    result = {**identity, "kinematics": kin, "passed": verified, **rendered}
    if mode == "q4_2d":
        result.update(integration="2x2_gauss", element_stress_strain_location="natural_center")
    _check_claimed(result, spec, exact_booleans=True)
    return result


def verify_fem_structure(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "truss_2d")).strip().lower()

    if mode in ("cst_2d", "q4_2d"):
        return _plane_mesh_response(spec)

    if mode == "truss_2d":
        return _truss_response(spec)

    if mode == "euler_bernoulli_beam_1d":
        return _beam_response(spec)

    if mode == "continuous_beam_multispan":
        from hamon_multispan_beam import solve_multispan

        try:
            result = solve_multispan({key: value for key, value in spec.items() if key != "domain"})
        except (KeyError, IndexError, TypeError, ValueError, ArithmeticError) as exc:
            return {"passed": False, "domain": "fem", "mode": mode, "reason": str(exc)}
        return {"passed": result["dual_verified"], "domain": "fem", "mode": mode, **result}

    return {"passed": False, "domain": "fem", "mode": mode, "reason": f"Unknown FEM mode: {mode}"}


def verify_wave5_engineering(spec: Dict[str, Any]) -> Dict[str, Any]:
    domain = str(spec.get("domain", "")).strip().lower()
    if domain in ("pde", "heat_pde", "fin"):
        return verify_pde_system(spec)
    if domain in ("control", "routh", "bode", "lti"):
        return verify_control_system(spec)
    if domain in ("psychrometrics", "psychrometric", "hvac_psychro"):
        return verify_psychrometric_process(spec)
    if domain in ("fem", "truss", "beam_fem", "structural_fem"):
        return verify_fem_structure(spec)
    mode = str(spec.get("mode", "")).strip().lower()
    if mode in ("convective_fin_1d", "heat_1d_transient", "laplace_2d_rect"):
        return verify_pde_system(spec)
    if mode in ("transfer_function", "state_space"):
        return verify_control_system(spec)
    if mode in ("state", "adiabatic_mixing", "cooling_coil"):
        return verify_psychrometric_process(spec)
    if mode in ("cst_2d", "q4_2d", "truss_2d", "euler_bernoulli_beam_1d", "continuous_beam_multispan"):
        return verify_fem_structure(spec)
    return {"passed": False, "reason": f"Unrecognized Wave-5 engineering domain/mode: domain={domain!r} mode={mode!r}"}


_WAVE5_KEYWORDS = {
    "pde": ("convective fin", "adiabatic tip", "fin efficiency", "heat equation", "crank-nicolson", "2d laplace"),
    "control": ("routh-hurwitz", "gain margin", "phase margin", "controllability matrix", "lyapunov equation", "hurwitz"),
    "psychrometrics": ("psychrometric", "humidity ratio", "wet-bulb", "dew-point", "cooling coil", "sensible heat ratio", "ashrae moist air"),
    "fem": ("stiffness matrix", "indeterminate truss", "pin-jointed truss", "hermite beam", "clapeyron"),
}


def detect_wave5_domain(prompt: str, discipline: str = "") -> Optional[str]:
    text = f"{discipline} {prompt}".lower()
    for dom, phrases in _WAVE5_KEYWORDS.items():
        if any(p in text for p in phrases):
            return dom
    return None


def presolve_wave5_prompt(prompt: str, discipline: str = "") -> Optional[Dict[str, Any]]:
    m = re.search(r"```wave5_spec\s*(\{.*?\})\s*```", prompt, flags=re.DOTALL)
    if m:
        try:
            spec = json.loads(m.group(1))
            res = verify_wave5_engineering(spec)
            if res.get("passed"):
                return res
        except Exception:
            return None
    dom = detect_wave5_domain(prompt, discipline)
    if dom is None:
        return None
    return {"domain": dom, "detected": True, "verifier": "hamon_pde_control_psychro_fem"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Wave-5 PDE/Control/Psychrometric/FEM Verifier")
    parser.add_argument("--json", dest="json_payload", default=None, help="JSON spec string")
    args = parser.parse_args(argv)
    raw = args.json_payload if args.json_payload is not None else sys.stdin.read()
    spec = json.loads(raw)
    out = verify_wave5_engineering(spec)
    sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    return 0 if out.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

