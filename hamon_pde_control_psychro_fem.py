"""Wave-5 Deterministic Engineering Verifiers: PDE, Control, Psychrometrics & 1D/2D FEM.

Self-contained zero-external-dependency (Python stdlib math/cmath) symbolic and
numerical cross-verification suite for Physical Engineering's Last Exam (PLE):
  1. PDE (`verify_pde_system`):
     - 1D convective extended-surface fin (adiabatic, convective, prescribed-tip, infinite)
       exact hyperbolic analytical solution cross-checked against 400-cell FVM Thomas solver.
     - 1D transient heat diffusion exact Fourier eigenmode series cross-checked against
       second-order unconditionally stable Crank-Nicolson solver.
     - 2D steady Laplace equation on rectangular domain exact harmonic eigenseries
       cross-checked against Gauss-Seidel SOR finite-difference solver.
  2. Control (`verify_control_system`):
     - Continuous-time LTI transfer-function Routh-Hurwitz stability table (with epsilon
       pivot regularization), companion/Durand-Kerner polished complex poles,
       exact Bode gain/phase crossover frequencies, Gain Margin (dB), Phase Margin (deg),
       unity-feedback closed-loop stability, and static DC gain.
     - State-space Faddeev-LeVerrier characteristic polynomial (M_1 = I), Cayley-Hamilton
       residual gate, Kalman controllability/observability matrix ranks, and continuous
       Lyapunov equation A^T P + P A = -Q Sylvester positive-definiteness check.
  3. Psychrometrics (`verify_psychrometric_process`):
     - ASHRAE Fundamentals (Hyland-Wexler 1983) saturation vapor pressure over ice & liquid,
       exact moist-air state inversion (from RH, wet-bulb, dew-point, or humidity ratio),
       2-stream adiabatic moist-air mixing conservation, and cooling/dehumidification coil
       mass & energy balance (condensate rate, total kW load, sensible heat ratio SHR).
     - Adiabatic mixing accepts condensate_model="equilibrium_ice" to resolve ice down
       to -60 C and coexisting ice and liquid at 0 C. The default is liquid condensate.
  4. Structural FEM (`verify_fem_structure`):
     - Arbitrary 2D pin-jointed determinate & statically indeterminate truss direct-stiffness
       solver with thermal strain support, kinematic mechanism detection, nodal reaction
       equilibrium, and Clapeyron work-strain-energy theorem verification.
     - 1D multi-element Euler-Bernoulli cubic Hermite beam solver with consistent distributed
       load vectors, shear/moment recovery, and Clapeyron strain-energy gate.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import re
import sys
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


def _solve_linear(a_in: Sequence[Sequence[float]], b_in: Sequence[float], tol: float = 1e-12) -> Optional[List[float]]:
    n = len(a_in)
    if n == 0:
        return []
    aug = [list(map(float, a_in[i])) + [float(b_in[i])] for i in range(n)]
    scale = max((abs(aug[i][j]) for i in range(n) for j in range(n)), default=0.0)
    if scale <= 0.0:
        return None
    piv_tol = tol * scale
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) <= piv_tol:
            return None
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        diag = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / diag
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x


def _matrix_rank(mat: Sequence[Sequence[float]], tol: float = 1e-9) -> int:
    if not mat or not mat[0]:
        return 0
    rows = len(mat)
    cols = len(mat[0])
    work = [list(map(float, r)) for r in mat]
    scale = max((abs(v) for r in work for v in r), default=0.0)
    if scale == 0.0:
        return 0
    eps = tol * scale
    rank = 0
    r_idx = 0
    for c_idx in range(cols):
        if r_idx >= rows:
            break
        piv = max(range(r_idx, rows), key=lambda r: abs(work[r][c_idx]))
        if abs(work[piv][c_idx]) <= eps:
            continue
        work[r_idx], work[piv] = work[piv], work[r_idx]
        diag = work[r_idx][c_idx]
        for r in range(r_idx + 1, rows):
            f = work[r][c_idx] / diag
            for c in range(c_idx, cols):
                work[r][c] -= f * work[r_idx][c]
        rank += 1
        r_idx += 1
    return rank


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


def _check_claimed(result: Dict[str, Any], spec: Dict[str, Any]) -> None:
    if "claimed_value" in spec and spec.get("claimed_field"):
        field = str(spec["claimed_field"])
        claimed = float(spec["claimed_value"])
        actual = float(result[field])
        rtol = float(spec.get("rtol", 1e-2))
        atol = float(spec.get("atol", 1e-4))
        err = abs(actual - claimed)
        limit = max(atol, rtol * max(abs(actual), 1e-12))
        result["claim_check"] = {
            "field": field,
            "claimed": claimed,
            "actual": actual,
            "abs_error": err,
            "within_tolerance": err <= limit,
        }
        if err > limit:
            result["passed"] = False
            result["reason"] = f"Claimed {field}={claimed} deviates from verified {actual:.6g} (err={err:.4g} > {limit:.4g})"


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
    scaled_time = alpha * time / length ** 2
    exact_decay = [math.exp(-scaled_time * (n * math.pi) ** 2) for n, _ in harmonics]
    for _ in range(16):
        angles = [math.pi * n / cells for n, _ in harmonics]
        # These are the eigenvalues of the centered second difference, not those
        # of the continuous Laplacian. Each CN time step applies its Cayley factor.
        rates = [scaled_time * (2 * cells * math.sin(angle / 2)) ** 2 for angle in angles]
        steps = max(minimum_steps, math.ceil(max(rates)))
        ratios = [rate / (2 * steps) for rate in rates]
        numerical_decay = [
            math.exp(steps * (math.log1p(-ratio) - math.log1p(ratio)))
            for ratio in ratios
        ]
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

    location = position / length * cells
    left = min(int(location), cells - 1)
    fraction = location - left

    def node_value(n, j):
        # Dirichlet endpoints and integral periods are exactly zero.
        phase = (n * j) % (2 * cells)
        if phase == 0 or phase == cells:
            return 0.0
        return math.sin(math.pi * phase / cells)

    u_numeric = math.fsum(
        amplitude * decay * ((1 - fraction) * node_value(n, left) + fraction * node_value(n, left + 1))
        for (n, amplitude), decay in zip(harmonics, numerical_decay)
    )
    u_exact = 0.0 if position == 0 or position == length else math.fsum(
        amplitude * decay * math.sin(math.pi * math.remainder(n * position / length, 2.0))
        for (n, amplitude), decay in zip(harmonics, exact_decay)
    )
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

        if min(k, ac, perim, h, length) <= 0.0 or not (0.0 <= x_eval <= length + 1e-12):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Non-positive fin geometry/property or out-of-range x_eval_m"}

        theta_b = t_b - t_inf
        m = math.sqrt((h * perim) / (k * ac))
        m_len = m * length
        cap_m = math.sqrt(h * perim * k * ac) * theta_b

        if tip_bc == "adiabatic":
            theta_x = theta_b * math.cosh(m * (length - x_eval)) / math.cosh(m_len)
            theta_tip = theta_b / math.cosh(m_len)
            q_fin = cap_m * math.tanh(m_len)
            h_tip = 0.0
        elif tip_bc == "convective":
            h_tip = float(spec.get("h_tip_w_m2k", h))
            if h_tip < 0.0:
                return {"passed": False, "domain": "pde", "mode": mode, "reason": "Negative h_tip_w_m2k"}
            u = h_tip / (m * k)
            denom = math.cosh(m_len) + u * math.sinh(m_len)
            theta_x = theta_b * (math.cosh(m * (length - x_eval)) + u * math.sinh(m * (length - x_eval))) / denom
            theta_tip = theta_b / denom
            q_fin = cap_m * (math.sinh(m_len) + u * math.cosh(m_len)) / denom
        elif tip_bc == "prescribed_temp":
            t_tip_spec = float(spec["t_tip_c"])
            theta_L = t_tip_spec - t_inf
            theta_x = (theta_L * math.sinh(m * x_eval) + theta_b * math.sinh(m * (length - x_eval))) / math.sinh(m_len)
            theta_tip = theta_L
            q_fin = cap_m * (math.cosh(m_len) - (theta_L / theta_b if abs(theta_b) > 1e-12 else 0.0)) / math.sinh(m_len)
            h_tip = 0.0
        elif tip_bc == "infinite":
            theta_x = theta_b * math.exp(-m * x_eval)
            theta_tip = theta_b * math.exp(-m_len)
            q_fin = cap_m
            h_tip = m * k
        else:
            return {"passed": False, "domain": "pde", "mode": mode, "reason": f"Unsupported tip_bc: {tip_bc}"}

        n_nodes = 401
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
        if tip_bc == "prescribed_temp":
            diag[n_nodes - 1] = 1.0
            rhs[n_nodes - 1] = theta_tip
        else:
            lower[n_nodes - 1] = -cond
            diag[n_nodes - 1] = cond + 0.5 * sink + h_tip * ac
            rhs[n_nodes - 1] = 0.0

        theta_fvm = _thomas_solve(lower, diag, upper, rhs)
        q_conv_fvm = 0.5 * sink * theta_fvm[0] + sum(sink * theta_fvm[i] for i in range(1, n_nodes - 1)) + 0.5 * sink * theta_fvm[-1]
        if tip_bc != "prescribed_temp":
            q_fvm = q_conv_fvm + h_tip * ac * theta_fvm[-1]
        else:
            q_fvm = cond * (theta_fvm[0] - theta_fvm[1]) + 0.5 * sink * theta_fvm[0]

        q_max = h * perim * length * theta_b + (h_tip * ac * theta_b if tip_bc == "convective" else 0.0)
        eta_fin = (q_fin / (h * perim * length * theta_b)) if abs(theta_b) > 1e-12 else 1.0
        eps_fin = (q_fin / (h * ac * theta_b)) if abs(theta_b) > 1e-12 else 0.0
        rel_q_err = abs(q_fvm - q_fin) / max(abs(q_fin), 1e-9)
        tip_err = abs(theta_fvm[-1] - theta_tip)
        passed = (rel_q_err < 1e-3) and (tip_err < max(1e-3, 1e-3 * abs(theta_b)))

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
        if a_len <= 0.0 or b_len <= 0.0 or not (0.0 <= xq <= a_len) or not (0.0 <= yq <= b_len):
            return {"passed": False, "domain": "pde", "mode": mode, "reason": "Invalid 2D Laplace domain or query point"}

        if top_profile == "sine":
            t_exact = t_top * math.sin(math.pi * xq / a_len) * (math.sinh(math.pi * yq / a_len) / math.sinh(math.pi * b_len / a_len))
        else:
            t_exact = 0.0
            for k_odd in range(1, 120, 2):
                bn = 4.0 * t_top / (k_odd * math.pi)
                arg_y = k_odd * math.pi * yq / a_len
                arg_b = k_odd * math.pi * b_len / a_len
                ratio = math.exp(arg_y - arg_b) * (1.0 - math.exp(-2.0 * arg_y)) / (1.0 - math.exp(-2.0 * arg_b))
                t_exact += bn * math.sin(k_odd * math.pi * xq / a_len) * ratio

        nx = 40
        ny = max(20, int(round(nx * b_len / a_len)))
        ny = min(ny, 60)
        dx = a_len / nx
        dy = b_len / ny
        grid = [[0.0] * (ny + 1) for _ in range(nx + 1)]
        for i in range(1, nx):
            xi = i * dx
            grid[i][ny] = t_top * math.sin(math.pi * xi / a_len) if top_profile == "sine" else t_top

        dx2 = dx * dx
        dy2 = dy * dy
        denom = 2.0 * (dx2 + dy2)
        omega = 1.72
        for _ in range(900):
            max_upd = 0.0
            for i in range(1, nx):
                for j in range(1, ny):
                    gs = ((grid[i + 1][j] + grid[i - 1][j]) * dy2 + (grid[i][j + 1] + grid[i][j - 1]) * dx2) / denom
                    upd = omega * (gs - grid[i][j])
                    grid[i][j] += upd
                    if abs(upd) > max_upd:
                        max_upd = abs(upd)
            if max_upd < 1e-8 * max(abs(t_top), 1.0):
                break

        fx = xq / dx
        fy = yq / dy
        i0 = min(int(math.floor(fx)), nx - 1)
        j0 = min(int(math.floor(fy)), ny - 1)
        tx = fx - i0
        ty = fy - j0
        t_fdm = (
            (1 - tx) * (1 - ty) * grid[i0][j0]
            + tx * (1 - ty) * grid[i0 + 1][j0]
            + (1 - tx) * ty * grid[i0][j0 + 1]
            + tx * ty * grid[i0 + 1][j0 + 1]
        )
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


def _bode_margins(num: Sequence[float], den: Sequence[float]) -> Dict[str, Any]:
    def g_jw(w: float) -> complex:
        jw = 1j * w
        d_val = _poly_eval_complex(den, jw)
        if abs(d_val) < 1e-30:
            return complex(float("inf"), 0.0)
        return _poly_eval_complex(num, jw) / d_val

    n_pts = 1600
    w_grid = [10.0 ** (-4.0 + 8.0 * k / (n_pts - 1)) for k in range(n_pts)]
    mags: List[float] = []
    phases_deg: List[float] = []
    prev_ph = 0.0
    for idx, w in enumerate(w_grid):
        val = g_jw(w)
        mags.append(abs(val))
        raw = math.degrees(cmath.phase(val))
        if idx == 0:
            ph = raw - 360.0 if raw > 5.0 else raw
        else:
            ph = raw
            while ph - prev_ph > 180.0:
                ph -= 360.0
            while ph - prev_ph < -180.0:
                ph += 360.0
        phases_deg.append(ph)
        prev_ph = ph

    def unwrap_near(w: float, ref_deg: float) -> float:
        ph = math.degrees(cmath.phase(g_jw(w)))
        while ph - ref_deg > 180.0:
            ph -= 360.0
        while ph - ref_deg < -180.0:
            ph += 360.0
        return ph

    w_gc: Optional[float] = None
    pm_deg: Optional[float] = None
    for k in range(n_pts - 1):
        f0 = mags[k] - 1.0
        f1 = mags[k + 1] - 1.0
        if f0 == 0.0 or f0 * f1 < 0.0:
            lo, hi = w_grid[k], w_grid[k + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if (abs(g_jw(lo)) - 1.0) * (abs(g_jw(mid)) - 1.0) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            w_gc = 0.5 * (lo + hi)
            ref_ph = 0.5 * (phases_deg[k] + phases_deg[k + 1])
            pm_deg = 180.0 + unwrap_near(w_gc, ref_ph)
            break

    w_pc: Optional[float] = None
    gm_db: Optional[float] = None
    for k in range(n_pts - 1):
        p0 = phases_deg[k] + 180.0
        p1 = phases_deg[k + 1] + 180.0
        if p0 * p1 < 0.0 or (abs(p0) < 1e-9 and k > 0):
            lo, hi = w_grid[k], w_grid[k + 1]
            ref_ph = 0.5 * (phases_deg[k] + phases_deg[k + 1])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if (unwrap_near(lo, ref_ph) + 180.0) * (unwrap_near(mid, ref_ph) + 180.0) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            w_pc = 0.5 * (lo + hi)
            mag_pc = abs(g_jw(w_pc))
            gm_db = -20.0 * math.log10(max(mag_pc, 1e-30))
            break

    return {
        "w_gc_rad_s": w_gc,
        "phase_margin_deg": pm_deg,
        "w_pc_rad_s": w_pc,
        "gain_margin_db": gm_db,
    }


def verify_control_system(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "transfer_function" if "den" in spec else "state_space")).strip().lower()

    if mode == "transfer_function":
        num = [float(c) for c in spec.get("num", [1.0])]
        den = [float(c) for c in spec["den"]]
        while len(den) > 1 and abs(den[0]) == 0.0:
            den.pop(0)
        if len(den) < 2:
            return {"passed": False, "domain": "control", "mode": mode, "reason": "Denominator degree must be >= 1"}
        if den[0] < 0.0:
            den = [-c for c in den]
            num = [-c for c in num]

        _, rhp_routh, zero_row = _routh_table(den)
        poles = _poly_roots(den)
        rhp_roots = sum(1 for p in poles if p.real > 1e-6)
        imag_roots = sum(1 for p in poles if abs(p.real) <= 1e-6)
        open_loop_hurwitz = (rhp_routh == 0) and (not zero_row) and all(p.real < -1e-6 for p in poles)
        routh_root_consistent = (rhp_routh == rhp_roots) or zero_row

        cl_len = max(len(den), len(num))
        den_pad = [0.0] * (cl_len - len(den)) + den
        num_pad = [0.0] * (cl_len - len(num)) + num
        cl_poly = [den_pad[i] + num_pad[i] for i in range(cl_len)]
        while len(cl_poly) > 1 and abs(cl_poly[0]) == 0.0:
            cl_poly.pop(0)
        cl_poles = _poly_roots(cl_poly)
        _, cl_rhp_routh, cl_zero_row = _routh_table(cl_poly if cl_poly[0] > 0 else [-c for c in cl_poly])
        closed_loop_hurwitz = (cl_rhp_routh == 0) and (not cl_zero_row) and all(p.real < -1e-6 for p in cl_poles)

        margins = _bode_margins(num, den)
        dc_gain = (num[-1] / den[-1]) if abs(den[-1]) > 1e-14 else float("inf")

        res: Dict[str, Any] = {
            "passed": bool(routh_root_consistent),
            "domain": "control",
            "mode": mode,
            "hurwitz_stable": open_loop_hurwitz,
            "closed_loop_hurwitz_stable": closed_loop_hurwitz,
            "rhp_poles_routh": rhp_routh,
            "rhp_poles_eigen": rhp_roots,
            "jw_axis_poles": imag_roots,
            "dc_gain": dc_gain,
            "poles_real_imag": [[p.real, p.imag] for p in sorted(poles, key=lambda z: (round(z.real, 7), round(z.imag, 7)))],
            "closed_loop_poles_real_imag": [[p.real, p.imag] for p in sorted(cl_poles, key=lambda z: (round(z.real, 7), round(z.imag, 7)))],
            **margins,
        }
        _check_claimed(res, spec)
        return res

    if mode == "state_space":
        a_mat = [[float(v) for v in row] for row in spec["A"]]
        n = len(a_mat)
        if n == 0 or any(len(r) != n for r in a_mat):
            return {"passed": False, "domain": "control", "mode": mode, "reason": "Matrix A must be non-empty square"}

        m_cur = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        char_poly = [1.0]
        for k in range(1, n + 1):
            am = _mat_mul(a_mat, m_cur)
            ck = -sum(am[i][i] for i in range(n)) / float(k)
            char_poly.append(ck)
            m_cur = [[am[i][j] + (ck if i == j else 0.0) for j in range(n)] for i in range(n)]

        ch_residual = max(abs(m_cur[i][j]) for i in range(n) for j in range(n))
        eigs = _poly_roots(char_poly)
        _, rhp_routh, zero_row = _routh_table(char_poly)
        hurwitz = (rhp_routh == 0) and (not zero_row) and all(z.real < -1e-6 for z in eigs)

        controllable: Optional[bool] = None
        ctrl_rank: Optional[int] = None
        if "B" in spec:
            b_mat = [[float(v) for v in (row if isinstance(row, list) else [row])] for row in spec["B"]]
            blocks = [b_mat]
            cur = b_mat
            for _ in range(1, n):
                cur = _mat_mul(a_mat, cur)
                blocks.append(cur)
            ctrb = []
            for i in range(n):
                row_full: List[float] = []
                for blk in blocks:
                    row_full.extend(blk[i])
                ctrb.append(row_full)
            ctrl_rank = _matrix_rank(ctrb)
            controllable = ctrl_rank == n

        observable: Optional[bool] = None
        obs_rank: Optional[int] = None
        if "C" in spec:
            c_raw = spec["C"]
            c_mat = [[float(v) for v in c_raw]] if (c_raw and not isinstance(c_raw[0], list)) else [[float(v) for v in r] for r in c_raw]
            obsv: List[List[float]] = []
            cur = c_mat
            for step in range(n):
                obsv.extend(cur)
                if step < n - 1:
                    cur = _mat_mul(cur, a_mat)
            obs_rank = _matrix_rank(obsv)
            observable = obs_rank == n

        q_mat = [[float(v) for v in r] for r in spec["Q"]] if "Q" in spec else [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        n2 = n * n
        kron = [[0.0] * n2 for _ in range(n2)]
        rhs_vec = [0.0] * n2
        for i in range(n):
            for j in range(n):
                row_idx = i * n + j
                rhs_vec[row_idx] = -q_mat[i][j]
                for k in range(n):
                    kron[row_idx][k * n + j] += a_mat[k][i]
                    kron[row_idx][i * n + k] += a_mat[k][j]

        p_vec = _solve_linear(kron, rhs_vec)
        lyapunov_pd = False
        p_mat: Optional[List[List[float]]] = None
        if p_vec is not None:
            p_mat = [[0.5 * (p_vec[i * n + j] + p_vec[j * n + i]) for j in range(n)] for i in range(n)]
            lyapunov_pd = True
            for m_sub in range(1, n + 1):
                sub = [[p_mat[r][c] for c in range(m_sub)] for r in range(m_sub)]
                ok_sub = True
                for c_idx in range(m_sub):
                    if sub[c_idx][c_idx] <= 1e-10:
                        ok_sub = False
                        break
                    for r_idx in range(c_idx + 1, m_sub):
                        f = sub[r_idx][c_idx] / sub[c_idx][c_idx]
                        for k_idx in range(c_idx, m_sub):
                            sub[r_idx][k_idx] -= f * sub[c_idx][k_idx]
                if not ok_sub:
                    lyapunov_pd = False
                    break

        scale_a = max((abs(a_mat[i][j]) for i in range(n) for j in range(n)), default=1.0) ** max(1, n)
        passed = (ch_residual <= 1e-7 * max(1.0, scale_a)) and (lyapunov_pd == hurwitz)
        res = {
            "passed": passed,
            "domain": "control",
            "mode": mode,
            "char_poly": char_poly,
            "cayley_hamilton_residual": ch_residual,
            "hurwitz_stable": hurwitz,
            "controllable": controllable,
            "controllability_rank": ctrl_rank,
            "observable": observable,
            "observability_rank": obs_rank,
            "lyapunov_pd": lyapunov_pd,
            "lyapunov_P": p_mat,
            "eigenvalues_real_imag": [[z.real, z.imag] for z in sorted(eigs, key=lambda z: (round(z.real, 7), round(z.imag, 7)))],
        }
        _check_claimed(res, spec)
        return res

    return {"passed": False, "domain": "control", "mode": mode, "reason": f"Unknown control mode: {mode}"}


def _pws_ashrae_pa(t_c: float, *, ice_below_c: float = 0.0) -> float:
    tk = t_c + 273.15
    if tk <= 0.0:
        raise ValueError("Temperature below absolute zero")
    if t_c < ice_below_c:
        c1 = -5.6745359e3
        c2 = 6.3925247e0
        c3 = -9.6778430e-3
        c4 = 6.2215701e-7
        c5 = 2.0747825e-9
        c6 = -9.4840240e-13
        c7 = 4.1635019e0
        ln_pws = c1 / tk + c2 + c3 * tk + c4 * (tk ** 2) + c5 * (tk ** 3) + c6 * (tk ** 4) + c7 * math.log(tk)
    else:
        c8 = -5.8002206e3
        c9 = 1.3914993e0
        c10 = -4.8640239e-2
        c11 = 4.1764768e-5
        c12 = -1.4452093e-8
        c13 = 6.5459673e0
        ln_pws = c8 / tk + c9 + c10 * tk + c11 * (tk ** 2) + c12 * (tk ** 3) + c13 * math.log(tk)
    return math.exp(ln_pws)


def _dew_point_from_pw(pw_pa: float) -> float:
    if pw_pa <= 0.0:
        return -100.0
    lo, hi = -80.0, 150.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _pws_ashrae_pa(mid) < pw_pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _w_from_twb(t_db_c: float, t_wb_c: float, p_atm_pa: float, *, ice_below_c: float = 0.0) -> float:
    pws_wb = _pws_ashrae_pa(t_wb_c, ice_below_c=ice_below_c)
    ws_wb = 0.621945 * pws_wb / (p_atm_pa - pws_wb)
    if t_wb_c >= 0.0:
        num = (2501.0 - 2.326 * t_wb_c) * ws_wb - 1.006 * (t_db_c - t_wb_c)
        den = 2501.0 + 1.86 * t_db_c - 4.186 * t_wb_c
    else:
        num = (2830.0 - 0.24 * t_wb_c) * ws_wb - 1.006 * (t_db_c - t_wb_c)
        den = 2830.0 + 1.86 * t_db_c - 2.1 * t_wb_c
    return max(0.0, num / den)


def _wet_bulb_from_state(t_db_c: float, w: float, p_atm_pa: float) -> float:
    lo, hi = -60.0, t_db_c
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _w_from_twb(t_db_c, mid, p_atm_pa) < w:
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
        w = _w_from_twb(t_db, t_wb, p_atm_pa, ice_below_c=ice_below_c)
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
    t_dp_calc = _dew_point_from_pw(pw) if pw > 0.0 else -100.0
    t_wb_calc = _wet_bulb_from_state(t_db, w, p_atm_pa)
    ws = 0.621945 * pws / (p_atm_pa - pws)
    deg_sat = w / ws if ws > 0 else 0.0

    return {
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


_FUSION_ENTHALPY_KJ_KG = 333.5


def _ice_enthalpy_kj_kg(temp_c: float) -> float:
    """Ice Ih relative to liquid water at 0 C, approximate from -60 to 0 C."""
    # Integral of a linear ice heat capacity, based on ASHRAE Fundamentals
    # (2025), Psychrometrics, Table 3. Fusion at 0 C is 333.5 kJ/kg.
    return -_FUSION_ENTHALPY_KJ_KG + temp_c * (2.097 + 0.00365 * temp_c)


def verify_psychrometric_process(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "state")).strip().lower()
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
        m1 = float(spec["stream1"].get("m_da_kg_s") or (float(spec["stream1"]["v_dot_m3_s"]) / s1["v_m3_per_kgda"]))
        m2 = float(spec["stream2"].get("m_da_kg_s") or (float(spec["stream2"]["v_dot_m3_s"]) / s2["v_m3_per_kgda"]))
        if m1 <= 0.0 or m2 <= 0.0:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Non-positive dry-air mass flow"}
        m3 = m1 + m2
        w3 = (m1 * s1["w_kg_per_kgda"] + m2 * s2["w_kg_per_kgda"]) / m3
        h3 = (m1 * s1["h_kj_per_kgda"] + m2 * s2["h_kj_per_kgda"]) / m3
        t_db3 = (h3 - 2501.0 * w3) / (1.006 + 1.86 * w3)
        pws3 = _pws_ashrae_pa(t_db3, ice_below_c=ice_below)
        pw3 = p_atm * w3 / (0.621945 + w3)
        rh3 = pw3 / pws3
        fogged = rh3 > (1.0 + 8.0 * math.ulp(1.0) if equilibrium else 1.0)
        vapor = w3
        ice = 0.0
        if fogged:
            frozen = False
            on_melting_plateau = False

            def saturated_outlet(temp: float) -> Tuple[float, float]:
                pws = _pws_ashrae_pa(temp, ice_below_c=ice_below)
                w_sat = 0.621945 * pws / (p_atm - pws)
                h_air = 1.006 * temp + w_sat * (2501.0 + 1.86 * temp)
                h_condensed = _ice_enthalpy_kj_kg(temp) if frozen else 4.186 * temp
                return w_sat, h_air + (w3 - w_sat) * h_condensed

            hi = max(s1["t_db_c"], s2["t_db_c"])
            lo = max(0.0, min(t_db3, hi))
            roundoff = 8.0 * math.ulp(max(abs(h3), 1.0))
            if hi < 0.0 or (t_db3 < 0.0 and saturated_outlet(lo)[1] > h3 + roundoff):
                if not equilibrium:
                    return {"passed": False, "domain": "psychrometrics", "mode": mode,
                            "reason": "Fog equilibrium requires an ice or mixed-phase model"}
                # At the melting point the air and total condensate mass are fixed.
                # The fusion enthalpy determines how much condensate is ice.
                vapor_zero, liquid_zero = saturated_outlet(0.0)
                frozen = True
                _, ice_zero = saturated_outlet(0.0)
                if hi >= 0.0 and w3 >= vapor_zero and ice_zero <= h3 <= liquid_zero:
                    on_melting_plateau = True
                    t_db3, vapor = 0.0, vapor_zero
                    ice = (liquid_zero - h3) / _FUSION_ENTHALPY_KJ_KG
                else:
                    lo, hi = max(-60.0, min(t_db3, hi)), min(0.0, hi)
            if not on_melting_plateau:
                if (lo > hi or saturated_outlet(lo)[1] > h3 + roundoff
                        or saturated_outlet(hi)[1] < h3 - roundoff):
                    return {"passed": False, "domain": "psychrometrics", "mode": mode,
                            "reason": "Inlets do not bracket a supported condensate equilibrium"}
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    if saturated_outlet(mid)[1] < h3:
                        lo = mid
                    else:
                        hi = mid
                t_db3 = 0.5 * (lo + hi)
                vapor = min(w3, saturated_outlet(t_db3)[0])
                if frozen:
                    ice = w3 - vapor
            rh3 = 1.0
        res: Dict[str, Any] = {
            "passed": True,
            "domain": "psychrometrics",
            "mode": mode,
            "m_da_total_kg_s": m3,
            "t_db_mix_c": t_db3,
            "w_mix_kg_per_kgda": w3,  # total water, including any condensate
            "w_vapor_kg_per_kgda": vapor,
            "w_condensate_kg_per_kgda": w3 - vapor,
            "h_mix_kj_per_kgda": h3,  # total stream enthalpy per kg dry air
            "rh_mix": min(rh3, 1.0),
            "fog_saturated": fogged,
        }
        if equilibrium:
            condensate = w3 - vapor
            ice = min(ice, condensate)
            res.update(condensate_model=model, w_ice_kg_per_kgda=ice,
                       w_liquid_kg_per_kgda=condensate - ice)
        _check_claimed(res, spec)
        return res

    if mode == "cooling_coil":
        s_in = _psychro_state(spec["inlet"], p_atm)
        s_out = _psychro_state(spec["outlet"], p_atm)
        if not s_in.get("passed") or not s_out.get("passed"):
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": s_in.get("reason") or s_out.get("reason")}
        m_da = float(spec.get("m_da_kg_s") or (float(spec["v_dot_in_m3_s"]) / s_in["v_m3_per_kgda"]))
        if m_da <= 0.0:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Non-positive dry-air mass flow"}
        if s_out["w_kg_per_kgda"] > s_in["w_kg_per_kgda"] + 1e-7 or s_out["t_db_c"] > s_in["t_db_c"] + 1e-7:
            return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": "Cooling coil cannot increase humidity ratio or dry-bulb"}

        m_cond_kg_s = m_da * max(0.0, s_in["w_kg_per_kgda"] - s_out["w_kg_per_kgda"])
        t_cond = float(spec.get("t_condensate_c", s_out["t_db_c"]))
        h_f_kj_kg = 4.186 * t_cond
        q_total_kw = m_da * (s_in["h_kj_per_kgda"] - s_out["h_kj_per_kgda"]) - m_cond_kg_s * h_f_kj_kg
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
            "q_total_kw": q_total_kw,
            "q_sensible_kw": q_sens_kw,
            "q_latent_kw": q_lat_kw,
            "sensible_heat_ratio": shr,
            "inlet_h_kj_per_kgda": s_in["h_kj_per_kgda"],
            "outlet_h_kj_per_kgda": s_out["h_kj_per_kgda"],
        }
        _check_claimed(res, spec)
        return res

    return {"passed": False, "domain": "psychrometrics", "mode": mode, "reason": f"Unknown psychrometric mode: {mode}"}


def verify_fem_structure(spec: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(spec.get("mode", "truss_2d")).strip().lower()

    if mode == "truss_2d":
        nodes = [[float(c[0]), float(c[1])] for c in spec["nodes"]]
        elements = spec["elements"]
        n_nodes = len(nodes)
        ndof = 2 * n_nodes
        k_glob = [[0.0] * ndof for _ in range(ndof)]
        f_ext = [0.0] * ndof
        f_th = [0.0] * ndof

        elem_meta = []
        for el in elements:
            ni = int(el["i"])
            nj = int(el["j"])
            e_mod = float(el["E_pa"])
            area = float(el["A_m2"])
            alpha = float(el.get("alpha_per_K", 0.0))
            dt_k = float(el.get("dT_K", 0.0))
            dx = nodes[nj][0] - nodes[ni][0]
            dy = nodes[nj][1] - nodes[ni][1]
            length = math.hypot(dx, dy)
            if length <= 1e-12 or e_mod <= 0.0 or area <= 0.0:
                return {"passed": False, "domain": "fem", "mode": mode, "reason": "Degenerate truss member geometry or stiffness"}
            c = dx / length
            s = dy / length
            k_axial = e_mod * area / length
            dofs = [2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1]
            b_vec = [-c, -s, c, s]
            for a_idx in range(4):
                for b_idx in range(4):
                    k_glob[dofs[a_idx]][dofs[b_idx]] += k_axial * b_vec[a_idx] * b_vec[b_idx]
            nth = e_mod * area * alpha * dt_k
            for a_idx in range(4):
                f_th[dofs[a_idx]] += nth * b_vec[a_idx]
            elem_meta.append((ni, nj, e_mod, area, length, c, s, alpha, dt_k, dofs, b_vec))

        for ld in spec.get("loads", []):
            nd = int(ld["node"])
            f_ext[2 * nd] += float(ld.get("fx_N", 0.0))
            f_ext[2 * nd + 1] += float(ld.get("fy_N", 0.0))

        prescribed: Dict[int, float] = {}
        for sup in spec.get("supports", []):
            nd = int(sup["node"])
            if sup.get("fix_x", False):
                prescribed[2 * nd] = float(sup.get("ux_m", 0.0))
            if sup.get("fix_y", False):
                prescribed[2 * nd + 1] = float(sup.get("uy_m", 0.0))

        free_dofs = [d for d in range(ndof) if d not in prescribed]
        if not free_dofs:
            u_full = [prescribed.get(d, 0.0) for d in range(ndof)]
        else:
            k_ff = [[k_glob[r][c] for c in free_dofs] for r in free_dofs]
            rhs_f = []
            for r in free_dofs:
                val = f_ext[r] + f_th[r]
                for p_dof, u_val in prescribed.items():
                    val -= k_glob[r][p_dof] * u_val
                rhs_f.append(val)
            u_free = _solve_linear(k_ff, rhs_f, tol=1e-11)
            if u_free is None:
                return {"passed": False, "domain": "fem", "mode": mode, "reason": "Singular stiffness matrix (kinematic mechanism or insufficient supports)"}
            u_full = [0.0] * ndof
            for d, u_val in prescribed.items():
                u_full[d] = u_val
            for idx, d in enumerate(free_dofs):
                u_full[d] = u_free[idx]

        reactions = [
            sum(k_glob[i][j] * u_full[j] for j in range(ndof)) - f_ext[i] - f_th[i]
            for i in range(ndof)
        ]

        axial_forces: List[float] = []
        axial_stresses: List[float] = []
        strain_energy = 0.0
        has_thermal_or_settlement = any(abs(v) > 0.0 for v in prescribed.values())
        for (_, _, e_mod, area, length, _, _, alpha, dt_k, dofs, b_vec) in elem_meta:
            ext_m = sum(b_vec[idx] * u_full[dofs[idx]] for idx in range(4))
            mech_ext = ext_m - alpha * dt_k * length
            if abs(dt_k) > 0.0 and abs(alpha) > 0.0:
                has_thermal_or_settlement = True
            n_force = (e_mod * area / length) * mech_ext
            axial_forces.append(n_force)
            axial_stresses.append(n_force / area)
            strain_energy += 0.5 * (n_force ** 2) * length / (e_mod * area)

        fx_sum = sum(f_ext[2 * i] + reactions[2 * i] for i in range(n_nodes))
        fy_sum = sum(f_ext[2 * i + 1] + reactions[2 * i + 1] for i in range(n_nodes))
        mom_z = sum(
            nodes[i][0] * (f_ext[2 * i + 1] + reactions[2 * i + 1]) - nodes[i][1] * (f_ext[2 * i] + reactions[2 * i])
            for i in range(n_nodes)
        )
        load_scale = max((abs(v) for v in f_ext), default=1.0)
        eq_ok = max(abs(fx_sum), abs(fy_sum), abs(mom_z)) <= 1e-6 * max(1.0, load_scale)

        ext_work = 0.5 * sum(f_ext[d] * u_full[d] for d in range(ndof))
        clapeyron_ok = True
        if not has_thermal_or_settlement:
            clapeyron_ok = abs(strain_energy - ext_work) <= 1e-7 * max(abs(strain_energy), 1e-12)

        res: Dict[str, Any] = {
            "passed": bool(eq_ok and clapeyron_ok),
            "domain": "fem",
            "mode": mode,
            "displacements_xy_m": [[u_full[2 * i], u_full[2 * i + 1]] for i in range(n_nodes)],
            "axial_forces_N": axial_forces,
            "axial_stresses_Pa": axial_stresses,
            "reactions_xy_N": [[reactions[2 * i], reactions[2 * i + 1]] for i in range(n_nodes)],
            "strain_energy_J": strain_energy,
            "external_work_J": ext_work,
            "equilibrium_residual_N": max(abs(fx_sum), abs(fy_sum)),
        }
        _check_claimed(res, spec)
        return res

    if mode == "euler_bernoulli_beam_1d":
        x_nodes = [float(x) for x in spec["x_nodes_m"]]
        n_nodes = len(x_nodes)
        if n_nodes < 2 or any(x_nodes[i + 1] <= x_nodes[i] for i in range(n_nodes - 1)):
            return {"passed": False, "domain": "fem", "mode": mode, "reason": "x_nodes_m must be strictly increasing with >= 2 nodes"}
        ndof = 2 * n_nodes
        k_glob = [[0.0] * ndof for _ in range(ndof)]
        f_eq = [0.0] * ndof

        default_ei = float(spec.get("EI_Nm2", 1.0))
        default_q = float(spec.get("q_uniform_N_m", 0.0))
        el_specs = spec.get("elements") or [
            {"EI_Nm2": default_ei, "q_N_m": default_q} for _ in range(n_nodes - 1)
        ]

        for e_idx in range(n_nodes - 1):
            le = x_nodes[e_idx + 1] - x_nodes[e_idx]
            ei = float(el_specs[e_idx].get("EI_Nm2", default_ei))
            q_val = float(el_specs[e_idx].get("q_N_m", default_q))
            if ei <= 0.0:
                return {"passed": False, "domain": "fem", "mode": mode, "reason": "Non-positive flexural rigidity EI_Nm2"}
            c_fac = ei / (le ** 3)
            ke = [
                [12.0 * c_fac, 6.0 * le * c_fac, -12.0 * c_fac, 6.0 * le * c_fac],
                [6.0 * le * c_fac, 4.0 * le * le * c_fac, -6.0 * le * c_fac, 2.0 * le * le * c_fac],
                [-12.0 * c_fac, -6.0 * le * c_fac, 12.0 * c_fac, -6.0 * le * c_fac],
                [6.0 * le * c_fac, 2.0 * le * le * c_fac, -6.0 * le * c_fac, 4.0 * le * le * c_fac],
            ]
            fe = [0.5 * q_val * le, (q_val * le * le) / 12.0, 0.5 * q_val * le, -(q_val * le * le) / 12.0]
            dofs = [2 * e_idx, 2 * e_idx + 1, 2 * (e_idx + 1), 2 * (e_idx + 1) + 1]
            for a_i in range(4):
                f_eq[dofs[a_i]] += fe[a_i]
                for b_i in range(4):
                    k_glob[dofs[a_i]][dofs[b_i]] += ke[a_i][b_i]

        for ld in spec.get("nodal_loads", []):
            nd = int(ld["node"])
            f_eq[2 * nd] += float(ld.get("fy_N", 0.0))
            f_eq[2 * nd + 1] += float(ld.get("mz_Nm", 0.0))

        prescribed = {}
        for sup in spec.get("supports", []):
            nd = int(sup["node"])
            if sup.get("fix_w", False):
                prescribed[2 * nd] = float(sup.get("w_m", 0.0))
            if sup.get("fix_theta", False):
                prescribed[2 * nd + 1] = float(sup.get("theta_rad", 0.0))

        free_dofs = [d for d in range(ndof) if d not in prescribed]
        k_ff = [[k_glob[r][c] for c in free_dofs] for r in free_dofs]
        rhs_f = [
            f_eq[r] - sum(k_glob[r][p] * u_p for p, u_p in prescribed.items())
            for r in free_dofs
        ]
        u_free = _solve_linear(k_ff, rhs_f, tol=1e-11)
        if u_free is None:
            return {"passed": False, "domain": "fem", "mode": mode, "reason": "Unstable beam boundary conditions"}

        u_full = [prescribed.get(d, 0.0) for d in range(ndof)]
        for idx, d in enumerate(free_dofs):
            u_full[d] = u_free[idx]

        reactions = [
            sum(k_glob[i][j] * u_full[j] for j in range(ndof)) - f_eq[i]
            for i in range(ndof)
        ]
        u_strain = 0.5 * sum(u_full[i] * sum(k_glob[i][j] * u_full[j] for j in range(ndof)) for i in range(ndof))
        w_ext = 0.5 * sum(f_eq[i] * u_full[i] for i in range(ndof))
        has_settlement = any(abs(v) > 0.0 for v in prescribed.values())
        clapeyron_ok = True if has_settlement else (abs(u_strain - w_ext) <= 1e-7 * max(abs(u_strain), 1e-12))

        res = {
            "passed": clapeyron_ok,
            "domain": "fem",
            "mode": mode,
            "deflections_m": [u_full[2 * i] for i in range(n_nodes)],
            "rotations_rad": [u_full[2 * i + 1] for i in range(n_nodes)],
            "tip_deflection_m": u_full[2 * (n_nodes - 1)],
            "reactions_V_M": [[reactions[2 * i], reactions[2 * i + 1]] for i in range(n_nodes)],
            "strain_energy_J": u_strain,
            "external_work_J": w_ext,
        }
        _check_claimed(res, spec)
        return res

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
    if mode in ("truss_2d", "euler_bernoulli_beam_1d"):
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
