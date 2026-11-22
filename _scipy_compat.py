"""Pure-Python / NumPy high-precision numerical fallbacks when SciPy is absent."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

try:
    from scipy import integrate as _sp_integrate
    from scipy import linalg as _sp_linalg
    from scipy import optimize as _sp_optimize
    HAS_SCIPY = True
except ModuleNotFoundError:
    _sp_integrate = None
    _sp_linalg = None
    _sp_optimize = None
    HAS_SCIPY = False


def brentq_fallback(f, a: float, b: float, args=(), xtol: float = 2e-12, rtol: float = 8.881784197001252e-16, maxiter: int = 200) -> float:
    xpre, xcur = float(a), float(b)
    fpre, fcur = float(f(xpre, *args)), float(f(xcur, *args))
    if fpre == 0.0:
        return xpre
    if fcur == 0.0:
        return xcur
    if (fpre > 0.0) == (fcur > 0.0):
        raise ValueError(f"f(a)={fpre} and f(b)={fcur} must have opposite signs")
    xblk, fblk = xpre, fpre
    spre = scur = xcur - xpre
    for _ in range(maxiter):
        if fcur == 0.0:
            return xcur
        if (fpre > 0.0) != (fcur > 0.0):
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        delta = 0.5 * (xtol + rtol * abs(xcur))
        sbis = 0.5 * (xblk - xcur)
        if fcur == 0.0 or abs(sbis) <= delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                denom = fcur - fpre
                stry = (-fcur * (xcur - xpre) / denom) if abs(denom) > 1e-300 else sbis
            else:
                dx1 = xpre - xcur
                dx2 = xblk - xcur
                dpre = ((fpre - fcur) / dx1) if abs(dx1) > 1e-300 else 0.0
                dblk = ((fblk - fcur) / dx2) if abs(dx2) > 1e-300 else 0.0
                denom = dblk * dpre * (fblk - fpre)
                stry = (-fcur * (fblk * dblk - fpre * dpre) / denom) if abs(denom) > 1e-300 else sbis
            if 2.0 * abs(stry) < min(abs(spre), 3.0 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre = scur = sbis
        else:
            spre = scur = sbis
        xpre, fpre = xcur, fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0.0 else -delta
        fcur = float(f(xcur, *args))
    return xcur


_GK15_X = (
    0.0,
    0.20778495500789848,
    0.4058451513773972,
    0.5860872354676911,
    0.7415311855993945,
    0.8648644233597691,
    0.9491079123427585,
    0.9914553711208126,
)
_GK15_W = (
    0.20948214108472782,
    0.20443294007529889,
    0.19035057806478542,
    0.1690047266392679,
    0.14065325971552592,
    0.10479001032225019,
    0.06309209262997856,
    0.022935322010529224,
)
_G7_W = (
    0.4179591836734694,
    0.3818300505051189,
    0.27970539148927664,
    0.1294849661688697,
)


def _gk15_panel(f, a: float, b: float, args):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    fc = float(f(mid, *args))
    ik = _GK15_W[0] * fc
    ig = _G7_W[0] * fc
    for idx in range(1, 8):
        dx = half * _GK15_X[idx]
        s = float(f(mid - dx, *args)) + float(f(mid + dx, *args))
        ik += _GK15_W[idx] * s
        if idx % 2 == 0:
            ig += _G7_W[idx // 2] * s
    ik *= half
    ig *= half
    err = abs(ik - ig)
    return ik, max(err, (200.0 * err) ** 1.5 if err < 1.0 else err)


def quad_fallback(func, a: float, b: float, args=(), epsabs: float = 1.49e-8, epsrel: float = 1.49e-8, limit: int = 200):
    sign = 1.0
    a_f, b_f = float(a), float(b)
    if a_f == b_f:
        return 0.0, 0.0
    if math.isinf(b_f) and not math.isinf(a_f):
        s_inf = 1.0 if b_f > 0 else -1.0
        def wrapped(t, *wargs):
            omt = 1.0 - t
            x = a_f + s_inf * t / omt
            return float(func(x, *wargs)) / (omt * omt)
        val, err = quad_fallback(wrapped, 0.0, 1.0 - 1e-15, args=args, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return s_inf * val, err
    if b_f < a_f:
        a_f, b_f = b_f, a_f
        sign = -1.0
    val0, err0 = _gk15_panel(func, a_f, b_f, args)
    stack = [(err0, a_f, b_f, val0)]
    total = val0
    total_err = err0
    subdivisions = 1
    while subdivisions < max(limit, 32):
        tol = max(epsabs, epsrel * abs(total))
        if total_err <= tol:
            break
        stack.sort(key=lambda item: item[0])
        err_top, lo, hi, val_top = stack.pop()
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            stack.append((err_top, lo, hi, val_top))
            break
        v1, e1 = _gk15_panel(func, lo, mid, args)
        v2, e2 = _gk15_panel(func, mid, hi, args)
        total += (v1 + v2) - val_top
        total_err += (e1 + e2) - err_top
        stack.append((e1, lo, mid, v1))
        stack.append((e2, mid, hi, v2))
        subdivisions += 1
    return sign * total, total_err


_DP_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)
_DP_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84),
)
_DP_B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
_DP_E = (71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40)


def _dp54_step(fun, t: float, y: np.ndarray, h: float, k1: np.ndarray):
    ks = [k1]
    for s in range(1, 7):
        ys = y.copy()
        for j, a_sj in enumerate(_DP_A[s]):
            if a_sj != 0.0:
                ys = ys + (h * a_sj) * ks[j]
        ks.append(np.asarray(fun(t + _DP_C[s] * h, ys), dtype=float))
    y_next = y.copy()
    err = np.zeros_like(y)
    for j in range(7):
        if _DP_B5[j] != 0.0:
            y_next = y_next + (h * _DP_B5[j]) * ks[j]
        if _DP_E[j] != 0.0:
            err = err + (h * _DP_E[j]) * ks[j]
    return y_next, err, ks


def _dp54_dense(y: np.ndarray, h: float, ks, theta: float) -> np.ndarray:
    b1 = theta * (1.0 + theta * (-1337/480 + theta * (1039/360 - 1163/1152 * theta)))
    b3 = 100.0 * theta * theta * (1054/9275 + theta * (-4682/27825 + 379/5565 * theta)) / 3.0
    b4 = -5.0 * theta * theta * (27/40 + theta * (-9/5 + 83/96 * theta)) / 2.0
    b5 = 18225.0 * theta * theta * (-3/250 + theta * (22/375 - 37/600 * theta)) / 848.0
    b6 = -22.0 * theta * theta * (-3/10 + theta * (29/30 - 17/24 * theta)) / 7.0
    return y + h * (b1 * ks[0] + b3 * ks[2] + b4 * ks[3] + b5 * ks[4] + b6 * ks[5])


def solve_ivp_fallback(fun, t_span, y0, method: str = "RK45", t_eval=None, dense_output: bool = False, events=None, vectorized: bool = False, args=None, **options):
    rtol = float(options.get("rtol", 1e-3))
    atol = options.get("atol", 1e-6)
    max_step = float(options.get("max_step", math.inf))
    if args is not None:
        raw_fun = fun
        fun = lambda t, y: raw_fun(t, y, *args)
        if events is not None:
            ev_list_raw = events if isinstance(events, (list, tuple)) else [events]
            wrapped_evs = []
            for ev in ev_list_raw:
                def _wrap_ev(t, y, _ev=ev):
                    return _ev(t, y, *args)
                _wrap_ev.terminal = getattr(ev, "terminal", False)
                _wrap_ev.direction = getattr(ev, "direction", 0.0)
                wrapped_evs.append(_wrap_ev)
            events = wrapped_evs
    t0, tf = float(t_span[0]), float(t_span[1])
    direction = 1.0 if tf >= t0 else -1.0
    y = np.asarray(y0, dtype=float).reshape(-1).copy()
    n = y.size
    atol_arr = np.asarray(atol, dtype=float)
    ev_list = [] if events is None else (list(events) if isinstance(events, (list, tuple)) else [events])
    t_events = [[] for _ in ev_list]
    y_events = [[] for _ in ev_list]

    k1 = np.asarray(fun(t0, y), dtype=float).reshape(-1)
    nfev = 1
    scale0 = atol_arr + rtol * np.abs(y)
    d0 = float(np.linalg.norm(y / scale0) / math.sqrt(max(n, 1)))
    d1 = float(np.linalg.norm(k1 / scale0) / math.sqrt(max(n, 1)))
    h_abs = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * (d0 / d1)
    h_abs = min(h_abs, abs(tf - t0), max_step)

    ts = [t0]
    ys = [y.copy()]
    segments = []
    ev_prev = [float(ev(t0, y)) for ev in ev_list]
    t = t0
    status = 0

    while direction * (tf - t) > 1e-15 * max(1.0, abs(t), abs(tf)):
        rem = abs(tf - t)
        h_abs = min(h_abs, rem, max_step)
        h = direction * h_abs
        y_next, err, ks = _dp54_step(fun, t, y, h, k1)
        nfev += 6
        sc = atol_arr + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err_norm = float(np.linalg.norm(err / sc) / math.sqrt(max(n, 1)))
        if err_norm <= 1.0 or h_abs <= 1e-15 * max(1.0, abs(t)):
            t_next = t + h
            ev_curr = [float(ev(t_next, y_next)) for ev in ev_list]
            term_hit = None
            for idx, ev in enumerate(ev_list):
                g0, g1 = ev_prev[idx], ev_curr[idx]
                ev_dir = float(getattr(ev, "direction", 0.0))
                crossed = False
                if g0 != 0.0 and (g1 == 0.0 or (g0 > 0.0) != (g1 > 0.0)):
                    if ev_dir == 0.0 or (ev_dir > 0.0 and g1 > g0) or (ev_dir < 0.0 and g1 < g0):
                        crossed = True
                if crossed:
                    def _ev_theta(th, _ev=ev, _y=y, _h=h, _ks=ks, _t=t):
                        yt = _dp54_dense(_y, _h, _ks, th)
                        return float(_ev(_t + th * _h, yt))
                    th_root = 1.0 if g1 == 0.0 else brentq_fallback(_ev_theta, 0.0, 1.0, xtol=1e-13, rtol=1e-13)
                    t_root = t + th_root * h
                    y_root = _dp54_dense(y, h, ks, th_root)
                    if getattr(ev, "terminal", False):
                        if term_hit is None or direction * t_root < direction * term_hit[0]:
                            term_hit = (t_root, y_root, idx)
                    else:
                        t_events[idx].append(t_root)
                        y_events[idx].append(y_root)
            if term_hit is not None:
                t_root, y_root, term_idx = term_hit
                for idx in range(len(ev_list)):
                    if idx == term_idx:
                        t_events[idx].append(t_root)
                        y_events[idx].append(y_root)
                    else:
                        keep = [i for i, te in enumerate(t_events[idx]) if direction * te <= direction * t_root]
                        t_events[idx] = [t_events[idx][i] for i in keep]
                        y_events[idx] = [y_events[idx][i] for i in keep]
                segments.append((t, t_root, y.copy(), h, ks))
                ts.append(t_root)
                ys.append(y_root.copy())
                status = 1
                break
            segments.append((t, t_next, y.copy(), h, ks))
            t = t_next
            y = y_next
            k1 = ks[6]
            ev_prev = ev_curr
            ts.append(t)
            ys.append(y.copy())
            factor = 5.0 if err_norm == 0.0 else min(5.0, max(0.2, 0.9 * (err_norm ** (-0.2))))
            h_abs *= factor
        else:
            factor = max(0.1, 0.9 * (err_norm ** (-0.25)))
            h_abs *= factor

    def _sol(tq):
        tq_arr = np.asarray(tq, dtype=float)
        scalar = tq_arr.ndim == 0
        tq_1d = np.atleast_1d(tq_arr)
        cols = []
        for te in tq_1d:
            if not segments:
                cols.append(ys[0])
                continue
            s_idx = 0
            while s_idx + 1 < len(segments) and direction * te > direction * segments[s_idx][1]:
                s_idx += 1
            st0, _st1, sy0, sh, sks = segments[s_idx]
            th = 0.0 if sh == 0.0 else max(0.0, min(1.0, (float(te) - st0) / sh))
            cols.append(_dp54_dense(sy0, sh, sks, th))
        out = np.column_stack(cols)
        return out[:, 0] if scalar else out

    if t_eval is not None:
        t_eval_arr = np.asarray(t_eval, dtype=float)
        t_end = ts[-1]
        mask = (direction * t_eval_arr >= direction * t0 - 1e-15) & (direction * t_eval_arr <= direction * t_end + 1e-15)
        t_out = t_eval_arr[mask]
        y_out = _sol(t_out) if t_out.size else np.empty((n, 0), dtype=float)
    else:
        t_out = np.asarray(ts, dtype=float)
        y_out = np.column_stack(ys)

    t_ev_out = [np.asarray(tev, dtype=float) for tev in t_events]
    y_ev_out = [np.vstack(yev) if yev else np.empty((0, n), dtype=float) for yev in y_events]
    return SimpleNamespace(
        t=t_out,
        y=y_out,
        sol=_sol if dense_output else None,
        t_events=t_ev_out if events is not None else None,
        y_events=y_ev_out if events is not None else None,
        nfev=nfev,
        njev=0,
        nlu=0,
        status=status,
        message="Termination event occurred." if status == 1 else "The solver successfully reached the end of the integration interval.",
        success=True,
    )


def lu_factor_fallback(a):
    mat = np.array(a, dtype=float, copy=True)
    n = mat.shape[0]
    piv = np.arange(n, dtype=int)
    for k in range(n):
        p = k + int(np.argmax(np.abs(mat[k:, k])))
        piv[k] = p
        if mat[p, k] == 0.0:
            continue
        if p != k:
            mat[[k, p], :] = mat[[p, k], :]
        mat[k + 1:, k] /= mat[k, k]
        mat[k + 1:, k + 1:] -= np.outer(mat[k + 1:, k], mat[k, k + 1:])
    return mat, piv


def lu_solve_fallback(lu_and_piv, b, trans: int = 0):
    lu, piv = lu_and_piv
    b_arr = np.array(b, dtype=float, copy=True)
    is_1d = b_arr.ndim == 1
    if is_1d:
        b_arr = b_arr[:, None]
    n = lu.shape[0]
    if trans == 0:
        for k in range(n):
            p = piv[k]
            if p != k:
                b_arr[[k, p], :] = b_arr[[p, k], :]
        for i in range(1, n):
            b_arr[i, :] -= lu[i, :i] @ b_arr[:i, :]
        for i in range(n - 1, -1, -1):
            b_arr[i, :] = (b_arr[i, :] - lu[i, i + 1:] @ b_arr[i + 1:, :]) / lu[i, i]
    else:
        for i in range(n):
            b_arr[i, :] = (b_arr[i, :] - lu[:i, i] @ b_arr[:i, :]) / lu[i, i]
        for i in range(n - 2, -1, -1):
            b_arr[i, :] -= lu[i + 1:, i] @ b_arr[i + 1:, :]
        for k in range(n - 1, -1, -1):
            p = piv[k]
            if p != k:
                b_arr[[k, p], :] = b_arr[[p, k], :]
    return b_arr[:, 0] if is_1d else b_arr


def eigh_fallback(a, b=None, eigvals_only: bool = False):
    a_sym = 0.5 * (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T)
    if b is None:
        return np.linalg.eigvalsh(a_sym) if eigvals_only else np.linalg.eigh(a_sym)
    b_sym = 0.5 * (np.asarray(b, dtype=float) + np.asarray(b, dtype=float).T)
    chol = np.linalg.cholesky(b_sym)
    chol_inv = np.linalg.inv(chol)
    c_mat = chol_inv @ a_sym @ chol_inv.T
    c_mat = 0.5 * (c_mat + c_mat.T)
    if eigvals_only:
        return np.linalg.eigvalsh(c_mat)
    vals, u_vecs = np.linalg.eigh(c_mat)
    vecs = chol_inv.T @ u_vecs
    return vals, vecs


class _IntegrateCompat:
    quad = staticmethod(_sp_integrate.quad if HAS_SCIPY else quad_fallback)
    solve_ivp = staticmethod(_sp_integrate.solve_ivp if HAS_SCIPY else solve_ivp_fallback)


class _OptimizeCompat:
    brentq = staticmethod(_sp_optimize.brentq if HAS_SCIPY else brentq_fallback)


class _LinalgCompat:
    eigh = staticmethod(_sp_linalg.eigh if HAS_SCIPY else eigh_fallback)
    lu_factor = staticmethod(_sp_linalg.lu_factor if HAS_SCIPY else lu_factor_fallback)
    lu_solve = staticmethod(_sp_linalg.lu_solve if HAS_SCIPY else lu_solve_fallback)


integrate = _IntegrateCompat()
optimize = _OptimizeCompat()
linalg = _LinalgCompat()
brentq = optimize.brentq
quad = integrate.quad
solve_ivp = integrate.solve_ivp
eigh = linalg.eigh
lu_factor = linalg.lu_factor
lu_solve = linalg.lu_solve
