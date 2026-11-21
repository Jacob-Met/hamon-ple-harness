"""Pure-Python / NumPy high-precision numerical fallbacks when SciPy is absent."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Sequence, Tuple

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


def brentq_fallback(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
    maxiter: int = 200,
) -> float:
    """Brent-Dekker bracketed root finder matching scipy.optimize.brentq."""
    xpre, xcur = float(a), float(b)
    fpre, fcur = float(f(xpre)), float(f(xcur))
    if fpre == 0.0:
        return xpre
    if fcur == 0.0:
        return xcur
    if (fpre > 0.0) == (fcur > 0.0):
        raise ValueError(f"f(a)={fpre} and f(b)={fcur} must have opposite signs")
    xblk, fblk = xpre, fpre
    spre = scur = xcur - xpre
    for _ in range(maxiter):
        if fcur != 0.0 and fblk != 0.0 and ((fcur > 0.0) != (fblk > 0.0)):
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
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
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
        fcur = float(f(xcur))
    return xcur


_GK15_X = (
    0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945,
    0.5860872354676911, 0.4058451513773972, 0.2077849550078985, 0.0,
)
_GK15_W = (
    0.0229353220105292, 0.0630920926299786, 0.1047900103222502, 0.1406532597155259,
    0.1690047266392679, 0.1903505780647854, 0.2044329400752989, 0.2094821410847278,
)
_G7_W = (
    0.0, 0.1294849661688697, 0.0, 0.2797053914892766,
    0.0, 0.3818300505051189, 0.0, 0.4179591836734694,
)


def _gk15_panel(f: Callable[[float], float], a: float, b: float) -> Tuple[float, float]:
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    fc = float(f(mid))
    i_k = _GK15_W[7] * fc
    i_g = _G7_W[7] * fc
    for k in range(7):
        dx = half * _GK15_X[k]
        fsum = float(f(mid - dx)) + float(f(mid + dx))
        i_k += _GK15_W[k] * fsum
        if _G7_W[k]:
            i_g += _G7_W[k] * fsum
    i_k *= half
    i_g *= half
    return i_k, abs(i_k - i_g)


def quad_fallback(
    func: Callable[[float], float],
    a: float,
    b: float,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 200,
) -> Tuple[float, float]:
    """Adaptive 15-point Gauss-Kronrod quadrature matching scipy.integrate.quad."""
    if a == b:
        return 0.0, 0.0
    i0, e0 = _gk15_panel(func, float(a), float(b))
    stack = [(float(a), float(b), i0, e0)]
    total = i0
    err_total = e0
    subdivs = 1
    while stack and subdivs < max(limit, 16):
        stack.sort(key=lambda item: item[3])
        la, lb, li, le = stack.pop()
        tol = max(epsabs, epsrel * abs(total))
        if err_total <= tol and le <= tol:
            break
        mid = 0.5 * (la + lb)
        i1, e1 = _gk15_panel(func, la, mid)
        i2, e2 = _gk15_panel(func, mid, lb)
        total += (i1 + i2) - li
        err_total = max(0.0, err_total + (e1 + e2) - le)
        stack.append((la, mid, i1, e1))
        stack.append((mid, lb, i2, e2))
        subdivs += 1
    return float(total), float(max(err_total, 0.0))


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
_DP_E = (71/57600, 0.0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40)


def solve_ivp_fallback(
    fun: Callable[[float, np.ndarray], Sequence[float]],
    t_span: Tuple[float, float],
    y0: Sequence[float],
    method: str = "RK45",
    rtol: float = 1e-9,
    atol: float = 1e-12,
    **_: Any,
) -> SimpleNamespace:
    """Adaptive Dormand-Prince 5(4) integrator matching scipy.integrate.solve_ivp."""
    t0, tf = float(t_span[0]), float(t_span[1])
    y = np.asarray(y0, dtype=float).copy()
    if tf == t0:
        return SimpleNamespace(t=np.array([t0]), y=y[:, None], success=True)
    direction = 1.0 if tf > t0 else -1.0
    rtol_eff = max(min(float(rtol), 1e-11), 1e-13)
    atol_eff = max(min(float(atol), 1e-13), 1e-15)
    h = direction * min(abs(tf - t0) / 32.0, 1e-2 * max(abs(tf - t0), 1.0))
    t = t0
    ts = [t0]
    ys = [y.copy()]
    k1 = np.asarray(fun(t, y), dtype=float)
    steps = 0
    while direction * (tf - t) > 1e-15 * max(1.0, abs(tf)) and steps < 200000:
        if direction * (t + h - tf) > 0.0:
            h = tf - t
        ks = [k1]
        for s in range(1, 7):
            y_stage = y.copy()
            for j, a_sj in enumerate(_DP_A[s]):
                if a_sj != 0.0:
                    y_stage = y_stage + (h * a_sj) * ks[j]
            ks.append(np.asarray(fun(t + _DP_C[s] * h, y_stage), dtype=float))
        y_next = y.copy()
        for j, b_j in enumerate(_DP_A[6]):
            if b_j != 0.0:
                y_next = y_next + (h * b_j) * ks[j]
        err_vec = np.zeros_like(y)
        for j, e_j in enumerate(_DP_E):
            if e_j != 0.0:
                err_vec = err_vec + (h * e_j) * ks[j]
        scale = atol_eff + rtol_eff * np.maximum(np.abs(y), np.abs(y_next))
        err_norm = float(np.sqrt(np.mean((err_vec / scale) ** 2)))
        if err_norm <= 1.0 or abs(h) <= 1e-15 * max(1.0, abs(t)):
            t = t + h
            y = y_next
            k1 = ks[6]
            ts.append(t)
            ys.append(y.copy())
            steps += 1
        factor = 2.5 if err_norm == 0.0 else min(2.5, max(0.2, 0.9 * (err_norm ** -0.2)))
        h *= factor
    return SimpleNamespace(t=np.asarray(ts, dtype=float), y=np.column_stack(ys), success=True)


def lu_factor_fallback(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Partial-pivoting LU decomposition matching scipy.linalg.lu_factor."""
    lu = np.asarray(a, dtype=float).copy()
    n = lu.shape[0]
    piv = np.arange(n, dtype=int)
    for k in range(n):
        p = int(k + np.argmax(np.abs(lu[k:, k])))
        piv[k] = p
        if p != k:
            lu[[k, p], :] = lu[[p, k], :]
        diag = lu[k, k]
        if abs(diag) < 1e-30:
            continue
        lu[k + 1 :, k] /= diag
        lu[k + 1 :, k + 1 :] -= np.outer(lu[k + 1 :, k], lu[k, k + 1 :])
    return lu, piv


def lu_solve_fallback(lu_and_piv: Tuple[np.ndarray, np.ndarray], b: np.ndarray) -> np.ndarray:
    """Forward/backward solve matching scipy.linalg.lu_solve."""
    lu, piv = lu_and_piv
    x = np.asarray(b, dtype=float).copy()
    n = lu.shape[0]
    for k in range(n):
        p = int(piv[k])
        if p != k:
            x[k], x[p] = x[p].copy(), x[k].copy()
    for i in range(1, n):
        x[i] -= np.dot(lu[i, :i], x[:i])
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - np.dot(lu[i, i + 1 :], x[i + 1 :])) / lu[i, i]
    return x


def eigh_fallback(a: np.ndarray, b: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric (generalized) eigensolver matching scipy.linalg.eigh(a, b)."""
    a_sym = 0.5 * (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T)
    if b is None:
        return np.linalg.eigh(a_sym)
    b_sym = 0.5 * (np.asarray(b, dtype=float) + np.asarray(b, dtype=float).T)
    chol_l = np.linalg.cholesky(b_sym)
    l_inv = np.linalg.inv(chol_l)
    c_sym = l_inv @ a_sym @ l_inv.T
    c_sym = 0.5 * (c_sym + c_sym.T)
    evals, u = np.linalg.eigh(c_sym)
    evecs = l_inv.T @ u
    return evals, evecs


integrate = _sp_integrate if HAS_SCIPY else SimpleNamespace(quad=quad_fallback, solve_ivp=solve_ivp_fallback)
optimize = _sp_optimize if HAS_SCIPY else SimpleNamespace(brentq=brentq_fallback)
linalg = _sp_linalg if HAS_SCIPY else SimpleNamespace(
    lu_factor=lu_factor_fallback,
    lu_solve=lu_solve_fallback,
    eigh=eigh_fallback,
)
