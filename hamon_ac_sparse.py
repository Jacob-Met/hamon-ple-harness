"""Sparse linear algebra for AC network verification.

The independent rectangular solve starts from the original flat voltage seed.
Its residual and derivatives use explicit branch terminals, never the polar
solver's bus-admittance matrix or its converged voltage as a starting guess.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def linear_step(matrix, residual):
    """Solve a Newton correction without making a sparse graph dense."""
    if not sparse.issparse(matrix):
        return np.linalg.solve(matrix, residual)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            correction = spsolve(matrix.tocsc(), residual)
    except MatrixRankWarning as exc:
        raise np.linalg.LinAlgError("Singular sparse AC Jacobian") from exc
    if not np.isfinite(correction).all():
        raise np.linalg.LinAlgError("Nonfinite sparse AC correction")
    return correction


def terminal_derivative(shunt, f, t, y, tap, yc):
    """Differentiate each endpoint current, accumulating parallel branches."""
    n = len(shunt)
    nodes = np.arange(n)
    row = np.r_[nodes, f, f, t, t]
    col = np.r_[nodes, f, t, f, t]
    values = np.r_[shunt, (y + yc) / (tap * tap.conjugate()),
                   -y / tap.conjugate(), -y / tap, y + yc]
    return sparse.coo_matrix((values, (row, col)), shape=(n, n)).tocsr()


def current_jacobian(voltage, current, terminal, nonslack, pq, pv, p, q):
    """Real Jacobian of PQ KCL, PV active power and squared voltage."""
    derivative = np.zeros(len(voltage), dtype=complex)
    derivative[pq] = np.conj((p[pq] + 1j * q[pq]) / voltage[pq] ** 2)
    constant_power = sparse.diags(derivative, format="csr")
    real_current = terminal + constant_power
    imag_current = 1j * (terminal - constant_power)
    vd = sparse.diags(voltage, format="csr")
    ic = sparse.diags(np.conj(current), format="csr")
    real_power = (ic + vd @ terminal.conjugate()).real
    imag_power = (1j * ic - 1j * vd @ terminal.conjugate()).real
    magnitude_real = sparse.diags(2 * voltage.real, format="csr")
    magnitude_imag = sparse.diags(2 * voltage.imag, format="csr")

    def select(matrix, rows):
        return matrix[rows, :][:, nonslack]

    return sparse.bmat([
        [select(real_current.real, pq), select(imag_current.real, pq)],
        [select(real_current.imag, pq), select(imag_current.imag, pq)],
        [select(real_power, pv), select(imag_power, pv)],
        [select(magnitude_real, pv), select(magnitude_imag, pv)],
    ], format="csc")


def dc_angle_seed(initial, unpack, nonslack, p, shunt, f, t, y, tap):
    """Independent lossless angle estimate, used only to start exact AC KCL.

    A zero-reactance branch or a singular signed-reactance graph makes this
    approximation unavailable. Neither the seed nor its DC equations can
    qualify the final AC result.
    """
    n, m = len(p), len(f)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reactance = (1 / y).imag
        weights = 1 / (reactance * np.abs(tap))
    if np.any(reactance == 0) or not np.isfinite(weights).all():
        return None
    edge = np.arange(m)
    incidence = sparse.coo_matrix(
        (np.r_[np.ones(m), -np.ones(m)], (np.r_[edge, edge], np.r_[f, t])),
        shape=(m, n),
    ).tocsr()
    susceptance = (incidence.T @ sparse.diags(weights) @ incidence).tocsr()
    flat = unpack(initial)
    slack = np.setdiff1d(np.arange(n), nonslack)
    angles = np.zeros(n)
    angles[slack] = np.angle(flat[slack])
    rhs = p - shunt.real + incidence.T @ (weights * np.angle(tap)) - susceptance @ angles
    try:
        angles[nonslack] = linear_step(susceptance[nonslack, :][:, nonslack], rhs[nonslack])
    except np.linalg.LinAlgError:
        return None
    voltage = np.abs(flat) * np.exp(1j * angles)
    return np.r_[voltage[nonslack].real, voltage[nonslack].imag]


def rectangular_current_solve(initial, unpack, balance, branch_currents,
                              nonslack, pq, pv, p, q, shunt, f, t, y, tap, yc):
    """Globalized independent KCL Newton solve; return the actual final iterate.

    Failure to converge never certifies an answer: the caller still checks this
    residual, agreement with its polar solve and emitted-voltage conservation.
    """
    terminal = terminal_derivative(shunt, f, t, y, tap, yc)

    def iterate(seed):
        x = seed.copy()
        for _ in range(60):
            residual = balance(x)
            norm = float(np.linalg.norm(residual, np.inf))
            if norm < 1e-11:
                return x
            voltage = unpack(x)
            if np.any(np.abs(voltage[pq]) < 1e-12):
                return x
            current = branch_currents(voltage)[0]
            jacobian = current_jacobian(voltage, current, terminal, nonslack, pq, pv, p, q)
            try:
                correction = linear_step(jacobian, residual)
            except np.linalg.LinAlgError:
                return x
            for exponent in range(24):
                trial = x - correction / 2**exponent
                trial_residual = balance(trial)
                if np.isfinite(trial_residual).all() and np.linalg.norm(trial_residual, np.inf) < norm:
                    x = trial
                    break
            else:
                return x
        return x

    flat_result = iterate(initial)
    flat_error = float(np.linalg.norm(balance(flat_result), np.inf))
    if flat_error < 1e-11:
        return flat_result
    seed = dc_angle_seed(initial, unpack, nonslack, p, shunt, f, t, y, tap)
    if seed is None:
        return flat_result
    other = iterate(seed)
    return other if np.linalg.norm(balance(other), np.inf) < flat_error else flat_result
