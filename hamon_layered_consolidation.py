"""Coupled, linear one-dimensional consolidation of perfectly contacting layers.

The governing equation is mv*u_t = (cv*mv*u_z)_z + mv*q_t. Pressure and
Darcy flux are continuous at internal interfaces. Only the outer faces drain.
A conservative cell-centred diffusion model is checked against an independent
Laplace-domain solution which solves each layer analytically in space.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.linalg import eigh_tridiagonal

from hamon_consolidation import _history_segments, _number


_FIELDS = frozenset({"calculation", "mode", "consolidation_layers", "time_s", "drainage",
                     "load_history", "proposed_settlement_m",
                     "proposed_mean_excess_pore_pressure_Pa"})
_LAYER_FIELDS = {"thickness_m", "cv_m2_s", "mv_per_Pa"}


def _rounded_dimension(exact):
    value = float(exact)
    if not math.isfinite(value) or (exact and value == 0):
        raise ValueError("A dimensional result is outside the finite nonzero output range")
    return value


def _finite_volume(layers, segments, factor, drainage, cells):
    """Conservative fluxes, exact modal time integration, and layer averages."""
    count = len(layers) * cells
    widths = np.concatenate([np.full(cells, h / cells) for h, _, _ in layers])
    storage = np.concatenate([np.full(cells, mv) for _, _, mv in layers])
    conductivities = np.concatenate([np.full(cells, cv * mv) for _, cv, mv in layers])
    mass = storage * widths
    resistance = widths / (2 * conductivities)
    conductance = 1 / (resistance[:-1] + resistance[1:])
    diagonal = np.zeros(count)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    diagonal[0] += 1 / resistance[0]
    if drainage == "double":
        diagonal[-1] += 1 / resistance[-1]
    diagonal /= mass
    off_diagonal = -conductance / np.sqrt(mass[:-1] * mass[1:])
    rates, modes = eigh_tridiagonal(diagonal, off_diagonal, lapack_driver="stev")
    if np.any(rates <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("The layer diffusion spectrum is not numerically resolved")
    forcing = np.sqrt(mass) @ modes
    effective = np.zeros(count)
    pressure = np.zeros(count)
    for start, end, initial, final in segments:
        dt = (end - start) * factor
        change = final - initial
        if dt == 0:
            if end != start:
                raise ValueError("A loading interval underflows the diffusion time scale")
            pressure += change
            continue
        x = rates * dt
        decay = np.exp(-x)
        retained = -np.expm1(-x) / x
        ramp = 1 - retained
        old = retained - decay
        small = x < 1e-5
        ramp[small] = x[small] * (0.5 - x[small] / 6 + x[small]**2 / 24)
        old[small] = x[small] * (0.5 - x[small] / 3 + x[small]**2 / 8)
        effective = effective * decay + initial * old + final * ramp
        pressure = pressure * decay + change * retained
        if final == 0:
            pressure = -effective
    effective_field = (modes @ (forcing * effective)) / np.sqrt(mass)
    pressure_field = (modes @ (forcing * pressure)) / np.sqrt(mass)
    means_e = effective_field.reshape(len(layers), cells).mean(axis=1)
    means_p = pressure_field.reshape(len(layers), cells).mean(axis=1)
    return np.array([means_e, means_p])


def _laplace_history(layers, history, elapsed, factor, scale, drainage, degree):
    """Independent exact-in-space layer transfer with de Hoog inversion.

    In the Laplace domain effective pressure w solves cv*w'' = s*w.
    Its prescribed drained-face value for a unit load step is 1/s.
    Layer dynamic stiffness enforces pressure and flux continuity directly.
    """
    from mpmath import mp

    ctx = mp.clone()
    ctx.dps = 60
    rows = [(ctx.mpf(h), ctx.mpf(cv), ctx.mpf(mv)) for h, cv, mv in layers]
    n = len(rows)
    cache = {}
    kernels = {}

    def transformed(s):
        key = (s.real, s.imag)
        if key in cache:
            return cache[key]
        stiffness = ctx.matrix(n + 1)
        roots = []
        for j, (h, cv, mv) in enumerate(rows):
            root = ctx.sqrt(s / cv)
            roots.append(root)
            a = cv * mv * root
            diagonal = a * ctx.coth(root * h)
            coupling = -a / ctx.sinh(root * h)
            stiffness[j, j] += diagonal
            stiffness[j + 1, j + 1] += diagonal
            stiffness[j, j + 1] += coupling
            stiffness[j + 1, j] += coupling
        known = {0: 1 / s}
        if drainage == "double":
            known[n] = 1 / s
        free = [j for j in range(n + 1) if j not in known]
        values = [known.get(j, ctx.zero) for j in range(n + 1)]
        if free:
            matrix = ctx.matrix([[stiffness[i, j] for j in free] for i in free])
            rhs = ctx.matrix([-ctx.fsum(stiffness[i, j] * v for j, v in known.items())
                              for i in free])
            solution = ctx.lu_solve(matrix, rhs)
            for j, value in zip(free, solution):
                values[j] = value
        result = tuple((values[j] + values[j + 1]) * ctx.tanh(roots[j] * h / 2)
                       / (roots[j] * h) for j, (h, _, _) in enumerate(rows))
        cache[key] = result
        return result

    def kernel(age, integral, pressure):
        if age == 0:
            return [ctx.one if pressure and not integral else ctx.zero] * n
        key = (age, integral, pressure)
        if key not in kernels:
            result = []
            for j in range(n):
                def transform(s):
                    value = transformed(s)[j]
                    if pressure:
                        value = 1 / s - value
                    return value / s if integral else value
                result.append(ctx.invertlaplace(transform, age, method="dehoog", degree=degree))
            kernels[key] = result
        return kernels[key]

    time = ctx.mpf(elapsed)
    diffusion_factor = ctx.mpf(factor)
    answer = []
    segments = _history_segments(history, time, ctx.mpf)
    for start, end, first, last in segments:
        if end > start and (first or last):
            young = (time - end) * diffusion_factor
            span = (end - start) * diffusion_factor
            if young + span == young:
                raise ValueError("Loading interval is unresolved at the independent inversion precision")
    for pressure in (False, True):
        terms = [[] for _ in rows]
        for start, end, first, last in segments:
            change = (last - first) / scale
            if not change:
                continue
            young = (time - end) * diffusion_factor
            span = (end - start) * diffusion_factor
            if span and young + span == young:
                raise ValueError("Loading interval is unresolved at the independent inversion precision")
            if not span:
                response = kernel(young, False, pressure)
            else:
                before = kernel(young + span, True, pressure)
                after = kernel(young, True, pressure)
                response = [(a - b) / span for a, b in zip(before, after)]
            for j, value in enumerate(response):
                terms[j].append(change * value)
        summed = [ctx.fsum(values) for values in terms]
        if any(value and float(value) == 0 for value in summed):
            raise ValueError("A normalized pressure component underflows the independent solver")
        answer.append([float(value) for value in summed])
    return np.array(answer)


def verify_layered_consolidation(spec):
    out = {"ok": False, "tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
           "calculation": "layered_consolidation_1d", "conservation_verified": False}
    try:
        if spec.get("calculation") != "layered_consolidation_1d" or spec.get("mode", "auto") not in (
                "auto", "civil_geotech", "civil", "geotech"):
            raise ValueError("Require the layered_consolidation_1d civil calculation")
        if set(spec) - _FIELDS:
            raise ValueError("Only constant linear layer properties and a uniform surcharge history are supported")
        raw = spec.get("consolidation_layers")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
            raise ValueError("Provide one to eight consolidation_layers in top-to-bottom order")
        layers = []
        for row in raw:
            if not isinstance(row, dict) or set(row) != _LAYER_FIELDS:
                raise ValueError("Each layer requires thickness_m, cv_m2_s and mv_per_Pa only")
            values = tuple(_number(row, key) for key in ("thickness_m", "cv_m2_s", "mv_per_Pa"))
            if min(values) <= 0:
                raise ValueError("Layer thickness, diffusivity and compressibility must be positive")
            layers.append(values)
        elapsed = _number(spec, "time_s")
        drainage = spec.get("drainage")
        if elapsed < 0 or drainage not in ("single", "double"):
            raise ValueError("Require nonnegative time_s and single or double outer-face drainage")
        history = spec.get("load_history")
        if not isinstance(history, list) or len(history) > 32:
            raise ValueError("Provide at most 32 load_history ordinates")
        segments = _history_segments(history, elapsed)
        height = math.fsum(h for h, _, _ in layers)
        cv_scale = max(cv for _, cv, _ in layers)
        mv_scale = max(mv for _, _, mv in layers)
        factor = (cv_scale / height) / height
        scale = max(max(first, last) for _, _, first, last in segments)
        if not math.isfinite(height) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("Layer dimensions do not resolve a finite positive diffusion time scale")
        if scale * mv_scale >= 1:
            raise ValueError("Applied loads exceed the linear small-strain model's physical range")
        normalized = [(h / height, cv / cv_scale, mv / mv_scale) for h, cv, mv in layers]
        if any(min(row) <= 0 for row in normalized):
            raise ValueError("Layer property ratios underflow the numerical representation")
        current = segments[-1][3]
        if scale == 0 or elapsed == 0:
            result = np.array([[0.] * len(layers), [current / scale if scale else 0.] * len(layers)])
            finite_volume = result.copy()
            inversion_residual = dual_residual = balance_residual = component_residual = 0.
        else:
            normalized_history = [(a, b, c / scale, d / scale) for a, b, c, d in segments]
            coarse = _finite_volume(normalized, normalized_history, factor, drainage, 64)
            fine = _finite_volume(normalized, normalized_history, factor, drainage, 128)
            finite_volume = (4 * fine - coarse) / 3
            first = _laplace_history(normalized, history, elapsed, factor, scale, drainage, 24)
            result = _laplace_history(normalized, history, elapsed, factor, scale, drainage, 36)
            inversion_residual = float(np.max(np.abs(result - first)))
            dual_residual = float(np.max(np.abs(result - finite_volume)))
            component_scale = np.maximum(np.abs(result), np.abs(finite_volume))
            component_error = np.divide(np.abs(result - finite_volume), component_scale,
                                        out=np.zeros_like(result), where=component_scale > 0)
            component_residual = float(np.max(component_error))
            balance_residual = float(np.max(np.abs(result.sum(axis=0) - current / scale)))
            if (not np.all(np.isfinite(result)) or not np.all(np.isfinite(finite_volume))
                    or inversion_residual >= 1e-9 or dual_residual >= 1e-6 or component_residual >= 1e-6
                    or balance_residual >= 1e-9 or np.min(result[0]) < -1e-10
                    or np.max(result[0]) > 1 + 1e-10 or np.max(np.abs(result[1])) > 1 + 1e-10):
                raise ValueError("Independent layer solutions or physical bounds did not converge below the residual gate")
            nonzero_duration_load = any(b > a and (first > 0 or last > 0)
                                        for a, b, first, last in segments)
            increasing_load = all(last >= first for _, _, first, last in segments)
            if ((nonzero_duration_load and np.any(result[0] <= 0))
                    or (increasing_load and np.any(result[1] <= 0))):
                raise ValueError("A strictly positive finite-time response is not numerically resolved")
        # Keep every binary64 factor exact until the final dimensional rounding.
        # A small strain or compliance may underflow even when h*mv*q is finite.
        F = lambda x: Fraction(float(x))
        load = F(scale)
        capacities = [F(h) * F(mv) for h, _, mv in layers]
        capacity = sum(capacities)
        settlement_terms = [c * load * F(e) for c, e in zip(capacities, result[0])]
        settlements = [_rounded_dimension(value) for value in settlement_terms]
        settlement = _rounded_dimension(sum(settlement_terms))
        layer_pressure = [_rounded_dimension(load * F(p)) for p in result[1]]
        layer_effective = [_rounded_dimension(load * F(e)) for e in result[0]]
        mean_pressure = _rounded_dimension(sum(F(h) * load * F(p) for (h, _, _), p in zip(layers, result[1]))
                                           / sum(F(h) for h, _, _ in layers))
        stored_pressure = _rounded_dimension(sum(c * load * F(p) for c, p in zip(capacities, result[1])) / capacity)
        ultimate = _rounded_dimension(capacity * F(current))
        fv_pressure = [_rounded_dimension(load * F(p)) for p in finite_volume[1]]
        for key, value in (("proposed_settlement_m", settlement),
                           ("proposed_mean_excess_pore_pressure_Pa", mean_pressure)):
            if key in spec and not math.isclose(_number(spec, key), value, rel_tol=1e-6, abs_tol=0.):
                raise ValueError("Proposed value disagrees with the independently checked layered solution")
        out.update(ok=True, conservation_verified=True, drainage=drainage, time_s=elapsed,
                   settlement_at_time_m=settlement, ultimate_settlement_at_current_load_m=ultimate,
                   mean_excess_pore_pressure_Pa=mean_pressure,
                   storage_weighted_excess_pore_pressure_Pa=stored_pressure,
                   layer_mean_excess_pore_pressure_Pa=layer_pressure,
                   layer_mean_effective_increment_Pa=layer_effective, layer_settlement_m=settlements,
                   finite_volume_layer_mean_excess_pore_pressure_Pa=fv_pressure,
                   dual_solver_rel_residual=dual_residual, inversion_rel_residual=inversion_residual,
                   component_rel_residual=component_residual,
                   load_balance_rel_residual=balance_residual, residual_load_scale_Pa=scale,
                   model="constant mv and cv, uniform loading, perfect hydraulic contact, outer-face drainage",
                   methods=["analytic layer transfer and de Hoog Laplace inversion",
                            "conservative finite volume with exact modal evolution and Richardson refinement"])
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        out["reason"] = str(exc)
    return out
