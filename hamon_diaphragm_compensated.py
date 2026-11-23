"""Bounded rectangular Q4 solve with replay of JSON binary64 displacement pairs."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import numpy as np
from scipy.linalg import qr, solve_triangular


def _decimal(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite compensated diaphragm coefficient or increment")
    return Decimal.from_float(number)


def _fraction(value):
    return Fraction.from_float(float(value))


def _decimal_ratio(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _finite(value, *, allow_underflow=False):
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("nonfinite compensated diaphragm output") from exc
    if not math.isfinite(number):
        raise ValueError("nonfinite compensated diaphragm output")
    if number == 0 and value != 0 and not allow_underflow:
        raise ValueError("nonzero compensated diaphragm output is below binary64 range")
    return number


def check_compensated_size(spec):
    rows = 5*len(spec["panels"]) + len(spec.get("bars", [])) + len(spec["springs"])
    if 2*len(spec["nodes_m"]) > 64 or rows > 1024:
        raise ValueError("compensated diaphragm mode supports at most 64 displacement DOFs and 1024 strain rows")


def _strain_rows(spec, xy, springs):
    panels, bars = spec["panels"], spec.get("bars", [])
    rows = 5 * len(panels) + len(bars) + len(springs)
    dofs = 2 * len(xy)
    shape = np.full((rows, dofs), Fraction(0), dtype=object)
    weights = []
    for panel in panels:
        sw, _, ne, _ = panel["nodes"]
        width, height = xy[ne] - xy[sw]
        dx, dy, shear = (_fraction(panel[k]) for k in ("Dx_N_m", "Dy_N_m", "G_N_m"))
        row = len(weights)
        # Affine strains and the two mixed Q4 displacement coefficients
        # are orthogonal in the area integral of rectangular panel energy.
        for node, (sx, sy) in zip(panel["nodes"], ((-1,-1), (1,-1), (1,1), (-1,1))):
            x, y = 2*node, 2*node + 1
            shape[row, x] = shape[row+2, y] = sx / (2*width)
            shape[row+1, y] = shape[row+2, x] = sy / (2*height)
            shape[row+3, x] = sx*sy / (2*width)
            shape[row+4, y] = sx*sy / (2*height)
        weights.extend(width*height*t for t in
                       (dx, dy, shear, (dx + shear*(width/height)**2)/3,
                        (dy + shear*(height/width)**2)/3))
    for bar in bars:
        a, b = bar["nodes"]
        edge = xy[b] - xy[a]
        length = Fraction(_decimal_ratio(sum(v*v for v in edge)).sqrt())
        shape[len(weights), 2*a:2*a+2] = -edge / length
        shape[len(weights), 2*b:2*b+2] = edge / length
        weights.append(_fraction(bar["EA_N"]) / length)
    for dof, k in springs:
        shape[len(weights), dof] = Fraction(1)
        weights.append(_fraction(k))
    return shape, np.array(weights, dtype=object)


def solve_compensated(spec, nodes, springs, bar_checks):
    with localcontext() as context:
        context.prec = 60
        xy = np.array([[_fraction(x), _fraction(y)] for x, y in nodes], dtype=object)
        exact_shape, exact_weights = _strain_rows(spec, xy, springs)
        # Round only after summing all binary64 loads on each degree of freedom.
        exact_force = np.full(2*len(xy), Fraction(0), dtype=object)
        for load in spec["loads"]:
            for axis, value in enumerate(load["force_N"]):
                exact_force[2*load["node"] + axis] += _fraction(value)
        to_decimal = np.vectorize(_decimal_ratio, otypes=[object])
        shape = to_decimal(exact_shape)
        weights = to_decimal(exact_weights)
        force = to_decimal(exact_force)
        weighted = shape.astype(float) * np.sqrt(weights.astype(float))[:, None]
        norms = np.linalg.norm(weighted, axis=0)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("invalid compensated diaphragm stiffness")
        scale = 1 / norms
        _, factor, order = qr(weighted * scale, mode="economic", pivoting=True)
        pivots = np.abs(factor.diagonal())
        if len(pivots) != len(force) or min(pivots) <= 1e-12*max(pivots):
            raise ValueError("unsupported or ill-conditioned compensated diaphragm")

        motion = np.full(len(force), Decimal(0), dtype=object)
        residual = force.copy()
        for _ in range(5):
            rhs = (residual.astype(float) * scale)[order]
            intermediate = solve_triangular(factor.T, rhs, lower=True)
            increment = solve_triangular(factor, intermediate)
            motion[order] += [_decimal(v) for v in scale[order] * increment]
            residual = force - shape.T @ (weights * (shape @ motion))

        primary = np.array([_finite(v) for v in motion])
        tail = np.array([_finite(v - _decimal(h), allow_underflow=True) for v, h in zip(motion, primary)])
        # The replay uses exact sums and products of the wire values and
        # element coefficients, independently of the solver's rounded RHS.
        decoded = np.array([_fraction(h) + _fraction(l) for h,l in zip(primary, tail)])
        strain = exact_shape @ decoded
        demands = exact_weights * strain
        panel_end = 5 * len(spec["panels"])
        bar_end = panel_end + len(spec.get("bars", []))
        spring_forces = demands[bar_end:]
        reactions = np.full(len(force), Fraction(0), dtype=object)
        for (dof, _), value in zip(springs, spring_forces):
            reactions[dof] += value
        balance = (exact_force - reactions).reshape(-1, 2)
        arms = xy - xy[0]
        error_force = balance.sum(axis=0)
        error_moment = arms[:,0] @ balance[:,1] - arms[:,1] @ balance[:,0]
        force_scale = max(sum(abs(exact_force)) + sum(abs(reactions)), Fraction(1, 10**30))
        span = max(max(axis)-min(axis) for axis in xy.T)
        work, energy = exact_force @ decoded, demands @ strain
        errors = [max(abs(exact_shape.T @ demands - exact_force)) / force_scale,
                  sum(abs(error_force)) / force_scale, abs(error_moment) / (span * force_scale),
                  abs(work-energy) / max(abs(work), abs(energy), Fraction(1, 10**30))]
        discrepancy = _finite(max(errors), allow_underflow=True)
        if discrepancy > 1e-7:
            raise ValueError("compensated diaphragm failed equilibrium or work-energy replay")
        # EA * edge·(ub-ua) / |edge|² is the axial force; this form
        # cancels both square roots even for an inclined bar.
        tensions = []
        for bar in spec.get("bars", []):
            a, b = bar["nodes"]
            edge = xy[b] - xy[a]
            relative = decoded[2*b:2*b+2] - decoded[2*a:2*a+2]
            tensions.append(_fraction(bar["EA_N"]) * (edge @ relative) / (edge @ edge))
        return {"displacements_m": primary.reshape(-1, 2).tolist(),
                "displacement_remainders_m": tail.reshape(-1, 2).tolist(),
                "verified_displacement_representation": "binary64_primary_plus_remainder",
                "panel_center_shear_N_m": [_finite(_fraction(p["G_N_m"])*strain[5*i+2])
                                           for i,p in enumerate(spec["panels"])],
                "bar_tension_N": [_finite(v) for v in tensions],
                "bar_capacity_checks": bar_checks(spec, tensions, convert=_fraction, output=_finite),
                "spring_transmitted_N": [_finite(v) for v in spring_forces],
                "force_residual_N": [_finite(v, allow_underflow=True) for v in error_force],
                "moment_residual_Nm": _finite(error_moment, allow_underflow=True), "strain_energy_J": _finite(energy/2),
                "external_work_J": _finite(work/2), "dual_rel_discrepancy": discrepancy,
                "dual_verified": True}
