"""Endpoint replay in the normalized circular restricted three-body problem.

The frame and state ordering follow NASA/JPL's Three-Body Periodic Orbits API.
The two primaries are fixed in the synodic frame and revolve in the inertial
frame. An endpoint certificate compares independently propagated states.
"""
import math

import numpy as np

from _scipy_compat import integrate


def _libration_state(mu, point):
    secondary = 1.0 - mu
    if point in ("L4", "L5"):
        sign = 1 if point == "L4" else -1
        return [0.5 - mu, sign * math.sqrt(3) / 2, 0.0, 0.0, 0.0, 0.0]

    # The axial force increases strictly between consecutive primaries. Bisect
    # from the open interval without evaluating at either singular endpoint.
    brackets = {"L1": (-mu, secondary), "L2": (secondary, 2.0), "L3": (-2.0, -mu)}
    if point not in brackets:
        raise ValueError("unknown_libration_point")
    lo, hi = brackets[point]

    def force(x):
        a, b = x + mu, x - secondary
        return x - math.copysign((1 - mu) / a**2, a) - math.copysign(mu / b**2, b)

    best = None
    for _ in range(100):
        x = (lo + hi) / 2
        if x == lo or x == hi:
            break
        value = force(x)
        if best is None or abs(value) < best[0]:
            best = (abs(value), x)
        if value < 0:
            lo = x
        else:
            hi = x
    if best is None or best[0] > 1e-10:
        raise ValueError("unresolved_libration_point")
    return [best[1], 0.0, 0.0, 0.0, 0.0, 0.0]


def verify_cr3bp_transfer(params):
    """Check an unpowered synodic initial/target state at a known flight time."""
    result = {"domain": "lagrangian_orbital", "subkind": "cr3bp_transfer", "verified": False}
    allowed = {"domain", "kind", "prompt", "subkind", "mass_ratio", "flight_time_tu",
               "initial_state", "target_state", "target_lagrange", "jacobi"}
    if set(params) - allowed:
        return dict(result, error="unsupported_cr3bp_obligation")

    def scalar(value):
        if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
            raise ValueError("invalid_cr3bp_number")
        return float(value)

    def state(values):
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise ValueError("expected_six_synodic_state_components")
        return [scalar(v) for v in values]

    try:
        mu, duration = scalar(params["mass_ratio"]), scalar(params["flight_time_tu"])
        if not 0 < mu <= 0.5 or duration <= 0:
            raise ValueError("invalid_mass_ratio_or_flight_time")
        initial = state(params["initial_state"])
        if ("target_state" in params) == ("target_lagrange" in params):
            raise ValueError("specify_one_target_state_or_lagrange_point")
        target = (state(params["target_state"]) if "target_state" in params
                  else _libration_state(mu, params["target_lagrange"]))
        declared = scalar(params["jacobi"]) if "jacobi" in params else None
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        return dict(result, error=str(exc))

    def gravity(x, y, z, cosine, sine):
        # The barycentric primary positions are -mu*u and (1-mu)*u.
        dx1, dy1 = x + mu*cosine, y + mu*sine
        dx2, dy2 = x - (1-mu)*cosine, y - (1-mu)*sine
        r1, r2 = math.hypot(dx1, dy1, z), math.hypot(dx2, dy2, z)
        if min(r1, r2) <= 1e-10:
            raise ValueError("unresolved_primary_encounter")
        f1, f2 = (1-mu)/r1**3, mu/r2**3
        return -f1*dx1-f2*dx2, -f1*dy1-f2*dy2, -(f1+f2)*z

    def jacobi(y):
        x, yy, z, vx, vy, vz = y
        r1 = math.hypot(x + mu, yy, z)
        r2 = math.hypot(x - (1-mu), yy, z)
        if min(r1, r2) <= 1e-10:
            raise ValueError("unresolved_primary_encounter")
        return x*x + yy*yy + 2*((1-mu)/r1 + mu/r2) - (vx*vx + vy*vy + vz*vz)

    def rotating(t, y):
        x, yy, z, vx, vy, vz = y
        ax, ay, az = gravity(x, yy, z, 1, 0)
        return [vx, vy, vz, ax + x + 2*vy, ay + yy - 2*vx, az]

    def inertial(t, y):
        x, yy, z, vx, vy, vz = y
        ax, ay, az = gravity(x, yy, z, math.cos(t), math.sin(t))
        return [vx, vy, vz, ax, ay, az]

    calls = 0
    def propagate(rhs, initial_state):
        def bounded(t, y):
            nonlocal calls
            calls += 1
            if calls > 60000:
                raise ValueError("cr3bp_integration_work_limit")
            return rhs(t, y)
        sol = integrate.solve_ivp(bounded, (0.0, duration), initial_state, method="DOP853",
                                  max_step=min(0.04, duration), rtol=2e-12, atol=2e-14)
        if not sol.success or abs(sol.t[-1] - duration) > 4*math.ulp(duration):
            raise ValueError("cr3bp_integration_incomplete")
        if not np.isfinite(sol.y).all():
            raise ValueError("nonfinite_cr3bp_trajectory")
        return sol.y

    try:
        reference_c = jacobi(initial)
        first = propagate(rotating, initial)
        x, y, z, vx, vy, vz = initial
        second = propagate(inertial, [x, y, z, vx-y, vy+x, vz])[:, -1]
        cosine, sine = math.cos(duration), math.sin(duration)
        ix, iy, iz, ivx, ivy, ivz = second
        sx, sy = cosine*ix + sine*iy, -sine*ix + cosine*iy
        replay = np.array([sx, sy, iz, cosine*ivx + sine*ivy + sy,
                           -sine*ivx + cosine*ivy - sx, ivz])
        endpoint = first[:, -1]
        cross_frame = float(np.max(np.abs(replay - endpoint)))
        position_error = float(np.max(np.abs(endpoint[:3] - target[:3])))
        velocity_error = float(np.max(np.abs(endpoint[3:] - target[3:])))
        drift = max(abs(jacobi(row) - reference_c) for row in first.T)
        declared_error = abs(declared - reference_c) if declared is not None else 0.0
        errors = (cross_frame, position_error, velocity_error, drift, declared_error)
        if not all(math.isfinite(x) for x in errors):
            raise ValueError("nonfinite_cr3bp_residual")
    except (ArithmeticError, TypeError, ValueError) as exc:
        return dict(result, error=str(exc))
    result.update(endpoint=endpoint.tolist(), target=target, jacobi=reference_c,
                  frame_residual=cross_frame, position_residual=position_error,
                  velocity_residual=velocity_error, jacobi_drift=drift, evaluations=calls,
                  declared_jacobi_residual=declared_error, residual=max(errors))
    result["verified"] = bool(max(errors) <= 1e-8)
    return result
