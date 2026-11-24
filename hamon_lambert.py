"""Two-body Lambert transfers with independent endpoint and revolution replay.

The time equation and radial/transverse reconstruction follow Izzo (2015),
https://arxiv.org/abs/1403.2705, equation 18 and Algorithm 1.
A bracket, rather than an unconstrained Newton step, stays on each time branch.
The returned binary64 velocities are separately propagated with universal
Kepler variables. Conservation alone cannot establish a specified arrival.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from mpmath import mp


def _dot(ctx, a, b):
    return ctx.fsum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def _norm(ctx, a):
    return ctx.sqrt(_dot(ctx, a, a))


def _finite(value):
    if isinstance(value, bool):
        raise ValueError("boolean_is_not_an_orbital_quantity")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite_orbital_input")
    return number


def _vector(ctx, value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("orbital_vectors_require_three_components")
    return [ctx.mpf(_finite(x)) for x in value]


def _stumpff(ctx, z):
    if abs(z) < ctx.mpf("0.01"):
        c, s = ctx.mpf("0.5"), ctx.mpf(1)/6
        tc, ts = c, s
        for k in range(1, 128):
            tc *= -z / ((2*k+1)*(2*k+2))
            ts *= -z / ((2*k+2)*(2*k+3))
            c += tc
            s += ts
            if max(abs(tc), abs(ts)) < ctx.eps:
                return c, s
        raise ArithmeticError("stumpff_series_budget")
    if z > 0:
        q = ctx.sqrt(z)
        return 2*(ctx.sin(q/2)/q)**2, (q-ctx.sin(q))/q**3
    q = ctx.sqrt(-z)
    return 2*(ctx.sinh(q/2)/q)**2, (ctx.sinh(q)-q)/q**3


def _propagate(ctx, r0, v0, dt):
    """Independent universal-Kepler initial-value solve, normalized to mu=1."""
    radius = _norm(ctx, r0)
    radial = _dot(ctx, r0, v0)
    alpha = 2/radius - _dot(ctx, v0, v0)

    def elapsed(chi):
        c, s = _stumpff(ctx, alpha*chi*chi)
        return radial*chi*chi*c + (1-alpha*radius)*chi**3*s + radius*chi

    lo, hi = ctx.zero, min(ctx.one, dt/radius)
    for _ in range(256):
        if elapsed(hi) >= dt:
            break
        hi *= 2
    else:
        raise ArithmeticError("kepler_bracket_budget")
    for _ in range(320):
        chi = (lo+hi)/2
        if elapsed(chi) < dt:
            lo = chi
        else:
            hi = chi
        if hi-lo <= ctx.mpf("1e-50")*hi:
            break
    else:
        raise ArithmeticError("kepler_iteration_budget")
    chi = (lo+hi)/2
    c, s = _stumpff(ctx, alpha*chi*chi)
    f = 1-chi*chi*c/radius
    g = dt-chi**3*s
    r = [f*a+g*b for a, b in zip(r0, v0)]
    end_radius = _norm(ctx, r)
    if end_radius == 0:
        raise ArithmeticError("collision_trajectory")
    fdot = (alpha*chi**3*s-chi)/(radius*end_radius)
    gdot = 1-chi*chi*c/end_radius
    v = [fdot*a+gdot*b for a, b in zip(r0, v0)]
    return r, v, abs(elapsed(chi)-dt)/dt


def _time_of_flight(ctx, x, lam, revolutions=0):
    if x == 1:
        return ctx.inf if revolutions else ctx.mpf(2)/3*(1-lam**3)
    delta = (1-x)*(1+x)
    y = ctx.sqrt(1-lam*lam*delta)
    if x < 1:
        # atan2 keeps the small-angle information lost by acos(1-epsilon).
        psi = ctx.atan2((y-lam*x)*ctx.sqrt(delta), x*y+lam*delta)
    else:
        psi = ctx.asinh((y-lam*x)*ctx.sqrt(-delta))
    return ((psi+revolutions*ctx.pi)/ctx.sqrt(abs(delta))-x+lam*y)/delta


def _minimum_time(ctx, lam, revolutions):
    """Bracket the unique elliptic minimum using Izzo equation 22."""
    lo, hi = ctx.zero, ctx.one/2
    def derivative(x):
        delta = (1-x)*(1+x)
        y = ctx.sqrt(1-lam*lam*delta)
        time = _time_of_flight(ctx, x, lam, revolutions)
        return (3*time*x-2+2*lam**3*x/y)/delta
    for _ in range(240):
        if derivative(hi) > 0:
            break
        hi = (hi+1)/2
    else:
        raise ArithmeticError("multirevolution_minimum_bracket_budget")
    for iteration in range(320):
        middle = (lo+hi)/2
        if derivative(middle) < 0:
            lo = middle
        else:
            hi = middle
        if hi-lo < ctx.mpf("1e-55"):
            x = (lo+hi)/2
            return x, _time_of_flight(ctx, x, lam, revolutions), iteration+1
    raise ArithmeticError("multirevolution_minimum_iteration_budget")


def _solve_x(ctx, lam, target):
    # For M=0, T decreases continuously from infinity at x=-1 to zero.
    # Choose brackets adaptively in both the elliptic and hyperbolic domains.
    lo, hi = ctx.zero, ctx.one
    if _time_of_flight(ctx, lo, lam) < target:
        for _ in range(240):
            lo = (lo-1)/2
            if _time_of_flight(ctx, lo, lam) >= target:
                break
        else:
            raise ArithmeticError("lambert_elliptic_bracket_budget")
    if _time_of_flight(ctx, hi, lam) > target:
        for _ in range(240):
            hi *= 2
            if _time_of_flight(ctx, hi, lam) <= target:
                break
        else:
            raise ArithmeticError("lambert_hyperbolic_bracket_budget")
    for iteration in range(320):
        x = (lo+hi)/2
        elapsed = _time_of_flight(ctx, x, lam)
        if abs(elapsed-target) <= target*ctx.mpf("1e-45"):
            return x, iteration+1, abs(elapsed-target)/target
        if elapsed > target:
            lo = x
        else:
            hi = x
    raise ArithmeticError("lambert_iteration_budget")


def solve_lambert_transfer(params):
    result = {"domain": "lagrangian_orbital", "subkind": "lambert_transfer",
              "model": "unperturbed_two_body_point_mass", "revolutions": 0,
              "verified": False, "endpoint_verified": False}
    ctx = mp.clone()
    ctx.dps = 90
    allowed = {"domain", "subkind", "mu", "r1_m", "r2_m", "tof_s",
               "transfer_path", "plane_normal", "revolutions",
               "proposed_v1_m_s", "proposed_v2_m_s"}
    try:
        if not isinstance(params, Mapping):
            raise ValueError("lambert_parameters_require_an_object")
        if set(params)-allowed:
            raise ValueError("unsupported_lambert_parameters:"+",".join(sorted(set(params)-allowed)))
        count = params.get("revolutions", 0)
        if type(count) is not int or count != 0:
            raise ValueError("only_zero_revolution_transfers_supported")
        mu, dt = (ctx.mpf(_finite(params[k])) for k in ("mu", "tof_s"))
        if mu <= 0 or dt <= 0:
            raise ValueError("mu_and_tof_must_be_positive")
        path = params.get("transfer_path", "short")
        if path not in ("short", "long"):
            raise ValueError("transfer_path_must_be_short_or_long")
        r1, r2 = (_vector(ctx, params[k]) for k in ("r1_m", "r2_m"))
        a, b = _norm(ctx, r1), _norm(ctx, r2)
        if a == 0 or b == 0:
            raise ValueError("zero_orbital_position")
        chord = _norm(ctx, [v-u for u, v in zip(r1, r2)])
        normal = _cross(r1, r2)
        normal_length = _norm(ctx, normal)
        alignment = _dot(ctx, r1, r2)
        short_angle = ctx.atan2(normal_length, alignment)
        if normal_length == 0:
            if alignment >= 0:
                raise ValueError("coincident_or_radial_geometry_is_ambiguous")
            if "plane_normal" not in params:
                raise ValueError("antipodal_transfer_requires_plane_normal")
            normal = _vector(ctx, params["plane_normal"])
            normal_length = _norm(ctx, normal)
            if normal_length == 0 or _dot(ctx, normal, r1) != 0:
                raise ValueError("plane_normal_must_be_nonzero_and_perpendicular")
        elif "plane_normal" in params:
            raise ValueError("plane_normal_is_only_used_for_antipodal_positions")
        unit_normal = [n/normal_length for n in normal]
        if path == "long":
            unit_normal = [-n for n in unit_normal]
        ur1, ur2 = [v/a for v in r1], [v/b for v in r2]
        ut1, ut2 = _cross(unit_normal, ur1), _cross(unit_normal, ur2)
        semiperimeter = (a+b+chord)/2
        lam_squared = (semiperimeter-chord)/semiperimeter
        if not 0 <= lam_squared < 1:
            raise ArithmeticError("unresolved_transfer_geometry")
        lam = ctx.sqrt(lam_squared)
        if path == "long":
            lam = -lam
        target = dt*ctx.sqrt(2*mu/semiperimeter**3)
        x, iterations, time_residual = _solve_x(ctx, lam, target)
        y = ctx.sqrt(1-lam*lam*(1-x*x))
        rho = (a-b)/chord
        sigma_squared = (1-rho)*(1+rho)
        if sigma_squared <= 0:
            raise ArithmeticError("unresolved_transverse_geometry")
        gamma = ctx.sqrt(mu*semiperimeter/2)
        radial1 = gamma*((lam*y-x)-rho*(lam*y+x))/a
        radial2 = -gamma*((lam*y-x)+rho*(lam*y+x))/b
        momentum = gamma*ctx.sqrt(sigma_squared)*(y+lam*x)
        exact_v1 = [radial1*u+momentum/a*t for u, t in zip(ur1, ut1)]
        exact_v2 = [radial2*u+momentum/b*t for u, t in zip(ur2, ut2)]

        def publish(value):
            wire = float(value)
            if not math.isfinite(wire) or (wire == 0 and value != 0 and abs(value) < ctx.mpf(2)**-1074):
                raise ArithmeticError("orbital_result_outside_numeric_range")
            return wire

        v1, v2 = [publish(v) for v in exact_v1], [publish(v) for v in exact_v2]
        # Reconstruct from precisely the published floats, never hidden solver values.
        speed_scale = ctx.sqrt(mu/a)
        rr1, rr2 = [v/a for v in r1], [v/a for v in r2]
        vv1, vv2 = [[ctx.mpf(v)/speed_scale for v in vv] for vv in (v1, v2)]
        normalized_dt = dt*speed_scale/a
        arrival, arrival_velocity, kepler_residual = _propagate(ctx, rr1, vv1, normalized_dt)
        endpoint_error = _norm(ctx, [u-v for u, v in zip(arrival, rr2)])/(b/a)
        velocity_error = _norm(ctx, [u-v for u, v in zip(arrival_velocity, vv2)])/max(_norm(ctx, vv2), ctx.sqrt(a/b))
        h1, h2 = _cross(rr1, vv1), _cross(rr2, vv2)
        hnorm = _norm(ctx, h1)
        if hnorm == 0:
            raise ArithmeticError("radial_collision_trajectory")
        angular_error = _norm(ctx, [u-v for u, v in zip(h1, h2)])/hnorm
        energy1 = _dot(ctx, vv1, vv1)/2-1
        energy2 = _dot(ctx, vv2, vv2)/2-a/b
        energy_scale = max(ctx.one, a/b, _dot(ctx, vv1, vv1)/2, _dot(ctx, vv2, vv2)/2)
        energy_error = abs(energy1-energy2)/energy_scale
        residual = max(time_residual, kepler_residual, endpoint_error, velocity_error, angular_error, energy_error)
        certificate = residual < ctx.mpf("1e-10")
        evec = [v-u for v, u in zip(_cross(vv1, h1), rr1)]
        eccentricity = _norm(ctx, evec)
        pericenter = a*hnorm*hnorm/(1+eccentricity)
        result.update(
            transfer_path=path, v1_m_s=v1, v2_m_s=v2,
            tof_s=float(dt), transfer_angle_rad=publish(
                short_angle if path == "short" else 2*ctx.pi-short_angle),
            eccentricity=publish(eccentricity), conic_periapsis_m=publish(pericenter),
            residual=publish(residual), endpoint_relative_error=publish(endpoint_error),
            arrival_velocity_relative_error=publish(velocity_error),
            specific_energy_relative_error=publish(energy_error),
            angular_momentum_relative_error=publish(angular_error),
            time_equation_relative_error=publish(time_residual),
            kepler_time_relative_error=publish(kepler_residual), iterations=iterations,
            endpoint_verified=bool(certificate), verified=bool(certificate),
        )
        claims = {}
        for field, expected in (("proposed_v1_m_s", v1), ("proposed_v2_m_s", v2)):
            if field in params:
                proposed = _vector(ctx, params[field])
                truth = [ctx.mpf(v) for v in expected]
                scale = max(_norm(ctx, truth), speed_scale)
                relative = _norm(ctx, [u-v for u, v in zip(proposed, truth)])/scale
                claims[field] = {"relative_error": publish(relative), "passed": bool(relative <= ctx.mpf("1e-10"))}
        if claims:
            result["claims"] = claims
            result["verified"] = bool(certificate and all(c["passed"] for c in claims.values()))
        if not certificate:
            result["error"] = "published_velocity_does_not_close_transfer"
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result.update(verified=False, error=str(exc))
    return result

