"""Constant-area perfect-gas flow with simultaneous wall friction and heat.

The uniform heat input fixes stagnation temperature as an affine function of
distance. An integrating factor solves momentum in velocity coordinates;
branch-bounded roots stop at the first sonic point. See docs/coupled_duct.md.
"""
from __future__ import annotations

import math
from functools import lru_cache
from fractions import Fraction

from mpmath import mp
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=2)
def _quadrature(order):
    nodes, weights = leggauss(order)
    return tuple(zip(nodes.tolist(), weights.tolist()))


def coupled_duct(params, constants):
    result = {"domain": "compressible_acoustics", "subkind": "fanno_rayleigh",
              "verified": False}
    try:
        return _solve(params, constants, result)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return dict(result, error=str(exc))


def _solve(params, constants, result):
    allowed = {"subkind", "M1", "T01", "P1", "L", "D_h", "f", "q_J_kg",
               "gamma", "R", "cp", "friction_factor_convention", "branch"}
    extra = sorted(set(params) - allowed)
    if extra:
        raise ValueError("Unsupported coupled-duct parameters: " + ", ".join(extra))
    ctx = mp.clone()
    ctx.dps = 100

    def number(key, default=None):
        value = params[key] if key in params else default
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(key + " must be a finite JSON number")
        return ctx.mpf(value)

    m, t0, p1, length, diameter, friction, heat = (
        number("M1"), number("T01"), number("P1", 101325.), number("L"),
        number("D_h"), number("f"), number("q_J_kg"))
    for key in ("gamma", "R", "cp"):
        if key in params:
            number(key)
    if min(m, t0, p1, diameter) <= 0 or min(length, friction) < 0:
        raise ValueError("Require positive M1, T01, P1, D_h and nonnegative L, f")
    if not length and heat:
        raise ValueError("A zero-length duct cannot deliver nonzero distributed heat")
    convention = params.get("friction_factor_convention", "fanning")
    if convention not in ("fanning", "darcy"):
        raise ValueError("friction_factor_convention must be fanning or darcy")
    g, gas, cp = (ctx.mpf(v.numerator) / v.denominator for v in constants)
    b = gas / (2 * cp)
    a = 1 - b
    t1 = t0 / (1 + (g - 1) * m*m / 2)
    w0 = m * ctx.sqrt(g * t1 / t0)
    eps = heat / (cp * t0)
    e = 1 + eps
    k = (4 if convention == "fanning" else 1) * friction * length / (2 * diameter)
    branch = "subsonic" if m < 1 else "supersonic" if m > 1 else "sonic"
    if "branch" in params and params["branch"] != branch:
        raise ValueError("The outlet must remain on the supplied inlet branch")
    result.update(M1=float(m), inlet_branch=branch, requested_length_m=float(length),
                  requested_heat_J_kg=float(heat), friction_factor_convention=convention,
                  model="constant_area_uniform_heat_constant_friction_perfect_gas")
    if e <= 0:
        return dict(result, requested_state_feasible=False, choked=False,
                    reason="Heat removal leaves nonpositive stagnation temperature")
    if m == 1 and (k or eps):
        return dict(result, reason="A nontrivial sonic inlet needs a separately specified transonic model")

    def wire(value):
        answer = float(value)
        if not math.isfinite(answer) or (value and answer == 0):
            raise ArithmeticError("Coupled-duct output is outside the finite reporting range")
        return answer

    def bisect(fn, left, right):
        fl, fr = fn(left), fn(right)
        if fl == 0:
            return left
        if fr == 0:
            return right
        if ctx.sign(fl) == ctx.sign(fr):
            raise ArithmeticError("Physical branch does not bracket an outlet")
        for _ in range(360):
            mid = (left + right) / 2
            fm = fn(mid)
            if fm == 0:
                return mid
            if ctx.sign(fm) == ctx.sign(fl):
                left, fl = mid, fm
            else:
                right, fr = mid, fm
            if abs(right-left) < ctx.mpf("1e-80") * max(abs(left), abs(right)):
                return (left + right) / 2
        raise ArithmeticError("Coupled-duct root did not resolve within its iteration budget")

    # Detect the constant-velocity cooling trajectory in the exact input
    # rationals. Squaring a rounded square root would invent a tiny force and
    # choose the singular neighbouring integrating-factor chart instead.
    gr, _, cpr = constants
    mr = Fraction(params['M1'])
    yr = gr*mr*mr/(1+(gr-1)*mr*mr/2)
    kr = ((4 if convention == 'fanning' else 1)*Fraction(params['f'])*
          Fraction(params['L'])/(2*Fraction(params['D_h'])))
    er = Fraction(params['q_J_kg'])/(cpr*Fraction(params['T01']))
    exact_force = kr*yr+er
    c0 = ctx.mpf(exact_force.numerator)/exact_force.denominator
    critical = None
    critical_w = None
    mode = "constant_velocity" if not c0 else "fanno" if not eps else "rayleigh" if not k else "coupled"
    sep = None
    sign_c = ctx.sign(c0)
    if mode == "constant_velocity":
        w = w0
        if eps:
            candidate = (a*w0*w0 - 1) / eps
            if candidate > 0:
                critical, critical_w = candidate, w0
        along_energy = lambda velocity: ctx.mpf(1)
        invariant = lambda velocity: velocity-w0
    elif mode == "fanno":
        def spent(velocity):
            return (1/(w0*w0) - 1/(velocity*velocity) - 2*a*ctx.log(velocity/w0)) / (2*k)
        critical_w = 1 / ctx.sqrt(a)
        critical = spent(critical_w)
        along_energy = lambda velocity: ctx.mpf(1)
        invariant = lambda velocity: spent(velocity)-1
    elif mode == "rayleigh":
        momentum = 1/w0 + a*w0
        critical_w = momentum / (2*a)
        candidate = (a*critical_w**2 - 1) / eps
        if candidate > 0:
            critical = candidate
        along_energy = lambda velocity: velocity*(momentum-a*velocity)
        invariant = lambda velocity: e/velocity+a*velocity-momentum
    else:
        c = eps/k
        root_c = ctx.sqrt(abs(c))
        if c > 0:
            primitive = lambda velocity: ctx.asinh(velocity/root_c)
        elif sign_c > 0:
            sep = root_c
            primitive = lambda velocity: ctx.acosh(velocity/root_c)
        else:
            sep = root_c
            primitive = lambda velocity: ctx.asin(velocity/root_c)
        def factor(velocity):
            return ctx.sqrt(abs(velocity*velocity+c))/velocity
        initial_factor = factor(w0)
        initial_primitive = primitive(w0)
        def integral(velocity):
            return a*c*sign_c*(primitive(velocity)-initial_primitive)
        def invariant(velocity):
            return e*factor(velocity)+integral(velocity)-initial_factor
        def along_energy(velocity):
            return (initial_factor-integral(velocity))/factor(velocity)
        def sonic_equation(velocity):
            return a*velocity*velocity*factor(velocity)+integral(velocity)-initial_factor
        # The sonic equation has derivative 2*a*w*factor(w)>0. The
        # physical trajectory can cross it at most once. Negative c0 moves
        # away from that crossing on either inlet branch.
        if sign_c > 0:
            if m < 1:
                lo, hi = w0, 2*w0
                for _ in range(400):
                    if sonic_equation(hi) > 0:
                        break
                    hi *= 2
                else:
                    raise ArithmeticError("Sonic bracket exceeds numeric budget")
                critical_w = bisect(sonic_equation, lo, hi)
            else:
                lo = sep if sep is not None else ctx.mpf(0)
                lower = (integral(lo)-initial_factor if lo == 0 else sonic_equation(lo))
                if lower < 0:
                    if lo == 0:
                        # Evaluate the analytic finite endpoint without factor(0).
                        def bounded_sonic(velocity):
                            return lower if not velocity else sonic_equation(velocity)
                        critical_w = bisect(bounded_sonic, lo, w0)
                    else:
                        critical_w = bisect(sonic_equation, lo, w0)
            if critical_w is not None:
                candidate = (a*critical_w**2 - 1)/eps
                if candidate > 0:
                    critical = candidate

    def state(velocity, total_ratio):
        temp = t0*(total_ratio-b*velocity*velocity)
        if temp <= 0:
            raise ArithmeticError("No finite positive-temperature outlet on the inlet branch")
        speed = ctx.sqrt(gas*t0)*velocity
        pressure = p1*(temp/t1)*(w0/velocity)
        return {"M": wire(speed/ctx.sqrt(g*gas*temp)), "P_Pa": wire(pressure),
                "T_K": wire(temp), "T0_K": wire(t0*total_ratio), "u_m_s": wire(speed)}

    if critical is not None:
        result["L_choke_m"] = wire(length*critical)
        result["q_choke_at_fixed_heat_rate_J_kg"] = wire(heat*critical) if heat else 0.
        if critical < 1:
            return dict(result, requested_state_feasible=False, choked=True,
                        reason="The coupled flow reaches sonic conditions before the requested exit",
                        reachable_sonic_state=state(critical_w, a*critical_w**2))

    if mode != "constant_velocity":
        sonic = ctx.sqrt(e/a)
        zero_temp = ctx.sqrt(e/b)
        lo, hi = ((ctx.mpf(0), sonic) if m < 1 else (sonic, zero_temp))
        if sep is not None:
            if sign_c > 0:
                lo = max(lo, sep)
            else:
                hi = min(hi, sep)
        direction = sign_c * (1 if m < 1 else -1)
        if direction > 0:
            lo = max(lo, w0)
        else:
            hi = min(hi, w0)
        if not lo < hi:
            return dict(result, requested_state_feasible=False, choked=False,
                        reason="No positive-temperature outlet on the connected inlet branch")
        def bounded_invariant(velocity):
            if velocity == 0:
                return -ctx.inf if mode == "fanno" else ctx.inf
            return invariant(velocity)
        if ctx.sign(bounded_invariant(lo)) == ctx.sign(bounded_invariant(hi)):
            return dict(result, requested_state_feasible=False, choked=False,
                        reason="No finite outlet on the connected physical branch")
        w = bisect(bounded_invariant, lo, hi)
        if w == zero_temp or w == sep or w == 0:
            return dict(result, requested_state_feasible=False, choked=False,
                        reason="The limiting outlet has no finite positive static state")
    output = state(w, e)
    if branch != "sonic" and (output["M"] < 1) != (m < 1) and not (critical == 1):
        return dict(result, requested_state_feasible=True, reason="Outlet Mach does not resolve its physical branch")

    # Re-evaluate the published binary64 velocity and temperature, rather than
    # certifying only the unreported high-precision root.
    uw = ctx.mpf(output["u_m_s"])
    tw = ctx.mpf(output["T_K"])
    pw = ctx.mpf(output["P_Pa"])
    ww = uw/ctx.sqrt(gas*t0)
    ew = (tw+uw*uw/(2*cp))/t0
    mass = pw*uw/(gas*tw)
    mass1 = p1*ctx.sqrt(gas*t0)*w0/(gas*t1)
    energy_residual = abs(ew-e)
    if mode == "constant_velocity":
        wall_integral = w0
        path_residual = abs(ww-w0)/w0
    elif not k:
        wall_integral = ctx.mpf(0)
        path_residual = abs(invariant(ww))/max(1, abs(1/w0+a*w0))
    else:
        # Integrate the wall force over the solved velocity path. This uses
        # dx/du directly, not an endpoint pressure-drop definition of drag.
        # The force is integrated to the physical exit. Inverting a rounded
        # velocity to a different length is ill-conditioned near the exact
        # constant-velocity cooling trajectory.
        midpoint, radius = (w+w0)/2, (w-w0)/2
        def wall_quad(order):
            values = []
            for node, weight in _quadrature(order):
                v = midpoint+radius*node
                values.append(weight*(along_energy(v)-a*v*v)/(k*v*v+eps))
            return radius*ctx.fsum(values)
        coarse, wall_integral = wall_quad(32), wall_quad(64)
        if abs(k*(wall_integral-coarse)) > ctx.mpf('1e-12')*(1/w0+a*w0):
            raise ArithmeticError("Independent wall-force quadrature did not converge")
        path_residual = abs(invariant(w))/max(1, abs(invariant(w0)))
    momentum_residual = abs(ew/ww+a*ww-(1/w0+a*w0)+k*wall_integral) / (1/w0+a*w0)
    residuals = {"mass": wire(abs(mass/mass1-1)), "energy": wire(energy_residual),
                 "momentum": wire(momentum_residual), "path": wire(path_residual)}
    result.update(M2=output["M"], P2_Pa=output["P_Pa"], T2_K=output["T_K"],
                  T02_K=output["T0_K"], u2_m_s=output["u_m_s"],
                  P2_over_P1=wire(pw/p1), T2_over_T1=wire(tw/t1),
                  mass_flux_kg_m2_s=wire(mass1), cp_J_kgK=wire(cp),
                  gamma_effective=wire(g), R_effective_J_kgK=wire(gas),
                  requested_state_feasible=True, choked=bool(critical == 1),
                  residuals=residuals, residual=max(residuals.values()),
                  verified=max(residuals.values()) < 1e-10)
    return result
