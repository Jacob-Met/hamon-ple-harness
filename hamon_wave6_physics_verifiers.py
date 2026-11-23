"""PLE Wave-6 Deterministic Physics & Engineering Verifiers + Pareto Fast-Path Pruner.

Implements exact analytical + SymPy/SciPy residual-certified solvers for four engineering domains:
1. 1D Compressible Gas Dynamics & Acoustics (isentropic nozzle/area-Mach, normal shock, Fanno, Rayleigh, plane-wave & Helmholtz acoustics).
2. RF Transmission Lines, Smith-Chart Single-Stub Matching & Second-Order RLC Transients.
3. Solid Mechanics: 2D/3D Mohr's Circle, Von Mises/Tresca, Thin & Thick-Walled Lame Pressure Vessels, Euler-Johnson Column Buckling.
4. Chemical Reactor Engineering: Arrhenius Kinetics, Isothermal & Adiabatic CSTR / PFR / N-CSTRs-in-Series Mole & Energy Balances.

Also provides deterministic fast-path token/cost pruning (`prune_deterministic_fast_path`) for solved
residual-certified engineering domains to maximize the PLE accuracy-cost Pareto frontier.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np
from _scipy_compat import HAS_SCIPY, integrate, optimize
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    class _DecimalSympyShim:
        pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494')
        @staticmethod
        def Float(val, prec=50):
            getcontext().prec = max(60, int(prec) + 10)
            return Decimal(str(val))
    sp = _DecimalSympyShim()
    HAS_SYMPY = False

WAVE6_DOMAINS: Tuple[str, ...] = (
    "compressible_acoustics",
    "rf_transmission_rlc",
    "solid_mohr_vessel_buckling",
    "reactor_kinetics",
)

DETERMINISTIC_FAST_PATH_DOMAINS: frozenset[str] = frozenset(
    {
        "compressible_acoustics",
        "rf_transmission_rlc",
        "solid_mohr_vessel_buckling",
        "reactor_kinetics",
        "pde_heat_wave_laplace",
        "control_bode_routh_ss",
        "psychrometric_hvac",
        "fem_frame_truss_beam",
        "conservation_balance",
        "cad_step_brep",
        "struct_step_brep",
        "civil_geotech",
        "mech_thermo_hx",
    }
)


def _isentropic_log_area(mach, gamma):
    """Log(A/A*) on the supplied positive Mach coordinate."""
    if mach == 1:
        return Decimal(0)
    half_delta = (gamma - 1) / 2
    temperature = 1 + half_delta * mach**2
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return exponent * (temperature / (1 + half_delta)).ln() - mach.ln()


def _nozzle_area_mach(area, gamma, regime):
    if area < 1:
        raise ValueError("Isentropic A/A* must be at least one")
    if regime not in ("sub", "subsonic", "sup", "supersonic"):
        raise ValueError("Area inversion requires the subsonic or supersonic branch")
    if area == 1:
        return 1.0
    subsonic = regime in ("sub", "subsonic")
    lo, hi = (math.ulp(0.0), 1.) if subsonic else (1., math.nextafter(math.inf, 0.))
    target = area.ln()

    def error(value):
        return _isentropic_log_area(Decimal.from_float(value), gamma) - target

    if error(lo if subsonic else hi) < 0:
        raise ValueError("Requested area has no representable Mach number on this branch")
    # Bisect in log Mach over the entire finite positive range. Finish on
    # adjacent representable Mach values so shallow sonic minima do not turn
    # either branch into a manufactured equality.
    while math.nextafter(lo, hi) < hi:
        mid = math.sqrt(lo) * math.sqrt(hi)
        if not lo < mid < hi:
            mid = lo + (hi-lo)/2
        residual = error(mid)
        if residual == 0:
            return mid
        if (residual > 0) == subsonic:
            lo = mid
        else:
            hi = mid
    return min((lo, hi), key=lambda value: abs(error(value)))


def _nozzle_brackets(mach, predicted, target, relation, increasing):
    """Check the specified ratio on the reported Mach grid, including sonic sensitivity."""
    if predicted == target:
        return True
    toward = math.inf if (target > predicted) == increasing else 0.
    neighbor = math.nextafter(float(mach), toward)
    if neighbor == 0 or not math.isfinite(neighbor):
        return False
    adjacent = relation(Decimal.from_float(neighbor))
    return min(predicted, adjacent) <= target <= max(predicted, adjacent)


def _joint_nozzle_mach(mach, area, pressure, gamma, regime):
    """Find a single continuous nozzle state that rounds to every supplied coordinate."""
    gm1 = gamma - 1
    one = Decimal(1)

    def cell(value):
        wire = float(value)
        down = Decimal.from_float(math.nextafter(wire, 0.))
        above = math.nextafter(wire, math.inf)
        up = Decimal.from_float(above) if math.isfinite(above) else value + (value-down)
        return (value+down)/2, (value+up)/2

    def inverse_pressure(value):
        return (2*((value.ln()*gm1/gamma).exp()-1)/gm1).sqrt()

    if pressure is not None:
        p_low, p_high = cell(pressure)
        # Finite slow flow can have a stagnation ratio that rounds to unity.
        # Other observations still locate its positive Mach within that cell.
        lo = max(inverse_pressure(max(one, p_low)), Decimal.from_float(math.ulp(0.))/2)
        hi = inverse_pressure(p_high)
        preferred = inverse_pressure(pressure)
    else:
        lo, hi = cell(mach)
        preferred = mach
    if mach is not None:
        ml, mh = cell(mach)
        lo, hi = max(lo, ml), min(hi, mh)
        preferred = mach
    if lo > hi:
        return None

    def log_matches(log_value, observation):
        low, high = cell(observation)
        return (low.ln() <= log_value <= high.ln()
                and float(log_value.exp()) == float(observation))

    def matches(candidate):
        if float(candidate) == 0 or (mach is not None and float(candidate) != float(mach)):
            return False
        if pressure is not None:
            log_pressure = (1+gm1*candidate**2/2).ln()*gamma/gm1
            if not log_matches(log_pressure, pressure):
                return False
        return area is None or log_matches(_isentropic_log_area(candidate, gamma), area)

    for sub in ([regime in ("sub", "subsonic")] if regime else [False, True]):
        left, right = (lo, min(hi, one)) if sub else (max(lo, one), hi)
        if left > right:
            continue
        candidate = min(max(preferred, left), right)
        if matches(candidate):
            return candidate
        if area is None:
            midpoint = (left+right)/2
            if matches(midpoint):
                return midpoint
            continue
        # Clip the area observation to this monotone branch of the same
        # pressure/Mach interval before choosing their shared state.
        al, ah = cell(area)
        ends = [_isentropic_log_area(m, gamma) for m in (left, right)]
        lower, upper = max(min(ends), max(one, al).ln()), min(max(ends), ah.ln())
        if lower > upper:
            continue
        target = (lower+upper)/2
        while left < right:
            middle = (left*right).sqrt() if right > 2*left else (left+right)/2
            if middle in (left, right):
                break
            observed = _isentropic_log_area(middle, gamma)
            if matches(middle):
                return middle
            if (observed > target) == sub:
                left = middle
            else:
                right = middle
        for endpoint in (left, right):
            if matches(endpoint):
                return endpoint
    return None


def _isentropic_nozzle(params, gamma_value, gas_constant):
    """One isentropic state must satisfy all supplied nozzle observations.

    Mach, A/A* and P0/P can each supply the operating coordinate. When supplied
    together they denote independently rounded measurements of the same state.
    A stated regime constrains that state. Area-only inversion defaults to the
    supersonic branch.
    """
    def positive(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Isentropic {name} must be finite and positive")
        return Decimal.from_float(value)

    def publish(value, name):
        rounded = float(value)
        if not math.isfinite(rounded) or rounded == 0:
            raise ValueError(f"Isentropic {name} is outside the positive binary64 range")
        return rounded

    gamma = positive(gamma_value, "gamma")
    if gamma <= 1:
        raise ValueError("Isentropic gamma must exceed one")
    gas = positive(gas_constant, "gas constant")
    t0 = positive(params.get("T0", 300.), "stagnation temperature")
    p0 = positive(params.get("P0", 101325.), "stagnation pressure")
    throat = positive(params.get("A_throat", .001), "throat area")
    with localcontext() as ctx:
        # The forward relation subtracts log Mach at a quadratic minimum.
        # Preserve its excess above unity, including when gamma is near one.
        ctx.prec = 80 + max(0, gamma.adjusted())
        gm1 = gamma - 1
        mach, area, pressure = (positive(params[key], label) if params.get(key) is not None else None
                                for key, label in (("mach", "Mach number"), ("area_ratio", "area ratio"),
                                                   ("p0_over_p", "stagnation pressure ratio")))
        regime = str(params["regime"]).strip().lower() if "regime" in params else None
        if regime is not None and regime not in ("sub", "subsonic", "sup", "supersonic"):
            raise ValueError("Nozzle regime must be subsonic or supersonic")
        if area is not None and area < 1:
            raise ValueError("Isentropic A/A* must be at least one")
        supplied = sum(value is not None for value in (mach, area, pressure))
        if pressure is not None and (pressure < 1 or (pressure == 1 and supplied == 1)):
            raise ValueError("A pressure-only flowing nozzle requires P0/P greater than one and joint ratios must be at least one")
        if supplied == 0:
            raise ValueError("isentropic requires 'mach', 'area_ratio', or 'p0_over_p'")
        joint = supplied > 1
        if joint:
            mach = _joint_nozzle_mach(mach, area, pressure, gamma, regime)
            if mach is None:
                return {"domain": "compressible_acoustics", "subkind": "isentropic", "verified": False,
                        "reason": "No common nozzle state matches the supplied Mach, area, pressure and regime"}
        elif mach is None and area is not None:
            regime = regime or "supersonic"
            mach = Decimal.from_float(_nozzle_area_mach(area, gamma, regime))
        elif mach is None:
            m = (2 * ((pressure.ln()*gm1/gamma).exp() - 1) / gm1).sqrt()
            mach = Decimal.from_float(publish(m, "Mach number"))

        t_ratio = 1 + gm1*mach**2/2
        log_p_ratio = t_ratio.ln() * gamma/gm1
        log_area = _isentropic_log_area(mach, gamma)
        max_log = Decimal.from_float(math.nextafter(math.inf, 0.)).ln()
        if max(log_p_ratio, log_area) > max_log:
            raise ValueError("Isentropic stagnation pressure or area ratio exceeds the finite range")
        p_ratio = log_p_ratio.exp()
        density_ratio = (t_ratio.ln()/gm1).exp()
        area_ratio = log_area.exp()
        ts = t0 / t_ratio
        ps = p0 / p_ratio
        velocity = mach*(gamma*gas*ts).sqrt()
        star_ratio = (gamma + 1)/2
        star_t = t0 / star_ratio
        star_p = p0 / (star_ratio.ln()*gamma/gm1).exp()
        choked = throat * star_p / (gas*star_t) * (gamma*gas*star_t).sqrt()
        result = {key: publish(value, key) for key, value in (
            ("mach", mach), ("T0_over_T", t_ratio), ("P0_over_P", p_ratio),
            ("rho0_over_rho", density_ratio), ("A_over_Astar", area_ratio),
            ("T_static_K", ts), ("P_static_Pa", ps), ("velocity_m_s", velocity),
            ("mdot_choked_kg_s", choked))}
        # Replay the published static pressure, temperature and velocity, using
        # the requested area itself when it determined the operating point.
        ts, ps, speed, flow = (Decimal.from_float(result[k]) for k in
                               ("T_static_K", "P_static_Pa", "velocity_m_s", "mdot_choked_kg_s"))
        local_area = area if area is not None else Decimal.from_float(result["A_over_Astar"])
        mass_error = abs(ps/(gas*ts) * speed * throat * local_area / flow - 1)
        energy_error = abs((ts + speed**2 * gm1/(2*gamma*gas)) / t0 - 1)
        pressure_error = abs(p0 / (ps*pressure) - 1) if pressure is not None else 0
        input_resolved = True
        if joint:
            input_resolved = ((area is None or float(area) == result["A_over_Astar"])
                              and (pressure is None or float(pressure) == result["P0_over_P"]))
        elif area is not None:
            subsonic = regime in ("sub", "subsonic")
            on_branch = mach == 1 if area == 1 else (mach < 1 if subsonic else mach > 1)
            input_resolved = on_branch and _nozzle_brackets(
                mach, log_area, area.ln(), lambda m: _isentropic_log_area(m, gamma), not subsonic)
        elif pressure is not None:
            input_resolved = _nozzle_brackets(
                mach, log_p_ratio, pressure.ln(), lambda m: (1 + gm1*m*m/2).ln()*gamma/gm1, True)
        if regime and mach != 1:
            input_resolved = input_resolved and ((mach < 1) == (regime in ("sub", "subsonic")))
        residual = float(max(mass_error, energy_error, pressure_error))
        return {"domain": "compressible_acoustics", "subkind": "isentropic", **result,
                "residual": residual, "verified": input_resolved and residual < 1e-10}


# Treat rounded critical inputs as the sonic section within this relative
# length/heat tolerance. There is no absolute window around a small budget.
_DUCT_SONIC_REL_TOL = 32 * math.ulp(1.0)


def _ideal_gas_constants(params, gamma, r_gas, model):
    """Use one calorically perfect gas for all supplied caloric constants."""
    if not (math.isfinite(gamma) and gamma > 1 and math.isfinite(r_gas) and r_gas > 0):
        raise ValueError(f"{model} requires gamma > 1 and a positive finite gas constant")
    g, gas = map(Fraction, (gamma, r_gas))
    if "cp" in params:
        cp = float(params["cp"])
        if not math.isfinite(cp) or cp <= 0:
            raise ValueError(f"{model} flow requires positive finite specific heat")
        c = Fraction(cp)
        if "R" in params:
            # Use one physical gas. Preserve the supplied cp and R
            # together when the implied ratio rounds to the declared gamma.
            if c <= gas:
                raise ValueError(f"{model} cp, R and gamma describe inconsistent gas properties")
            implied_gamma = c/(c-gas)
            if "gamma" in params and float(implied_gamma) != gamma:
                raise ValueError(f"{model} cp, R and gamma describe inconsistent gas properties")
            g = implied_gamma
        else:
            gas = c*(g-1)/g
    else:
        c = g*gas/(g-1)

    return g, gas, c


def _fanno_duct(params, gamma, r_gas):
    """Steady adiabatic constant-area flow at fixed inlet Mach and friction."""
    from mpmath import mp

    context = mp.clone()
    context.dps = 90
    m1, length, diameter, friction = (float(params[key]) for key in ("M1", "L", "D_h", "f"))
    if not all(map(math.isfinite, (m1, length, diameter, friction))) or min(m1, diameter, friction) <= 0 or length < 0:
        raise ValueError("Fanno flow requires finite positive Mach, diameter and friction factor, and nonnegative length")
    convention = str(params.get("friction_factor_convention", "fanning")).lower()
    if convention not in ("fanning", "darcy"):
        raise ValueError("Fanno friction_factor_convention must be fanning or darcy")
    g, _, _ = _ideal_gas_constants(params, gamma, r_gas, "Fanno")
    branch = "subsonic" if m1 < 1 else "supersonic" if m1 > 1 else "sonic"
    if "branch" in params and params["branch"] != branch:
        raise ValueError("Fanno outlet must approach the sonic state on the inlet branch")

    def real(rational):
        return context.mpf(rational.numerator) / rational.denominator

    def wire(value):
        result = float(value)
        if not math.isfinite(result) or (value and result == 0):
            raise ArithmeticError("Fanno flow output outside binary64 range")
        return result

    gm1, gp1 = real(g - 1), real(g + 1)
    factor = real((g + 1) / (2 * g))
    m = context.mpf(m1)
    inlet = 1 / m**2
    friction_per_m = (4 if convention == "fanning" else 1) * context.mpf(friction) / diameter
    requested = context.mpf(length) * friction_per_m

    def spent(outlet):
        # z = 1/M^2 advances monotonically between the inlet and 1. For
        # x = 2(z-1)/(gamma+1), F = factor*(x-log1p(x)). Difference the
        # two x values directly so finite large M retains its 1/M^2 term.
        x2 = 2 * (outlet - 1) / gp1
        ratio = 2 * (inlet - outlet) / (gm1 + 2*outlet)
        if abs(ratio) < context.mpf("0.01"):
            # ratio-log1p(ratio) starts at second order near sonic.
            power = ratio * ratio
            remainder = power / 2
            for n in range(3, 100):
                power *= -ratio
                term = power / n
                remainder += term
                if abs(term) <= context.eps * abs(remainder):
                    break
        else:
            # The positive analytic complement also retains tiny gamma-1 and
            # 1/M^2 where forming 1+ratio would erase them.
            remainder = ratio - context.log((gm1 + 2*inlet) / (gm1 + 2*outlet))
        return factor * (ratio*x2 + remainder)

    def state(z):
        ratio_t = z * (gm1 + 2*inlet) / (inlet * (gm1 + 2*z))
        return (wire(1/context.sqrt(z)), wire(context.sqrt(z/inlet * ratio_t)), wire(ratio_t))

    base = {"domain": "compressible_acoustics", "subkind": "fanno", "M1": m1,
            "requested_length_m": length, "inlet_branch": branch,
            "friction_factor_convention": convention, "verified": False}
    try:
        critical_budget = spent(1)
        critical = wire(critical_budget/friction_per_m)
        base["L_choke_m"] = critical
        # The correctly rounded critical length is the sonic endpoint. A
        # distinct longer binary64 request cannot share that endpoint.
        if length > critical:
            ms, ps, ts = state(context.mpf(1))
            return dict(base, choked=True, requested_state_feasible=False,
                        reason="Specified length exceeds the fixed-inlet sonic limit.",
                        reachable_sonic_state={"M": ms, "length_m": critical,
                                               "P_over_P1": ps, "T_over_T1": ts},
                        residual=wire(requested-critical_budget))
        choked = length == critical
        if choked:
            z = context.mpf(1)
        elif length == 0:
            z = inlet
        else:
            # Geometric bisection resolves both Mach branches, including
            # finite inlets far out on the asymptotic supersonic friction line.
            left, right = inlet, context.mpf(1)
            for _ in range(200):
                z = context.sqrt(left*right)
                if spent(z) < requested:
                    left = z
                else:
                    right = z
                if abs(right-left) <= context.mpf("1e-35")*min(left, right):
                    break
            if state(left) != state(right):
                return dict(base, choked=False, requested_state_feasible=True,
                            reason="Fanno outlet not resolved to one rounded state")
            z = context.sqrt(left*right)
        m2, pressure, temperature = state(z)
        if 0 < length < critical and (m2 == 1 or m2 == m1):
            return dict(base, choked=False, requested_state_feasible=True,
                        reason="Finite Fanno advance unresolved by the reported Mach number")
        residual = abs(spent(1/context.mpf(m2)**2) - requested)
        return dict(base, M2=m2, choked=choked, requested_state_feasible=True,
                    P2_over_P1=pressure, T2_over_T1=temperature,
                    residual=float(residual), verified=residual < 1e-10)
    except ArithmeticError as exc:
        return dict(base, reason=str(exc))


def _rayleigh_duct(params, gamma, r_gas):
    """Fixed-inlet Rayleigh line with finite-Mach branch and sonic endpoints."""
    mach, t_in, heat = (float(params[key]) for key in ("M1", "T01", "q_J_kg"))
    if not all(map(math.isfinite, (mach, t_in, heat))) or min(mach, t_in) <= 0:
        raise ValueError("Rayleigh flow requires positive finite Mach and inlet temperature")

    def rounding_cell(value):
        center = Fraction(value)
        lower = (Fraction(math.nextafter(value, 0.)) + center) / 2
        next_value = math.nextafter(value, math.inf)
        upper = ((center + Fraction(next_value)) / 2 if math.isfinite(next_value)
                 else center + (center-lower))
        return lower, upper

    g, gas, c = _ideal_gas_constants(params, gamma, r_gas, "Rayleigh")
    branch = params.get("branch")
    if branch not in (None, "subsonic", "supersonic"):
        raise ValueError("Rayleigh branch must be subsonic or supersonic")
    m, t, q = map(Fraction, (mach, t_in, heat))
    inlet_branch = "subsonic" if m < 1 else "supersonic" if m > 1 else None
    if inlet_branch and branch and branch != inlet_branch:
        raise ValueError("Rayleigh outlet must stay on the inlet branch")
    branch = inlet_branch or branch
    # z=(M²-1)/(1+gamma*M²) is negative below sonic and positive above.
    # T0/T0* = 1-z². The finite supersonic line ends at z=1/gamma.
    z1 = (m*m - 1) / (1 + g*m*m)
    star = t / (1 - z1*z1)
    scale = c * star
    budget = scale * z1*z1
    squared = (budget - q) / scale

    def as_decimal(value):
        return Decimal(value.numerator) / Decimal(value.denominator)

    base = {"domain": "compressible_acoustics", "subkind": "rayleigh", "M1": mach,
            "requested_heat_J_kg": heat, "verified": False}
    try:
        if float(gas) == 0 or (budget and float(budget) == 0):
            raise OverflowError
        base.update(T0_star_K=float(star), q_choke_J_kg=float(budget),
                    gamma_effective=float(g), cp_J_kgK=float(c), R_effective_J_kgK=float(gas))
        # Retain the documented relative sonic tolerance for rounded critical inputs.
        at_limit = abs(q-budget) <= Fraction(_DUCT_SONIC_REL_TOL) * max(abs(q), budget)
        if q > budget and not at_limit:
            return dict(base, requested_state_feasible=False, choked=True,
                        reason="Specified heat input exceeds the sonic limit at the supplied inlet Mach number.",
                        reachable_sonic_state={"M": 1., "heat_J_kg": float(budget),
                                               "T0_K": float(star), "P_over_P1": float((1+g*m*m)/(1+g))},
                        residual=float((q-budget)/scale))
        if squared >= 1:
            raise ValueError("Specified heat removal gives nonpositive stagnation temperature")
        if branch is None and not at_limit:
            if squared < 1/(g*g):
                return dict(base, requested_state_feasible=True, choked=False,
                            available_branches=["subsonic", "supersonic"],
                            reason="Select a branch for cooling from a sonic inlet with two finite outlet states.")
            branch = "subsonic"
        if branch == "supersonic" and squared >= 1/(g*g):
            return dict(base, requested_state_feasible=False, choked=False,
                        limiting_heat_J_kg=float(budget - scale/(g*g)),
                        reason="No finite supersonic outlet at or below the infinite-Mach cooling limit.")
        with localcontext() as context:
            context.prec = 85
            if at_limit:
                outlet = Decimal(1)
            elif q == 0:
                outlet = Decimal.from_float(mach)
            else:
                gap = as_decimal(squared).sqrt()
                if branch == "subsonic":
                    m2sq = as_decimal(1-squared) / ((1+gap)*(1+as_decimal(g)*gap))
                else:
                    m2sq = (1+gap)*(1+as_decimal(g)*gap) / as_decimal(1-g*g*squared)
                outlet = m2sq.sqrt()
            m2 = float(outlet)
            pressure = float(as_decimal(1+g*m*m) / (1+as_decimal(g)*outlet**2))
        if not all(math.isfinite(v) and v > 0 for v in (m2, pressure)):
            raise OverflowError
        # Decode the reported M and P. Mass continuity sets u2/u1, so the
        # ideal-gas law gives T2/T1 and both original balances can be replayed.
        mw, pw = Fraction(m2), Fraction(pressure)
        velocity = pw * (mw/m)**2
        temp_ratio = pw * velocity
        recovered_t0 = t * temp_ratio * (2+(g-1)*mw*mw) / (2+(g-1)*m*m)
        energy_error = c*(recovered_t0-t)-q
        momentum_error = pw + g*m*m*velocity - (1+g*m*m)
        residual = max(abs(energy_error)/scale, abs(momentum_error)/(1+g*m*m))
        # Confirm one physical state in the intersection of the two reported
        # rounding cells. Along the fixed momentum line, heat is quadratic in
        # u2/u1 and has its maximum at the sonic velocity.
        def velocity_at_mach(value):
            return (value/m)**2 * (1+g*m*m) / (1+g*value*value)

        def heat_at_velocity(value):
            pressure_at_v = 1+g*m*m*(1-value)
            total_ratio = value*(2*pressure_at_v+(g-1)*m*m*value)/(2+(g-1)*m*m)
            return c*t*(total_ratio-1)

        mlo, mhi = rounding_cell(m2)
        plo, phi = rounding_cell(pressure)
        vlo = max(velocity_at_mach(mlo), 1+(1-phi)/(g*m*m))
        vhi = min(velocity_at_mach(mhi), 1+(1-plo)/(g*m*m))
        sonic = velocity_at_mach(Fraction(1))
        if branch == "supersonic":
            vlo = max(vlo, sonic)
        elif branch == "subsonic":
            vhi = min(vhi, sonic)
        ends = [heat_at_velocity(vlo), heat_at_velocity(vhi)]
        if vlo <= sonic <= vhi:
            ends.append(budget)
        target = budget if at_limit else q
        enclosed = vlo <= vhi and min(ends) <= target <= max(ends)
        resolved = at_limit or (m2 != 1 and (q == 0 or m2 != mach or pressure != 1))

        return dict(base, M2=m2, P2_over_P1=pressure, choked=at_limit,
                    requested_state_feasible=True, residual=float(residual),
                    wire_rounding_verified=bool(enclosed),
                    verified=bool(resolved and enclosed and residual < 1e-10))
    except OverflowError:
        return dict(base, reason="Rayleigh outlet or sonic reference exceeds the finite output range.")


def _passive_acoustic_interface(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Normal-incidence pressure and power at a lossless, passive fluid interface."""
    result = {"domain": "compressible_acoustics", "subkind": "acoustics", "verified": False}

    oblique = "incidence_angle_deg" in params or "incidence_sine" in params

    def positive(value):
        if oblique:
            if type(value) is int and value > 0:
                return Fraction(value)
            if type(value) is not float:
                raise ValueError("Oblique acoustic properties must be positive finite JSON numbers")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Acoustic material and resonator parameters must be finite and positive")
        return Fraction(value)

    def represent(value, name):
        rounded = float(value)
        if not math.isfinite(rounded) or (value != 0 and rounded == 0):
            raise ValueError(f"{name} is outside the representable output range")
        return rounded

    try:
        rho1, c1, rho2, c2 = (positive(params.get(name, default)) for name, default in
                               (("rho1", 1.204), ("c1", 343.0), ("rho2", 998.0), ("c2", 1481.0)))
        z1, z2 = rho1 * c1, rho2 * c2
        frequency = None
        dimensions = {"A_neck", "V_cavity", "L_eff"}
        supplied = dimensions.intersection(params)
        if supplied:
            if supplied != dimensions:
                raise ValueError("Helmholtz resonance requires A_neck, V_cavity and L_eff together")
            area, volume, length = (positive(params[key]) for key in ("A_neck", "V_cavity", "L_eff"))
            squared = c1 * c1 * area / (volume * length)
            with localcontext() as context:
                context.prec = 60
                root = (Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt()
                frequency = represent(root / Decimal.from_float(math.tau), "resonant frequency")

        if oblique:
            from hamon_oblique_acoustic_interface import oblique_interface
            return {**result, **oblique_interface(params, rho1, c1, rho2, c2),
                    "helmholtz_f0_Hz": frequency}

        total = z1 + z2
        r = (z2 - z1) / total
        t = 2 * z2 / total
        power = t * t * z1 / z2

        # Near a match, 1 - R² can round to one although the loss is measurable
        # in the returned numeric type. At high contrast, never floor the power.
        if abs(r) <= Fraction(1, 2):
            loss = -10.0 * math.log1p(-float(r * r)) / math.log(10.0)
        else:
            with localcontext() as context:
                context.prec = 60
                power_decimal = Decimal(power.numerator) / Decimal(power.denominator)
                loss = float(-10 * power_decimal.log10())

        # Check the reported binary coefficients against the two interface
        # boundary equations (the velocity equation is impedance-scaled).
        r_out = represent(r, "pressure reflection")
        t_out = represent(t, "pressure transmission")
        power_out = represent(power, "intensity transmission")
        rr, tt = Fraction(r_out), Fraction(t_out)
        pressure_error = abs(1 + rr - tt)
        velocity_error = abs(z2 * (1 - rr) - z1 * tt) / total
        energy_error = abs(rr * rr + Fraction(power_out) - 1)
        residual = float(max(pressure_error, velocity_error, energy_error))
        return {
            **result, "Z1_rayl": represent(z1, "Z1"), "Z2_rayl": represent(z2, "Z2"),
            "pressure_reflection_R": r_out, "pressure_transmission_T": t_out,
            "intensity_transmission_TI": power_out, "transmission_loss_dB": loss,
            "helmholtz_f0_Hz": frequency, "residual": residual,
            "verified": 0 < power_out <= 1 and abs(r_out) <= 1 and residual < 1e-11,
        }
    except (TypeError, ValueError, OverflowError) as exc:
        return {**result, "error": str(exc)}


def solve_compressible_gas_acoustics(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for 1D compressible gas dynamics & acoustic systems."""
    subkind = str(params.get("subkind", "isentropic")).strip().lower()
    gamma = float(params.get("gamma", 1.4))
    r_gas = float(params.get("R", 287.05))

    if subkind == "isentropic":
        return _isentropic_nozzle(params, gamma, r_gas)

    if subkind == "normal_shock":
        from hamon_normal_shock import normal_shock_state
        out = {"domain": "compressible_acoustics", "subkind": "normal_shock", "verified": False}
        try:
            g, gas, cp = _ideal_gas_constants(params, gamma, r_gas, "Normal shock")
            state = (float(params["M1"]), float(params.get("P1", 101325.0)),
                     float(params.get("T1", 288.15)))
            if not all(map(math.isfinite, state)) or min(state[1:]) <= 0:
                raise ValueError("Require finite Mach and positive finite absolute pressure and temperature")
            m1, p1, t1 = map(Fraction, state)
            if m1 < 1:
                raise ValueError("A compressive normal shock requires supersonic inflow")
            effective = (float(g), float(gas), float(cp))
            if not all(math.isfinite(v) and v > 0 for v in effective):
                raise ValueError("Effective caloric properties are outside the reporting range")
            return dict(out, **normal_shock_state(m1, g, gas, p1, t1),
                        gamma_effective=effective[0], R_effective_J_kgK=effective[1], cp_J_kgK=effective[2])
        except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
            return dict(out, error=str(exc))

    if subkind == "fanno":
        return _fanno_duct(params, gamma, r_gas)

    if subkind == "rayleigh":
        return _rayleigh_duct(params, gamma, r_gas)

    if subkind == "acoustics":
        return _passive_acoustic_interface(params)

    raise ValueError(f"Unsupported compressible_acoustics subkind: {subkind}")


def _rlc_matrix_product(a, b):
    """Multiply two row-major 2x2 Decimal matrices."""
    return (a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3])


def _rlc_matrix_vector(matrix, vector):
    return (matrix[0] * vector[0] + matrix[1] * vector[1],
            matrix[2] * vector[0] + matrix[3] * vector[1])


def _rlc_matrix_reference(r, l, c, vs, v0, i0, time_, digits=50):
    """Independent affine matrix exponential, without modal roots.

    Use energy coordinates (v, i*sqrt(L/C)) and propagate the constant
    source column directly. This preserves tiny zero-state responses that
    would disappear when subtracting two equilibrium-sized quantities.
    """
    if time_ == 0 or (v0 == vs and i0 == 0):
        return v0, i0, 0
    with localcontext() as context:
        context.prec = digits
        rd, ld, cd, sd, vd, id_, td = map(
            lambda value: Decimal(str(value)), (r, l, c, vs, v0, i0, time_))
        omega = Decimal(1) / (ld * cd).sqrt()
        alpha = rd / (2 * ld)
        # Infinity norm of the augmented constant-source generator.
        norm = 2 * td * (omega + alpha)
        context.prec += max(0, norm.adjusted() + 1) if norm else 0
        omega = Decimal(1) / (ld * cd).sqrt()
        alpha = rd / (2 * ld)
        impedance = (ld / cd).sqrt()
        norm = 2 * td * (omega + alpha)
        scale, squarings = Decimal(1), 0
        while norm / scale > Decimal("0.125"):
            scale *= 2
            squarings += 1
        h = (Decimal(0), td * omega / scale,
             -td * omega / scale, -td * 2 * alpha / scale)
        forcing = (Decimal(0), td * omega / scale)
        h_norm = norm / scale
        term = total = (Decimal(1), Decimal(0), Decimal(0), Decimal(1))
        source_term = source_total = (Decimal(0), Decimal(0))
        term_bound = Decimal(1)
        tolerance = Decimal(1).scaleb(-context.prec + 8)
        for degree in range(1, max(64, context.prec)):
            term = tuple(value / degree for value in _rlc_matrix_product(term, h))
            source_term = (forcing if degree == 1 else tuple(
                value / degree for value in _rlc_matrix_vector(h, source_term)))
            total = tuple(a + b for a, b in zip(total, term))
            source_total = tuple(a + b for a, b in zip(source_total, source_term))
            term_bound *= h_norm / degree
            tail_bound = (term_bound * h_norm / (degree + 1)
                          / (1 - h_norm / (degree + 2)))
            # A zero-state voltage first appears in degree two.
            if degree >= 2 and tail_bound <= tolerance:
                break
        else:
            raise ArithmeticError("RLC matrix exponential did not meet its tail bound")
        for _ in range(squarings):
            transported = _rlc_matrix_vector(total, source_total)
            source_total = tuple(a + b for a, b in zip(source_total, transported))
            total = _rlc_matrix_product(total, total)
        scaled_current = id_ * impedance
        voltage = total[0] * vd + total[1] * scaled_current + source_total[0] * sd
        current = (total[2] * vd + total[3] * scaled_current + source_total[1] * sd) / impedance
        return float(voltage), float(current), squarings


def _rlc_overdamped_modes(r, l, c, vs, v0, i0, time_):
    """Project the initial state on the two real poles in physical coordinates."""
    rd, ld, cd, sd, vd, id_, td = map(
        lambda value: Decimal(str(value)), (r, l, c, vs, v0, i0, time_))
    with localcontext() as context:
        context.prec = 60
        discriminant = Fraction(rd)**2 * Fraction(cd) - 4 * Fraction(ld)
        if discriminant <= 0:
            # Rounded binary rates can straddle critical damping. The existing
            # continuous propagator handles this near-coincident limit.
            return None
        x = Fraction(vd) - Fraction(sd)
        current = Fraction(id_)
        # L(i-C*f*x)(i-C*s*x), with x=v0-vs and s,f the slow/fast poles.
        # Form it exactly so a pure eigenmode has zero complementary amplitude.
        invariant = (Fraction(ld) * current**2 + Fraction(rd) * Fraction(cd) * x * current
                     + Fraction(cd) * x**2)
        for _ in range(2):
            radicand = Decimal(discriminant.numerator) / Decimal(discriminant.denominator) / cd
            delta = Decimal(x.numerator) / Decimal(x.denominator)
            fast = -(rd + radicand.sqrt()) / (2 * ld)
            slow = 1 / (ld * cd * fast)
            slow_numerator = id_ - cd * fast * delta
            fast_numerator = id_ - cd * slow * delta
            product = Decimal(invariant.numerator) / Decimal(invariant.denominator) / ld
            # Divide the exact product by the larger projection, avoiding
            # subtraction of two almost equal quantities in the smaller one.
            if abs(slow_numerator) < abs(fast_numerator):
                slow_numerator = product / fast_numerator
            elif slow_numerator:
                fast_numerator = product / slow_numerator
            denominator = cd * (slow - fast)
            slow_v = slow_numerator / denominator * (slow * td).exp()
            fast_v = -fast_numerator / denominator * (fast * td).exp()
            slow_i, fast_i = cd * slow * slow_v, cd * fast * fast_v
            voltage, current_end = sd + slow_v + fast_v, slow_i + fast_i
            # An applied source or a crossing can cancel the final sum. Use
            # enough digits to resolve a binary64 endpoint even from 1e308.
            sums = ((voltage, max(abs(sd), abs(slow_v), abs(fast_v))),
                    (current_end, max(abs(slow_i), abs(fast_i))))
            if all(not size or (value and size.adjusted() - value.adjusted() < context.prec - 25)
                   for value, size in sums):
                break
            context.prec = max(context.prec, 380 + max(size.adjusted() for _, size in sums))
        return float(voltage), float(current_end)


def _rlc_analytic_state(alpha, omega, impedance, vs, v0, i0, time_, r, l, c):
    """Stable modal propagator; voltage and impedance-scaled current."""
    if alpha == omega:
        regime = "critically_damped"
    elif alpha < omega:
        regime = "underdamped"
    else:
        regime = "overdamped"
    if time_ == 0 or (v0 == vs and i0 == 0):
        return v0, i0, regime
    wt, at = omega * time_, alpha * time_
    if wt + 2 * at <= 0.125:
        # Affine increments keep the O(t^2) zero-state voltage explicit.
        initial_current = i0 * impedance
        increment = (wt * initial_current,
                     wt * (vs - v0) - 2 * at * initial_current)
        voltages, currents = [v0], [initial_current]
        for degree in range(1, 49):
            voltages.append(increment[0])
            currents.append(increment[1])
            voltage, current = math.fsum(voltages), math.fsum(currents)
            if (degree >= 2
                    and abs(increment[0]) <= 2e-16 * max(abs(voltage), math.ulp(0.0))
                    and abs(increment[1]) <= 2e-16 * max(abs(current), math.ulp(0.0))):
                break
            increment = (wt * increment[1] / (degree + 1),
                         (-wt * increment[0] - 2 * at * increment[1]) / (degree + 1))
        return voltage, current / impedance, regime
    if regime == "overdamped":
        if v0 or i0:
            endpoint = _rlc_overdamped_modes(r, l, c, vs, v0, i0, time_)
            if endpoint is not None:
                return endpoint[0], endpoint[1], regime
        beta = math.sqrt(alpha - omega) * math.sqrt(alpha + omega)
        denominator = alpha + beta
        slow = -omega * (omega / denominator)
        fast = -denominator
        gap = 2 * beta
        if not all(math.isfinite(value) for value in (beta, denominator, gap)):
            raise ValueError("RLC modal rates exceed the supported finite range")
        e_slow, e_fast = math.exp(slow * time_), math.exp(fast * time_)
        if gap == 0 or gap * time_ == 0:
            k = time_ * e_slow
        else:
            k = e_slow * (-math.expm1(-gap * time_)) / gap
        p00, p11 = e_slow - slow * k, e_fast + slow * k
        source_gain = -math.expm1(slow * time_) + slow * k
    else:
        decay = math.exp(-alpha * time_)
        if not decay:
            return vs, 0.0, regime
        if regime == "critically_damped":
            k = time_ * decay
            cosine = decay
            source_gain = -math.expm1(-alpha * time_) - alpha * k
        else:
            frequency = math.sqrt(omega - alpha) * math.sqrt(omega + alpha)
            phase = frequency * time_
            if not math.isfinite(phase):
                raise ValueError("RLC oscillation phase exceeds the supported finite range")
            sinc = math.sin(phase) / phase if phase else 1.0
            k = time_ * decay * sinc
            cosine = decay * math.cos(phase)
            source_gain = (-math.expm1(-alpha * time_)
                           + 2 * decay * math.sin(0.5 * phase) ** 2 - alpha * k)
        p00, p11 = cosine + alpha * k, cosine - alpha * k
    cross = omega * k
    shifted, scaled_current = v0 - vs, i0 * impedance
    voltage = p00 * v0 + source_gain * vs + cross * scaled_current
    current = (-cross * shifted + p11 * scaled_current) / impedance
    return voltage, current, regime


def _solve_rlc_transient(params):
    r, l, c = (float(params[name]) for name in ("R", "L", "C"))
    vs, v0, i0 = (float(params.get(name, default))
                  for name, default in (("V_s", 1.0), ("v_c0", 0.0), ("i_l0", 0.0)))
    time_ = float(params["t"])
    if not all(math.isfinite(value) for value in (r, l, c, vs, v0, i0, time_)):
        raise ValueError("RLC parameters must be finite")
    if r < 0 or l <= 0 or c <= 0 or time_ < 0:
        raise ValueError("Passive RLC requires R >= 0, L > 0, C > 0 and t >= 0")
    alpha = (0.5 * r) / l
    lc = l * c
    omega = (1.0 / math.sqrt(lc) if math.isfinite(lc) and lc > 0
             else (1.0 / math.sqrt(l)) / math.sqrt(c))
    impedance = math.sqrt(l) / math.sqrt(c)
    if not all(math.isfinite(value) and value > 0 for value in (omega, impedance)):
        raise ValueError("RLC natural scales exceed the supported finite range")
    if not math.isfinite(alpha) or not math.isfinite(alpha / omega):
        raise ValueError("RLC damping exceeds the supported finite range")
    if not math.isfinite(v0 - vs) or not math.isfinite(i0 * impedance):
        raise ValueError("RLC initial state exceeds the supported finite range")
    voltage, current, regime = _rlc_analytic_state(
        alpha, omega, impedance, vs, v0, i0, time_, r, l, c)
    if not math.isfinite(voltage) or not math.isfinite(current):
        raise ValueError("RLC solution exceeds the supported finite range")
    decayed = regime == "overdamped" and (abs(voltage) < abs(v0) or abs(current) < abs(i0))
    digits = 50
    if decayed:
        # Budget decimal places for cancellation of initial/source contributions.
        z = Decimal(str(impedance))
        v_scale = max(Decimal(str(abs(v0))), Decimal(str(abs(vs))), Decimal(str(abs(i0))) * z)
        smallest = Decimal(str(math.ulp(0.0)))
        lost_digits = max(scale.adjusted() - max(Decimal(str(abs(value))), smallest).adjusted()
                          for scale, value in ((v_scale, voltage), (v_scale / z, current)))
        digits = max(digits, 32 + lost_digits)
    args = (r, l, c, vs, v0, i0, time_)
    reference_v, reference_i, squarings = _rlc_matrix_reference(*args, digits)
    if not math.isfinite(reference_v) or not math.isfinite(reference_i):
        raise ValueError("RLC solution exceeds the supported finite range")
    voltage_scale = max(abs(reference_v), math.ulp(0.0))
    current_scale = max(abs(reference_i), math.ulp(0.0))
    reference_converged = True
    if decayed:
        # Require two independent precision settings to round to the same
        # endpoints before qualifying relative error on the decayed signal.
        check_v, check_i, _ = _rlc_matrix_reference(*args, digits + 16)
        reference_converged = (reference_v, reference_i) == (check_v, check_i)
    else:
        voltage_scale = max(voltage_scale, abs(v0))
        current_scale = max(current_scale, abs(i0))
    residual = max(abs(voltage - reference_v) / voltage_scale,
                   abs(current - reference_i) / current_scale)
    zeta = alpha / omega
    if alpha < omega:
        frequency = math.sqrt(omega - alpha) * math.sqrt(omega + alpha)
        overshoot = 100.0 * math.exp(-math.pi * alpha / frequency)
    else:
        overshoot = 0.0
    return {
        "domain": "rf_transmission_rlc",
        "subkind": "rlc_transient",
        "regime": regime,
        "omega0_rad_s": omega,
        "zeta": zeta,
        "v_c_V": voltage,
        "i_l_A": current,
        "percent_overshoot": overshoot,
        "residual": residual,
        "verified": reference_converged and residual < 1e-8,
        "reference_converged": reference_converged if decayed else None,
        "verification_method": "decimal_scaled_matrix_exponential",
        "reference_squarings": squarings,
    }


def _transmission_port(params):
    """Propagate voltage and Z0-scaled current without dividing at a port node."""
    from mpmath import mp
    result = {"domain": "rf_transmission_rlc", "subkind": "transmission_line", "verified": False}

    def scalar(key, default):
        value = float(params.get(key, default))
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite {key}")
        return Fraction(value)

    def wire(value):
        f = float(value)
        if not math.isfinite(f) or (value and f == 0):
            raise ArithmeticError("Unrepresentable transmission line output")
        return f

    def chart(numerator, denominator):
        if denominator == 0:
            return None, None
        value = numerator/denominator
        try:
            return wire(value.real), wire(value.imag)
        except ArithmeticError:
            return None, None

    try:
        characteristic = complex(params.get("Z0", 50.))
        if not characteristic or not all(map(math.isfinite, (characteristic.real, characteristic.imag))):
            raise ValueError("Characteristic impedance must be finite and nonzero")
        r, x = scalar("R_L", 100.), scalar("X_L", 0.)
        length = scalar("length_m", 1.)
        attenuation = scalar("alpha_np_m", 0.)*length
        in_turns = params.get("length_wavelengths") is not None
        landmark = 0
        if in_turns:
            turns = scalar("length_wavelengths", 0.)
            landmark = round(8*turns)
            remaining = turns-Fraction(landmark, 8)
        else:
            remaining = scalar("beta_rad_m", 0.)*length

        ctx = mp.clone()
        # Retain the weak wave as well as the strong one, including a small
        # departure from an exact standing-wave node. Radian input can require
        # additional bits to resolve the phase of a long propagation interval.
        scales = [abs(s) for s in (r, x, Fraction(characteristic.real),
                  Fraction(characteristic.imag), attenuation, remaining, 1) if s]
        contrast = max(scales)/min(scales)
        ctx.prec = 160+2*(contrast.numerator.bit_length()-contrast.denominator.bit_length())

        def real(q):
            return ctx.mpf(q.numerator)/q.denominator

        line = ctx.mpc(characteristic.real, characteristic.imag)
        load = ctx.mpc(real(r), real(x))
        if load+line == 0:
            raise ValueError("Load reflection coefficient is undefined")
        phase = real(remaining)*(2*ctx.pi if in_turns else 1)
        loss = real(attenuation)
        # At exact eighth turns, use only {0,±1,±i}. The common scale of V
        # and I is arbitrary, so no independently rounded sqrt(2) is needed.
        a, b = ((load, line), (load+ctx.j*line, line+ctx.j*load),
                (line, load), (load-ctx.j*line, line-ctx.j*load))[landmark % 4]
        if not remaining and not attenuation:
            v, i = a, b
        else:
            t = ctx.tanh(ctx.mpc(loss,phase))
            v, i = a+b*t, b+a*t
            if not v or not i:
                raise ArithmeticError("Unresolved finite transmission port")

        # Independently propagate the forward/backward traveling waves. Their
        # common attenuation cancels, leaving one relative reflection factor.
        reflection = (load-line)/(load+line)
        eighth = (1, -ctx.j, -1, ctx.j)[landmark % 4]
        reflected = reflection*eighth*ctx.exp(-2*ctx.mpc(loss,phase))
        v_check, i_check = 1+reflected, 1-reflected
        check_scale = max(abs(v), abs(i))*max(abs(v_check), abs(i_check))
        mismatch = abs(v*i_check-i*v_check)/check_scale
        kind = "open" if not i else "short" if not v else "finite"
        if abs(reflected) < ctx.mpf(".25"):
            # Preserve the small complex correction to a matched line: complex
            # division of two nearly identical port phasors can invent reactance.
            zin = line+2*line*reflected/(1-reflected)
            yin = 1/line-2*reflected/(line*(1+reflected))
            zr, zi = chart(zin, 1)
            yr, yi = chart(yin, 1)
        else:
            zr, zi = chart(line*v, i)
            yr, yi = chart(i, line*v)
        if kind == "finite" and zr is None and yr is None:
            raise ArithmeticError("Both finite port charts exceed the output range")

        for reported, reference in (((zr,zi), (line*v,i)), ((yr,yi), (i,line*v))):
            if reported[0] is not None:
                measured = ctx.mpc(*reported)*reference[1]
                scale = max(abs(measured), abs(reference[0]))
                if scale:
                    mismatch = max(mismatch, abs(measured-reference[0])/scale)

        # The lossless forward/backward load-wave ratio can be exactly one
        # even though its voltage standing-wave ratio has no finite value.
        plus, minus = abs(load+line), abs(load-line)
        gamma = abs(reflection)
        power = 4*ctx.re(load*line.conjugate())
        vswr, vswr_status = None, "unbounded"
        if power:
            try:
                vswr = wire((plus+minus)**2/abs(power))
                vswr_status = "finite"
            except ArithmeticError:
                vswr_status = "outside_numeric_range"
        return_loss, loss_status = None, "infinite"
        if gamma:
            try:
                # 1-|Gamma|² equals the normalized real load-wave power.
                # Its zero and sign survive a purely reactive termination.
                power_fraction = power/plus**2
                loss_db = (-10*ctx.log1p(-power_fraction)/ctx.log(10)
                           if abs(power_fraction) <= .5 else -20*ctx.log10(gamma))
                return_loss = wire(loss_db)
                loss_status = "finite"
            except ArithmeticError:
                loss_status = "outside_numeric_range"
        transformer = None
        if not x and r >= 0 and characteristic.real >= 0 and not characteristic.imag:
            transformer = wire(ctx.sqrt(real(r)*line.real))
        return dict(result, Z_in_kind=kind, Z_in_real=zr, Z_in_imag=zi,
                    Y_in_real=yr, Y_in_imag=yi, gamma_load_mag=wire(gamma),
                    gamma_load_deg=wire(ctx.degrees(ctx.arg(reflection))),
                    VSWR=vswr, VSWR_status=vswr_status,
                    return_loss_dB=return_loss, return_loss_status=loss_status,
                    quarter_wave_Z_T=transformer, residual=float(mismatch), verified=mismatch < 1e-10)
    except (ArithmeticError, ValueError, TypeError) as exc:
        return dict(result, error=str(exc))


def _smith_coordinate(ctx, turns):
    quarter = int(ctx.nint(4 * turns))
    offset = turns - ctx.mpf(quarter) / 4
    return {"quarter_waves": quarter, "offset_wavelengths": ctx.nstr(offset, ctx.dps - 12)}


def _smith_rotation(ctx, quarter, offset):
    # Rotate the small offset before applying its exact quarter-wave anchor.
    c, s = ctx.cospi(2 * offset), ctx.sinpi(2 * offset)
    return ((c, s), (-s, c), (-c, -s))[quarter]


def _smith_residual(ctx, load, stub, line_rotation, stub_rotation):
    # Independent normalized ABCD propagation. Combine the shunt admittance
    # in homogeneous coordinates so an open or short never needs a tangent.
    c, s = line_rotation
    c_stub, s_stub = stub_rotation
    voltage = load * c + ctx.j * s
    current = c + ctx.j * load * s
    if stub == "short":
        num, den = -ctx.j * c_stub, s_stub
    else:
        num, den = ctx.j * s_stub, c_stub
    total_current = current * den + voltage * num
    if not total_current:
        return None
    error = float(abs((voltage * den - total_current) / total_current))
    return error if math.isfinite(error) else None


def _solve_smith_stub(params):
    """Synthesize both lossless shunt matches and verify their serialized lengths."""
    from mpmath import mp

    z0 = float(params.get("Z0", 50.0))
    resistance = float(params["R_L"])
    reactance = float(params.get("X_L", 0.0))
    if not all(map(math.isfinite, (z0, resistance, reactance))) or min(z0, resistance) <= 0:
        raise ValueError("Smith stub matching requires finite R_L > 0, Z0 > 0 and finite X_L")
    stub = str(params.get("stub_type", "short")).strip().lower()
    if stub.startswith("short"):
        stub = "short"
    elif stub.startswith("open"):
        stub = "open"
    else:
        raise ValueError("stub_type must select an open or short termination")
    representation = params.get("length_representation", "wavelengths")
    if representation not in ("wavelengths", "endpoint_relative"):
        raise ValueError("length_representation must be wavelengths or endpoint_relative")
    branch = params.get("branch", 0)
    if isinstance(branch, bool) or branch not in (0, 1, "0", "1"):
        raise ValueError("branch must select 0 or 1")

    exponents = [math.frexp(abs(x))[1] for x in (z0, resistance, reactance) if x]
    ctx = mp.clone()
    # Resolve the weak wave and small angular offsets across the input contrast.
    ctx.prec = 192 + 3 * (max(exponents) - min(exponents))
    load = ctx.mpc(resistance, reactance) / ctx.mpf(z0)
    magnitude = abs(load - 1) / ctx.sqrt(load.real)
    gamma_load = (load - 1) / (load + 1)

    def decode(coordinate):
        return _smith_rotation(ctx, coordinate["quarter_waves"], ctx.mpf(coordinate["offset_wavelengths"]))

    solutions = []
    for susceptance in ((magnitude, -magnitude) if magnitude else (ctx.zero,)):
        if susceptance:
            # The two unit-conductance intersections have y = 1 +/- j|b|.
            gamma_match = -ctx.j * susceptance / (2 + ctx.j * susceptance)
            phase = ctx.arg(gamma_load * ctx.conj(gamma_match))
            distance = (phase / (4 * ctx.pi)) % ctx.mpf("0.5")
        else:
            distance = ctx.zero
        if stub == "short":
            length = ctx.atan2(1, susceptance) / (2 * ctx.pi)
        else:
            length = (ctx.atan(-susceptance) / (2 * ctx.pi)) % ctx.mpf("0.5")

        # Map a length rounded to a half wavelength to its equivalent zero.
        d, l_stub = float(distance) % 0.5, float(length) % 0.5
        d_coord, l_coord = _smith_coordinate(ctx, distance), _smith_coordinate(ctx, length)
        plain = _smith_residual(ctx, load, stub, _smith_rotation(ctx, 0, ctx.mpf(d)),
                                _smith_rotation(ctx, 0, ctx.mpf(l_stub)))
        relative = _smith_residual(ctx, load, stub, decode(d_coord), decode(l_coord))
        residual = relative if representation == "endpoint_relative" else plain
        solutions.append((distance, {
            "d_over_lambda": d, "l_stub_over_lambda": l_stub,
            "d_coordinate": d_coord, "l_stub_coordinate": l_coord,
            "absolute_length_residual": plain,
            "absolute_lengths_verified": plain is not None and plain < 1e-10,
            "endpoint_relative_residual": relative,
            "endpoint_relative_verified": relative is not None and relative < 1e-10,
            "residual": residual, "verified": residual is not None and residual < 1e-10,
        }))

    all_branches = [row for _, row in sorted(solutions, key=lambda item: item[0])]
    chosen = all_branches[min(int(branch), len(all_branches) - 1)]
    return {"domain": "rf_transmission_rlc", "subkind": "smith_stub", "stub_type": stub,
            "length_representation": representation, **chosen, "all_branches": all_branches}


def solve_transmission_line_rlc(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for lossless/lossy transmission lines, Smith-chart stubs, and RLC transients."""
    subkind = str(params.get("subkind", "transmission_line")).strip().lower()

    if subkind == "transmission_line":
        return _transmission_port(params)

    if subkind == "smith_stub":
        return _solve_smith_stub(params)

    if subkind == "rlc_transient":
        return _solve_rlc_transient(params)

    raise ValueError(f"Unsupported rf_transmission_rlc subkind: {subkind}")


def _rounded_stress_root(compare):
    """Round a real algebraic stress quantity by exact comparisons with binary64 values."""
    import struct

    def value(bits):
        return struct.unpack(">d", bits.to_bytes(8, "big"))[0]

    sign = compare(Fraction(0))
    if not sign:
        return 0.0
    low, high = 0, 0x7ff0000000000000
    while high-low > 1:
        mid = (low+high)//2
        candidate = value(mid)
        side = sign*compare(sign*Fraction(candidate))
        if side == 0:
            return sign*candidate
        if side > 0:
            low = mid
        else:
            high = mid
    a = Fraction(value(low))
    b = Fraction(1 << 1024) if high == 0x7ff0000000000000 else Fraction(value(high))
    side = sign*compare(sign*(a+b)/2)
    chosen = low if side < 0 or (side == 0 and low%2 == 0) else high
    if chosen == 0x7ff0000000000000 or chosen == 0:
        raise ValueError("Nonzero stress result exceeds the binary64 reporting range")
    return sign*value(chosen)


def _mohr_stress(params):
    """Recover pressure-independent shear and yield from both Cartesian and principal invariants."""
    def sign(v):
        return int(v > 0) - int(v < 0)

    def number(key):
        v = float(params.get(key, 0.0))
        if not math.isfinite(v):
            raise ValueError("Stress inputs must be finite")
        return Fraction(v)

    def compare_root(index, quadratic, linear, constant):
        # The shifted monic cubic has three real roots. Sign variations
        # count those strictly above zero, including their multiplicities.
        signs = [v > 0 for v in (1, quadratic, linear, constant) if v]
        greater = sum(a != b for a, b in zip(signs, signs[1:]))
        equal = (1 + int(linear == 0) + int(linear == 0 and quadratic == 0)) if constant == 0 else 0
        return 1 if index < greater else (0 if index < greater+equal else -1)

    def positive_sqrt(square, threshold):
        return 1 if threshold < 0 else sign(square-threshold*threshold)

    try:
        x, y, z = (number(k) for k in ("sigma_x", "sigma_y", "sigma_z"))
        xy, xz, yz = (number(k) for k in ("tau_xy", "tau_xz", "tau_yz"))
        strength = number("sigma_yield") if params.get("sigma_yield") is not None else None
        if strength is not None and strength < 0:
            raise ValueError("Yield strength must be nonnegative")
        mean = (x+y+z)/3
        a, b, c = x-mean, y-mean, z-mean
        j2 = ((x-y)**2+(y-z)**2+(z-x)**2)/6 + xy*xy+xz*xz+yz*yz
        j3 = a*b*c + 2*xy*xz*yz - a*yz*yz - b*xz*xz - c*xy*xy

        def principal(index, threshold):
            v = threshold-mean
            return compare_root(index, 3*v, 3*v*v-j2, v*(v*v-j2)-j3)

        # The three squared pairwise principal differences form another
        # real-rooted cubic. Its largest root is the squared Tresca stress.
        gap_a, gap_b = -6*j2, 9*j2*j2
        discriminant = 4*j2**3-27*j3*j3

        def gap(threshold):
            if threshold < 0:
                return 1
            v = threshold*threshold
            return compare_root(0, 3*v+gap_a, 3*v*v+2*gap_a*v+gap_b,
                                ((v+gap_a)*v+gap_b)*v-discriminant)

        def safety(compare):
            if strength is None or not j2:
                return None
            if not strength:
                return 0.0
            return _rounded_stress_root(lambda t: 1 if t <= 0 else -compare(strength/t))

        vm = lambda t: positive_sqrt(3*j2, t)
        ux, uy = x-y, 2*xy
        scale = max(abs(ux), abs(uy))
        angle = 0.5*math.degrees(math.atan2(float(uy/scale), float(ux/scale))) if scale else 0.
        principal_wire = ([_rounded_stress_root(lambda t: principal(i, t)) for i in range(3)]
                          if j2 else [float(mean)]*3)
        residual = 0.
        if j2:
            # Independent numerical spectrum in deviatoric coordinates. The
            # pressure shift cannot hide a wrong J2, determinant or stress gap.
            unit = max(map(abs, (a,b,c,xy,xz,yz)))
            tensor = np.array([[float(v/unit) for v in row] for row in
                               ((a,xy,xz), (xy,b,yz), (xz,yz,c))])
            spectrum = np.linalg.eigvalsh(tensor)
            ref_gap = _rounded_stress_root(lambda t: gap(unit*t))
            residual = max(abs(float(spectrum.sum())),
                           abs(float(spectrum @ spectrum)/2 - float(j2/unit**2)),
                           abs(float(np.prod(spectrum)) - float(j3/unit**3)),
                           abs(float(spectrum[-1]-spectrum[0])-ref_gap))
        return {"sigma_1": principal_wire[0], "sigma_2": principal_wire[1], "sigma_3": principal_wire[2],
                "tau_max_inplane": _rounded_stress_root(lambda t: positive_sqrt(ux*ux/4+xy*xy, t)),
                "tau_max_abs": _rounded_stress_root(lambda t: gap(2*t)),
                "theta_p_deg": angle, "sigma_von_mises": _rounded_stress_root(vm),
                "safety_factor_vm": safety(vm), "safety_factor_tresca": safety(gap),
                "residual": residual, "verified": residual < 1e-11}
    except (ArithmeticError, ValueError) as exc:
        return {"verified": False, "error": str(exc)}


def _lame_annulus(params):
    """Resolve a common wall and sample radius before evaluating its tractions."""
    def observation(name):
        raw = params[name]
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        if type(raw) is int:
            value = Fraction(raw)
            return value, value, value, True
        f = float(raw)
        if not math.isfinite(f):
            raise ValueError(f"{name} must be finite")
        value = Fraction(f)
        low = math.nextafter(f, -math.inf)
        high = math.nextafter(f, math.inf)
        lo = Fraction(low) if math.isfinite(low) else 2*value-Fraction(high)
        hi = Fraction(high) if math.isfinite(high) else 2*value-Fraction(low)
        return value, (value+lo)/2, (value+hi)/2, False

    def agrees(coord, value):
        center,lo,hi,exact = coord
        if exact:
            return value == center
        if lo < value < hi:
            return True
        if value != lo and value != hi:
            return False
        # Endpoint ties belong to a stored significand with an even low bit.
        return (center/Fraction(math.ulp(float(center)))) % 2 == 0

    inner = observation("r_i")
    outer = observation("r_o") if params.get("r_o") is not None else None
    wall = observation("t") if params.get("t") is not None else None
    sample = observation("r_eval") if params.get("r_eval") is not None else None
    if not (outer or wall):
        raise ValueError("Supply an outer radius or a wall thickness")
    if any(u[0] <= 0 for u in (inner, outer, wall, sample) if u is not None):
        raise ValueError("Radii and thickness must be positive")

    if outer is None:
        candidates = [(inner[0], inner[0]+wall[0])]
    elif wall is None:
        candidates = [(inner[0], outer[0])]
    else:
        # Prefer supplied centers when they form an admissible annulus. If
        # all three measurements constrain it off-center, project ri+t=ro
        # onto the outer-radius interval and take an interior representative.
        candidates = [(inner[0], outer[0]), (inner[0], inner[0]+wall[0]),
                      (outer[0]-wall[0], outer[0])]
        ri_lo, ri_hi = inner[1:3]
        ro_lo, ro_hi = outer[1:3]
        if sample:
            ri_hi = min(ri_hi, sample[2])
            ro_lo = max(ro_lo, sample[1])
        lo = max(ro_lo, ri_lo+wall[1])
        hi = min(ro_hi, ri_hi+wall[2])
        if ri_lo <= ri_hi and ro_lo <= ro_hi and lo <= hi:
            ro = (lo+hi)/2
            inside_lo, inside_hi = max(ri_lo,ro-wall[2]), min(ri_hi,ro-wall[1])
            if inside_lo <= inside_hi:
                candidates.append(((inside_lo+inside_hi)/2,ro))

    for ri,ro in candidates:
        if not 0 < ri < ro or not agrees(inner,ri):
            continue
        if outer and not agrees(outer,ro) or wall and not agrees(wall,ro-ri):
            continue
        r = min(ro,max(ri,sample[0])) if sample else ri
        if sample is None or agrees(sample,r):
            return ri,ro,r
    raise ValueError("No positive annular wall and material radius satisfy all supplied dimensions")


def _lame_stress(params):
    result = {"domain":"solid_mohr_vessel_buckling", "subkind":"pressure_vessel", "verified":False}
    try:
        def pressure(name):
            raw = params.get(name,0) if name == "p_o" else params[name]
            if isinstance(raw,bool):
                raise ValueError(f"{name} must be a finite pressure")
            return Fraction(raw) if type(raw) is int else Fraction(float(raw))

        ri,ro,r = _lame_annulus(params)
        pi,po = pressure("p_i"),pressure("p_o")
        closed = params.get("closed_ends", True)
        if type(closed) is int and closed in (0, 1):
            closed = bool(closed)
        if type(closed) is not bool:
            raise ValueError("closed_ends must be a boolean")
        area = (ro-ri)*(ro+ri)
        uniform = (pi*ri**2-po*ro**2)/area
        b = (pi-po)*ri**2*ro**2/area
        amplitude = b/r**2
        sr,st = uniform-amplitude,uniform+amplitude
        sz = uniform if closed else Fraction(0)
        mean = (sr+st+sz)/3
        # Radial equilibrium, diametral cut force and end closure are
        # checked in the original pressure/stress coordinates.
        if (uniform-b/ri**2 != -pi or uniform-b/ro**2 != -po or
                uniform*(ro-ri)+b*(1/ri-1/ro) != pi*ri-po*ro or
                sz*area != (pi*ri**2-po*ro**2 if closed else 0)):
            raise ValueError("Annular pressure tractions do not balance")
        vm2 = 3*amplitude**2+(sz-uniform)**2
        if 2*vm2 != (st-sr)**2+(sr-sz)**2+(sz-st)**2:
            raise ValueError("Deviatoric invariant does not match the principal stress differences")

        errors = []
        def report(x):
            shown = float(x)
            if not math.isfinite(shown):
                raise ValueError("Annular result exceeds finite reporting range")
            if x:
                errors.append(abs(Fraction(shown)/x-1))
            return shown
        def vm_compare(trial):
            if trial < 0:
                return 1
            radicand = vm2-trial**2
            return (radicand>0)-(radicand<0)
        vm = _rounded_stress_root(vm_compare)
        if vm2:
            errors.append(abs(Fraction(vm)**2/vm2-1))
        membrane = (pi-po)*ri/(ro-ri)
        out = dict(r_i=report(ri),r_o=report(ro),r_eval=report(r),wall_thickness=report(ro-ri),
                   sigma_hoop_lame=report(st),sigma_radial_lame=report(sr),sigma_axial_lame=report(sz),
                   sigma_hoop_thin=report(-po+membrane),
                   sigma_axial_thin=report(-po+membrane/2) if closed else 0.,sigma_von_mises=vm,
                   thin_wall_model="Hoop mean includes external pressure. Closed ends use the inner-radius axial membrane approximation. Freely open ends carry zero axial stress.")
        out["stress_split"] = dict(uniform_in_plane=report(uniform), radial_hoop_amplitude=report(amplitude),
                                   mean_normal_stress=report(mean), deviatoric_radial=report(sr-mean),
                                   deviatoric_hoop=report(st-mean), deviatoric_axial=report(sz-mean))
        residual = max(errors,default=Fraction(0))
        if residual >= Fraction(1,10**11):
            raise ValueError("Resolved wall geometry or stress exceeds the supported reporting resolution")
        result.update(out,verified=True,residual=float(residual))
    except (ArithmeticError,KeyError,ValueError,TypeError) as exc:
        result["error"] = str(exc)
    return result


def solve_mohr_vessel_buckling(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for 2D/3D stress states, Lame/thin vessels, and Euler-Johnson buckling."""
    subkind = str(params.get("subkind", "mohr_stress")).strip().lower()

    if subkind == "mohr_stress":
        return {"domain": "solid_mohr_vessel_buckling", "subkind": subkind, **_mohr_stress(params)}

    if subkind == "pressure_vessel":
        return _lame_stress(params)

    if subkind == "column_buckling":
        from hamon_column_buckling import column_buckling
        return column_buckling(params)

    raise ValueError(f"Unsupported solid_mohr_vessel_buckling subkind: {subkind}")



def _cstr_poly_trim(p):
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def _cstr_poly_divmod(p, q):
    from fractions import Fraction
    p, q = _cstr_poly_trim(p[:]), _cstr_poly_trim(q[:])
    quotient = [Fraction(0)] * max(1, len(p) - len(q) + 1)
    while len(p) >= len(q) and any(p):
        factor, shift = p[-1] / q[-1], len(p) - len(q)
        quotient[shift] = factor
        for i, c in enumerate(q):
            p[i + shift] -= factor*c
        _cstr_poly_trim(p)
    return _cstr_poly_trim(quotient), p


def _cstr_poly_gcd(p, q):
    while any(q):
        p, q = q, _cstr_poly_divmod(p, q)[1]
    return [c/p[-1] for c in p]


def _cstr_stationary_points(coefficients, *, intervals=False, precision_bits=190):
    """Isolate every distinct open-unit-interval root using exact Sturm signs."""
    from fractions import Fraction

    def value(p, x):
        y = Fraction(0)
        for c in reversed(p):
            y = y * x + c
        return y

    p = _cstr_poly_trim(coefficients[:])
    if not any(p):
        raise ValueError("indeterminate CSTR stationary polynomial")
    while len(p) > 1 and p[0] == 0:
        p.pop(0)
    while len(p) > 1 and value(p, Fraction(1)) == 0:
        # Exact division by x - 1 removes physical-interval endpoints.
        q = [Fraction(0)] * (len(p) - 1)
        q[-1] = p[-1]
        for j in range(len(q) - 2, -1, -1):
            q[j] = p[j + 1] + q[j + 1]
        p = _cstr_poly_trim(q)
    if len(p) == 1:
        return []
    # Repeated factors must be removed before evaluating a Sturm chain at an
    # exact split: otherwise every chain member may vanish at that boundary.
    derivative = [i*p[i] for i in range(1, len(p))]
    p = _cstr_poly_divmod(p, _cstr_poly_gcd(p, derivative))[0]
    chain = [p, [i * p[i] for i in range(1, len(p))]]
    while any(chain[-1]):
        rem = _cstr_poly_trim([-x for x in _cstr_poly_divmod(chain[-2], chain[-1])[1]])
        if not any(rem):
            break
        # Positive rescaling preserves every Sturm sign.
        scale = abs(rem[-1])
        chain.append([x / scale for x in rem])

    def variations(x):
        signs = [1 if y > 0 else -1 for q in chain if (y := value(q, x)) != 0]
        return sum(a != b for a, b in zip(signs, signs[1:]))

    left, right = Fraction(0), Fraction(1)
    pending = [(left, right, variations(left), variations(right), 0)]
    roots = []
    while pending:
        lo, hi, vlo, vhi, depth = pending.pop()
        count = vlo - vhi
        if count == 0:
            continue
        if count == 1:
            if value(p, hi) == 0:
                roots.append((hi, hi))
                continue
            if value(p, (lo+hi)/2) == 0:
                roots.append(((lo+hi)/2, (lo+hi)/2))
                continue
            if lo > 0 and hi < 1 and hi-lo <= min(lo, 1-hi)/2**precision_bits:
                roots.append((lo, hi))
                continue
        if depth > 2048:
            raise ValueError("CSTR stationary points are not numerically separable")
        mid = (lo + hi) / 2
        vmid = variations(mid)
        pending.extend(((mid, hi, vmid, vhi, depth + 1),
                        (lo, mid, vlo, vmid, depth + 1)))
    roots.sort()
    return roots if intervals else [(lo+hi)/2 for lo, hi in roots]


class _CstrRoot(NamedTuple):
    lower: Fraction
    upper: Fraction
    monotone_from: Fraction
    monotone_to: Fraction
    direction: int

    def coordinates(self):
        midpoint = (self.lower + self.upper) / 2
        return float(midpoint), float(1 - midpoint)


def _cstr_conversion_branches(params, volume, c0, feed, t0, top, delta, epsilon, order, gas, inlet_x, refine):
    """Enumerate the single-CSTR power-law/Arrhenius steady-state equation.

    Exact stationary-polynomial isolation partitions the physical interval.
    High-precision log balances avoid a conversion grid, exponential overflow,
    and a single bracketing search silently discarding other branches.
    Conversion and remaining reactant are rounded separately. The smaller
    coordinate determines its complement when checking the physical balance.
    No dynamical stability classification follows from this steady-state model.
    """
    inputs = (volume, c0, feed, t0, top, delta, epsilon, order)
    if not all(math.isfinite(x) for x in inputs):
        raise ValueError("CSTR inputs must be finite")
    if volume < 0 or min(c0, feed, t0) <= 0 or epsilon <= -1:
        raise ValueError("CSTR requires nonnegative volume, positive feed/concentration/temperature and epsilon_A > -1")
    slope = delta if abs(delta) > 1e-15 else 0.0
    inlet = t0 if slope else top
    if min(inlet, inlet + slope) <= 0:
        raise ValueError("CSTR temperature must stay positive on 0 <= conversion <= 1")
    if volume == 0:
        return [_CstrRoot(inlet_x, inlet_x, 0, 1, 1)]
    if inlet_x == 1:
        return [_CstrRoot(Fraction(1), Fraction(1), 1, 1, 0)] if order > 0 else []

    # Use precisely the same precedence as the maintained rate law.
    if "A_preexp" in params and "E_a" in params:
        rate_kind = "preexponential"
        rate = float(params["A_preexp"])
        activation = float(params["E_a"])
        reference = 1.0
    elif "k_ref" in params and "T_ref" in params and "E_a" in params:
        rate_kind = "reference"
        rate = float(params["k_ref"])
        activation = float(params["E_a"])
        reference = float(params["T_ref"])
    else:
        rate_kind = "constant"
        rate = float(params["k"])
        activation = 0.0
        reference = 1.0
    if not all(math.isfinite(x) for x in (rate, activation, reference)) or min(rate, reference) <= 0:
        raise ValueError("CSTR kinetic rate and reference temperature must be finite and positive")
    F = Fraction
    x, remaining, expansion, temperature = [-inlet_x, F(1.)], [F(1.), F(-1.)], [F(1.), F(epsilon)], [F(inlet), F(slope)]

    def multiply(*polys):
        out = [Fraction(1)]
        for p in polys:
            product = [Fraction(0)] * (len(out) + len(p) - 1)
            for i, a in enumerate(out):
                for j, b in enumerate(p):
                    product[i+j] += a*b
            out = product
        return out

    coefficients = [Fraction(0)] * 5
    terms = [
        (Fraction(1), multiply(remaining, expansion, temperature, temperature)),
        (F(order), multiply(x, expansion, temperature, temperature)),
        (F(order)*F(epsilon), multiply(x, remaining, temperature, temperature)),
        (F(order)*F(slope) if gas else Fraction(0), multiply(x, remaining, expansion, temperature)),
        (-F(activation)/F(8.31446261815324)*F(slope), multiply(x, remaining, expansion)),
    ]
    for factor, poly in terms:
        for i, c in enumerate(poly):
            coefficients[i] += factor*c
    def poly_value(p, q):
        value = Fraction(0)
        for c in reversed(p):
            value = value*q+c
        return value

    # At constant kinetics the balance is algebraic. Clear only positive
    # denominators and use its exact common factors with the stationary
    # polynomial. Evaluating an approximate X cannot recognize a fold at 1/3.
    rational_order = F(order)
    numerator, denominator = rational_order.numerator, rational_order.denominator
    algebraic = activation == 0.0 and denominator + 2*abs(numerator) <= 64
    common = [Fraction(1)]
    rational_roots = []
    if algebraic:
        concentration_num = multiply([F(c0)], remaining)
        concentration_den = expansion
        if gas:
            concentration_num = multiply(concentration_num, [F(t0)])
            concentration_den = multiply(concentration_den, temperature)
        if numerator < 0:
            concentration_num, concentration_den = concentration_den, concentration_num
        left = multiply(*([multiply([F(feed)], x)]*denominator +
                          [concentration_den]*abs(numerator)))
        right = multiply(*([[F(volume)*F(rate)]]*denominator +
                           [concentration_num]*abs(numerator)))
        residual_poly = [Fraction(0)]*max(len(left), len(right))
        for i, c in enumerate(left):
            residual_poly[i] += c
        for i, c in enumerate(right):
            residual_poly[i] -= c
        residual_poly = _cstr_poly_trim(residual_poly)
        if len(residual_poly) == 2:
            root = -residual_poly[0]/residual_poly[1]
            if not inlet_x < root <= 1 or (root == 1 and order != 0):
                return []
            if root < 1 and min(float(root), float(1-root)) == 0:
                raise ValueError("interior CSTR conversion is not representable in complementary coordinates")
            return [_CstrRoot(root, root, inlet_x, F(1), 1)]
        if len(residual_poly) == 3:
            a0, a1, a2 = residual_poly
            disc = a1*a1 - 4*a2*a0
            if disc >= 0:
                sqrt_disc = Fraction(math.isqrt(disc.numerator), math.isqrt(disc.denominator))
                if sqrt_disc*sqrt_disc == disc:
                    rational_roots = [(-a1 + sign*sqrt_disc)/(2*a2) for sign in (-1, 1)]
        common = _cstr_poly_gcd(coefficients, residual_poly)
    ordinary = coefficients[:]
    while len((shared := _cstr_poly_gcd(ordinary, common))) > 1:
        ordinary = _cstr_poly_divmod(ordinary, shared)[0]
    exact_reference = []
    if rate_kind == "reference" and activation != 0 and slope != 0:
        q = (F(reference)-F(inlet))/F(slope)
        if inlet_x < q < 1 and poly_value(coefficients, q) == 0:
            concentration = F(c0)*(1-q)/(1+F(epsilon)*q)
            if gas:
                concentration *= F(t0)/F(reference)
            ratio = F(feed)*(q-inlet_x)/(F(volume)*F(rate))
            if (denominator + abs(numerator) <= 64 and
                    ratio**denominator == concentration**numerator):
                exact_reference.append((q, q))
                while len(ordinary) > 1 and poly_value(ordinary, q) == 0:
                    ordinary = _cstr_poly_divmod(ordinary, [-q, Fraction(1)])[0]

    with localcontext() as context:
        context.prec = min(400, 80 + max(0, math.ceil(math.log10(abs(order))) if order else 0))
        if 0 < inlet_x < 1:
            context.prec = min(400, context.prec + max(0, math.ceil(-math.log10(float(min(inlet_x, 1-inlet_x))))))
        def D(value):
            f = Fraction(value)
            return Decimal(f.numerator) / Decimal(f.denominator)
        dc, df, dt0, di, ds, de, dn, dr = map(D, (c0, feed, t0, inlet, slope, epsilon, order, rate))
        ea = D(activation) / D(8.31446261815324)
        tref = D(reference)
        target = D(volume).ln()
        log_rate, log_concentration = dr.ln(), dc.ln()
        exact_power = activation == 0.0 and order.is_integer() and abs(order) <= 32
        fixed = df.ln()-log_rate-dn*log_concentration-target

        def balance(z):
            if z <= D(inlet_x):
                return Decimal("-Infinity")
            temp = di + ds*z
            log_k = log_rate
            if rate_kind == "preexponential":
                log_k -= ea/temp
            elif rate_kind == "reference":
                log_k -= ea*(1/temp - 1/tref)
            if z == 1 and dn != 0:
                return Decimal("Infinity") if dn > 0 else Decimal("-Infinity")
            if exact_power:
                # Preserve exact endpoint/fold equalities and the adjacent
                # binary64 volumes. Independently rounded logarithms can change
                # a zero into a sign even in 4*x*(1-x) == 1 at x == 1/2.
                q = Fraction(z)
                concentration = F(c0)*(1-q)/(1+F(epsilon)*q)
                if gas:
                    concentration *= F(t0)/(F(inlet)+F(slope)*q)
                mass_balance = F(feed)*(q-inlet_x) - F(volume)*F(rate)*concentration**int(order)
                return Decimal((mass_balance > 0) - (mass_balance < 0))
            log_c = log_concentration - (1 + de*z).ln()
            if dn:
                log_c += (1-z).ln()
            if gas:
                log_c -= (temp/dt0).ln()
            return fixed + (z-D(inlet_x)).ln() + (log_rate-log_k) - dn*(log_c-log_concentration)

        def certified_cut_sign(bounds):
            # Interval bounds use exact rational arguments. Decimal ln is
            # correctly rounded; one outward ulp encloses each logarithm.
            from decimal import ROUND_FLOOR, ROUND_CEILING
            a, b = bounds

            def decimal_bound(q, rounding):
                with localcontext() as directed:
                    directed.rounding = rounding
                    return Decimal(q.numerator)/Decimal(q.denominator)

            def log_interval(a_, b_):
                lo = decimal_bound(a_, ROUND_FLOOR)
                hi = decimal_bound(b_, ROUND_CEILING)
                # ln(1) is exact. Its adjacent Decimal is a context-minimum
                # subnormal, whose Fraction denominator can have a million
                # digits; it is neither needed nor useful for this enclosure.
                return (Decimal(0) if lo == 1 else lo.ln().next_minus(),
                        Decimal(0) if hi == 1 else hi.ln().next_plus())

            terms = [(Fraction(1), log_interval(F(feed), F(feed))),
                     (Fraction(1), log_interval(a-inlet_x, b-inlet_x)),
                     (Fraction(-1), log_interval(F(rate), F(rate))),
                     (Fraction(-1), log_interval(F(volume), F(volume))),
                     (-F(order), log_interval(F(c0), F(c0)))]
            if order:
                terms.append((-F(order), log_interval(1-b, 1-a)))
                expansions = sorted((1+F(epsilon)*a, 1+F(epsilon)*b))
                terms.append((F(order), log_interval(*expansions)))
            temperatures = sorted((F(inlet)+F(slope)*a, F(inlet)+F(slope)*b))
            if gas and order:
                terms.append((F(order), log_interval(temperatures[0]/F(t0), temperatures[1]/F(t0))))
            lower = upper = Fraction(0)
            for factor, (l, h) in terms:
                values = (factor*Fraction(l), factor*Fraction(h))
                lower += min(values)
                upper += max(values)
            if rate_kind != "constant":
                offsets = [F(activation)/F(8.31446261815324)/t for t in temperatures]
                if rate_kind == "reference":
                    offsets = [v-F(activation)/F(8.31446261815324)/F(reference) for v in offsets]
                lower += min(offsets)
                upper += max(offsets)
            return 1 if lower > 0 else -1 if upper < 0 else None

        for bits in (190, 1100):
            stationary = [(bounds, True) for bounds in _cstr_stationary_points(common, intervals=True, precision_bits=bits)]
            stationary += [(bounds, True) for bounds in exact_reference]
            stationary += [(bounds, False) for bounds in _cstr_stationary_points(ordinary, intervals=True, precision_bits=bits)]
            stationary = [(bounds, fold) for bounds, fold in stationary if bounds[1] > inlet_x]
            if any(bounds[0] <= inlet_x for bounds, _ in stationary):
                continue
            stationary.sort(key=lambda item: sum(item[0]))
            signs = [0 if exact else certified_cut_sign(bounds) for bounds, exact in stationary]
            if all(sign is not None for sign in signs):
                break
        else:
            raise ValueError("CSTR stationary balance sign is unresolved at supported precision")
        cuts = [D(inlet_x)] + [Decimal((a+b).numerator)/Decimal((a+b).denominator)/2
                               for (a, b), _ in stationary] + [Decimal(1)]
        values = [balance(Decimal(0))] + signs + [balance(Decimal(1))]
        found = []
        bounds = [(inlet_x, inlet_x)] + [pair for pair, _ in stationary] + [(F(1), F(1))]
        for (left, right), value_ in zip(bounds, values):
            if value_ == 0:
                found.append(_CstrRoot(left, right, left, right, 0))
        for index, (lo, hi, flo, fhi) in enumerate(zip(cuts, cuts[1:], values, values[1:])):
            if flo == 0 or fhi == 0 or (flo > 0) == (fhi > 0):
                continue
            for _ in range(1250):
                mid = (lo+hi)/2
                if mid == lo or mid == hi:
                    if context.prec >= 400:
                        raise ValueError("CSTR root refinement exceeds supported precision")
                    # A small complementary coordinate can need more digits
                    # than an ordinary state. Increase precision only there.
                    context.prec = min(400, 2*context.prec)
                    ea = D(activation)/D(8.31446261815324)
                    target = D(volume).ln()
                    log_rate, log_concentration = dr.ln(), dc.ln()
                    fixed = df.ln()-log_rate-dn*log_concentration-target
                    flo, fhi = balance(lo), balance(hi)
                    continue
                fm = balance(mid)
                if fm == 0:
                    if exact_power:
                        lo = hi = mid
                    else:
                        lo, hi = mid.next_minus(), mid.next_plus()
                    break
                if (flo > 0) == (fm > 0):
                    lo, flo = mid, fm
                else:
                    hi, fhi = mid, fm
                if (float(lo) == float(hi) and float(1-lo) == float(1-hi)
                        and (not refine or hi-lo <= min(lo, 1-hi) * Decimal("1e-30"))):
                    break
            else:
                raise ValueError("CSTR root refinement did not converge")
            root = (lo+hi)/2
            if not float(root) > 0 or not float(1-root) > 0:
                raise ValueError("interior CSTR conversion is not representable in complementary coordinates")
            found.append(_CstrRoot(F(lo), F(hi), bounds[index][1], bounds[index+1][0],
                                   1 if fhi > 0 else -1))
        # A rational root must enter the next stage exactly when it can
        # coincide with a downstream fold. Snap only on polynomial identity.
        for i, root in enumerate(found):
            for q in rational_roots:
                if root.lower <= q <= root.upper:
                    found[i] = root._replace(lower=q, upper=q)
                    break
        found.sort(key=lambda root: root.lower)
        coordinates = [root.coordinates() for root in found]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("distinct CSTR branches are not separable in output precision")
        return found


def _cstr_stage_roots(inputs, inlet_bounds, refine):
    """Track roots throughout the preceding stage's conversion enclosure."""
    low, high = inlet_bounds
    left = _cstr_conversion_branches(*inputs, low, refine)
    if low == high:
        return left
    right = _cstr_conversion_branches(*inputs, high, refine)
    message = "CSTR series branch topology is unresolved at supported precision"
    if len(left) != len(right):
        raise ValueError(message)
    roots = []
    for a, b in zip(left, right):
        lo, hi = min(a.lower, b.lower), max(a.upper, b.upper)
        # At fixed X the unlogged material balance is affine in the inlet.
        # Outside these corridors its endpoint signs agree. Inside, the
        # logarithmic balance's derivative numerator is also affine in the
        # inlet, so the same monotonicity at both extremes covers the interval.
        if (a.direction == 0 or a.direction != b.direction or
                lo <= max(a.monotone_from, b.monotone_from) or
                hi >= min(a.monotone_to, b.monotone_to) or
                (roots and lo <= roots[-1].upper)):
            raise ValueError(message)
        root = _CstrRoot(lo, hi, max(a.monotone_from, b.monotone_from),
                         min(a.monotone_to, b.monotone_to), a.direction)
        if a.coordinates() != b.coordinates():
            raise ValueError("CSTR series outlet rounding is unresolved at supported precision")
        roots.append(root)
    return roots




def _constant_rate_pfr(params, concentration, feed, flow, order, rate, inlet_t, heat_rise):
    """Integrate dU/dV = -(k*C_in**n/F_in)*U**n for remaining fraction U."""
    from mpmath import mp

    ctx = mp.clone()
    # At one unit of exposure, the positive remaining fraction can be of
    # order n itself. Preserve n in the exponent n-1 on both sides of zero.
    ctx.dps = 120 + (max(0, -math.floor(math.log10(abs(order)))) if order else 0)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("PFR kinetic rate must be positive and representable")
    effective_c = ctx.mpf(concentration.numerator)/concentration.denominator
    coefficient = ctx.mpf(rate)*ctx.power(effective_c, order)/feed
    exponent = ctx.mpf(order)-1

    def dose(log_inverse_remaining):
        if not exponent:
            return log_inverse_remaining
        return ctx.expm1(exponent*log_inverse_remaining)/exponent

    def relative(a, b):
        scale = max(abs(a), abs(b))
        return abs(a-b)/scale if scale else ctx.zero

    sizing = params.get("conversion") is not None
    if sizing:
        x = ctx.mpf(float(params["conversion"]))
        u = 1-x
        if x == 1:
            if exponent >= 0:
                raise ValueError("complete PFR conversion requires infinite volume for order >= 1")
            exposure = -1/exponent
        else:
            exposure = dose(-ctx.log1p(-x))
        volume = float(exposure/coefficient)
    else:
        volume = float(params["volume"])
        exposure = ctx.mpf(volume)*coefficient
        if exponent < 0 and exposure >= -1/exponent:
            # Positive fractional-order rates vanish at depletion, permitting
            # downstream volume at zero rate. A nonpositive order cannot do so.
            if exposure > -1/exponent and order <= 0:
                raise ValueError("PFR volume exceeds the physical depletion limit")
            x, u = ctx.one, ctx.zero
        else:
            log_inverse = ctx.log1p(exponent*exposure)/exponent if exponent else exposure
            x, u = -ctx.expm1(-log_inverse), ctx.exp(-log_inverse)

    if not math.isfinite(volume):
        raise ValueError("PFR volume is not representable")
    time = ctx.mpf(volume)/flow
    space_time = float(time)
    if not math.isfinite(space_time) or (time and space_time == 0):
        raise ValueError("PFR space time is not representable")
    conversion, remaining = float(x), float(u)
    if (x and conversion == 0) or (u and remaining == 0) or (exposure and volume == 0):
        raise ValueError("PFR state is not representable in conversion, remaining fraction and volume")
    returned_x, returned_u = ctx.mpf(conversion), ctx.mpf(remaining)
    reported_exposure = ctx.mpf(volume)*coefficient
    if remaining == 0:
        required = -1/exponent
        closure = (ctx.zero if not sizing and order > 0 and reported_exposure >= required
                   else relative(required, reported_exposure))
    else:
        log_inverse = -ctx.log1p(-returned_x) if conversion <= remaining else -ctx.log(returned_u)
        closure = relative(dose(log_inverse), reported_exposure)
    temperature = (ctx.mpf(inlet_t) + ctx.mpf(heat_rise)*x if x <= u else
                   (ctx.mpf(inlet_t) + heat_rise) - ctx.mpf(heat_rise)*u)
    exit_t = float(temperature)
    if not math.isfinite(exit_t) or exit_t <= 0:
        raise ValueError("PFR exit temperature is not representable")
    residual = float(max(closure, relative(x, returned_x), relative(u, returned_u),
                         relative(time, ctx.mpf(space_time)), relative(temperature, ctx.mpf(exit_t))))
    return {"domain": "reactor_kinetics", "reactor_type": "pfr", "conversion_X": conversion,
            "unreacted_fraction": remaining, "volume_m3": volume, "space_time_s": space_time,
            "exit_T_K": exit_t, "k_exit": rate, "residual": residual, "verified": residual < 1e-9}


def solve_reactor_kinetics(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for CSTR, PFR, N-CSTRs, Arrhenius, and adiabatic reactor design."""
    reactor_type = str(params.get("reactor_type", "pfr")).strip().lower()
    order_n = float(params.get("order", 1.0))
    c_a0 = float(params.get("C_A0", 1.0))
    v0 = float(params.get("v0", 1.0))
    f_a0 = float(params.get("F_A0", c_a0 * v0))
    eps_a = float(params.get("epsilon_A", 0.0))
    t0_k = float(params.get("T0", 300.0))
    t_op = float(params.get("T", t0_k))
    delta_t_ad = float(params.get("delta_T_adiabatic", 0.0))
    r_gas = 8.31446261815324

    def k_at_temp(temp_k: float) -> float:
        if "A_preexp" in params and "E_a" in params:
            return float(params["A_preexp"]) * math.exp(-float(params["E_a"]) / (r_gas * temp_k))
        if "k_ref" in params and "T_ref" in params and "E_a" in params:
            return float(params["k_ref"]) * math.exp(
                -(float(params["E_a"]) / r_gas) * (1.0 / temp_k - 1.0 / float(params["T_ref"]))
            )
        return float(params["k"])

    gas_phase = bool(params.get("gas_phase", abs(eps_a) > 1e-15))

    if reactor_type == "pfr":
        if not all(map(math.isfinite, (c_a0, v0, f_a0, order_n, t0_k, t_op, eps_a, delta_t_ad))):
            raise ValueError("PFR feed, temperature, order and expansion must be finite")
        heat_rise = delta_t_ad if abs(delta_t_ad) > 1e-15 else 0.
        inlet_t = t0_k if heat_rise else t_op
        if min(c_a0, v0, f_a0, t0_k, inlet_t) <= 0:
            raise ValueError("PFR requires positive feed and inlet temperature")
        if params.get("conversion") is not None:
            target_x = float(params["conversion"])
            if not 0 <= target_x <= 1:
                raise ValueError("PFR conversion must lie in [0, 1]")
            if target_x == 1 and order_n >= 1:
                raise ValueError("complete PFR conversion requires infinite volume for order >= 1")
            target = Fraction(target_x)
            if Fraction(inlet_t) + Fraction(heat_rise)*target <= 0 or 1 + Fraction(eps_a)*target <= 0:
                raise ValueError("PFR exit temperature and volumetric flow must remain positive")
        elif params.get("volume") is None:
            raise ValueError("PFR requires a conversion or a volume")
        elif not math.isfinite(float(params["volume"])) or float(params["volume"]) < 0:
            raise ValueError("PFR volume must be finite and nonnegative")
        if "A_preexp" in params and "E_a" in params:
            prefactor, reference, activation = float(params["A_preexp"]), 1., float(params["E_a"])
        elif all(key in params for key in ("k_ref", "T_ref", "E_a")):
            prefactor, reference, activation = map(float, (params["k_ref"], params["T_ref"], params["E_a"]))
        else:
            prefactor, reference, activation = float(params["k"]), 1., 0.
        if min(prefactor, reference) <= 0 or not all(map(math.isfinite, (prefactor, reference, activation))):
            raise ValueError("PFR kinetic rate and reference temperature must be finite and positive")
        arrhenius = activation != 0
        if eps_a == 0 and (not heat_rise or (not gas_phase and not arrhenius)):
            concentration = Fraction(c_a0)
            if gas_phase:
                concentration *= Fraction(t0_k)/Fraction(inlet_t)
            return _constant_rate_pfr(params, concentration, f_a0, v0, order_n,
                                      k_at_temp(inlet_t), inlet_t, heat_rise)

    elif reactor_type not in ("cstr", "cstr_series"):
        raise ValueError(f"Unsupported reactor_type: {reactor_type}")

    def rate_minus_ra(x_conv: float) -> float:
        temp_x = (t0_k + delta_t_ad * x_conv) if abs(delta_t_ad) > 1e-15 else t_op
        k_val = k_at_temp(temp_x)
        vol_factor = (1.0 + eps_a * x_conv) * (temp_x / t0_k if gas_phase else 1.0)
        ca_x = c_a0 * max(1.0 - x_conv, 0.0) / vol_factor
        return k_val * (ca_x ** order_n)

    def _cstr_series_exit_x(tau_stage: float, n_stages: int) -> float:
        if tau_stage <= 0.0:
            return 0.0
        x_prev = 0.0
        for _ in range(n_stages):
            fn = lambda x_cur, xp=x_prev: (f_a0 * (x_cur - xp) / max(rate_minus_ra(x_cur), 1e-30)) - (tau_stage * v0)
            gap = 0.5 * (1.0 - x_prev)
            x_hi = 1.0 - gap
            for _ in range(60):
                if fn(x_hi) > 0.0:
                    break
                gap *= 0.5
                x_hi = 1.0 - gap
            x_prev = float(optimize.brentq(fn, x_prev, x_hi, xtol=1e-13, rtol=1e-13))
        return x_prev

    def volume_for_conversion(x_conv: float) -> float:
        if x_conv <= 0.0:
            return 0.0
        if reactor_type == "cstr":
            return f_a0 * x_conv / rate_minus_ra(x_conv)
        if reactor_type == "pfr":
            val, _ = integrate.quad(lambda xi: f_a0 / rate_minus_ra(xi), 0.0, x_conv, epsabs=1e-12, epsrel=1e-12, limit=200)
            return float(val)
        if reactor_type == "cstr_series":
            n_stages = int(params.get("n_stages", 2))
            fn_tau = lambda tau_stage: _cstr_series_exit_x(tau_stage, n_stages) - x_conv
            tau_hi = max(1e-6, (c_a0 * x_conv / max(rate_minus_ra(0.5 * x_conv), 1e-12)))
            for _ in range(40):
                if fn_tau(tau_hi) > 0.0:
                    break
                tau_hi *= 2.0
            tau_s = float(optimize.brentq(fn_tau, 0.0, tau_hi, xtol=1e-13, rtol=1e-13))
            return n_stages * tau_s * v0
        raise ValueError(f"Unsupported reactor_type: {reactor_type}")

    cstr_branches = None
    if "conversion" in params and params["conversion"] is not None:
        x_out = float(params["conversion"])
        vol_m3 = volume_for_conversion(x_out)
        residual = abs(volume_for_conversion(x_out) - vol_m3) / max(vol_m3, 1e-15)
    elif "volume" in params and params["volume"] is not None:
        vol_m3 = float(params["volume"])
        if reactor_type in ("cstr", "cstr_series"):
            if not math.isfinite(v0) or v0 <= 0 or not math.isfinite(vol_m3) or vol_m3 < 0:
                raise ValueError("CSTR inlet flow must be positive and vessel volume finite and nonnegative")
            count = params.get("n_stages", 2) if reactor_type == "cstr_series" else 1
            n_stages = int(count)
            if isinstance(count, bool) or n_stages < 1 or Fraction(count) != n_stages:
                raise ValueError("n_stages must be a positive integer")
            stage_volume = Fraction(vol_m3) / n_stages

            def stage_state(conversion, unreacted, inlet_x):
                # Check the returned coordinates without subnormal product
                # underflow or an absolute denominator floor. Use the smaller
                # coordinate as the accurate chart and recover its complement.
                with localcontext() as check_context:
                    small = min(conversion, unreacted)
                    digits = -math.floor(math.log10(small)) if small > 0 else 0
                    check_context.prec = min(400, 80+max(0, digits))
                    D = Decimal.from_float
                    xd, ud = D(conversion), D(unreacted)
                    if conversion <= unreacted:
                        ud = 1-xd
                    else:
                        xd = 1-ud
                    td = D(t0_k)+D(delta_t_ad)*xd if abs(delta_t_ad) > 1e-15 else D(t_op)
                    if "A_preexp" in params and "E_a" in params:
                        kd = D(float(params["A_preexp"]))*((-D(float(params["E_a"]))/(D(r_gas)*td)).exp())
                    elif "k_ref" in params and "T_ref" in params and "E_a" in params:
                        kd = D(float(params["k_ref"]))*((-D(float(params["E_a"]))/D(r_gas)*
                              (1/td-1/D(float(params["T_ref"])))).exp())
                    else:
                        kd = D(float(params["k"]))
                    cd = D(c_a0)*ud/(1+D(eps_a)*xd)
                    if gas_phase:
                        cd *= D(t0_k)/td
                    feed_consumed = D(f_a0)*(xd - Decimal(inlet_x.numerator)/Decimal(inlet_x.denominator))
                    reactor_consumed = Decimal(stage_volume.numerator)/Decimal(stage_volume.denominator)*kd*(cd**D(order_n) if order_n else Decimal(1))
                    scale = max(abs(feed_consumed), abs(reactor_consumed))
                    error = float(abs(feed_consumed-reactor_consumed)/scale) if scale else 0.0
                    temperature = float(td)
                    energy_error = float(abs(D(temperature)-td)/td)
                error = max(error, energy_error)
                return {"conversion_X": conversion, "unreacted_fraction": unreacted,
                        "exit_T_K": temperature, "k_exit": k_at_temp(temperature),
                        "residual": error, "verified": error < 1e-9}

            # Each tank can have several steady outlets even for one inlet.
            # Keep complete trajectories: an inlet with no downstream root
            # does not eliminate a different, physically realizable path.
            inputs = (params, stage_volume, c_a0, f_a0, t0_k, t_op, delta_t_ad, eps_a, order_n, gas_phase)
            branch = params.get("steady_state_branch")
            ambiguity = ("multiple physical CSTR steady states; set steady_state_branch to 'all' "
                         "or an ascending zero-based branch index")
            paths = [([], (Fraction(0), Fraction(0)), Fraction(0))]
            for stage in range(n_stages):
                following = []
                for history, inlet_bounds, inlet_report in paths:
                    for root in _cstr_stage_roots(inputs, inlet_bounds, stage + 1 < n_stages):
                        conversion, remaining = root.coordinates()
                        row = stage_state(conversion, remaining, inlet_report)
                        outlet = Fraction(conversion) if conversion <= remaining else 1-Fraction(remaining)
                        following.append((history+[row], (root.lower, root.upper), outlet))
                paths = following
                # A positive-order tank always has a continuation between its
                # inlet and depletion. These distinct paths therefore persist.
                if branch is None and order_n > 0 and len(paths) > 1:
                    raise ValueError(ambiguity)
                if not paths:
                    raise ValueError("no physical CSTR steady state for the supplied volume")
            cstr_branches = []
            for history, _, _ in sorted(paths, key=lambda path: path[1]):
                row = history[-1].copy()
                if reactor_type == "cstr_series":
                    row.update(stages=history, residual=max(r["residual"] for r in history),
                               verified=all(r["verified"] for r in history))
                cstr_branches.append(row)
            if branch == "all":
                return {"domain": "reactor_kinetics", "reactor_type": reactor_type, "volume_m3": vol_m3,
                        "space_time_s": vol_m3 / v0, "steady_states": cstr_branches,
                        "steady_state_count": len(cstr_branches), "steady_state_branch": "all",
                        "residual": max(row["residual"] for row in cstr_branches),
                        "verified": all(row["verified"] for row in cstr_branches)}
            if branch is None:
                if len(cstr_branches) != 1:
                    raise ValueError(ambiguity)
                branch = 0
            if type(branch) is not int or not 0 <= branch < len(cstr_branches):
                raise ValueError("steady_state_branch must be 'all' or a valid ascending zero-based branch index")
            x_out = cstr_branches[branch]["conversion_X"]
        elif vol_m3 <= 0.0:
            x_out = 0.0
        else:
            fn_v = lambda x: volume_for_conversion(x) - vol_m3
            limit = Fraction(1)
            if heat_rise < 0:
                limit = min(limit, -Fraction(inlet_t)/Fraction(heat_rise))
            if eps_a < 0:
                limit = min(limit, -1/Fraction(eps_a))
            # Search only through the actual physical domain. A temperature
            # or expansion singularity beyond the outlet is not traversed.
            largest = float(limit)
            closed = (limit == 1 and order_n == 0 and 1+Fraction(eps_a) > 0 and
                      Fraction(inlet_t)+Fraction(heat_rise) > 0)
            if not closed and Fraction(largest) >= limit:
                largest = math.nextafter(largest, 0.)
            gap = float(limit)
            for _ in range(60):
                gap *= .5
                x_hi = min(float(limit)-gap, largest)
                if fn_v(x_hi) >= 0.0:
                    break
                if x_hi == largest:
                    raise ValueError("no representable PFR outlet in the physical integration interval")
            x_out = float(optimize.brentq(fn_v, 0.0, x_hi, xtol=1e-13, rtol=1e-13))
        if cstr_branches is not None:
            residual = cstr_branches[branch]["residual"]
        else:
            replay = volume_for_conversion(x_out)
            scale = max(abs(replay), abs(vol_m3))
            residual = abs(replay-vol_m3)/scale if scale else 0.0
    else:
        raise ValueError("reactor_kinetics requires either 'conversion' or 'volume'")

    tau_total = vol_m3 / v0
    temp_exit = (t0_k + delta_t_ad * x_out) if abs(delta_t_ad) > 1e-15 else t_op
    result = {
        "domain": "reactor_kinetics",
        "reactor_type": reactor_type,
        "conversion_X": x_out,
        "volume_m3": vol_m3,
        "space_time_s": tau_total,
        "exit_T_K": temp_exit,
        "k_exit": k_at_temp(temp_exit),
        "residual": residual,
        "verified": residual < 1e-9,
    }
    if cstr_branches is not None:
        result.update(steady_states=cstr_branches, steady_state_count=len(cstr_branches),
                      steady_state_branch=branch)
        result["exit_T_K"] = cstr_branches[branch]["exit_T_K"]
        result["k_exit"] = cstr_branches[branch]["k_exit"]
        result["unreacted_fraction"] = cstr_branches[branch]["unreacted_fraction"]
        result["residual"] = cstr_branches[branch]["residual"]
        result["verified"] = cstr_branches[branch]["verified"]
        if reactor_type == "cstr_series":
            result["stages"] = cstr_branches[branch]["stages"]
    return result


def infer_wave6_domain(text_or_domain: str) -> Optional[str]:
    s = (text_or_domain or "").strip().lower()
    if s in WAVE6_DOMAINS:
        return s
    if any(k in s for k in ("isentropic", "normal shock", "fanno", "rayleigh", "mach", "helmholtz", "acoustic impedance")):
        return "compressible_acoustics"
    if any(k in s for k in ("smith chart", "transmission line", "vswr", "single-stub", "rlc transient", "underdamped rlc")):
        return "rf_transmission_rlc"
    if any(k in s for k in ("mohr", "von mises", "lame", "pressure vessel", "hoop stress", "column buckling", "euler-johnson", "slenderness")):
        return "solid_mohr_vessel_buckling"
    if any(k in s for k in ("cstr", "pfr", "plug flow reactor", "arrhenius", "space time", "levenspiel", "adiabatic reactor")):
        return "reactor_kinetics"
    return None


def dispatch_wave6_verifier(domain_or_spec: Any, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(domain_or_spec, Mapping):
        spec = domain_or_spec
        domain = infer_wave6_domain(str(spec.get("domain") or spec.get("kind") or spec.get("prompt") or ""))
        eff_params = dict(spec.get("params") or spec)
    else:
        domain = infer_wave6_domain(str(domain_or_spec))
        eff_params = dict(params or {})

    if domain == "compressible_acoustics":
        return solve_compressible_gas_acoustics(eff_params)
    if domain == "rf_transmission_rlc":
        return solve_transmission_line_rlc(eff_params)
    if domain == "solid_mohr_vessel_buckling":
        return solve_mohr_vessel_buckling(eff_params)
    if domain == "reactor_kinetics":
        return solve_reactor_kinetics(eff_params)
    return {"domain": domain, "verified": False, "error": f"unrecognized_wave6_domain:{domain_or_spec}"}


from hamon_wave7_engineering_verifiers import WAVE7_DOMAINS, infer_wave7_domain

WAVE1_5_CERTIFIED_FAST_PATH_DOMAINS = frozenset(
    {
        "dxf_takeoff",
        "dxf_curved_takeoff",
        "mep_routing_clearance_3d",
        "step_brep_mass_inertia",
        "geotech_bearing_earthwork",
        "thermo_hx_pinch_conservation",
        "spice_ac_kcl_tellegen",
        "pde",
        "convective_fin_1d",
        "control",
        "transfer_function",
        "state_space",
        "psychrometrics",
        "cooling_coil",
        "fem",
        "truss_2d",
        "euler_bernoulli_beam_1d",
    }
) | frozenset(WAVE7_DOMAINS)


def prune_deterministic_fast_path(
    tier: str,
    est_cost_usd: float,
    *,
    presolve_verified: bool,
    domain: str = "",
    residual: float = 0.0,
    has_unparsed_visual: bool = False,
    light_cost_usd: float = 1.9331,
) -> Dict[str, Any]:
    norm_domain = (domain or "").strip().lower()
    domain_ok = (
        not norm_domain
        or norm_domain in DETERMINISTIC_FAST_PATH_DOMAINS
        or norm_domain in WAVE1_5_CERTIFIED_FAST_PATH_DOMAINS
        or infer_wave6_domain(norm_domain) is not None
        or infer_wave7_domain(norm_domain) is not None
    )
    eligible = (
        bool(presolve_verified)
        and math.isfinite(float(residual))
        and abs(float(residual)) < 1e-6
        and not has_unparsed_visual
        and domain_ok
    )
    if eligible and tier in ("low", "medium", "high", "heavy", "frontier"):
        pruned_cost = min(float(est_cost_usd), float(light_cost_usd))
        return {
            "tier": "light",
            "original_tier": tier,
            "est_cost_usd": round(pruned_cost, 6),
            "cost_saved_usd": round(max(0.0, float(est_cost_usd) - pruned_cost), 6),
            "fast_path_pruned": True,
        }
    return {
        "tier": tier,
        "original_tier": tier,
        "est_cost_usd": round(float(est_cost_usd), 6),
        "cost_saved_usd": 0.0,
        "fast_path_pruned": False,
    }
