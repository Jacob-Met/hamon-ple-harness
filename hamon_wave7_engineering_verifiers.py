"""PLE Wave-7 First-Principles Deterministic Engineering Verifiers.

Covers four residual-certified engineering domains with dual-solver cross-checks
and First/Second-Law & dynamical conservation invariants:
  1. thermo_cycles: Regenerative/irreversible Brayton, Rankine, Otto, Diesel &
     vapor-compression refrigeration/heat-pump cycles (Carnot & Gouy-Stodola guards).
  2. exchangers_radiation: Forward/inverse epsilon-NTU & LMTD dual heat exchangers
     plus N-surface diffuse-gray Oppenheim radiative enclosure networks.
  3. lagrangian_orbital: 3D Euler rigid-body kinetics (RK4 vs DOP853, kinetic energy
     & Casimir |H|^2 invariants), N-DOF conservative Lagrangian eigenmodes (Cholesky
     vs generalized eigensolver), and Keplerian vis-viva / Hohmann / plane-change orbits.
  4. electrochem_separations: Nernst-Gibbs-Faraday-Overpotential electrochemistry and
     Fenske / dual-solver Underwood / stage-by-stage McCabe-Thiele binary distillation.
"""
from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from _scipy_compat import HAS_SCIPY, integrate, linalg, optimize

SIGMA_SB = 5.670374419e-8
FARADAY_C_MOL = 96485.33212
R_GAS_UNIV = 8.314462618

WAVE7_DOMAINS = (
    "thermo_cycles",
    "exchangers_radiation",
    "lagrangian_orbital",
    "electrochem_separations",
)


def _rankine_cycle(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Energy balance and phase information for the first turbine outlet."""
    result = {"domain": "thermo_cycles", "subkind": "rankine", "verified": False}

    def number(name, default=None):
        value = params.get(name, default)
        if isinstance(value, bool) or value is None:
            raise ValueError(f"invalid_or_missing_{name}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"nonfinite_{name}")
        return Fraction(str(parsed))

    def bracketed(value, liquid, latent):
        if value < liquid:
            return "below_saturated_liquid", None
        if value > liquid + latent:
            return "above_saturated_vapor", None
        fraction = (value - liquid) / latent
        if fraction == 0:
            return "saturated_liquid", fraction
        if fraction == 1:
            return "saturated_vapor", fraction
        return "liquid_vapor_mixture", fraction

    try:
        p1, p2, v1 = number("P1"), number("P2"), number("v1")
        h1, h3 = number("h1"), number("h3")
        eta_p, eta_t = number("eta_p", 1), number("eta_t", 1)
        if p2 <= p1 or v1 <= 0 or not (0 < eta_p <= 1 and 0 < eta_t <= 1):
            raise ValueError("invalid_rankine_parameters")
        declared_x = number("x4s") if "x4s" in params else None
        if declared_x is not None and not 0 <= declared_x <= 1:
            raise ValueError("invalid_declared_turbine_quality")

        have_h = "h_f" in params or "h_fg" in params
        if have_h:
            hf, hfg = number("h_f"), number("h_fg")
            if hfg <= 0:
                raise ValueError("nonpositive_saturation_enthalpy_span")
        entropy_state = None
        if "s_f" in params or "s_fg" in params:
            sf, sfg = number("s_f"), number("s_fg")
            if sfg <= 0:
                raise ValueError("nonpositive_saturation_entropy_span")
            if "s3" in params:
                entropy_state = bracketed(number("s3"), sf, sfg)
        if "h4s" in params:
            h4s = number("h4s")
            source = "direct_enthalpy"
        else:
            if not have_h or entropy_state is None:
                raise ValueError("saturation_enthalpy_and_entropy_brackets_required")
            fraction = entropy_state[1]
            if fraction is None:
                raise ValueError("isentropic_entropy_outside_saturation_bracket")
            h4s = hf + fraction * hfg
            source = "saturation_entropy"

        phase_s, xs = bracketed(h4s, hf, hfg) if have_h else (entropy_state or ("unknown", None))
        if have_h and entropy_state is not None:
            entropy_phase, entropy_x = entropy_state
            if xs is None or entropy_x is None:
                if entropy_phase != phase_s:
                    raise ValueError("contradictory_isentropic_phase")
            elif not math.isclose(float(xs), float(entropy_x), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("contradictory_isentropic_quality")
        can_check_quality = have_h or entropy_state is not None
        if declared_x is not None and can_check_quality:
            if xs is None or not math.isclose(float(xs), float(declared_x), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("declared_quality_inconsistent_with_saturation_bracket")
        if h4s > h3:
            raise ValueError("negative_first_turbine_work")
        wp = v1 * (p2 - p1) / eta_p
        h2 = h1 + wp
        heat_boiler = h3 - h2
        if heat_boiler < 0:
            raise ValueError("negative_boiler_heat_addition")
        h4 = h3 - eta_t * (h3 - h4s)
        phase, x = bracketed(h4, hf, hfg) if have_h else ("unknown", None)
        if not have_h and eta_t == 1:
            phase, x = phase_s, xs
        wt = h3 - h4
        heat_reheat = 0
        exhaust = h4
        if any(k in params for k in ("h5", "h6s", "eta_t2")):
            if "h5" not in params or "h6s" not in params:
                raise ValueError("incomplete_rankine_reheat_stage")
            h5, h6s = number("h5"), number("h6s")
            eta_t2 = number("eta_t2", float(eta_t))
            if not 0 < eta_t2 <= 1:
                raise ValueError("invalid_reheat_turbine_efficiency")
            if h5 < h4 or h6s > h5:
                raise ValueError("invalid_reheat_component_direction")
            exhaust = h5 - eta_t2 * (h5 - h6s)
            heat_reheat = h5 - h4
            wt += h5 - exhaust
        qin = heat_boiler + heat_reheat
        qout = exhaust - h1
        net = wt - wp
        if net <= 0 or qin <= 0:
            raise ValueError("non_positive_rankine_work")
        if qout <= 0:
            raise ValueError("nonpositive_rankine_heat_rejection")
        thermal = net / qin
        bounded = "T_cond" in params or "T_boiler_max" in params
        carnot_ok = True
        if bounded:
            cold, hot = number("T_cond"), number("T_boiler_max")
            if not 0 < cold < hot:
                raise ValueError("invalid_rankine_reservoirs")
            carnot_ok = float(thermal - (1-cold/hot)) <= 1e-11
        qin_f, qout_f, work_f = map(float, (qin, qout, net))
        if min(qin_f, qout_f, work_f) <= 0:
            raise ValueError("unrepresentable_rankine_energy")
        residual = abs((qin_f - qout_f) - work_f) / max(abs(qin_f), 1e-12)
        result.update(
            w_pump_J_kg=float(wp), w_turbine_J_kg=float(wt), w_net_J_kg=work_f,
            q_in_J_kg=qin_f, q_out_J_kg=qout_f, eta_thermal=float(thermal),
            isentropic_outlet_enthalpy_J_kg=float(h4s), first_turbine_outlet_enthalpy_J_kg=float(h4),
            isentropic_outlet_source=source, isentropic_outlet_phase=phase_s,
            x4s_quality=None if xs is None else float(xs),
            first_turbine_outlet_phase=phase, x4_quality=None if x is None else float(x),
            carnot_checked=bounded, residual=residual, verified=carnot_ok and residual < 1e-10,
        )
        if declared_x is not None:
            result["declared_x4s_quality"] = float(declared_x)
            result["declared_x4s_quality_checked"] = can_check_quality
    except (TypeError, ValueError, OverflowError) as exc:
        result["error"] = str(exc)
    return result


def _vapor_compression_cycle(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Throttle isenthalpically and balance the compressor from its suction datum."""
    result = {"domain": "thermo_cycles", "subkind": "vapor_compression", "verified": False}

    def quantity(name, default=None):
        raw = params.get(name, default)
        if isinstance(raw, bool):
            raise ValueError
        return Fraction.from_float(float(raw))

    try:
        h1, h2s, h3 = (quantity(key) for key in ("h1", "h2s", "h3"))
        eta, flow = quantity("eta_c", 1), quantity("m_dot", 1)
        if h2s <= h1 or h1 <= h3 or not (0 < eta <= 1) or flow <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return dict(result, error="invalid_vcr_enthalpies")

    # These extensive differences are invariant to a common enthalpy datum.
    isentropic_rise = h2s - h1
    work = isentropic_rise / eta
    cold = h1 - h3
    hot = cold + work
    cop_r = cold / work
    cop_hp = hot / work
    carnot_checked = "T_L" in params and "T_H" in params
    carnot_ok = True
    if carnot_checked:
        try:
            low, high = quantity("T_L"), quantity("T_H")
            if low <= 0 or high <= low:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            return dict(result, error="invalid_vcr_reservoirs")
        carnot_ok = cop_r <= low / (high - low) + Fraction.from_float(1e-9)

    try:
        h2 = h1 + work
        primary = float(h2)
        correction = float(h2 - Fraction(primary))
        exact_values = {"q_evap": cold, "q_cond": hot, "w_comp": work,
                        "Q_evap_W": flow * cold, "W_comp_W": flow * work,
                        "COP_R": cop_r, "COP_HP": cop_hp}
        wire = {name: float(value) for name, value in exact_values.items()}
    except (OverflowError, ValueError):
        return dict(result, error="vcr_result_outside_numeric_range")

    # Replay the emitted values in the original equations. The correction lets
    # receivers reconstruct a sub-ULP discharge rise even at a large datum.
    decoded = {name: Fraction(value) for name, value in wire.items()}
    rise = Fraction(primary) + Fraction(correction) - h1
    residuals = [abs(decoded[key] - value) / value for key, value in exact_values.items()]
    residuals.extend((
        abs(eta * rise - isentropic_rise) / isentropic_rise,
        abs(decoded["w_comp"] - rise) / work,
        abs(decoded["q_evap"] + decoded["w_comp"] - decoded["q_cond"]) / hot,
        abs((decoded["COP_HP"] - decoded["COP_R"]) - 1),
    ))
    residual = max(residuals)
    result.update(wire, h2=primary, h2_correction=correction, h4=float(h3),
                  carnot_checked=carnot_checked, residual=float(residual),
                  verified=bool(carnot_ok and residual < 1e-11))
    return result


def solve_thermodynamic_cycle(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(params.get("subkind") or params.get("cycle_type") or "brayton").strip().lower()

    if subkind == "brayton":
        t1 = float(params["T1"])
        t3 = float(params["T3"])
        r_p = float(params["r_p"])
        gamma = float(params.get("gamma", 1.4))
        cp = float(params.get("cp", 1005.0))
        eta_c = float(params.get("eta_c", 1.0))
        eta_t = float(params.get("eta_t", 1.0))
        eps_reg = float(params.get("regenerator_eff", 0.0))

        if t1 <= 0.0 or t3 <= t1 or r_p <= 1.0 or gamma <= 1.0 or cp <= 0.0:
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_brayton_parameters"}
        if not (0.0 < eta_c <= 1.0 and 0.0 < eta_t <= 1.0 and 0.0 <= eps_reg <= 1.0):
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_brayton_efficiencies"}

        r_spec = cp * (gamma - 1.0) / gamma
        tau_s = r_p ** ((gamma - 1.0) / gamma)
        t2s = t1 * tau_s
        t2 = t1 + (t2s - t1) / eta_c
        t4s = t3 / tau_s
        t4 = t3 - eta_t * (t3 - t4s)

        if eps_reg > 0.0 and t4 <= t2:
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "regenerator_temperature_crossover"}

        t2x = t2 + eps_reg * (t4 - t2) if eps_reg > 0.0 else t2
        t4y = t4 - (t2x - t2)

        w_c = cp * (t2 - t1)
        w_t = cp * (t3 - t4)
        w_net = w_t - w_c
        if w_net <= 0.0:
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "non_positive_net_work"}

        q_in = cp * (t3 - t2x)
        q_out = cp * (t4y - t1)
        eta_th = w_net / q_in
        bwr = w_c / w_t
        eta_carnot = 1.0 - t1 / t3

        ds_c = cp * math.log(t2 / t1) - r_spec * math.log(r_p)
        ds_t = cp * math.log(t4 / t3) + r_spec * math.log(r_p)
        ds_reg = (cp * math.log(t2x / t2) + cp * math.log(t4y / t4)) if eps_reg > 0.0 else 0.0
        s_gen = ds_c + ds_t + ds_reg

        first_law_res = float(abs((q_in - q_out) - w_net) / max(abs(q_in), 1e-12))
        second_law_ok = bool((ds_c >= -1e-11) and (ds_t >= -1e-11) and (ds_reg >= -1e-11) and (eta_th <= eta_carnot + 1e-11))
        return {
            "domain": "thermo_cycles",
            "subkind": subkind,
            "T2_K": float(t2),
            "T2x_K": float(t2x),
            "T4_K": float(t4),
            "T4y_K": float(t4y),
            "w_compressor_J_kg": float(w_c),
            "w_turbine_J_kg": float(w_t),
            "w_net_J_kg": float(w_net),
            "q_in_J_kg": float(q_in),
            "q_out_J_kg": float(q_out),
            "back_work_ratio": float(bwr),
            "eta_thermal": float(eta_th),
            "eta_carnot": float(eta_carnot),
            "s_gen_J_kgK": float(s_gen),
            "residual": first_law_res,
            "verified": bool(second_law_ok and first_law_res < 1e-10),
        }

    if subkind == "rankine":
        return _rankine_cycle(params)

    if subkind in ("otto", "diesel"):
        r = float(params["r"])
        t1 = float(params.get("T1", 300.0))
        gamma = float(params.get("gamma", 1.4))
        cv = float(params.get("cv", 718.0))
        p1 = float(params.get("P1", 101325.0))
        if r <= 1.0 or t1 <= 0.0 or gamma <= 1.0 or cv <= 0.0 or p1 <= 0.0:
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_ic_cycle_parameters"}
        cp = gamma * cv
        r_spec = cp - cv
        t2 = t1 * (r ** (gamma - 1.0))

        if subkind == "otto":
            if "T3" in params:
                t3 = float(params["T3"])
                q_in = cv * (t3 - t2)
            else:
                q_in = float(params["q_in"])
                t3 = t2 + q_in / cv
            if t3 <= t2:
                return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "non_positive_heat_addition"}
            t4 = t3 / (r ** (gamma - 1.0))
            q_out = cv * (t4 - T1_val if (T1_val := t1) else t1)
            eta_exact = 1.0 - (r ** (1.0 - gamma))
        else:
            r_c = float(params["r_c"])
            if r_c <= 1.0 or r_c >= r:
                return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_cutoff_ratio"}
            t3 = t2 * r_c
            t4 = t3 * ((r_c / r) ** (gamma - 1.0))
            q_in = cp * (t3 - t2)
            q_out = cv * (t4 - t1)
            eta_exact = 1.0 - (1.0 / (r ** (gamma - 1.0))) * ((r_c ** gamma - 1.0) / (gamma * (r_c - 1.0)))

        w_net = q_in - q_out
        eta_th = w_net / q_in
        v1 = r_spec * t1 / p1
        v2 = v1 / r
        mep_pa = w_net / (v1 - v2)
        residual = float(abs(eta_th - eta_exact))
        return {
            "domain": "thermo_cycles",
            "subkind": subkind,
            "T2_K": float(t2),
            "T3_K": float(t3),
            "T4_K": float(t4),
            "w_net_J_kg": float(w_net),
            "q_in_J_kg": float(q_in),
            "q_out_J_kg": float(q_out),
            "eta_thermal": float(eta_th),
            "eta_closed_form": float(eta_exact),
            "MEP_Pa": float(mep_pa),
            "residual": residual,
            "verified": bool(residual < 1e-11),
        }

    if subkind == "vapor_compression":
        return _vapor_compression_cycle(params)

    return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": f"unsupported_cycle:{subkind}"}


def _unmixed_crossflow_small(ntu: float, cr: float) -> float:
    """Evaluate E[min(Pois(a), Pois(b))]/b for 0 < a <= 1 and b=a*Cr."""
    b = ntu * cr
    mass_a = ntu * math.exp(-ntu)
    mass_b_per_b = math.exp(-b)
    masses = []
    # At a,b <= 1, the first omitted mass divided by its mean is <= 1/25!.
    for count in range(1, 25):
        masses.append((mass_a, mass_b_per_b))
        mass_a *= ntu / (count + 1)
        mass_b_per_b *= b / (count + 1)
    tail_a = tail_b_per_b = result = 0.0
    for mass_a, mass_b_per_b in reversed(masses):
        tail_a += mass_a
        tail_b_per_b += mass_b_per_b
        result += tail_a * tail_b_per_b
    return result


@lru_cache(maxsize=1)
def _unmixed_quadrature():
    return np.polynomial.legendre.leggauss(32)


def _unmixed_crossflow_deficit(ntu: float, cr: float) -> float:
    """Unequilibrated outlet fraction from the ideal both-unmixed flow equations."""
    if ntu <= 0.0:
        return 1.0
    if cr == 0.0:
        return math.exp(-ntu)
    if math.isnan(ntu) or math.isnan(cr):
        return math.nan
    if math.isinf(ntu):
        return 0.0
    ratio = math.sqrt(cr)
    # This difference retains precision when the capacity rates almost coincide.
    difference = (1.0 - cr) / (1.0 + ratio)
    root_ntu = math.sqrt(ntu)
    root_z = root_ntu * math.sqrt(2.0 * ratio)
    gap = root_ntu * difference
    if gap >= 28.0:
        return 0.0  # exp(-gap**2) is below the binary64 subnormal range.

    # The ideal unmixed-flow PDE gives eps = E[min(X,Y)]/b for independent
    # X~Pois(NTU), Y~Pois(b=Cr*NTU). The remaining fraction is E[(Y-X)+]/b.
    # Integrating the Skellam Bessel series by parts yields a positive angular
    # integral with denominator (1-r)^2 + 4*r*sin(theta/2)^2. Its exponential
    # is exp(-2*(sqrt(z)*sin(theta/2))^2); stop after its exponent reaches -80.
    right = 2.0 * math.asin(min(1.0, math.sqrt(40.0) / root_z))
    nodes, weights = _unmixed_quadrature()
    edge = min(right, difference / math.sqrt(ratio)) if difference else right
    left = total = 0.0
    while left < right:
        half_width = (edge - left) * 0.5
        angle = left + (nodes + 1.0) * half_width
        sine_half = np.sin(angle * 0.5)
        binding = np.hypot(difference, 2.0 * math.sqrt(ratio) * sine_half)
        integrand = np.exp(-2.0 * (root_z * sine_half)**2) * (np.sin(angle) / binding)**2
        total += half_width * float(np.dot(weights, integrand))
        left, edge = edge, min(right, 8.0 * edge)
    return (2.0 / math.pi) * math.exp(-gap * gap) * total


def _unmixed_crossflow_ntu(eps: float, cr: float):
    """Locate the inverse for the supplied effectiveness using the numerical model."""
    if eps < 1e-16:
        return eps, (eps, math.nextafter(eps, math.inf))

    def below_target(log_ntu):
        ntu = math.exp(log_ntu)
        if eps > 0.5:
            # Invert the directly evaluated deficit, which stays resolved when
            # effectiveness rounds to 1 at physically different NTU values.
            return _unmixed_crossflow_deficit(ntu, cr) > 1.0 - eps
        return _hx_eps_from_ntu(ntu, cr, "crossflow_unmixed") < eps

    lower = math.log(0.5 * eps)  # Effectiveness cannot exceed NTU.
    upper = 1.0
    while below_target(upper):
        upper *= 2.0
    for _ in range(96):
        middle = (lower + upper) * 0.5
        if middle == lower or middle == upper:
            break
        if below_target(middle):
            lower = middle
        else:
            upper = middle
    lo, hi = math.exp(lower), math.exp(upper)
    return lo + (hi - lo) * 0.5, (lo, hi)


def _hx_eps_from_ntu(ntu: float, cr: float, arrangement: str) -> float:
    if ntu <= 0.0:
        return 0.0
    if arrangement == "crossflow_unmixed":
        if ntu <= 1.0:
            return _unmixed_crossflow_small(ntu, cr)
        return 1.0 - _unmixed_crossflow_deficit(ntu, cr)
    if arrangement == "counterflow":
        delta = 1.0 - cr
        if delta == 0.0:
            return ntu / (1.0 + ntu)
        drop = -math.expm1(-ntu * delta)
        return drop / (delta + cr * drop)
    if cr == 0.0 or arrangement in ("condenser", "boiler"):
        return -math.expm1(-ntu)
    if arrangement == "parallel":
        return -math.expm1(-ntu * (1.0 + cr)) / (1.0 + cr)
    if arrangement == "shell_tube_1_2":
        factor = math.hypot(1.0, cr)
        t = math.tanh(ntu * factor / 2.0)
        return 2.0 * t / (factor + (1.0 + cr) * t)
    if arrangement == "crossflow_cmax_mixed":
        return -math.expm1(cr * math.expm1(-ntu)) / cr
    if arrangement == "crossflow_cmin_mixed":
        return -math.expm1(math.expm1(-cr * ntu) / cr)
    raise ValueError(f"Unsupported heat exchanger arrangement: {arrangement}")


def _hx_inverse(arrangement, field, measurement, hot, cold, c_hot, c_cold, c_min, eps_approx):
    """Invert a measured duty, temperature or effectiveness without first rounding away its pinch gap."""
    if eps_approx < 1e-15:
        return eps_approx  # Every supported effectiveness curve is NTU + O(NTU²) at zero.
    c_max = max(c_hot, c_cold)
    phase = arrangement in ("condenser", "boiler")
    with localcontext() as ctx:
        # Mixed-flow logarithms can involve the product of a tiny capacity ratio
        # and the effectiveness. Input values retain their exact binary64 meaning.
        ctx.prec = 80 if phase else 80 + max(0, Decimal(c_max).adjusted() - Decimal(c_min).adjusted())
        ch, cc, cm = map(Decimal, (c_hot, c_cold, c_min))
        hi, ci, reading = map(Decimal, (hot, cold, measurement))
        qmax = cm * (hi - ci)
        if field == "Q":
            e = reading / qmax
        elif field == "T_h_out":
            e = ch * (hi - reading) / qmax
        elif field == "T_c_out":
            e = cc * (reading - ci) / qmax
        else:
            e = reading
        r = Decimal(0) if phase else cm / Decimal(c_max)
        if not 0 < e < 1:
            raise ValueError("Observation cannot identify a finite positive UA")
        if r == 0:
            return float(-(1 - e).ln())
        if arrangement == "counterflow":
            odds = e / (1 - e)
            return float(odds if r == 1 else (1 + odds * (1 - r)).ln() / (1 - r))
        if arrangement == "parallel":
            remainder, factor = 1 - e * (1 + r), 1 + r
        elif arrangement == "shell_tube_1_2":
            factor = (1 + r*r).sqrt()
            remainder = (2 - e * (1 + r + factor)) / (2 - e * (1 + r - factor))
        elif arrangement == "crossflow_cmax_mixed":
            remainder, factor = 1 + (1 - r*e).ln() / r, 1
        else:  # Cmin mixed, Cmax unmixed.
            remainder, factor = 1 + r * (1 - e).ln(), r
        if remainder <= 0:
            raise ValueError("Observation exceeds the arrangement's finite-NTU domain")
        return float(-remainder.ln() / factor)


def _isothermal_radiation_components(factors, temperatures, adiabatic):
    """Return each anchored component's common prescribed temperature, when unique."""
    uniform = {}
    unseen = set(range(len(temperatures)))
    while unseen:
        first = unseen.pop()
        component, pending, fixed = {first}, [first], set()
        while pending:
            i = pending.pop()
            if i not in adiabatic:
                fixed.add(float(temperatures[i]))
            for j in np.flatnonzero((factors[i] != 0) | (factors[:, i] != 0)):
                if j in unseen:
                    unseen.remove(j)
                    component.add(j)
                    pending.append(j)
        if len(fixed) == 1:
            uniform.update(dict.fromkeys(component, fixed.pop()))
    return uniform


def _resolved_radiation_balance(factors, areas, emissivities, temperatures, adiabatic):
    """Resolve a weakly anchored diffuse-gray network before rounding its heat flux."""
    from sympy import Matrix, Rational, SparseMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    n = len(areas)
    matrix = SparseMatrix.zeros(n, n)
    rhs = Matrix.zeros(n, 1)
    exchange = []
    emit = []
    blackbody = []
    for i in range(n):
        # On a closed surface J-G = sum_j Fij (Ji-Jj). Self-view contributes
        # exactly zero. This retains the small emissive term in its own right.
        e = Rational(0) if i in adiabatic else Rational(float(emissivities[i]))
        eb = Rational(0) if i in adiabatic else Rational(SIGMA_SB) * Rational(float(temperatures[i]))**4
        links = [(j, Rational(float(factors[i, j]))) for j in np.flatnonzero(factors[i]) if j != i]
        for j, weight in links:
            matrix[i, int(j)] = -(1 - e) * weight
        matrix[i, i] = e + (1 - e) * sum(w for _, w in links)
        rhs[i] = e * eb
        exchange.append(links)
        emit.append(e)
        blackbody.append(eb)

    system = matrix.to_DM().to_field()
    try:
        numerator, divisor = system.solve_den(rhs.to_DM().convert_to(system.domain))
    except DMNonInvertibleMatrixError:
        return {"verified": False, "error": "singular_radiation_balance"}
    radiosity = list(numerator.to_Matrix() / system.domain.to_sympy(divisor))
    heat = []
    surface_errors = []
    for i, ji in enumerate(radiosity):
        outward = sum(w * (ji - radiosity[j]) for j, w in exchange[i])
        heat.append(Rational(float(areas[i])) * outward)
        # Replay both the gray-surface law and the outward flux against the
        # submitted view factors, including the independently supplied self-view.
        irradiation = Rational(float(factors[i, i])) * ji + sum(w * radiosity[j] for j, w in exchange[i])
        scale = max(abs(ji), abs(irradiation))
        mismatch = ji - emit[i] * blackbody[i] - (1 - emit[i]) * irradiation
        if scale:
            surface_errors.extend([abs(mismatch) / scale, abs(ji - irradiation - outward) / scale])
    balance = abs(sum(heat)) / max(sum(map(abs, heat)), 1)
    published_j = [float(j) for j in radiosity]
    published_q = [float(q) for q in heat]
    if not all(math.isfinite(v) for v in published_j + published_q) or any(j < 0 for j in radiosity):
        return {"verified": False, "error": "unrepresentable_radiation_balance"}
    def rounding_error(published, exact):
        return max((abs(Rational(v) - x) / abs(x) for v, x in zip(published, exact) if x), default=0)

    q_rounding = rounding_error(published_q, heat)
    j_rounding = rounding_error(published_j, radiosity)
    residual = float(max([balance, q_rounding, j_rounding, *surface_errors]))
    return {
        "radiosity_W_m2": published_j,
        "Q_net_W": published_q,
        "T_surface_K": [(float((radiosity[i] / Rational(SIGMA_SB)) ** Rational(1, 4))
                         if i in adiabatic else float(temperatures[i])) for i in range(n)],
        "residual": residual,
        "verified": residual < 1e-9,
        **({"error": "unrepresentable_radiation_heat"} if q_rounding >= 1e-9 else {}),
        **({"error": "unrepresentable_radiation_radiosity"} if j_rounding >= 1e-9 else {}),
    }


def solve_exchanger_or_radiation(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(params.get("subkind") or ("radiation_network" if "F" in params else "heat_exchanger")).strip().lower()

    if subkind == "heat_exchanger":
        arr = str(params.get("flow_arrangement", "counterflow")).strip().lower()
        arrangements = {"counterflow", "parallel", "shell_tube_1_2", "crossflow_unmixed",
                        "crossflow_cmax_mixed", "crossflow_cmin_mixed", "condenser", "boiler"}
        if arr not in arrangements:
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "unknown_flow_arrangement"}
        observation_keys = ("UA", "effectiveness", "Q", "T_h_out", "T_c_out")
        try:
            observed = {key: float(params[key]) for key in observation_keys if key in params}
        except (TypeError, ValueError, OverflowError):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "invalid_observation"}
        if any(isinstance(params[key], bool) or not math.isfinite(value) for key, value in observed.items()):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "invalid_observation"}
        t_hi = float(params["T_h_in"])
        t_ci = float(params["T_c_in"])
        if not (math.isfinite(t_hi) and math.isfinite(t_ci) and t_hi > t_ci):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "invalid_inlet_temperatures"}

        c_h = float(params.get("C_h", 0.0))
        c_c = float(params.get("C_c", 0.0))
        if arr == "condenser":
            sensible = [c_c]
        elif arr == "boiler":
            sensible = [c_h]
        else:
            sensible = [c_h, c_c]
        if (not all(math.isfinite(c) and c > 0.0 for c in sensible)
                or not (c_h >= 0.0 and c_c >= 0.0)):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "non_positive_heat_capacity_rates"}
        c_min = min(sensible)
        cr = c_min / max(sensible) if len(sensible) == 2 else 0.0

        crossflow_inverse = None
        q_max = c_min * (t_hi - t_ci)
        if not math.isfinite(q_max) or q_max <= 0.0:
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "unrepresentable_duty_scale"}
        # Admissibility applies to every supplied measurement, including those
        # that are close to a computed answer only because it is near a pinch.
        limit = 1.0
        if arr == "parallel":
            limit = 1.0 / (1.0 + cr)
        elif arr == "shell_tube_1_2":
            limit = 2.0 / (1.0 + cr + math.hypot(1.0, cr))
        elif arr == "crossflow_cmax_mixed" and cr > 0.0:
            limit = -math.expm1(-cr) / cr
        elif arr == "crossflow_cmin_mixed" and cr > 0.0:
            limit = -math.expm1(-1.0 / cr)
        max_duty = limit * q_max
        hot_low = t_hi if arr == "condenser" else max(t_ci, t_hi - max_duty / c_h)
        cold_high = t_ci if arr == "boiler" else min(t_hi, t_ci + max_duty / c_c)
        bounds = {"effectiveness": (0., limit), "Q": (0., max_duty),
                  "T_h_out": (hot_low, t_hi), "T_c_out": (t_ci, cold_high)}
        for key, (low, high) in bounds.items():
            if key not in observed:
                continue
            # The finite-capacity ceiling can accumulate rounding, whereas zero
            # duty and the unit effectiveness bound are exact physical limits.
            low_error = 2 * math.ulp(low) if key.startswith("T_") else 0.0
            high_error = 2 * math.ulp(high) if key.startswith("T_") or limit < 1.0 else 0.0
            if not low - low_error <= observed[key] <= high + high_error:
                return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False,
                        "error": "second_law_crossover_or_unreachable_effectiveness",
                        "invalid_observation": key}
        if "UA" in observed:
            ua = observed["UA"]
            if ua < 0.0:
                return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "negative_UA"}
            ntu = ua / c_min
            eps = _hx_eps_from_ntu(ntu, cr, arr)
        else:
            # A saturated readout or an isothermal outlet alone cannot identify UA.
            # Let another supplied measurement size the exchanger if it is informative.
            driver = next((key for key in ("effectiveness", "Q", "T_h_out", "T_c_out")
                           if key in observed and bounds[key][0] < observed[key] < bounds[key][1]), None)
            # With a nonzero inlet span, exact zero duty uniquely gives UA=0.
            # Prefer an interior measurement if a sensible outlet merely rounds
            # to its inlet. A phase-changing outlet stays fixed at every UA.
            zero_values = {"effectiveness": 0.0, "Q": 0.0}
            if arr != "condenser":
                zero_values["T_h_out"] = t_hi
            if arr != "boiler":
                zero_values["T_c_out"] = t_ci
            if driver is None:
                driver = next((key for key, zero in zero_values.items()
                               if observed.get(key) == zero), None)
            if driver is None:
                return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False,
                        "error": "second_law_crossover_or_unreachable_effectiveness" if observed else "missing_rating_or_sizing_spec"}
            reading = observed[driver]
            if driver == "effectiveness":
                eps_target = reading
            elif driver == "Q":
                eps_target = reading / q_max
            elif driver == "T_h_out":
                eps_target = c_h * (t_hi - reading) / q_max
            else:
                eps_target = c_c * (reading - t_ci) / q_max
            if reading == zero_values[driver]:
                ntu = 0.0
            elif arr == "crossflow_unmixed":
                if not 0.0 < eps_target < 1.0:
                    return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False,
                            "error": "unresolved_effectiveness"}
                ntu, crossflow_inverse = _unmixed_crossflow_ntu(eps_target, cr)
            else:
                try:
                    ntu = _hx_inverse(arr, driver, reading, t_hi, t_ci, c_h, c_c, c_min, eps_target)
                except ValueError:
                    return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False,
                            "error": "second_law_crossover_or_unreachable_effectiveness"}
            ua = ntu * c_min
            eps = _hx_eps_from_ntu(ntu, cr, arr)

        if not math.isfinite(ntu) or not math.isfinite(ua):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "unrepresentable_rating"}
        q_actual = eps * q_max
        t_ho = t_hi if arr == "condenser" else (t_hi - q_actual / c_h)
        t_co = t_ci if arr == "boiler" else (t_ci + q_actual / c_c)

        second_law_ok = bool((t_ho >= t_ci - 1e-9) and (t_co <= t_hi + 1e-9))
        if arr == "parallel":
            second_law_ok = bool(second_law_ok and (t_ho >= t_co - 1e-9))

        def stream_residual(capacity, inlet_temp, outlet_temp):
            rounding = 2 * max(math.ulp(inlet_temp), math.ulp(outlet_temp))
            discrepancy = abs(capacity * abs(inlet_temp - outlet_temp) - q_actual)
            return max(0., discrepancy - capacity * rounding) / max(q_actual, 1e-12)

        eb_h = 0.0 if arr == "condenser" else stream_residual(c_h, t_hi, t_ho)
        eb_c = 0.0 if arr == "boiler" else stream_residual(c_c, t_ci, t_co)
        lmtd_res = 0.0
        lmtd_val = None
        lmtd_bounds = None
        if arr in ("counterflow", "parallel", "condenser", "boiler") and q_actual > 0.0:
            # LMTD is monotone in both ends. The rounded outlet temperatures
            # independently bound Q/UA even when a true pinch gap rounds to zero.
            ends = ((t_hi, t_ci), (t_ho, t_co)) if arr == "parallel" else ((t_hi, t_co), (t_ho, t_ci))
            span = t_hi - t_ci
            bounds_at_ends = []
            for hot, cold in ends:
                gap = hot - cold
                rounding = 2 * (math.ulp(hot) + math.ulp(cold))
                bounds_at_ends.append((max(0., gap - rounding), min(span, gap + rounding)))

            def log_mean(x, y):
                low, high = sorted((x, y))
                if low <= 0.0:
                    return 0.0
                if low == high:
                    return low
                diff = high - low
                ratio_log = math.log1p(diff / low) if diff < low else math.log(high) - math.log(low)
                return diff / ratio_log

            lower, upper = (log_mean(*pair) for pair in zip(*bounds_at_ends))
            lmtd_bounds = [lower, upper]
            lmtd_val = q_actual / ua
            if lmtd_val == 0.0:
                return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "unrepresentable_mean_temperature_difference"}
            lmtd_res = max(lower - lmtd_val, lmtd_val - upper, 0.) / lmtd_val

        # A supplied rating can determine the state while other observations test it.
        # Compare each outlet on its actual stream temperature-change scale.
        expected = {"UA": ua, "effectiveness": eps, "Q": q_actual, "T_h_out": t_ho, "T_c_out": t_co}
        inlet = {"T_h_out": t_hi, "T_c_out": t_ci}
        observation_residuals = {}
        for key, measured in observed.items():
            value = expected[key]
            reference = inlet.get(key, 0.0)
            scale = max(abs(value - reference), abs(measured - reference))
            roundoff = 2 * max(math.ulp(value), math.ulp(measured)) if key in inlet else 0.0
            difference = max(0.0, abs(value - measured) - roundoff)
            observation_residuals[key] = difference / scale if scale else 0.0
        residual = float(max(eb_h, eb_c, lmtd_res, *observation_residuals.values()))
        return {
            "domain": "exchangers_radiation",
            "subkind": subkind,
            "flow_arrangement": arr,
            "C_r": float(cr),
            "NTU": float(ntu),
            "UA_W_K": float(ua),
            "effectiveness": float(eps),
            "Q_W": float(q_actual),
            "T_h_out_K": float(t_ho),
            "T_c_out_K": float(t_co),
            "LMTD_K": lmtd_val,
            **({"LMTD_bounds_K": lmtd_bounds} if lmtd_bounds is not None else {}),
            "observation_residuals": observation_residuals,
            **({"effectiveness_model": "ideal_single_pass_both_unmixed"} if arr == "crossflow_unmixed" else {}),
            **({"NTU_numerical_bracket": list(crossflow_inverse)} if crossflow_inverse is not None else {}),
            "residual": residual,
            "verified": bool(second_law_ok and residual < 1e-9),
        }

    if subkind == "radiation_network":
        f_mat = np.asarray(params["F"], dtype=float)
        areas = np.asarray(params["areas"], dtype=float)
        eps_vec = np.asarray(params["emissivities"], dtype=float)
        t_raw = list(params["T_K"])
        n = f_mat.shape[0]
        if f_mat.shape != (n, n) or areas.shape != (n,) or eps_vec.shape != (n,) or len(t_raw) != n:
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "dimension_mismatch"}
        if (not all(np.isfinite(v).all() for v in (areas, eps_vec, f_mat)) or np.any(areas <= 0.0)
                or np.any(eps_vec <= 0.0) or np.any(eps_vec > 1.0) or np.any(f_mat < -1e-12)):
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "invalid_surface_properties"}

        row_sum_err = float(np.max(np.abs(np.sum(f_mat, axis=1) - 1.0)))
        af = areas[:, None] * f_mat
        # Reciprocity is local to a pair of surfaces. Express the mismatch
        # as a view-factor error on the smaller surface of each pair.
        pair_area = np.minimum(areas[:, None], areas[None, :])
        recip_err = float(np.max(np.abs(af - af.T) / pair_area))
        if row_sum_err > 1e-5 or recip_err > 1e-5:
            return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "view_factor_enclosure_or_reciprocity_violation"}

        adiabatic = set(int(i) for i in params.get("adiabatic_indices", []))
        for idx, tv in enumerate(t_raw):
            if tv is None:
                adiabatic.add(idx)

        # In a closed enclosure, an adiabatic surface sets J_i = sum(F_ij J_j).
        # A component containing only such equations leaves its common J free.
        # Walk backwards from prescribed emitters along the actual positive factors.
        anchored = np.array([i not in adiabatic and eps_vec[i] > 0.0 for i in range(n)])
        pending = np.flatnonzero(anchored).tolist()
        while pending:
            j = pending.pop()
            predecessors = np.flatnonzero((f_mat[:, j] > 0.0) & ~anchored)
            anchored[predecessors] = True
            pending.extend(predecessors.tolist())
        if not np.all(anchored):
            return {
                "domain": "exchangers_radiation",
                "subkind": subkind,
                "verified": False,
                "error": "underdetermined_closed_radiation_enclosure",
                "surfaces_without_temperature_path": np.flatnonzero(~anchored).tolist(),
            }

        for i in range(n):
            if i not in adiabatic and not 0 < float(t_raw[i]) < math.inf:
                return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "non_positive_surface_temperature"}

        def resolved():
            return {
                "domain": "exchangers_radiation", "subkind": subkind,
                "row_sum_error": row_sum_err, "reciprocity_error": recip_err,
                **_resolved_radiation_balance(f_mat, areas, eps_vec, t_raw, adiabatic),
            }

        uniform = _isothermal_radiation_components(f_mat, t_raw, adiabatic)
        blackbody = {t: Fraction(SIGMA_SB) * Fraction(t)**4
                     for t in {float(t_raw[i]) for i in range(n) if i not in adiabatic}}
        m_sys = np.zeros((n, n), dtype=float)
        b_rhs = np.zeros(n, dtype=float)
        try:
            for i in range(n):
                if i in uniform:
                    # Every surface has the same exact radiosity and zero heat
                    # transfer in an isothermal, closed, anchored component.
                    m_sys[i, i] = 1.
                    b_rhs[i] = float(blackbody[uniform[i]])
                    continue
                emission = 0. if i in adiabatic else eps_vec[i]
                reflectivity = 1. - emission
                m_sys[i] = -reflectivity * f_mat[i]
                outward = math.fsum(f_mat[i, j] for j in range(n) if j != i)
                m_sys[i, i] = emission + reflectivity * outward
                if i not in adiabatic:
                    b_rhs[i] = float(Fraction(float(emission)) * blackbody[float(t_raw[i])])
        except OverflowError:
            return resolved()

        # An almost reflecting surface can lose its emissive term in 1-eps.
        # Weak exchange with an adiabatic component can similarly lose its
        # physical anchor in I-F. Replay when binary64 cannot resolve the
        # surface balance at the verification tolerance.
        resolution = np.finfo(float).eps / 1e-9
        active_eps = [eps_vec[i] for i in range(n) if i not in adiabatic and i not in uniform]
        if ((active_eps and min(active_eps) <= resolution)
                or (adiabatic.difference(uniform) and np.linalg.cond(m_sys) * resolution >= 1.)):
            return resolved()

        j_primary = np.linalg.solve(m_sys, b_rhs)
        if not np.isfinite(j_primary).all() or np.any(j_primary < np.finfo(float).tiny):
            return resolved()
        lu, piv = linalg.lu_factor(m_sys)
        j_dual = linalg.lu_solve((lu, piv), b_rhs)

        q_net = np.zeros(n, dtype=float)
        surf_res_err = 0.0
        t_effective: List[float] = []
        for i in range(n):
            outward = float(np.sum(f_mat[i] * (j_primary[i] - j_primary)))
            q_encl = float(areas[i]) * outward
            if not math.isfinite(q_encl):
                return resolved()
            q_net[i] = 0. if i in adiabatic or i in uniform else q_encl
            irradiation = float(f_mat[i] @ j_primary)
            level = max(abs(j_primary[i]), abs(irradiation))
            if level:
                surf_res_err = max(surf_res_err, abs(j_primary[i] - irradiation - outward) / level)
            if i in adiabatic and level:
                surf_res_err = max(surf_res_err, abs(outward) / level)
            if i not in adiabatic and i not in uniform:
                exchange_level = math.fsum(abs(f_mat[i, j]) * (abs(j_primary[i]) + abs(j_primary[j]))
                                           for j in range(n) if j != i)
                if (exchange_level and abs(outward) <= resolution * exchange_level) or abs(q_encl) < np.finfo(float).tiny:
                    return resolved()
            if i in uniform:
                t_effective.append(uniform[i])
            elif i in adiabatic:
                t_effective.append(float((max(float(j_primary[i]), 0.0) / SIGMA_SB) ** 0.25))
            else:
                ti = float(t_raw[i])
                t_effective.append(ti)
                if eps_vec[i] < 1.0 - 1e-12:
                    q_surf = float((areas[i] * eps_vec[i] / (1.0 - eps_vec[i])) * (float(blackbody[ti]) - j_primary[i]))
                    surf_res_err = max(surf_res_err, abs(q_surf - q_encl) / max(abs(q_encl), 1.0))

        enclosure_balance_err = abs(float(np.sum(q_net))) / max(float(np.sum(np.abs(q_net))), 1.0)
        dual_err = float(np.linalg.norm(j_primary - j_dual)) / max(float(np.linalg.norm(j_primary)), 1.0)
        residual = float(max(enclosure_balance_err, surf_res_err, dual_err))
        return {
            "domain": "exchangers_radiation",
            "subkind": subkind,
            "radiosity_W_m2": [float(x) for x in j_primary],
            "Q_net_W": [float(x) for x in q_net],
            "T_surface_K": t_effective,
            "row_sum_error": row_sum_err,
            "reciprocity_error": recip_err,
            "residual": residual,
            "verified": bool(residual < 1e-9),
        }

    return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": f"unsupported_exchanger_radiation:{subkind}"}


def _dyadic_matrix(matrix):
    """Return integer entries and one denominator for the supplied binary64 matrix."""
    ratios = [float(value).as_integer_ratio() for value in matrix.flat]
    denominator = max(d for _, d in ratios)
    entries = [a * (denominator // d) for a, d in ratios]
    return np.asarray(entries, dtype=object).reshape(matrix.shape), denominator


def _inertia_counts(matrix):
    """Count negative and zero directions by exact symmetric fraction-free elimination."""
    block = matrix.copy()
    previous = 1
    negative = 0
    while len(block):
        pivot_index = next((i for i, v in enumerate(block.diagonal()) if v), None)
        if pivot_index is None:
            mixed = np.argwhere(block != 0)
            if not len(mixed):
                return negative, len(block)
            # An off-diagonal pair with two zero diagonals supplies a nonzero
            # quadratic pivot after the unimodular basis change e_i <- e_i+e_j.
            i, j = mixed[0]
            block[i, :] += block[j, :]
            block[:, i] += block[:, j]
            pivot_index = i
        if pivot_index:
            block[[0, pivot_index], :] = block[[pivot_index, 0], :]
            block[:, [0, pivot_index]] = block[:, [pivot_index, 0]]
        pivot = block[0, 0]
        negative += (pivot < 0) != (previous < 0)
        column = block[1:, :1]
        block = (pivot * block[1:, 1:] - column * column.T) // previous
        previous = pivot
    return negative, 0


def _energy_bounds(matrix, vectors):
    """Diagonal lower/upper forms from an exact congruence and absolute row sums."""
    reduced = vectors.T @ (matrix @ vectors)
    center = reduced.diagonal()
    radius = np.abs(reduced).sum(axis=1) - np.abs(center)
    return center - radius, center + radius


def _modal_spectrum_certificate(mass, stiffness, vectors, estimates):
    """Certify sorted omega² intervals of this binary64 symmetric pencil."""
    k, kd = _dyadic_matrix(stiffness)
    m, md = _dyadic_matrix(mass)
    x, _ = _dyadic_matrix(vectors)
    ml, mu = _energy_bounds(m, x)
    have_basis = all(a > 0 for a in ml)
    if not have_basis and _inertia_counts(m) != (0, 0):
        raise ValueError("non_spd_mass_matrix")
    kl, ku = _energy_bounds(k, x)
    if have_basis and all(a > 0 for a in kl):
        nullity = 0
    else:
        negative, nullity = _inertia_counts(k)
        if negative:
            raise ValueError("unstable_negative_stiffness_eigenvalue")

    # For t >= 0, diag(kl)-t*diag(mu) <= X'(K-tM)X <=
    # diag(ku)-t*diag(ml), with common dyadic scales restored. Counts of
    # negative directions bracket each ordered generalized eigenvalue.
    lower, upper = [], []
    if have_basis:
        lower = sorted(Fraction(a * md, b * kd) for a, b in zip(kl, mu))
        upper = sorted(Fraction(a * md, b * kd) for a, b in zip(ku, ml))

    def counts_at(t):
        return _inertia_counts(k * (md * t.denominator) - m * (kd * t.numerator))

    tolerance = Fraction(1, 10**10)
    bounds = [(Fraction(0), Fraction(0))] * nullity
    for i in range(nullity, len(estimates)):
        if not math.isfinite(estimates[i]) or estimates[i] <= 0:
            raise ArithmeticError("unresolved_positive_modal_frequency")
        value = Fraction(float(estimates[i]))
        low, high = value * (1 - tolerance), value * (1 + tolerance)
        if have_basis and lower[i] >= low and upper[i] <= high:
            bounds.append((min(lower[i], value), max(upper[i], value)))
            continue
        # Only ambiguous intervals pay for the two exact shifted-pencil tests.
        # Equality at either endpoint is allowed (including repeated roots).
        n_below, _ = counts_at(low)
        n_above, n_equal = counts_at(high)
        if n_below > i or n_above + n_equal <= i:
            raise ArithmeticError("unresolved_positive_modal_frequency")
        bounds.append((low, high))
    return nullity, bounds


def _binary_modal_solution(m, k):
    try:
        chol = np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        # A floating failure alone does not establish invalid mass.
        if _inertia_counts(_dyadic_matrix(m)[0]) != (0, 0):
            raise ValueError("non_spd_mass_matrix")
        raise ArithmeticError("unresolved_mass_factorization")

    values, vectors = linalg.eigh(k, m)
    if not (np.isfinite(values).all() and np.isfinite(vectors).all()):
        raise ArithmeticError("unresolved_modal_eigensystem")
    nullity, bounds = _modal_spectrum_certificate(m, k, vectors, values)
    values[:nullity] = 0.0
    left = np.linalg.solve(chol, k)
    whitened = np.linalg.solve(chol, left.T).T
    dual = np.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    positive, dual_positive = values[nullity:], dual[nullity:]
    if np.any(dual_positive <= 0) or not np.isfinite(dual).all():
        raise ArithmeticError("unresolved_dual_modal_frequency")
    dual_error = float(np.max(np.abs(positive-dual_positive) /
                              np.maximum(positive, dual_positive), initial=0.0))
    denominator = (np.linalg.norm(k, np.inf) + np.abs(values)*np.linalg.norm(m, np.inf)) * np.max(np.abs(vectors), axis=0)
    error = np.max(np.abs(k @ vectors - (m @ vectors)*values), axis=0)
    eigen_error = float(np.max(np.divide(error, denominator, out=np.zeros_like(error), where=denominator > 0)))
    if len(positive):
        # Mass-normalized residual against each physical mode's own scale.
        weighted = np.linalg.solve(chol, k @ vectors - (m @ vectors)*values)
        mode_scale = np.where(values > 0, values, positive[0])
        eigen_error = max(eigen_error, float(np.max(np.linalg.norm(weighted / mode_scale, axis=0))))
    orth_error = float(np.linalg.norm(vectors.T @ m @ vectors - np.eye(len(m)), ord=np.inf))
    residual = max(eigen_error, orth_error, dual_error)
    if not math.isfinite(residual) or residual >= 1e-10:
        raise ArithmeticError("unresolved_modal_solver_residual")

    return values, nullity, bounds, residual


def _refined_modal_solution(m, k):
    """Recover weak positive modes and retain the exact shifted-pencil certificate."""
    from mpmath import mp

    if _inertia_counts(_dyadic_matrix(m)[0]) != (0, 0):
        raise ValueError("non_spd_mass_matrix")
    negative, nullity = _inertia_counts(_dyadic_matrix(k)[0])
    if negative:
        raise ValueError("unstable_negative_stiffness_eigenvalue")
    ctx = mp.clone()
    n = len(m)
    previous = None
    for bits in (128, 256, 512, 1024, 2048, 4096):
        ctx.prec = bits
        mass, stiffness = ctx.matrix(m.tolist()), ctx.matrix(k.tolist())
        try:
            inv_l = ctx.cholesky(mass)**-1
        except ValueError:
            continue
        c = inv_l * stiffness * inv_l.T
        eigen, orth = ctx.eigsy((c+c.T)/2)
        gap = eigen[nullity] if nullity < n else ctx.one
        if gap <= 0:
            continue
        condition = ctx.mnorm(mass) * ctx.mnorm(inv_l)**2
        if ctx.eps * n**2 * condition * ctx.mnorm(inv_l)**2 * ctx.mnorm(stiffness) >= gap * ctx.mpf('1e-18'):
            continue
        values = [0.]*nullity + [float(eigen[i]) for i in range(nullity, n)]
        if not all(math.isfinite(v) and v > 0 for v in values[nullity:]):
            raise ArithmeticError("unrepresentable_modal_frequency")
        vectors = inv_l.T * orth
        residual = ctx.mnorm(vectors.T * mass * vectors - ctx.eye(n))
        for i, value in enumerate(values):
            mode = vectors[:, i]
            error = inv_l * (stiffness * mode - ctx.mpf(value) * (mass * mode))
            residual = max(residual, ctx.norm(error) / (value if value else gap))
        if previous == values and residual < ctx.mpf('1e-10'):
            values = np.asarray(values)
            basis = np.array(vectors.tolist(), dtype=float)
            count, bounds = _modal_spectrum_certificate(m, k, basis, values)
            return values, count, bounds, float(residual)
        previous = values
    raise ArithmeticError("unresolved_modal_solver_residual")


def _lagrangian_modes(params):
    result = {"domain": "lagrangian_orbital", "subkind": "lagrangian_modes", "verified": False}
    try:
        m = np.asarray(params["M"], dtype=float)
        k = np.asarray(params["K"], dtype=float)
        if m.ndim != 2 or not len(m) or m.shape[0] != m.shape[1] or k.shape != m.shape:
            raise ValueError("matrix_shape_mismatch")
        if not (np.isfinite(m).all() and np.isfinite(k).all()):
            raise ValueError("nonfinite_mass_or_stiffness")
        if not (np.array_equal(m, m.T) and np.array_equal(k, k.T)):
            raise ValueError("non_symmetric_mass_or_stiffness")

        # Normalize the independent mass and stiffness units before either solve.
        def scaled(a):
            exp = math.frexp(float(np.abs(a).max()))[1]
            exp -= exp % 2  # Even powers preserve the Cholesky square-root scaling.
            normalized = np.ldexp(a, -exp)
            if not np.array_equal(np.ldexp(normalized, exp), a):
                raise ArithmeticError("unresolved_modal_matrix_dynamic_range")
            return normalized, exp

        m, me = scaled(m)
        k, ke = scaled(k)
        try:
            values, nullity, bounds, residual = _binary_modal_solution(m, k)
        except (ArithmeticError, np.linalg.LinAlgError):
            values, nullity, bounds, residual = _refined_modal_solution(m, k)

        unit = Fraction(2) ** (ke-me)
        def outward(q, upward):
            number = float(q)
            if not math.isfinite(number):
                raise ArithmeticError("unrepresentable_modal_frequency")
            represented = Fraction(number)
            if (upward and represented < q) or (not upward and represented > q):
                number = math.nextafter(number, math.inf if upward else -math.inf)
            if not math.isfinite(number):
                raise ArithmeticError("unrepresentable_modal_frequency")
            return number

        physical = [float(Fraction(float(v))*unit) for v in values]
        intervals = [[outward(lo*unit, False), outward(hi*unit, True)] for lo, hi in bounds]
        # Restoring physical units can consume most of a subnormal
        # eigenvalue's significant bits after the normalized solve passed.
        for i in range(nullity, len(physical)):
            value = physical[i]
            if value >= np.finfo(float).tiny:
                continue
            if value <= 0:
                raise ArithmeticError("unrepresentable_modal_frequency")
            published = Fraction(value)
            target = published / unit
            k_int, kd = _dyadic_matrix(k)
            m_int, md = _dyadic_matrix(m)
            below, equal = _inertia_counts(k_int*(md*target.denominator) - m_int*(kd*target.numerator))
            if below <= i < below+equal:
                intervals[i] = [value, value]
                continue
            relative = max(abs(bound*unit-published)/published for bound in bounds[i])
            if relative >= Fraction(1, 10**10):
                raise ArithmeticError("unrepresentable_modal_frequency")
            residual = max(residual, float(relative))
        if any(low <= 0.0 or val <= 0.0 for (low, _), val in zip(intervals[nullity:], physical[nullity:])):
            raise ArithmeticError("unrepresentable_modal_frequency")
        omega = [math.sqrt(val) for val in physical]
        result.update(eigenvalues_omega2=physical, eigenvalue_intervals_omega2=intervals,
                      rigid_mode_count=nullity, natural_frequencies_rad_s=omega,
                      natural_frequencies_Hz=[w/(2*math.pi) for w in omega],
                      residual=residual, verified=True)
    except (KeyError, TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result["error"] = str(exc)
    return result


def _orbital_transfer(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Two circular orbits joined by a Hohmann ellipse, with tilt at the higher apsis."""
    from mpmath import mp

    result = {"domain": "lagrangian_orbital", "subkind": "orbital_transfer", "verified": False}
    ctx = mp.clone()

    def number(key, default=None):
        value = params.get(key, default)
        if value is None or isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number

    try:
        mu, r1, r2 = (number(key) for key in ("mu", "r1", "r2"))
        if min(mu, r1, r2) <= 0:
            raise ValueError
        degrees = "delta_inc_rad" not in params
        angle = number("delta_inc_deg" if degrees else "delta_inc_rad", 0.0)
        if degrees:
            angle = math.remainder(angle, 360.0)
        isp = number("isp_s") if "isp_s" in params else None
        g0 = number("g0", 9.80665) if isp is not None else None
        if isp is not None and min(isp, g0) <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return dict(result, error="invalid_orbital_parameters")

    # Direct energy replay at periapsis loses log2(ra/rp) bits for long ellipses.
    ctx.prec = 220 + abs(math.frexp(r1)[1] - math.frexp(r2)[1])
    mu, r1, r2, angle = map(ctx.mpf, (mu, r1, r2, angle))
    radii = r1 + r2
    axis = radii / 2
    eccentricity = abs(r2 - r1) / radii
    if degrees:
        sine, cosine = ctx.sinpi(angle / 360), ctx.cospi(angle / 360)
    else:
        sine, cosine = ctx.sin(angle / 2), ctx.cos(angle / 2)

    momentum = ctx.sqrt(2 * mu * r1 * r2 / radii)
    energy = -mu / radii
    residuals = []

    def relative_error(reported, reference):
        if reference == 0:
            return 0 if reported == 0 else ctx.inf
        return abs(reported - reference) / abs(reference)

    def publish(value):
        wire = float(value)
        if not math.isfinite(wire):
            raise ArithmeticError
        residuals.append(relative_error(ctx.mpf(wire), value))
        return wire

    def impulse(radius, other, tilted):
        circular = ctx.sqrt(mu / radius)
        factor = ctx.sqrt(2 * other / radii)
        transfer = circular * factor
        # Rationalize (factor - 1) so distinct adjacent radii retain both burns.
        difference = circular * (other - radius) / (radii * (factor + 1))
        if tilted:
            # Components along the bisector of the velocity vectors and its normal.
            magnitude = ctx.hypot(difference * cosine, (transfer + circular) * sine)
        else:
            magnitude = abs(difference)
        wire = publish(magnitude)

        # Replay the published burn in a frame aligned with the transfer velocity.
        # Angular momentum gives an independent transfer speed at this apsis.
        speed = momentum / radius
        axial = speed - circular if r1 != r2 else ctx.zero
        normal = ctx.zero
        if tilted:
            axial += 2 * circular * sine**2
            normal = 2 * circular * sine * cosine
        residuals.extend((relative_error(ctx.mpf(wire), ctx.hypot(axial, normal)),
                          relative_error(radius * transfer, momentum),
                          relative_error(transfer**2 / 2 - mu / radius, energy)))
        return wire, magnitude

    try:
        first, d1 = impulse(r1, r2, r1 > r2)
        second, d2 = impulse(r2, r1, r2 >= r1)
        total = d1 + d2
        wire_eccentricity = publish(eccentricity)
        if wire_eccentricity >= 1:
            raise ArithmeticError
        result.update({
            "a_transfer_m": publish(axis), "eccentricity_transfer": wire_eccentricity,
            "dv1_m_s": first, "dv2_m_s": second, "dv_total_m_s": publish(total),
            "transfer_time_s": publish(ctx.pi * ctx.sqrt(axis**3 / mu)),
            "propellant_mass_fraction": (publish(-ctx.expm1(-total / (ctx.mpf(isp) * g0)))
                                         if isp is not None else None),
        })
    except (ArithmeticError, OverflowError):
        return dict(result, error="orbital_result_outside_numeric_range")
    residual = max(residuals)
    result.update(residual=float(residual), verified=bool(residual < 1e-11))
    return result


def _symmetric_euler_motion(moments, initial, torque, duration, axis):
    """Integrate the transverse Euler equation in its rotating coordinates."""
    from mpmath import mp

    i, j = (axis+1) % 3, (axis+2) % 3
    it, iz = moments[i], moments[axis]
    coupling = (iz-it)/it
    az = torque[axis]/iz
    fx, fy = torque[i]/it, torque[j]/it
    ctx = mp.clone()

    def number(q):
        return ctx.mpf(q.numerator)/q.denominator

    def rational(v):
        if not v:
            return Fraction(0)
        f, e = ctx.frexp(v)
        return Fraction(int(ctx.ldexp(f, ctx.prec))) * Fraction(2)**(e-ctx.prec)

    def advance(start, t, guard):
        alpha = coupling*start[axis]*t
        beta = coupling*az*t*t/2
        if not alpha and not beta:
            end = [w+t*f for w, f in zip(start, (tau/I for tau, I in zip(torque, moments)))]
            work = [tau*t*(w+v)/2 for tau,w,v in zip(torque,start,end)]
            return end, sum(work), sum(I*d for I,d in zip(moments,work))
        phase_size = max(abs(alpha), abs(beta), Fraction(1))
        force = (t*fx, t*fy)
        imbalance = (force[0]-alpha*start[j], force[1]+alpha*start[i])
        if beta and any(force):
            # Completing the square introduces an absolute phase alpha²/(4 beta).
            phase_size = max(phase_size, abs(alpha*alpha/(4*beta)))
        # Near a force-balanced endpoint the first nonzero coordinate can be
        # quadratic in a small spin phase, and full and split intervals could
        # otherwise round that same coordinate to zero before comparison.
        small_phase = max(abs(alpha), abs(beta))
        if any(force) and small_phase < 1:
            phase_size = max(phase_size, 1/small_phase**2)
        ctx.prec = guard + max(0, phase_size.numerator.bit_length()-phase_size.denominator.bit_length())
        turn = number(alpha+beta)
        z0 = ctx.mpc(number(start[i]), number(start[j]))
        impulse = ctx.mpc(*(number(v) for v in force))
        if not impulse:
            z1, lateral_work = z0*ctx.expj(turn), Fraction(0)
        elif not beta:
            # Propagate the exact unbalanced initial acceleration. Separate its
            # small increment from the finite initial state when computing work.
            change = ctx.expm1(ctx.j*turn)/(ctx.j*turn) * ctx.mpc(*(number(v) for v in imbalance))
            z1 = z0+change
            dx, dy = rational(change.real), rational(change.imag)
            lateral_work = it*(start[i]*dx + start[j]*dy + (dx*dx+dy*dy)/2)
        else:
            # J = integral_0^1 exp(i*(alpha*s + beta*s²)) ds. Evaluate
            # impulse work separately from the endpoint kinetic energy.
            b = number(beta)
            factor = ctx.sqrt(2*abs(b)/ctx.pi)
            v0 = factor*number(alpha/(2*beta))
            v1 = v0+factor
            delta = (ctx.fresnelc(v1)-ctx.fresnelc(v0)
                     + ctx.j*ctx.sign(b)*(ctx.fresnels(v1)-ctx.fresnels(v0)))
            integral = ctx.expj(number(-alpha*alpha/(4*beta))) * delta/factor
            transported = impulse*ctx.conj(integral)
            z1 = ctx.expj(turn)*(z0+transported)
            lateral_work = it*rational(ctx.re(ctx.conj(impulse)*z0*integral) + abs(transported)**2/2)
        end = list(start)
        end[i], end[j], end[axis] = rational(z1.real), rational(z1.imag), start[axis]+az*t
        axial_work = torque[axis]*t*(start[axis]+end[axis])/2
        return end, lateral_work+axial_work, it*lateral_work+iz*axial_work

    # Resolve each signed observable against a separate split-interval evaluation.
    # The smallest wire value sets only the absolute floor for a mathematical zero.
    floor = Fraction(2)**-1074
    for guard in (150, 300, 600, 1200, 2400, 4800):
        full, work, impulse_h2 = advance(initial, duration, guard)
        middle, w1, h1 = advance(initial, duration/2, guard+80)
        halves, w2, h2 = advance(middle, duration/2, guard+80)
        pairs = list(zip(full, halves)) + [(work, w1+w2), (impulse_h2, h1+h2)]
        residual = max(abs(v-u)/max(abs(v), abs(u), floor) for v,u in pairs)
        if residual < Fraction(1, 10**15):
            return full, work, impulse_h2, float(residual)
    raise ArithmeticError("unresolved_symmetric_top_motion")


def _euler_rigid_body(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Euler equations for positive principal moments and constant body-frame torque."""
    result = {"domain": "lagrangian_orbital", "subkind": "euler_rigid_body", "verified": False}

    def vector(name, default=None):
        raw = np.asarray(params.get(name, default), dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError(name)
        return tuple(Fraction(float(v)) for v in raw)

    try:
        moments = vector("I")
        initial = vector("omega0")
        torque = vector("torque", [0, 0, 0])
        duration = Fraction(float(params.get("t_final", 1)))
        steps = max(1, int(params.get("n_steps", 2000)))
        if min(moments) <= 0 or duration < 0:
            raise ValueError("invalid_rigid_body_inputs")
    except (TypeError, ValueError, OverflowError):
        return {**result, "error": "invalid_rigid_body_inputs"}

    # I_j + I_k - I_i = 2 integral(x_i^2 dm). Equality allows planar bodies.
    # Check the exact parsed binary values before rounding any sums or ratios.
    if 2 * max(moments) > sum(moments):
        return {**result, "error": "unrealizable_principal_moments"}

    free = not any(torque)
    acceleration = [tau / inertia for tau, inertia in zip(torque, moments)]
    rotation_scale = max(max(map(abs, initial)), duration * max(map(abs, acceleration))) or 1
    moment_scale = max(moments)

    def representable(x):
        value = float(x)
        if not math.isfinite(value) or (x != 0 and value == 0):
            raise FloatingPointError("unrepresentable_rigid_body_quantity")
        return value

    def relative_balance(start, end, integral):
        scale = max(abs(start), abs(end), abs(integral))
        return float(abs(end - start - integral) / scale) if scale else 0.0

    try:
        # s=t/t_final, u=omega/rotation_scale. The angular acceleration and
        # initial velocity set the same scale for independently integrated paths.
        omega_scale = representable(rotation_scale)
        axis = next((k for k in range(3) if moments[(k+1)%3] == moments[(k+2)%3]
                     and moments[k] != moments[(k+1)%3]), None)
        if axis is not None and duration:
            endpoint, work, half_h2_change, dual_error = _symmetric_euler_motion(
                moments, initial, torque, duration, axis)
            omega = [representable(w) for w in endpoint]
            physical_work = representable(work)
            scaled_half_h2 = float(half_h2_change/(moment_scale*rotation_scale)**2)
            # Replay torque work against the actually reported endpoint.
            f = list(map(Fraction, omega))
            energy_balance = relative_balance(sum(i*w*w for i,w in zip(moments,initial))/2,
                                              sum(i*w*w for i,w in zip(moments,f))/2, work)
            momentum_balance = relative_balance(sum((i*w)**2 for i,w in zip(moments,initial))/2,
                                                sum((i*w)**2 for i,w in zip(moments,f))/2, half_h2_change)
        else:
            u0 = np.array([representable(x / rotation_scale) for x in initial])
            inertia = np.array([representable(x / moment_scale) for x in moments])
            forcing = np.array([representable(duration * a / rotation_scale) for a in acceleration])
            spin = np.array([
                representable(duration * rotation_scale * (moments[j] - moments[k]) / moments[i])
                for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            ])
            def velocity_rhs(_s, u):
                x, y, z = u
                return spin * np.array([y*z, z*x, x*y]) + forcing

            # Accumulate work and torque contribution to half |H|^2 along the
            # primary trajectory. These follow dT/dt=tau.omega, d(|H|^2/2)/dt=tau.H.
            scaled_torque = np.array([representable(tau * duration / (moment_scale * rotation_scale))
                                      for tau in torque])
            def primary_rhs(s, state):
                u = state[:3]
                return np.r_[velocity_rhs(s, u), scaled_torque @ u, scaled_torque @ (inertia * u)]

            state = np.r_[u0, 0.0, 0.0]
            dual = u0
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if duration:
                    ds = 1.0 / steps
                    for step in range(steps):
                        s = step * ds
                        k1 = primary_rhs(s, state)
                        k2 = primary_rhs(s + ds/2, state + ds*k1/2)
                        k3 = primary_rhs(s + ds/2, state + ds*k2/2)
                        k4 = primary_rhs(s + ds, state + ds*k3)
                        state += (ds/6) * (k1 + 2*k2 + 2*k3 + k4)
                    solution = integrate.solve_ivp(velocity_rhs, (0.0, 1.0), u0, method="DOP853",
                                                   rtol=1e-12, atol=1e-14)
                    if not solution.success or solution.t[-1] != 1.0:
                        return {**result, "error": "incomplete_rigid_body_integrator"}
                    dual = solution.y[:, -1]
                if not np.isfinite(state).all() or not np.isfinite(dual).all():
                    raise FloatingPointError("nonfinite_rigid_body_solution")
                u = state[:3]
                # Finite endpoint norms can have nonrepresentable squares. Evaluate
                # the two small endpoint quadratic forms exactly on parsed floats.
                def invariants(v):
                    h = [(i / moment_scale) * Fraction(float(w)) for i, w in zip(moments, v)]
                    return (sum(x * Fraction(float(w)) for x, w in zip(h, v)) / 2,
                            sum(x*x for x in h) / 2)
                e0, h02 = invariants(u0)
                ef, hf2 = invariants(u)
                energy_balance = relative_balance(e0, ef, Fraction(float(state[3])))
                momentum_balance = relative_balance(h02, hf2, Fraction(float(state[4])))
                dual_error = math.hypot(*(u-dual)) / max(1., math.hypot(*u), math.hypot(*dual))
                for axis in range(3):
                    plane = [k for k in range(3) if k != axis]
                    h = max(math.hypot(*v[plane]) for v in (u0, u, dual, forcing))
                    if h:
                        dual_error = max(dual_error, math.hypot(*(u-dual)[plane])/h)
                    if free:
                        # This independent invariant cancels the axial spin
                        # analytically before checking the smaller free wobble.
                        weights = [moments[k]*(moments[axis]-moments[k]) for k in plane]
                        old = [w*Fraction(float(u0[k]))**2 for w,k in zip(weights,plane)]
                        new = [w*Fraction(float(u[k]))**2 for w,k in zip(weights,plane)]
                        scale = max(sum(map(abs,old)),sum(map(abs,new)))
                        if scale:
                            dual_error = max(dual_error, float(abs(sum(new)-sum(old))/scale))
            omega = ([representable(rotation_scale * Fraction(float(x))) for x in u]
                     if duration else list(map(float, initial)))
            physical_work = representable(Fraction(float(state[3])) * moment_scale * rotation_scale**2)
            half_h2_change = Fraction(float(state[4])) * (moment_scale * rotation_scale)**2
            scaled_half_h2 = float(state[4])
        final = [Fraction(x) for x in omega]
        kinetic = representable(sum(i*w*w for i, w in zip(moments, final)) / 2)
        momentum = math.hypot(*(representable(i*w) for i, w in zip(moments, final)))
        # Squared SI momenta can exceed binary64 even when the angular momentum
        # and its direction are well resolved. Preserve a scaled representation.
        try:
            physical_half_h2_change = representable(half_h2_change)
        except (OverflowError, FloatingPointError):
            physical_half_h2_change = None

        x, y, z = initial
        initial_phase_rate = None
        if x or y:
            dx = ((moments[1] - moments[2]) * y*z + torque[0]) / moments[0]
            dy = ((moments[2] - moments[0]) * z*x + torque[1]) / moments[1]
            initial_phase_rate = representable((x*dy - y*dx) / (x*x + y*y))
        rate = initial_phase_rate if free and moments[0] == moments[1] else None
        h = [i*w for i, w in zip(moments, initial)]
        h2 = sum(xi*xi for xi in h)
        direction_speed = None
        if h2:
            # Inertial dH/dt = torque. The norm can be evaluated in the
            # instantaneous body basis without choosing an initial attitude.
            direction_speed = math.hypot(*(representable((h[j]*torque[k] - h[k]*torque[j]) / h2)
                                           for j, k in ((1, 2), (2, 0), (0, 1))))
        residual = max(dual_error, energy_balance, momentum_balance)
        if not all(map(math.isfinite, (momentum, residual, direction_speed or 0.0))):
            raise FloatingPointError("nonfinite_rigid_body_output")
    except (ArithmeticError, ValueError, RuntimeError):
        return {**result, "error": "unresolved_rigid_body_numerics"}

    return {**result, "omega_final_rad_s": omega, "kinetic_energy_J": kinetic,
            "angular_momentum_norm_Nms": momentum, "precession_rate_rad_s": rate,
            "initial_body_omega_azimuth_rate_rad_s": initial_phase_rate,
            "initial_inertial_H_direction_speed_rad_s": direction_speed,
            "work_done_J": physical_work, "torque_half_H2_change_Nms2": physical_half_h2_change,
            "torque_half_H2_change_scaled": scaled_half_h2,
            "moment_scale_kg_m2": float(moment_scale), "omega_scale_rad_s": omega_scale,
            "torque_free": free, "solver_residual": float(dual_error),
            "work_balance_residual": float(energy_balance),
            "angular_momentum_balance_residual": float(momentum_balance),
            "residual": float(residual), "verified": bool(residual < 1e-8)}


def solve_lagrangian_orbital(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(
        params.get("subkind")
        or ("euler_rigid_body" if "I" in params else ("lagrangian_modes" if "K" in params else "orbital_transfer"))
    ).strip().lower()

    if subkind == "euler_rigid_body":
        return _euler_rigid_body(params)

    if subkind == "lagrangian_modes":
        return _lagrangian_modes(params)

    if subkind == "orbital_transfer":
        return _orbital_transfer(params)

    return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": f"unsupported_lagrangian_orbital:{subkind}"}


def _electrochemistry(params: Mapping[str, Any]) -> Dict[str, Any]:
    """One written reaction and one identified product, at constant current."""
    result = {"domain": "electrochem_separations", "subkind": "electrochemistry"}

    def value(key, default=None):
        raw = params.get(key, default)
        if raw is None or isinstance(raw, bool):
            raise ValueError(f"invalid_{key}")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"invalid_{key}")
        return number

    def rational(number):
        return Fraction(str(number))

    def represent(quantity):
        number = float(quantity)
        if not math.isfinite(number) or (quantity and number == 0.0):
            raise ArithmeticError("electrochem_result_outside_numeric_range")
        return number

    try:
        potential = value("E0_V")
        electrons = value("n_electrons")
        temperature = value("T_K", 298.15)
        quotient = value("Q_reaction", 1.0)
        current = value("current_A", 0.0)
        resistance = value("R_internal_ohm", 0.0)
        activation = value("eta_activation_V", 0.0)
        duration = value("time_s", 0.0)
        efficiency = value("faradaic_efficiency", 1.0)
        coefficient = value("product_stoich", 1.0)
        molar_mass = value("molar_mass_g_mol") if "molar_mass_g_mol" in params else None
        target = value("target_mass_g") if "target_mass_g" in params else None
        mode = str(params.get("mode", "galvanic")).strip().lower()
        if (min(electrons, temperature, quotient, coefficient) <= 0
                or min(current, resistance, activation, duration) < 0
                or not 0 <= efficiency <= 1
                or (molar_mass is not None and molar_mass <= 0)
                or (target is not None and target < 0)
                or mode not in ("galvanic", "electrolytic")):
            raise ValueError("invalid_electrochem_parameters")

        # Charge and yield refer to the complete written reaction. Scaling its equation
        # changes extent inversely, leaving the amount of the selected species invariant.
        faraday = rational(FARADAY_C_MOL)
        n, nu = rational(electrons), rational(coefficient)
        total_charge = rational(current) * rational(duration)
        reaction_charge = rational(efficiency) * total_charge
        extent = reaction_charge / (n * faraday)
        product = nu * extent
        mass = product * rational(molar_mass) if molar_mass is not None else None
        result.update({
            "n_electrons": electrons,
            "product_stoich": coefficient,
            "charge_passed_C": represent(total_charge),
            "reaction_charge_C": represent(reaction_charge),
            "reaction_extent_mol": represent(extent),
            "moles_reacted": represent(extent),
            "product_moles": represent(product),
            "mass_deposited_g": represent(mass) if mass is not None else (0.0 if product == 0 else None),
        })

        rt = R_GAS_UNIV * temperature
        nf = electrons * FARADAY_C_MOL
        log_q = math.log(quotient)
        equilibrium = potential - (rt / nf) * log_q
        standard_gibbs = -nf * potential
        gibbs = -nf * equilibrium
        loss = activation + current * resistance
        terminal = equilibrium + loss if mode == "electrolytic" else equilibrium - loss
        ln_k = -standard_gibbs / rt
        thermo = [equilibrium, terminal, standard_gibbs, gibbs, ln_k]
        if not all(map(math.isfinite, thermo)):
            raise ArithmeticError("electrochem_result_outside_numeric_range")

        # Check both reported molar quantities against the selected reaction charge.
        def balance(charge):
            return float(abs(charge - reaction_charge) / max(abs(reaction_charge), Fraction(1)))

        residual = max(
            abs(gibbs - (standard_gibbs + rt * log_q)) / max(abs(gibbs), abs(standard_gibbs), rt, 1),
            balance(rational(result["reaction_extent_mol"]) * n * faraday),
            balance(rational(result["product_moles"]) * n * faraday / nu),
        )
        if not math.isfinite(residual):
            raise ArithmeticError("electrochem_result_outside_numeric_range")
        result.update({"E_nernst_V": equilibrium, "V_cell_V": terminal,
                       "delta_G0_J_mol": standard_gibbs, "delta_G_J_mol": gibbs,
                       "ln_K_eq": ln_k, "residual": residual, "verified": residual < 1e-11})

        if target is not None:
            result["target_time_s"] = None
            rate = rational(current) * rational(efficiency) * nu / (n * faraday)
            if target == 0:
                result["target_mass_status"] = "all_nonnegative_times" if rate == 0 else "unique"
                if rate:
                    result["target_time_s"] = 0.0
            elif rate == 0 or molar_mass is None:
                status = "unreachable_at_zero_rate" if rate == 0 else "molar_mass_required"
                result.update({"target_mass_status": status, "verified": False,
                               "error": f"target_mass_{status}"})
            else:
                time = rational(target) / (rate * rational(molar_mass))
                result.update({"target_mass_status": "unique", "target_time_s": represent(time)})
        return result
    except (ArithmeticError, TypeError, ValueError) as exc:
        return {"domain": "electrochem_separations", "subkind": "electrochemistry",
                "verified": False, "error": str(exc)}


def _distillation_underwood(z, xd, xb, volatility, quality):
    """Solve in u=(theta-1)/(alpha-1), keeping both distances to the poles."""
    width = volatility-1
    a = width*(1-quality)
    b = 1+width*z-a
    c = 1-z
    # P(u)=a*u²+b*u-c is negative at zero and positive at one. Using its
    # exact sign avoids evaluating the rational Underwood law at either pole.
    lo, hi = Fraction(0), Fraction(1)
    while hi-lo > min(lo, 1-hi)/2**75:
        mid = (lo+hi)/2
        value = (a*mid+b)*mid-c
        if not value:
            lo = hi = mid
            break
        if value < 0:
            lo = mid
        else:
            hi = mid
    u = (lo+hi)/2
    v = 1-u
    with localcontext() as context:
        context.prec = 90
        def dec(q):
            return Decimal(q.numerator)/Decimal(q.denominator)

        disc = dec(b*b+4*a*c).sqrt()
        def positive_root(aa, bb, cc):
            if bb >= 0:
                return 2*dec(cc)/(disc+dec(bb))
            return (disc-dec(bb))/(2*dec(aa))

        # A second solution for each complementary distance, using the
        # non-cancelling quadratic form for P(u) and for -P(1-v).
        ua = positive_root(a, b, c)
        va = positive_root(-a, 2*a+b, volatility*z)
        error = max(abs(dec(u)/ua-1), abs(dec(v)/va-1))
        # Fenske in the same round-trip decimal input convention as the
        # section balances; preserve log(alpha) when the keys are close.
        n_min = float((dec(xd)*(1-dec(xb))/(dec(xb)*(1-dec(xd)))).ln()/dec(volatility).ln())
    uw, vw = float(u), float(v)
    wire_error = max(abs(Fraction(uw)/u-1),abs(Fraction(vw)/v-1))
    # Rebuild the reported root from its smaller coordinate, exactly as
    # reported. Qualify the physical feed equation in that coordinate.
    public_u = Fraction(uw) if u <= v else 1-Fraction(vw)
    public_v = 1-public_u
    terms = (volatility*z*public_u, -(1-z)*public_v, -a*public_u*public_v)
    law_error = abs(sum(terms))/sum(map(abs,terms))
    pinch = volatility*xd/(width*v) - (1-xd)/(width*u) - 1
    return n_min, float(1+width*u), uw, vw, float(pinch), max(float(error),float(wire_error),float(law_error))


class _TrayPrecisionError(ArithmeticError):
    pass


def _distillation_trays(xd, xb, alpha, intersection, rect_slope, strip_slope, one_stage):
    """Cross each section by doubling its equilibrium/operating-line map."""
    from mpmath import mp
    from mpmath.ctx_iv import MPIntervalContext

    wire = mp.clone()
    wire.prec = 53

    def solve(ctx):
        def number(q):
            return q if ctx is None else ctx.mpf(q.numerator)/q.denominator

        def less(a, b):
            if ctx is None:
                return a < b
            if a.b < b.a:
                return True
            if a.a >= b.b:
                return False
            raise _TrayPrecisionError

        def reported(value):
            if ctx is None:
                return float(value)
            low, high = float(wire.mpf(value.a)), float(wire.mpf(value.b))
            if low != high:
                raise _TrayPrecisionError
            return low

        top, bottom, volatility, contact = map(number, (xd, xb, alpha, intersection))
        zero, one = number(Fraction(0)), number(Fraction(1))
        rect = number(rect_slope)
        strip = None if strip_slope is None else number(strip_slope)

        def cross(start, target, slope, intercept):
            # M(x) = (s*x+b)/(alpha-(alpha-1)*(s*x+b)). Its denominator is
            # positive throughout the traversed section. M(x)-x has a convex
            # quadratic numerator. Negative endpoint signs prove descent over
            # the entire interval; contact with a fixed point takes infinite trays.
            y0 = slope*start + intercept
            den = volatility - (volatility-one)*y0
            if not slope:
                endpoint = y0/den
                if less(target, endpoint):
                    raise ValueError("unreachable_distillation_section")
                return 1, start, endpoint
            yt = slope*target + intercept
            if (not less(yt, target*(volatility-(volatility-one)*yt))
                    or not less(y0, start*den)):
                raise ValueError("infinite_or_unreachable_distillation_section")

            # Work with distance traveled from the starting composition. All
            # four entries are nonnegative, avoiding cancellation under powering.
            matrix = (slope*(one+(volatility-one)*start), start*den-y0,
                      (volatility-one)*slope, den)
            gap = start-target

            def advance(m, distance):
                a,b,c,d = m
                return (a*distance+b)/(c*distance+d)

            powers = [matrix]
            while less(advance(powers[-1], zero), gap):
                a,b,c,d = powers[-1]
                entries = (a*a+b*c, b*(a+d), c*(a+d), b*c+d*d)
                # A common positive normalization leaves the map unchanged.
                scale = entries[3] if ctx is None else entries[3].b
                squared = tuple(e/scale for e in entries)
                if ctx is None and any(max(e.numerator.bit_length(),e.denominator.bit_length()) > 4096
                                       for e in squared):
                    raise _TrayPrecisionError
                powers.append(squared)

            steps, distance = 0, zero
            for bit in reversed(range(len(powers)-1)):
                candidate = advance(powers[bit], distance)
                if less(candidate, gap):
                    steps += 1 << bit
                    distance = candidate
            return steps+1, start-distance, start-advance(matrix, distance)

        first = xd/(alpha-(alpha-1)*xd)
        previous, current = top, number(first)
        count = 1
        feed_stage = 1 if first <= intersection < xd else None
        if one_stage:
            return 1, 1., 1
        reached_bottom = first <= xb
        if not reached_bottom and first > intersection:
            # A zero-boilup section meets the rectifier exactly at xB. Use
            # input-rational endpoint identities before interval conversion.
            feed_before_product = intersection > xb
            target = contact if feed_before_product else bottom
            steps, previous, current = cross(current, target, rect, (one-rect)*top)
            count += steps
            if intersection >= xb:
                feed_stage = count
            reached_bottom = not feed_before_product or not less(bottom, current)
        if not reached_bottom:
            slope, intercept = ((rect,(one-rect)*top) if strip is None else (strip,(one-strip)*bottom))
            steps, previous, current = cross(current, bottom, slope, intercept)
            count += steps
        fraction = (previous-bottom)/(previous-current)
        return count, reported((count-1)+fraction), feed_stage or count

    # Exact rational arithmetic handles an exact arrival at a feed or product
    # composition. On long columns switch from expanding integers to outward
    # rounded intervals and certify both sides of each counted crossing.
    try:
        return solve(None)
    except _TrayPrecisionError:
        pass
    for digits in (80, 160, 320, 640):
        ctx = MPIntervalContext()
        ctx.dps = digits
        try:
            return solve(ctx)
        except _TrayPrecisionError:
            continue
    raise ValueError("unresolved_distillation_tray_boundary")


def solve_electrochem_or_separation(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(
        params.get("subkind") or ("distillation" if "alpha" in params else "electrochemistry")
    ).strip().lower()

    if subkind == "equilibrium_flash":
        from hamon_fixed_k_flash import fixed_k_flash
        return fixed_k_flash(params)

    if subkind == "electrochemistry":
        return _electrochemistry(params)

    if subkind == "distillation":
        z_f = float(params["z_F"])
        x_d = float(params["x_D"])
        x_b = float(params["x_B"])
        alpha = float(params["alpha"])
        q = float(params.get("q", 1.0))
        f_flow = float(params.get("F_mol_s", 100.0))

        if (not all(math.isfinite(v) for v in (x_b, z_f, x_d, alpha, q, f_flow))
                or not 0.0 < x_b < z_f < x_d < 1.0 or alpha <= 1.0 or f_flow <= 0.0):
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False, "error": "invalid_distillation_compositions_or_volatility"}

        # Use the round-trip decimal inputs for exact flow signs and boundary contacts.
        xb, z, xd, volatility, quality, feed = (
            Fraction(str(value)) for value in (x_b, z_f, x_d, alpha, q, f_flow)
        )
        distillate = feed * (z - xb) / (xd - xb)
        bottoms = feed - distillate
        d_flow, b_flow = float(distillate), float(bottoms)

        n_min, theta_brent, theta_fraction, theta_complement, pinch_reflux, theta_err = (
            _distillation_underwood(z, xd, xb, volatility, quality))
        # L = R D >= 0 and V' = (R+1)D - (1-q)F >= 0. Then L' = V' + B > 0.
        flow_floor = max(Fraction(0), (1 - quality) * feed / distillate - 1)

        def feed_contact(reflux):
            # Eliminate the section flows from the total and component balances.
            x = ((reflux + 1) * z + (quality - 1) * xd) / (reflux + quality)
            y = (reflux * x + xd) / (reflux + 1)
            equilibrium_gap = volatility * x / (1 + (volatility - 1) * x) - y
            return x, y, equilibrium_gap

        _, _, floor_gap = feed_contact(flow_floor)
        # A pinch outside the working composition interval cannot set its minimum.
        r_min = float(flow_floor) if floor_gap >= 0 else max(float(flow_floor), pinch_reflux)
        limits = {"R_min": r_min, "R_flow_floor": float(flow_floor), "R_feed_pinch": pinch_reflux}
        if params.get("R") is None:
            multiplier = float(params.get("R_mult", 1.25))
            if not math.isfinite(multiplier) or multiplier < 0:
                return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                        "error": "invalid_reflux_multiplier", **limits}
            r_op = multiplier * r_min
        else:
            r_op = float(params["R"])
        if not math.isfinite(r_op):
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                    "error": "nonfinite_reflux", **limits}
        # A multiplier applied at an exact flow floor should preserve its boundary.
        if params.get("R") is None and floor_gap >= 0:
            reflux = Fraction(str(multiplier)) * flow_floor
        else:
            reflux = Fraction(str(r_op))
        liquid = reflux * distillate
        vapor = liquid + distillate
        strip_vapor = vapor - (1 - quality) * feed
        strip_liquid = strip_vapor + bottoms
        flow_report = {"L_rect_mol_s": float(liquid), "V_rect_mol_s": float(vapor),
                       "L_strip_mol_s": float(strip_liquid), "V_strip_mol_s": float(strip_vapor)}
        conditions = {**limits, **flow_report, "R_operating": r_op}
        if min(liquid, strip_vapor) < 0:
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                    "error": "negative_internal_section_flow", **conditions}
        xi, yi, gap = feed_contact(reflux)
        one_stage = xd * (1 + (volatility - 1) * xb) == volatility * xb
        flash = liquid == 0 and strip_vapor == 0
        if flash and gap:
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                    "error": "zero_circulation_flash_products_not_in_equilibrium", **conditions}
        if not flash and gap <= 0:
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                    "error": "reflux_below_minimum_pinch", **conditions}
        if flash:
            section_mode = "single_equilibrium_flash"
        elif liquid == 0:
            section_mode = "stripping_only"
        elif strip_vapor == 0:
            section_mode = "rectifying_only"
        else:
            section_mode = "two_section"

        try:
            stages, frac_stages, feed_stage = _distillation_trays(
                xd, xb, volatility, xi, reflux/(reflux+1),
                strip_liquid/strip_vapor if strip_vapor else None, one_stage)
        except ValueError as exc:
            return {"domain": "electrochem_separations", "subkind": subkind,
                    "verified": False, "error": str(exc), **conditions}

        mb_err = abs(d_flow * x_d + b_flow * x_b - f_flow * z_f) / max(f_flow * z_f, 1e-12)
        # Verify the section component balance without dividing by vanishing boilup.
        strip_err = abs(Fraction(flow_report["V_strip_mol_s"])*(yi-xb) -
                        Fraction(flow_report["L_strip_mol_s"])*(xi-xb))/feed
        rect_err = abs(Fraction(flow_report["V_rect_mol_s"])*yi -
                       Fraction(flow_report["L_rect_mol_s"])*xi - Fraction(d_flow)*xd)/feed
        residual = float(max(mb_err, theta_err, strip_err, rect_err))
        return {
            "domain": "electrochem_separations",
            "subkind": subkind,
            "D_mol_s": float(d_flow),
            "B_mol_s": float(b_flow),
            "N_min_fenske": float(n_min),
            "underwood_theta": theta_brent,
            "underwood_theta_fraction": theta_fraction,
            "underwood_complement_fraction": theta_complement,
            **conditions,
            "section_mode": section_mode,
            "N_stages_integer": int(stages),
            "N_theoretical": float(frac_stages),
            "feed_stage": feed_stage if feed_stage is not None else stages,
            "residual": residual,
            "verified": bool((one_stage or frac_stages > n_min) and residual < 1e-10),
        }

    return {"domain": "electrochem_separations", "subkind": subkind, "verified": False, "error": f"unsupported_electrochem_separation:{subkind}"}


def infer_wave7_domain(text_or_domain: str) -> Optional[str]:
    s = (text_or_domain or "").strip().lower()
    if s in WAVE7_DOMAINS:
        return s
    if any(k in s for k in ("brayton", "rankine", "otto", "diesel", "vapor_compression", "vapor-compression", "refrigeration cop", "regenerator", "back-work")):
        return "thermo_cycles"
    if any(k in s for k in ("epsilon-ntu", "ntu", "heat exchanger", "shell_tube", "radiosity", "view factor", "oppenheim", "gray enclosure", "radiation_network")):
        return "exchangers_radiation"
    if any(k in s for k in ("euler_rigid_body", "rigid body", "principal inertia", "lagrangian", "eigenmode", "natural frequenc", "hohmann", "vis-viva", "orbital_transfer")):
        return "lagrangian_orbital"
    if any(k in s for k in ("nernst", "faraday", "overpotential", "electrochem", "mccabe-thiele", "mccabe", "fenske", "underwood", "distillation", "equilibrium_flash")):
        return "electrochem_separations"
    return None


def dispatch_wave7_verifier(domain_or_spec: Any, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(domain_or_spec, Mapping):
        spec = domain_or_spec
        domain = infer_wave7_domain(str(spec.get("domain") or spec.get("kind") or spec.get("prompt") or ""))
        eff_params = dict(spec.get("params") or spec)
    else:
        domain = infer_wave7_domain(str(domain_or_spec))
        eff_params = dict(params or {})

    if domain == "thermo_cycles":
        return solve_thermodynamic_cycle(eff_params)
    if domain == "exchangers_radiation":
        return solve_exchanger_or_radiation(eff_params)
    if domain == "lagrangian_orbital":
        return solve_lagrangian_orbital(eff_params)
    if domain == "electrochem_separations":
        return solve_electrochem_or_separation(eff_params)
    return {"domain": domain, "verified": False, "error": f"unrecognized_wave7_domain:{domain_or_spec}"}
