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
        h1 = float(params["h1"])
        h2s = float(params["h2s"])
        h3 = float(params["h3"])
        eta_c = float(params.get("eta_c", 1.0))
        m_dot = float(params.get("m_dot", 1.0))
        if h2s <= h1 or h1 <= h3 or not (0.0 < eta_c <= 1.0) or m_dot <= 0.0:
            return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_vcr_enthalpies"}
        h4 = h3
        h2 = h1 + (h2s - h1) / eta_c
        w_in = h2 - h1
        q_l = h1 - h4
        q_h = h2 - h3
        cop_r = q_l / w_in
        cop_hp = q_h / w_in

        carnot_ok = True
        if "T_L" in params and "T_H" in params:
            t_l = float(params["T_L"])
            t_h = float(params["T_H"])
            if t_l <= 0.0 or t_h <= t_l:
                return {"domain": "thermo_cycles", "subkind": subkind, "verified": False, "error": "invalid_vcr_reservoirs"}
            carnot_ok = bool(cop_r <= (t_l / (t_h - t_l)) + 1e-9)

        residual = float(
            max(
                abs((cop_hp - cop_r) - 1.0),
                abs((q_l + w_in) - q_h) / max(abs(q_h), 1e-12),
            )
        )
        return {
            "domain": "thermo_cycles",
            "subkind": subkind,
            "h2": float(h2),
            "h4": float(h4),
            "q_evap": float(q_l),
            "q_cond": float(q_h),
            "w_comp": float(w_in),
            "Q_evap_W": float(m_dot * q_l),
            "W_comp_W": float(m_dot * w_in),
            "COP_R": float(cop_r),
            "COP_HP": float(cop_hp),
            "residual": residual,
            "verified": bool(carnot_ok and residual < 1e-11),
        }

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
        if np.any(areas <= 0.0) or np.any(eps_vec <= 0.0) or np.any(eps_vec > 1.0) or np.any(f_mat < -1e-12):
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

        m_sys = np.zeros((n, n), dtype=float)
        b_rhs = np.zeros(n, dtype=float)
        for i in range(n):
            if i in adiabatic:
                m_sys[i, :] = -f_mat[i, :]
                m_sys[i, i] += 1.0
                b_rhs[i] = 0.0
            else:
                ti = float(t_raw[i])
                if ti <= 0.0:
                    return {"domain": "exchangers_radiation", "subkind": subkind, "verified": False, "error": "non_positive_surface_temperature"}
                eb_i = SIGMA_SB * (ti ** 4)
                m_sys[i, :] = -(1.0 - eps_vec[i]) * f_mat[i, :]
                m_sys[i, i] += 1.0
                b_rhs[i] = eps_vec[i] * eb_i

        j_primary = np.linalg.solve(m_sys, b_rhs)
        lu, piv = linalg.lu_factor(m_sys)
        j_dual = linalg.lu_solve((lu, piv), b_rhs)

        q_net = np.zeros(n, dtype=float)
        surf_res_err = 0.0
        t_effective: List[float] = []
        for i in range(n):
            q_encl = float(areas[i]) * float(np.sum(f_mat[i, :] * (j_primary[i] - j_primary)))
            q_net[i] = q_encl
            if i in adiabatic:
                t_effective.append(float((max(float(j_primary[i]), 0.0) / SIGMA_SB) ** 0.25))
            else:
                ti = float(t_raw[i])
                t_effective.append(ti)
                if eps_vec[i] < 1.0 - 1e-12:
                    q_surf = float((areas[i] * eps_vec[i] / (1.0 - eps_vec[i])) * (SIGMA_SB * (ti ** 4) - j_primary[i]))
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


def solve_lagrangian_orbital(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(
        params.get("subkind")
        or ("euler_rigid_body" if "I" in params else ("lagrangian_modes" if "K" in params else "orbital_transfer"))
    ).strip().lower()

    if subkind == "euler_rigid_body":
        inertia = np.asarray(params["I"], dtype=float)
        w0 = np.asarray(params["omega0"], dtype=float)
        t_f = float(params.get("t_final", 1.0))
        torque = np.asarray(params.get("torque", [0.0, 0.0, 0.0]), dtype=float)
        n_steps = int(params.get("n_steps", 2000))
        if inertia.shape != (3,) or w0.shape != (3,) or torque.shape != (3,) or np.any(inertia <= 0.0) or t_f < 0.0:
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "invalid_rigid_body_inputs"}

        i1, i2, i3 = float(inertia[0]), float(inertia[1]), float(inertia[2])

        def rhs(_t: float, w: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    ((i2 - i3) * w[1] * w[2] + torque[0]) / i1,
                    ((i3 - i1) * w[2] * w[0] + torque[1]) / i2,
                    ((i1 - i2) * w[0] * w[1] + torque[2]) / i3,
                ],
                dtype=float,
            )

        dt = t_f / max(n_steps, 1)
        w_rk = w0.copy()
        t_cur = 0.0
        for _ in range(max(n_steps, 1) if t_f > 0.0 else 0):
            k1 = rhs(t_cur, w_rk)
            k2 = rhs(t_cur + 0.5 * dt, w_rk + 0.5 * dt * k1)
            k3 = rhs(t_cur + 0.5 * dt, w_rk + 0.5 * dt * k2)
            k4 = rhs(t_cur + dt, w_rk + dt * k3)
            w_rk = w_rk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t_cur += dt

        if t_f > 0.0:
            sol = integrate.solve_ivp(rhs, (0.0, t_f), w0, method="DOP853", rtol=1e-12, atol=1e-14)
            w_dop = sol.y[:, -1]
        else:
            w_dop = w0.copy()

        t_ke0 = 0.5 * float(np.sum(inertia * (w0 ** 2)))
        t_kef = 0.5 * float(np.sum(inertia * (w_rk ** 2)))
        h_0_sq = float(np.sum((inertia * w0) ** 2))
        h_f_sq = float(np.sum((inertia * w_rk) ** 2))

        solver_diff = float(np.linalg.norm(w_rk - w_dop)) / max(float(np.linalg.norm(w0)), 1.0)
        if float(np.linalg.norm(torque)) < 1e-15:
            ke_err = abs(t_kef - t_ke0) / max(t_ke0, 1e-15)
            casimir_err = abs(h_f_sq - h_0_sq) / max(h_0_sq, 1e-15)
            residual = float(max(solver_diff, ke_err, casimir_err))
        else:
            residual = float(solver_diff)

        precession_rate = float((i3 - i1) * w0[2] / i1) if abs(i1 - i2) < 1e-12 * max(i1, i2) else None
        return {
            "domain": "lagrangian_orbital",
            "subkind": subkind,
            "omega_final_rad_s": [float(x) for x in w_rk],
            "kinetic_energy_J": float(t_kef),
            "angular_momentum_norm_Nms": float(math.sqrt(h_f_sq)),
            "precession_rate_rad_s": precession_rate,
            "residual": residual,
            "verified": bool(residual < 1e-8),
        }

    if subkind == "lagrangian_modes":
        m_mat = np.asarray(params["M"], dtype=float)
        k_mat = np.asarray(params["K"], dtype=float)
        n = m_mat.shape[0]
        if m_mat.shape != (n, n) or k_mat.shape != (n, n):
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "matrix_shape_mismatch"}
        sym_err = max(
            float(np.linalg.norm(m_mat - m_mat.T)),
            float(np.linalg.norm(k_mat - k_mat.T)),
        ) / max(float(np.linalg.norm(k_mat)), 1.0)
        if sym_err > 1e-10:
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "non_symmetric_mass_or_stiffness"}

        try:
            chol_l = np.linalg.cholesky(m_mat)
        except np.linalg.LinAlgError:
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "non_spd_mass_matrix"}

        evals, evecs = linalg.eigh(k_mat, m_mat)
        if np.any(evals < -1e-10):
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "unstable_negative_stiffness_eigenvalue"}
        evals_pos = np.maximum(evals, 0.0)
        omegas = np.sqrt(evals_pos)

        l_inv = np.linalg.inv(chol_l)
        k_tilde = l_inv @ k_mat @ l_inv.T
        k_tilde = 0.5 * (k_tilde + k_tilde.T)
        evals_dual = np.maximum(np.linalg.eigvalsh(k_tilde), 0.0)

        eig_res = float(np.linalg.norm(k_mat @ evecs - (m_mat @ evecs) * evals[None, :])) / max(float(np.linalg.norm(k_mat)), 1.0)
        orth_res = float(np.linalg.norm(evecs.T @ m_mat @ evecs - np.eye(n)))
        dual_res = float(np.linalg.norm(evals_pos - evals_dual)) / max(float(np.linalg.norm(evals_pos)), 1.0)
        residual = float(max(eig_res, orth_res, dual_res))
        return {
            "domain": "lagrangian_orbital",
            "subkind": subkind,
            "eigenvalues_omega2": [float(x) for x in evals_pos],
            "natural_frequencies_rad_s": [float(x) for x in omegas],
            "natural_frequencies_Hz": [float(x / (2.0 * math.pi)) for x in omegas],
            "residual": residual,
            "verified": bool(residual < 1e-10),
        }

    if subkind == "orbital_transfer":
        mu = float(params["mu"])
        r1 = float(params["r1"])
        r2 = float(params["r2"])
        d_inc = float(params.get("delta_inc_rad", math.radians(float(params.get("delta_inc_deg", 0.0)))))
        if mu <= 0.0 or r1 <= 0.0 or r2 <= 0.0:
            return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": "invalid_orbital_parameters"}

        vc1 = math.sqrt(mu / r1)
        vc2 = math.sqrt(mu / r2)
        a_t = 0.5 * (r1 + r2)
        e_t = abs(r2 - r1) / (r1 + r2)
        vt1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
        vt2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

        if r2 >= r1:
            dv1 = abs(vt1 - vc1)
            dv2 = math.sqrt(max(0.0, vt2 * vt2 + vc2 * vc2 - 2.0 * vt2 * vc2 * math.cos(d_inc)))
        else:
            dv1 = math.sqrt(max(0.0, vc1 * vc1 + vt1 * vt1 - 2.0 * vc1 * vt1 * math.cos(d_inc)))
            dv2 = abs(vc2 - vt2)

        dv_tot = dv1 + dv2
        t_trans = math.pi * math.sqrt((a_t ** 3) / mu)
        h_expect = math.sqrt(mu * a_t * (1.0 - e_t * e_t))
        h_err = max(abs(r1 * vt1 - h_expect), abs(r2 * vt2 - h_expect)) / max(h_expect, 1e-12)
        eps_t = -mu / (2.0 * a_t)
        e_err = max(
            abs((0.5 * vt1 * vt1 - mu / r1) - eps_t),
            abs((0.5 * vt2 * vt2 - mu / r2) - eps_t),
        ) / max(abs(eps_t), 1e-12)
        residual = float(max(h_err, e_err))

        prop_frac = None
        if "isp_s" in params:
            isp = float(params["isp_s"])
            g0 = float(params.get("g0", 9.80665))
            prop_frac = float(1.0 - math.exp(-dv_tot / (isp * g0)))

        return {
            "domain": "lagrangian_orbital",
            "subkind": subkind,
            "a_transfer_m": float(a_t),
            "eccentricity_transfer": float(e_t),
            "dv1_m_s": float(dv1),
            "dv2_m_s": float(dv2),
            "dv_total_m_s": float(dv_tot),
            "transfer_time_s": float(t_trans),
            "propellant_mass_fraction": prop_frac,
            "residual": residual,
            "verified": bool(residual < 1e-11),
        }

    return {"domain": "lagrangian_orbital", "subkind": subkind, "verified": False, "error": f"unsupported_lagrangian_orbital:{subkind}"}


def solve_electrochem_or_separation(params: Mapping[str, Any]) -> Dict[str, Any]:
    subkind = str(
        params.get("subkind") or ("distillation" if "alpha" in params else "electrochemistry")
    ).strip().lower()

    if subkind == "electrochemistry":
        e0 = float(params["E0_V"])
        n_el = float(params["n_electrons"])
        t_k = float(params.get("T_K", 298.15))
        q_rxn = float(params.get("Q_reaction", 1.0))
        cur_a = float(params.get("current_A", 0.0))
        r_int = float(params.get("R_internal_ohm", 0.0))
        eta_act = float(params.get("eta_activation_V", 0.0))
        time_s = float(params.get("time_s", 0.0))
        mw_g = float(params.get("molar_mass_g_mol", 0.0))
        eta_f = float(params.get("faradaic_efficiency", 1.0))
        mode = str(params.get("mode", "galvanic")).strip().lower()

        if n_el <= 0.0 or t_k <= 0.0 or q_rxn <= 0.0 or cur_a < 0.0 or r_int < 0.0 or eta_act < 0.0 or not (0.0 < eta_f <= 1.0):
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False, "error": "invalid_electrochem_parameters"}

        e_nernst = e0 - (R_GAS_UNIV * t_k / (n_el * FARADAY_C_MOL)) * math.log(q_rxn)
        dg0 = -n_el * FARADAY_C_MOL * e0
        dg = -n_el * FARADAY_C_MOL * e_nernst
        ln_keq = (n_el * FARADAY_C_MOL * e0) / (R_GAS_UNIV * t_k)

        if mode == "electrolytic":
            v_cell = e_nernst + eta_act + cur_a * r_int
        else:
            v_cell = e_nernst - eta_act - cur_a * r_int

        charge_c = cur_a * time_s
        moles_prod = (eta_f * charge_c) / (n_el * FARADAY_C_MOL)
        mass_g = moles_prod * mw_g

        gibbs_res = abs(dg - (dg0 + R_GAS_UNIV * t_k * math.log(q_rxn))) / max(abs(dg0), R_GAS_UNIV * t_k, 1.0)
        faraday_res = abs(moles_prod * n_el * FARADAY_C_MOL - eta_f * charge_c) / max(charge_c, 1.0)
        residual = float(max(gibbs_res, faraday_res))
        return {
            "domain": "electrochem_separations",
            "subkind": subkind,
            "E_nernst_V": float(e_nernst),
            "V_cell_V": float(v_cell),
            "delta_G0_J_mol": float(dg0),
            "delta_G_J_mol": float(dg),
            "ln_K_eq": float(ln_keq),
            "moles_reacted": float(moles_prod),
            "mass_deposited_g": float(mass_g),
            "residual": residual,
            "verified": bool(residual < 1e-11),
        }

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

        n_min = math.log((x_d * (1.0 - x_b)) / ((1.0 - x_d) * x_b)) / math.log(alpha)

        def underwood_fn(th: float) -> float:
            return (alpha * z_f) / (alpha - th) + (1.0 - z_f) / (1.0 - th) - (1.0 - q)

        eps_b = 1e-12 * (alpha - 1.0)
        theta_brent = float(optimize.brentq(underwood_fn, 1.0 + eps_b, alpha - eps_b, xtol=1e-14, rtol=1e-14))

        a_q = 1.0 - q
        b_q = -((1.0 - q) * (alpha + 1.0) - (alpha - 1.0) * z_f - 1.0)
        c_q = -alpha * q
        if abs(a_q) < 1e-12:
            theta_alg = -c_q / b_q
        else:
            disc = math.sqrt(max(0.0, b_q * b_q - 4.0 * a_q * c_q))
            r_plus = (-b_q + disc) / (2.0 * a_q)
            r_minus = (-b_q - disc) / (2.0 * a_q)
            theta_alg = r_plus if (1.0 < r_plus < alpha) else r_minus

        pinch_reflux = (alpha * x_d) / (alpha - theta_brent) + (1.0 - x_d) / (1.0 - theta_brent) - 1.0
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
        x_int, y_int = float(xi), float(yi)
        one_stage = xd * (1 + (volatility - 1) * xb) == volatility * xb
        flash = liquid == 0 and strip_vapor == 0
        if flash and gap:
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False,
                    "error": "zero_circulation_flash_products_not_in_equilibrium", **conditions}
        if not flash and (gap <= 0 or (floor_gap < 0 and r_op <= r_min * (1.0 + 1e-7))):
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

        y_cur = x_d
        x_prev = x_d
        stages = 0
        feed_stage = None
        frac_stages = 0.0
        for step in range(1, 500):
            x_eq = y_cur / (alpha - (alpha - 1.0) * y_cur)
            stages = step
            if feed_stage is None and x_eq <= x_int and x_prev > x_int:
                feed_stage = step
            if one_stage:
                # These exact products share an equilibrium tie line; Fenske equality is attainable.
                frac_stages = 1.0
                break
            if x_eq <= x_b:
                frac = (x_prev - x_b) / (x_prev - x_eq)
                frac_stages = (step - 1) + frac
                break
            x_prev = x_eq
            if strip_vapor == 0 or x_eq > x_int:
                y_cur = (r_op / (r_op + 1.0)) * x_eq + x_d / (r_op + 1.0)
            else:
                # The stripping component balance, defined also when top reflux is zero.
                y_cur = x_b + float(strip_liquid / strip_vapor) * (x_eq - x_b)
        else:
            return {"domain": "electrochem_separations", "subkind": subkind, "verified": False, "error": "mccabe_thiele_did_not_converge"}

        mb_err = abs(d_flow * x_d + b_flow * x_b - f_flow * z_f) / max(f_flow * z_f, 1e-12)
        theta_err = abs(theta_brent - theta_alg) / max(theta_brent, 1.0)
        # Verify the section component balance without dividing by vanishing boilup.
        strip_err = abs(float(strip_vapor) * (y_int - x_b) - float(strip_liquid) * (x_int - x_b)) / f_flow
        residual = float(max(mb_err, theta_err, strip_err))
        return {
            "domain": "electrochem_separations",
            "subkind": subkind,
            "D_mol_s": float(d_flow),
            "B_mol_s": float(b_flow),
            "N_min_fenske": float(n_min),
            "underwood_theta": float(theta_brent),
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
    if any(k in s for k in ("nernst", "faraday", "overpotential", "electrochem", "mccabe-thiele", "mccabe", "fenske", "underwood", "distillation")):
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
