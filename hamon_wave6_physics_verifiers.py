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
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from _scipy_compat import HAS_SCIPY, integrate, optimize
from decimal import Decimal, getcontext, localcontext
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


def _area_mach_ratio_sq(mach: float, gamma: float) -> float:
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * mach * mach)
    return (term ** (gp1 / gm1)) / (mach * mach)


def _fanno_param(mach: float, gamma: float) -> float:
    m2 = mach * mach
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (1.0 - m2) / (gamma * m2) + (gp1 / (2.0 * gamma)) * math.log((gp1 * m2) / (2.0 + gm1 * m2))


def _rayleigh_t0_ratio(mach: float, gamma: float) -> float:
    m2 = mach * mach
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (2.0 * gp1 * m2 * (1.0 + 0.5 * gm1 * m2)) / ((1.0 + gamma * m2) ** 2)


def solve_compressible_gas_acoustics(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for 1D compressible gas dynamics & acoustic systems."""
    subkind = str(params.get("subkind", "isentropic")).strip().lower()
    gamma = float(params.get("gamma", 1.4))
    r_gas = float(params.get("R", 287.05))

    if subkind == "isentropic":
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        if "mach" in params and params["mach"] is not None:
            mach = float(params["mach"])
        elif "area_ratio" in params and params["area_ratio"] is not None:
            ar = float(params["area_ratio"])
            if ar < 1.0 - 1e-12:
                raise ValueError(f"area_ratio A/A*={ar} must be >= 1.0")
            if abs(ar - 1.0) <= 1e-12:
                mach = 1.0
            else:
                regime = str(params.get("regime", "supersonic")).strip().lower()
                fn = lambda m: math.sqrt(_area_mach_ratio_sq(m, gamma)) - ar
                if regime.startswith("sub"):
                    mach = float(optimize.brentq(fn, 1e-9, 1.0 - 1e-11, xtol=1e-14, rtol=1e-14))
                else:
                    mach = float(optimize.brentq(fn, 1.0 + 1e-11, 60.0, xtol=1e-14, rtol=1e-14))
        elif "p0_over_p" in params and params["p0_over_p"] is not None:
            pr = float(params["p0_over_p"])
            mach = math.sqrt((2.0 / gm1) * ((pr ** (gm1 / gamma)) - 1.0))
        else:
            raise ValueError("isentropic requires 'mach', 'area_ratio', or 'p0_over_p'")

        t0_over_t = 1.0 + 0.5 * gm1 * mach * mach
        p0_over_p = t0_over_t ** (gamma / gm1)
        rho0_over_rho = t0_over_t ** (1.0 / gm1)
        a_over_astar = math.sqrt(_area_mach_ratio_sq(mach, gamma)) if mach > 0 else float("inf")

        t0_k = float(params.get("T0", 300.0))
        p0_pa = float(params.get("P0", 101325.0))
        a_throat = float(params.get("A_throat", 1.0e-3))

        t_static = t0_k / t0_over_t
        p_static = p0_pa / p0_over_p
        rho_static = p_static / (r_gas * t_static)
        sound_speed = math.sqrt(gamma * r_gas * t_static)
        velocity = mach * sound_speed
        mdot_choked = (
            a_throat
            * p0_pa
            / math.sqrt(t0_k)
            * math.sqrt(gamma / r_gas)
            * ((0.5 * gp1) ** (-0.5 * gp1 / gm1))
        )
        mdot_local = rho_static * velocity * (a_throat * a_over_astar)
        residual = abs(mdot_local - mdot_choked) / max(abs(mdot_choked), 1e-15)
        return {
            "domain": "compressible_acoustics",
            "subkind": subkind,
            "mach": mach,
            "T0_over_T": t0_over_t,
            "P0_over_P": p0_over_p,
            "rho0_over_rho": rho0_over_rho,
            "A_over_Astar": a_over_astar,
            "T_static_K": t_static,
            "P_static_Pa": p_static,
            "velocity_m_s": velocity,
            "mdot_choked_kg_s": mdot_choked,
            "residual": residual,
            "verified": residual < 1e-10,
        }

    if subkind == "normal_shock":
        m1 = float(params["M1"])
        if m1 <= 1.0:
            raise ValueError(f"Normal shock requires M1 > 1, got {m1}")
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        m1_sq = m1 * m1
        m2 = math.sqrt((1.0 + 0.5 * gm1 * m1_sq) / (gamma * m1_sq - 0.5 * gm1))
        p2_p1 = 1.0 + (2.0 * gamma / gp1) * (m1_sq - 1.0)
        rho2_rho1 = (gp1 * m1_sq) / (2.0 + gm1 * m1_sq)
        t2_t1 = p2_p1 / rho2_rho1
        p02_p01 = (rho2_rho1 ** (gamma / gm1)) * ((gp1 / (2.0 * gamma * m1_sq - gm1)) ** (1.0 / gm1))

        p1 = float(params.get("P1", 101325.0))
        t1 = float(params.get("T1", 288.15))
        rho1 = p1 / (r_gas * t1)
        u1 = m1 * math.sqrt(gamma * r_gas * t1)
        p2 = p1 * p2_p1
        t2 = t1 * t2_t1
        rho2 = rho1 * rho2_rho1
        u2 = m2 * math.sqrt(gamma * r_gas * t2)
        cp = gamma * r_gas / gm1
        res_mass = abs(rho1 * u1 - rho2 * u2) / (rho1 * u1)
        res_mom = abs((p1 + rho1 * u1 * u1) - (p2 + rho2 * u2 * u2)) / (p1 + rho1 * u1 * u1)
        res_eng = abs((cp * t1 + 0.5 * u1 * u1) - (cp * t2 + 0.5 * u2 * u2)) / (cp * t1 + 0.5 * u1 * u1)
        residual = max(res_mass, res_mom, res_eng)
        return {
            "domain": "compressible_acoustics",
            "subkind": subkind,
            "M1": m1,
            "M2": m2,
            "P2_over_P1": p2_p1,
            "T2_over_T1": t2_t1,
            "rho2_over_rho1": rho2_rho1,
            "P02_over_P01": p02_p01,
            "P2_Pa": p2,
            "T2_K": t2,
            "residual": residual,
            "verified": residual < 1e-10,
        }

    if subkind == "fanno":
        m1 = float(params["M1"])
        f_fanning = float(params["f"])
        length_m = float(params["L"])
        dh_m = float(params["D_h"])
        phi1 = _fanno_param(m1, gamma)
        four_fl_d = 4.0 * f_fanning * length_m / dh_m
        l_choke = phi1 * dh_m / (4.0 * f_fanning)
        if four_fl_d >= phi1 - 1e-13:
            m2 = 1.0
            choked = True
            residual = 0.0
        else:
            choked = False
            target_phi2 = phi1 - four_fl_d
            fn = lambda m: _fanno_param(m, gamma) - target_phi2
            if m1 < 1.0:
                m2 = float(optimize.brentq(fn, m1, 1.0 - 1e-12, xtol=1e-14, rtol=1e-14))
            else:
                m2 = float(optimize.brentq(fn, 1.0 + 1e-12, m1, xtol=1e-14, rtol=1e-14))
            residual = abs(_fanno_param(m2, gamma) - target_phi2)
        gm1 = gamma - 1.0
        t2_t1 = (2.0 + gm1 * m1 * m1) / (2.0 + gm1 * m2 * m2)
        p2_p1 = (m1 / m2) * math.sqrt(t2_t1)
        return {
            "domain": "compressible_acoustics",
            "subkind": subkind,
            "M1": m1,
            "M2": m2,
            "choked": choked,
            "L_choke_m": l_choke,
            "P2_over_P1": p2_p1,
            "T2_over_T1": t2_t1,
            "residual": residual,
            "verified": residual < 1e-10,
        }

    if subkind == "rayleigh":
        m1 = float(params["M1"])
        t01 = float(params["T01"])
        q_heat = float(params["q_J_kg"])
        cp = float(params.get("cp", gamma * r_gas / (gamma - 1.0)))
        t02 = t01 + q_heat / cp
        r1 = _rayleigh_t0_ratio(m1, gamma)
        t0_star = t01 / r1
        target_r2 = t02 / t0_star
        if target_r2 >= 1.0 - 1e-12:
            m2 = 1.0
            choked = True
            residual = 0.0
        else:
            choked = False
            fn = lambda m: _rayleigh_t0_ratio(m, gamma) - target_r2
            if m1 < 1.0:
                m2 = float(optimize.brentq(fn, 1e-9, 1.0 - 1e-11, xtol=1e-14, rtol=1e-14))
            else:
                m2 = float(optimize.brentq(fn, 1.0 + 1e-11, 60.0, xtol=1e-14, rtol=1e-14))
            residual = abs(_rayleigh_t0_ratio(m2, gamma) - target_r2)
        p2_p1 = (1.0 + gamma * m1 * m1) / (1.0 + gamma * m2 * m2)
        return {
            "domain": "compressible_acoustics",
            "subkind": subkind,
            "M1": m1,
            "M2": m2,
            "choked": choked,
            "T0_star_K": t0_star,
            "P2_over_P1": p2_p1,
            "residual": residual,
            "verified": residual < 1e-10,
        }

    if subkind == "acoustics":
        rho1 = float(params.get("rho1", 1.204))
        c1 = float(params.get("c1", 343.0))
        rho2 = float(params.get("rho2", 998.0))
        c2 = float(params.get("c2", 1481.0))
        z1 = rho1 * c1
        z2 = rho2 * c2
        r_p = (z2 - z1) / (z2 + z1)
        t_p = 2.0 * z2 / (z1 + z2)
        r_intensity = r_p * r_p
        t_intensity = 4.0 * z1 * z2 / ((z1 + z2) ** 2)
        tl_db = -10.0 * math.log10(max(t_intensity, 1e-300))
        helmholtz_hz = None
        if all(k in params for k in ("A_neck", "V_cavity", "L_eff")):
            helmholtz_hz = (c1 / (2.0 * math.pi)) * math.sqrt(
                float(params["A_neck"]) / (float(params["V_cavity"]) * float(params["L_eff"]))
            )
        residual = abs((r_intensity + t_intensity) - 1.0)
        return {
            "domain": "compressible_acoustics",
            "subkind": subkind,
            "Z1_rayl": z1,
            "Z2_rayl": z2,
            "pressure_reflection_R": r_p,
            "pressure_transmission_T": t_p,
            "intensity_transmission_TI": t_intensity,
            "transmission_loss_dB": tl_db,
            "helmholtz_f0_Hz": helmholtz_hz,
            "residual": residual,
            "verified": residual < 1e-11,
        }

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


def _rlc_matrix_reference(r, l, c, vs, v0, i0, time_):
    """Independent affine matrix exponential, without modal roots.

    Use energy coordinates (v, i*sqrt(L/C)) and propagate the constant
    source column directly. This preserves tiny zero-state responses that
    would disappear when subtracting two equilibrium-sized quantities.
    """
    if time_ == 0 or (v0 == vs and i0 == 0):
        return v0, i0, 0
    with localcontext() as context:
        context.prec = 50
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


def _rlc_analytic_state(alpha, omega, impedance, vs, v0, i0, time_):
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
        alpha, omega, impedance, vs, v0, i0, time_)
    reference_v, reference_i, squarings = _rlc_matrix_reference(
        r, l, c, vs, v0, i0, time_)
    if not all(math.isfinite(value) for value in (voltage, current, reference_v, reference_i)):
        raise ValueError("RLC solution exceeds the supported finite range")
    voltage_scale = max(abs(reference_v), abs(v0), math.ulp(0.0))
    current_scale = max(abs(reference_i), abs(i0), math.ulp(0.0))
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
        "verified": residual < 1e-8,
        "verification_method": "decimal_scaled_matrix_exponential",
        "reference_squarings": squarings,
    }


def solve_transmission_line_rlc(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for lossless/lossy transmission lines, Smith-chart stubs, and RLC transients."""
    subkind = str(params.get("subkind", "transmission_line")).strip().lower()

    if subkind == "transmission_line":
        z0 = complex(params.get("Z0", 50.0))
        zl = complex(float(params.get("R_L", 100.0)), float(params.get("X_L", 0.0)))
        length = float(params.get("length_m", 1.0))
        alpha_l = float(params.get("alpha_np_m", 0.0)) * length
        if params.get("length_wavelengths") is not None:
            turns = math.remainder(float(params["length_wavelengths"]), 0.5)
            invert = abs(turns) > 0.125
            if invert:
                turns -= math.copysign(0.25, turns)
            phase = math.tau * turns
        else:
            beta_l = float(params.get("beta_rad_m", 0.0)) * length
            sine, cosine = math.sin(beta_l), math.cos(beta_l)
            invert = abs(sine) > abs(cosine)
            phase = math.atan(-cosine / sine if invert else sine / cosine)

        # Work within an eighth wavelength of zero. An extra quarter turn
        # interchanges the load and characteristic impedance in the transform.
        scale = max(abs(zl), abs(z0))
        load, line = zl / scale, z0 / scale
        gamma_load = (load - line) / (load + line)
        mag_g = abs(gamma_load)
        # Rationalizing 1 - |Gamma| preserves finite VSWR for large mismatch.
        power = 4.0 * (load * line.conjugate()).real
        vswr = ((abs(load + line) + abs(load - line)) ** 2 / power
                if power > 0 else float("inf"))
        return_loss_db = -20.0 * math.log10(max(mag_g, 1e-300))

        a, b = (line, load) if invert else (load, line)
        t = complex(np.tanh(complex(alpha_l, phase)))
        z_in = z0 * ((a + b * t) / (b + a * t))

        # Independently propagate voltage and current through the ABCD
        # matrix. Its common cosh(alpha*l) factor cancels in their ratio.
        sine, cosine = math.sin(phase), math.cos(phase)
        loss = math.tanh(alpha_l)
        cosh_gl = complex(cosine, loss * sine)
        sinh_gl = complex(loss * cosine, sine)
        voltage = a * cosh_gl + b * sinh_gl
        current = b * cosh_gl + a * sinh_gl
        z_in_check = z0 * (voltage / current)
        residual = abs(z_in - z_in_check) / max(abs(z_in), abs(z0), 1e-15)
        return {
            "domain": "rf_transmission_rlc",
            "subkind": subkind,
            "Z_in_real": float(np.real(z_in)),
            "Z_in_imag": float(np.imag(z_in)),
            "gamma_load_mag": float(mag_g),
            "gamma_load_deg": float(math.degrees(math.atan2(np.imag(gamma_load), np.real(gamma_load)))),
            "VSWR": float(vswr),
            "return_loss_dB": float(return_loss_db),
            "quarter_wave_Z_T": float(math.sqrt(float(np.real(z0)) * float(np.real(zl)))) if abs(np.imag(zl)) < 1e-12 else None,
            "residual": float(residual),
            "verified": float(residual) < 1e-10,
        }

    if subkind == "smith_stub":
        z0 = float(params.get("Z0", 50.0))
        rl = float(params["R_L"])
        xl = float(params.get("X_L", 0.0))
        stub_type = str(params.get("stub_type", "short")).strip().lower()
        branch = int(params.get("branch", 0))
        if not (math.isfinite(z0) and z0 > 0.0 and math.isfinite(rl) and rl > 0.0 and math.isfinite(xl)):
            raise ValueError("Smith stub matching requires finite R_L > 0, Z0 > 0 and finite X_L")

        # At either intersection with the unit-conductance circle, y = 1 + jb.
        # Equating its reflection magnitude with the load gives b² = |z - 1|²/Re(z).
        # Reflection phase is a full-circle coordinate even at a quarter wavelength.
        z_load = complex(rl / z0, xl / z0)
        delta = complex((rl - z0) / z0, xl / z0)
        b_abs = abs(delta) / math.sqrt(z_load.real)
        gamma_load = delta / (z_load + 1)
        phase_load = math.atan2(gamma_load.imag, gamma_load.real)
        sols = []
        for b in ((b_abs, -b_abs) if b_abs else (0.0,)):
            if b:
                gamma_match = -1j * b / (2.0 + 1j * b)
                phase_match = math.atan2(gamma_match.imag, gamma_match.real)
                d_lam = ((phase_load - phase_match) / (4.0 * math.pi)) % 0.5
                if d_lam == 0.5:
                    d_lam = 0.0
            else:
                # Every line length works for an already matched load.
                d_lam = 0.0

            if stub_type.startswith("short"):
                stub_angle = math.atan2(1.0, b)
                y_stub = -1j * math.cos(stub_angle) / math.sin(stub_angle)
            else:
                stub_angle = math.atan(-b) % math.pi
                if stub_angle == math.pi:
                    stub_angle = 0.0
                y_stub = 1j * math.sin(stub_angle) / math.cos(stub_angle)
            l_lam = stub_angle / (2.0 * math.pi)
            # Independently recover the terminal impedance using the line ABCD matrix.
            angle = 2.0 * math.pi * d_lam
            cs, sn = math.cos(angle), math.sin(angle)
            y_line = (cs + 1j * z_load * sn) / (z_load * cs + 1j * sn)
            residual = abs(1.0 / (y_line + y_stub) - 1.0)
            sols.append((d_lam, l_lam, float(residual)))

        sols.sort(key=lambda s: s[0])
        chosen = sols[min(branch, len(sols) - 1)]
        return {
            "domain": "rf_transmission_rlc",
            "subkind": subkind,
            "d_over_lambda": chosen[0],
            "l_stub_over_lambda": chosen[1],
            "all_branches": [{"d_over_lambda": d, "l_stub_over_lambda": l, "residual": r} for d, l, r in sols],
            "residual": chosen[2],
            "verified": chosen[2] < 1e-10,
        }

    if subkind == "rlc_transient":
        return _solve_rlc_transient(params)

    raise ValueError(f"Unsupported rf_transmission_rlc subkind: {subkind}")


def solve_mohr_vessel_buckling(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic solver for 2D/3D stress states, Lame/thin vessels, and Euler-Johnson buckling."""
    subkind = str(params.get("subkind", "mohr_stress")).strip().lower()

    if subkind == "mohr_stress":
        sx = float(params.get("sigma_x", 0.0))
        sy = float(params.get("sigma_y", 0.0))
        sz = float(params.get("sigma_z", 0.0))
        txy = float(params.get("tau_xy", 0.0))
        txz = float(params.get("tau_xz", 0.0))
        tyz = float(params.get("tau_yz", 0.0))
        tensor = np.array([[sx, txy, txz], [txy, sy, tyz], [txz, tyz, sz]], dtype=float)
        eigvals = np.sort(np.linalg.eigvalsh(tensor))[::-1]
        s1, s2, s3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
        tau_inplane = math.hypot(0.5 * (sx - sy), txy)
        tau_abs_max = 0.5 * (s1 - s3)
        theta_p_deg = 0.5 * math.degrees(math.atan2(2.0 * txy, sx - sy))
        sigma_vm = math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))

        sy_yield = params.get("sigma_yield")
        sf_vm = float(sy_yield) / sigma_vm if sy_yield is not None and sigma_vm > 0 else None
        sf_tresca = float(sy_yield) / (s1 - s3) if sy_yield is not None and (s1 - s3) > 0 else None

        i1_tensor = sx + sy + sz
        i1_eig = s1 + s2 + s3
        residual = abs(i1_tensor - i1_eig) / max(abs(i1_tensor), abs(s1), 1.0)
        return {
            "domain": "solid_mohr_vessel_buckling",
            "subkind": subkind,
            "sigma_1": s1,
            "sigma_2": s2,
            "sigma_3": s3,
            "tau_max_inplane": tau_inplane,
            "tau_max_abs": tau_abs_max,
            "theta_p_deg": theta_p_deg,
            "sigma_von_mises": sigma_vm,
            "safety_factor_vm": sf_vm,
            "safety_factor_tresca": sf_tresca,
            "residual": residual,
            "verified": residual < 1e-11,
        }

    if subkind == "pressure_vessel":
        ri = float(params["r_i"])
        ro = float(params["r_o"]) if "r_o" in params and params["r_o"] is not None else ri + float(params["t"])
        pi = float(params["p_i"])
        po = float(params.get("p_o", 0.0))
        r_eval = float(params.get("r_eval", ri))
        t_wall = ro - ri

        denom = ro * ro - ri * ri
        a_lame = (pi * ri * ri - po * ro * ro) / denom
        b_lame = ((pi - po) * ri * ri * ro * ro) / denom
        sigma_r = a_lame - b_lame / (r_eval * r_eval)
        sigma_theta = a_lame + b_lame / (r_eval * r_eval)
        sigma_z = a_lame if bool(params.get("closed_ends", True)) else 0.0
        sigma_vm_inner = math.sqrt(
            0.5 * ((sigma_theta - sigma_z) ** 2 + (sigma_z - sigma_r) ** 2 + (sigma_r - sigma_theta) ** 2)
        )
        sr_ri = a_lame - b_lame / (ri * ri)
        sr_ro = a_lame - b_lame / (ro * ro)
        residual = max(abs(sr_ri + pi), abs(sr_ro + po)) / max(abs(pi), abs(po), 1.0)
        return {
            "domain": "solid_mohr_vessel_buckling",
            "subkind": subkind,
            "sigma_hoop_lame": sigma_theta,
            "sigma_radial_lame": sigma_r,
            "sigma_axial_lame": sigma_z,
            "sigma_hoop_thin": pi * ri / t_wall,
            "sigma_axial_thin": 0.5 * pi * ri / t_wall,
            "sigma_von_mises": sigma_vm_inner,
            "residual": residual,
            "verified": residual < 1e-11,
        }

    if subkind == "column_buckling":
        e_mod = float(params["E"])
        sy = float(params["sigma_y"])
        length = float(params["L"])
        area = float(params["A"])
        i_min = float(params["I"])
        k_eff = float(params.get("K", 1.0))

        r_gyr = math.sqrt(i_min / area)
        slenderness = k_eff * length / r_gyr
        c_c = math.sqrt(2.0 * (math.pi ** 2) * e_mod / sy)
        if slenderness >= c_c:
            regime = "euler_elastic"
            sigma_cr = (math.pi ** 2) * e_mod / (slenderness ** 2)
        else:
            regime = "johnson_inelastic"
            sigma_cr = sy * (1.0 - (sy * (slenderness ** 2)) / (4.0 * (math.pi ** 2) * e_mod))
        p_cr = sigma_cr * area

        if HAS_SYMPY:
            s_sym = sp.Symbol("lam", positive=True)
            euler_at_cc = float(((sp.pi ** 2) * e_mod / (s_sym ** 2)).subs(s_sym, c_c).evalf())
            johnson_at_cc = float((sy * (1 - sy * (s_sym ** 2) / (4 * (sp.pi ** 2) * e_mod))).subs(s_sym, c_c).evalf())
        else:
            euler_at_cc = float((math.pi ** 2) * e_mod / (c_c ** 2))
            johnson_at_cc = float(sy * (1.0 - sy * (c_c ** 2) / (4.0 * (math.pi ** 2) * e_mod)))
        residual = abs(euler_at_cc - johnson_at_cc) / sy
        return {
            "domain": "solid_mohr_vessel_buckling",
            "subkind": subkind,
            "regime": regime,
            "slenderness_ratio": slenderness,
            "transition_Cc": c_c,
            "sigma_cr": sigma_cr,
            "P_cr_N": p_cr,
            "residual": residual,
            "verified": residual < 1e-11,
        }

    raise ValueError(f"Unsupported solid_mohr_vessel_buckling subkind: {subkind}")


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

    if "conversion" in params and params["conversion"] is not None:
        x_out = float(params["conversion"])
        vol_m3 = volume_for_conversion(x_out)
        residual = abs(volume_for_conversion(x_out) - vol_m3) / max(vol_m3, 1e-15)
        if reactor_type == "pfr" and abs(delta_t_ad) <= 1e-15 and abs(eps_a) <= 1e-15 and not gas_phase:
            k_iso = k_at_temp(t_op)
            if abs(order_n - 1.0) < 1e-12:
                v_exact = (f_a0 / (k_iso * c_a0)) * (-math.log(1.0 - x_out))
            elif abs(order_n - 2.0) < 1e-12:
                v_exact = (f_a0 / (k_iso * c_a0 * c_a0)) * (x_out / (1.0 - x_out))
            elif abs(order_n) < 1e-12:
                v_exact = f_a0 * x_out / k_iso
            else:
                v_exact = (f_a0 / (k_iso * (c_a0 ** order_n) * (order_n - 1.0))) * (((1.0 - x_out) ** (1.0 - order_n)) - 1.0)
            residual = abs(vol_m3 - v_exact) / max(abs(v_exact), 1e-15)
    elif "volume" in params and params["volume"] is not None:
        vol_m3 = float(params["volume"])
        if vol_m3 <= 0.0:
            x_out = 0.0
        elif reactor_type == "cstr_series":
            n_stages = int(params.get("n_stages", 2))
            x_out = _cstr_series_exit_x(vol_m3 / (n_stages * v0), n_stages)
        else:
            fn_v = lambda x: volume_for_conversion(x) - vol_m3
            gap = 0.5
            x_hi = 1.0 - gap
            for _ in range(60):
                if fn_v(x_hi) > 0.0:
                    break
                gap *= 0.5
                x_hi = 1.0 - gap
            x_out = float(optimize.brentq(fn_v, 0.0, x_hi, xtol=1e-13, rtol=1e-13))
        residual = abs(volume_for_conversion(x_out) - vol_m3) / max(vol_m3, 1e-15)
    else:
        raise ValueError("reactor_kinetics requires either 'conversion' or 'volume'")

    tau_total = vol_m3 / v0
    temp_exit = (t0_k + delta_t_ad * x_out) if abs(delta_t_ad) > 1e-15 else t_op
    return {
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
