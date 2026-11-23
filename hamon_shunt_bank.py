"""Size one balanced ideal shunt bank for a positive three-phase real load."""
import math
from fractions import Fraction

import mpmath


def _input(spec, name):
    value = spec[name]
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except (ValueError, OverflowError, TypeError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction(value)


def signed_shunt_bank(spec):
    p, voltage, frequency, initial_pf, target_pf = (
        _input(spec, name) for name in ("P_kW", "V_ll_V", "freq_Hz", "pf_initial", "pf_target"))
    initial = spec.get("initial_pf_direction", "lagging")
    target = spec.get("target_pf_direction", "lagging")
    connection = spec.get("bank_connection", "delta")
    if min(p, voltage, frequency) <= 0 or not (0 < initial_pf <= 1 and 0 < target_pf <= 1):
        raise ValueError("bank sizing requires positive real load, voltage, frequency and PF in (0,1]")
    if initial not in ("lagging", "leading") or target not in ("lagging", "leading"):
        raise ValueError("initial_pf_direction and target_pf_direction must be lagging or leading")
    if connection not in ("delta", "wye"):
        raise ValueError("bank_connection must be delta or wye")

    ctx = mpmath.mp.clone()
    ctx.dps = 90

    def number(q):
        return ctx.mpf(q.numerator) / q.denominator

    def report(value):
        shown = float(value)
        if not math.isfinite(shown) or (value and
                (not shown or abs(ctx.mpf(shown)/value-1) > ctx.mpf("1e-10"))):
            raise ValueError("bank quantity is outside the finite reporting resolution")
        return shown

    # Q/P = +/-tan(phi). Keep tan² as an exact rational until after
    # subtraction, so adjacent PF settings cannot collapse to a zero bank.
    i2, t2 = 1/initial_pf**2-1, 1/target_pf**2-1
    i_tan, t_tan = ctx.sqrt(number(i2)), ctx.sqrt(number(t2))
    i_sign = 1 if initial == "lagging" else -1
    t_sign = 1 if target == "lagging" else -1
    power = number(p)
    q_initial, q_target = power*i_sign*i_tan, power*t_sign*t_tan
    if i_sign == t_sign and i_tan + t_tan:
        q_shunt = power*t_sign*number(t2-i2)/(t_tan+i_tan)
    else:
        q_shunt = q_target-q_initial

    # Leg voltage squared is exact for both connections. Inductors absorb
    # reactive power and capacitors supply it in the passive sign convention.
    v2 = voltage**2 if connection == "delta" else voltage**2/3
    omega = 2*ctx.pi*number(frequency)
    v_sq = number(v2)
    c_leg = l_leg = None
    delta_c = 0.0
    if q_shunt < 0:
        bank = "capacitor"
        capacitance = -q_shunt*1000/(3*omega*v_sq)
        c_leg = report(capacitance)
        delta_c = report(-q_shunt*1000/(3*omega*number(voltage**2)))
        leg_admittance = ctx.j*omega*ctx.mpf(c_leg)
    elif q_shunt > 0:
        bank = "reactor"
        l_leg = report(3*v_sq/(omega*q_shunt*1000))
        delta_c = None
        leg_admittance = 1/(ctx.j*omega*ctx.mpf(l_leg))
    else:
        bank = "none"
        leg_admittance = ctx.mpc(0)

    # Independently replay S = 3 V_leg conj(I_leg) with the component value
    # actually reported, and check the change on the bank's own VAR scale.
    leg_voltage = ctx.sqrt(v_sq)
    leg_current = leg_admittance*leg_voltage
    replay = 3*leg_voltage*ctx.conj(leg_current)/1000
    if replay.real:
        raise ValueError("a passive ideal shunt must exchange zero real power")
    error = abs(replay.imag/q_shunt-1) if q_shunt else abs(replay.imag)
    if error > ctx.mpf("1e-10"):
        raise ValueError("published shunt bank fails the reactive power balance")
    return {
        "Q_initial_kvar": report(q_initial), "Q_target_kvar": report(q_target),
        "Qc_required_kvar": report(-q_shunt), "Q_shunt_kvar": report(q_shunt),
        "initial_pf_direction": initial if initial_pf < 1 else "unity",
        "target_pf_direction": target if target_pf < 1 else "unity",
        "bank_type": bank, "bank_connection": connection,
        "bank_leg_voltage_V": report(leg_voltage),
        "bank_capacitance_per_leg_F": c_leg, "bank_inductance_per_leg_H": l_leg,
        "C_equiv_delta_per_phase_F": delta_c,
        "Q_shunt_replay_kvar": report(replay.imag),
        "relative_replay_error": float(error), "dual_verified": True,
    }
