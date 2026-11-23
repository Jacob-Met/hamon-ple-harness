"""Passive RC transient from a specified initial voltage toward a constant DC bus."""
from fractions import Fraction
import math
from mpmath import mp


def rc_precharge_state(spec):
    publication_errors = []
    ctx = mp.clone()
    ctx.prec = 180

    def quantity(key):
        raw = spec[key]
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be finite")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return Fraction(value)

    def real(value):
        return ctx.mpf(value.numerator) / value.denominator

    def wire(value):
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("RC result lies outside the finite output range") from exc
        if not math.isfinite(result) or (value and not result):
            raise ValueError("RC result lies outside the finite output range")
        if value:
            reference = real(value) if isinstance(value, Fraction) else value
            publication_errors.append(abs(ctx.mpf(result) / reference - 1))
        return result

    def rel(a, b):
        scale = max(abs(a), abs(b))
        return abs(a - b) / scale if scale else 0

    bus, microfarads, resistance = map(quantity, ("V_bus_V", "C_uF", "R_precharge_ohm"))
    if min(resistance, microfarads) <= 0:
        raise ValueError("Passive RC precharge requires R,C > 0")
    initial = quantity("V_initial_V") if "V_initial_V" in spec else Fraction(0)
    if ("target_fraction" in spec) == ("V_target_V" in spec):
        raise ValueError("Specify exactly one of target_fraction or V_target_V")
    if "target_fraction" in spec:
        fraction = quantity("target_fraction")
        if not 0 <= fraction <= 1:
            raise ValueError("target_fraction must be in [0,1]")
        target = fraction * bus
    else:
        target = quantity("V_target_V")

    capacitance = microfarads / 1_000_000
    tau = resistance * capacitance
    gap = bus - initial
    rise = target - initial
    remaining = bus - target
    initial_energy = capacitance * initial ** 2 / 2
    result = dict(tau_s=wire(tau), stored_energy_J=wire(capacitance * bus ** 2 / 2),
                  initial_capacitor_voltage_V=wire(initial), initial_capacitor_energy_J=wire(initial_energy),
                  requested_capacitor_voltage_V=wire(target))
    if not rise:
        status, time_s = "already_at_target", 0.
    elif not gap or not 0 < rise / gap <= 1:
        result.update(target_time_status="unreachable", t_target_s=None, dual_verified=False,
                      dual_rel_discrepancy=None, target_capacitor_voltage_V=None, target_resistor_voltage_V=None,
                      target_capacitor_charge_C=None, target_current_A=None, target_capacitor_energy_J=None,
                      capacitor_energy_change_J=None, source_work_J=None, resistor_heat_J=None)
        return result
    elif not remaining:
        status, time_s = "asymptotic", None
    else:
        # Select the well-resolved coordinate both near t=0 and near equilibrium.
        fraction = rise / gap
        elapsed = -ctx.log1p(-real(fraction)) if fraction <= Fraction(1, 2) else -ctx.log(real(remaining / gap))
        status, time_s = "finite", wire(real(tau) * elapsed)

    charge = capacitance * target
    energy_change = capacitance * rise * (target + initial) / 2
    work = bus * capacitance * rise
    # Integral R I² dt, using the factored difference of squared voltage gaps.
    heat = capacitance * rise * (gap + remaining) / 2
    result.update(target_time_status=status, t_target_s=time_s, target_capacitor_voltage_V=wire(target),
                  target_resistor_voltage_V=wire(remaining), target_current_A=wire(remaining / resistance),
                  target_capacitor_charge_C=wire(charge), target_capacitor_energy_J=wire(charge * target / 2),
                  capacitor_energy_change_J=wire(energy_change), source_work_J=wire(work), resistor_heat_J=wire(heat))

    voltage, resistor_v, current, q, energy, delta, source, loss = (Fraction(result[k]) for k in (
        "target_capacitor_voltage_V", "target_resistor_voltage_V", "target_current_A", "target_capacitor_charge_C",
        "target_capacitor_energy_J", "capacitor_energy_change_J", "source_work_J", "resistor_heat_J"))
    # Scale balance residuals by the terms in the sum, including when the ideal
    # bus supplies zero work and heating exactly consumes previous stored energy.
    def balance(left, first, second):
        scale = max(abs(left), abs(first), abs(second))
        return abs(left - first - second) / scale if scale else 0

    errors = [balance(bus, voltage, resistor_v), rel(resistance * current, resistor_v),
              rel(q, capacitance * voltage), rel(2 * energy, q * voltage),
              balance(energy, Fraction(result["initial_capacitor_energy_J"]), delta),
              balance(source, delta, loss)]
    if status == "asymptotic":
        errors.extend((rel(source, bus * capacitance * gap), rel(loss, capacitance * gap ** 2 / 2)))
    elif gap:
        elapsed = ctx.mpf(time_s) / real(tau)
        errors.extend((rel(-ctx.expm1(-elapsed), real(rise / gap)),
                       rel(ctx.exp(-elapsed), real(remaining / gap)),
                       rel(-real(bus * capacitance * gap) * ctx.expm1(-elapsed), real(source)),
                       rel(-real(capacitance * gap ** 2 / 2) * ctx.expm1(-2 * elapsed), real(loss))))
    error = max(map(float, errors + publication_errors))
    result.update(dual_rel_discrepancy=error, dual_verified=error < 1e-10)
    return result
