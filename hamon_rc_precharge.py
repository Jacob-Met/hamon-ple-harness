"""Passive precharge of an initially discharged capacitor from a constant DC bus."""
from fractions import Fraction
import math
from mpmath import mp


def rc_precharge_state(spec):
    def quantity(key):
        raw = spec[key]
        if isinstance(raw, bool):
            raise ValueError(f"{key} must be finite")
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError(f"{key} must be finite")
        return Fraction(v)

    def wire(x):
        try:
            value = float(x)
        except OverflowError as exc:
            raise ValueError("RC result lies outside the finite output range") from exc
        if not math.isfinite(value) or (x and not value):
            raise ValueError("RC result lies outside the finite output range")
        return value

    bus, microfarads, resistance, f = map(quantity, (
        "V_bus_V", "C_uF", "R_precharge_ohm", "target_fraction"))
    if min(resistance, microfarads) <= 0 or not 0 <= f <= 1:
        raise ValueError("Passive RC precharge requires R,C > 0 and a target fraction in [0,1]")
    capacitance = microfarads/1_000_000
    tau = resistance*capacitance
    remaining = 1-f
    capacitor_v = f*bus
    resistor_v = remaining*bus
    charge = capacitance*capacitor_v
    full_energy = capacitance*bus**2/2
    target_energy = charge*capacitor_v/2
    source_work = bus*charge
    # Integral of R I(t)^2 from start to the endpoint, written as f(2-f)
    # so a small delivered fraction survives subtraction from the full energy.
    heat = full_energy*f*(2-f)

    ctx = mp.clone()
    ctx.prec = 180

    def real(x):
        return ctx.mpf(x.numerator)/x.denominator

    def rel(a,b):
        scale = max(abs(a),abs(b))
        return abs(a-b)/scale if scale else 0

    if not f or not bus:
        time_s, status = 0., "already_at_target"
    elif not remaining:
        time_s, status = None, "asymptotic"
    else:
        time_s, status = wire(-real(tau)*ctx.log1p(-real(f))), "finite"

    result = dict(tau_s=wire(tau), t_target_s=time_s, target_time_status=status,
                  stored_energy_J=wire(full_energy), target_capacitor_energy_J=wire(target_energy),
                  target_capacitor_voltage_V=wire(capacitor_v), target_resistor_voltage_V=wire(resistor_v),
                  target_capacitor_charge_C=wire(charge), target_current_A=wire(resistor_v/resistance),
                  source_work_J=wire(source_work), resistor_heat_J=wire(heat))

    # Check the published time against both voltage coordinates and the
    # integral of resistor power. Separately replay charge, KVL and the work
    # balance using the actual reported SI values.
    V, Vr, Q, I, U, W, H = (Fraction(result[k]) for k in (
        "target_capacitor_voltage_V", "target_resistor_voltage_V", "target_capacitor_charge_C",
        "target_current_A", "target_capacitor_energy_J", "source_work_J", "resistor_heat_J"))
    errors = [rel(Fraction(result["tau_s"]),tau), rel(V+Vr,bus), rel(resistance*I,Vr),
              rel(Q,capacitance*V), rel(2*U,Q*V), rel(W,bus*Q), rel(W,U+H)]
    if time_s is not None and bus:
        elapsed = ctx.mpf(time_s)/real(tau)
        errors.extend((rel(-ctx.expm1(-elapsed),real(f)), rel(ctx.exp(-elapsed),real(remaining)),
                       rel(-real(full_energy)*ctx.expm1(-2*elapsed),real(H))))
    error = max(map(float, errors))
    result.update(dual_rel_discrepancy=error, dual_verified=error < 1e-10)
    return result
