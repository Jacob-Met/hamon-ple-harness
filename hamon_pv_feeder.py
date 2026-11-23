"""Balanced two-bus, constant-power interconnection through a per-phase series impedance."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math


def solve_pv_interconnection(spec):
    """Use grid V_LL as rated voltage and R+jX per phase from grid to PCC.

    Positive P is export. The specified absorbing factor fixes Q/P, so
    reversing signed P also reverses Q. Nameplate kVA bounds line current
    at grid rated voltage. Voltage limits constrain the signed magnitude change.
    """
    def given(key):
        value = float(spec[key])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite feeder input {key}")
        return Fraction(value)

    v, p, pf, r, x, rating, limit = map(given, (
        "V_ll_V", "P_export_W", "pf_absorbing", "R_feeder_ohm", "X_feeder_ohm",
        "transformer_rating_kVA", "max_vrise_pu"))
    if v <= 0 or not 0 < pf <= 1 or r < 0 or rating <= 0 or limit < 0:
        raise ValueError("Require positive grid voltage, nameplate and PF, PF <= 1, R >= 0 and nonnegative limit")

    # Current is positive from the PCC into the stiff grid. S=P+jQ is the
    # three-phase injection, Q/P=-t, and u is PCC/grid phase voltage.
    # |u|² - conj(u) = Z conj(S)/V_LL² = a + j b. Thus Im(u)=b and
    # Re(u)=(1+sqrt(d))/2 on the root continuous from u=1 at no load.
    t2 = 1/pf**2 - 1
    pr, px = p*r/v**2, p*x/v**2
    a0, a1 = pr, -px                 # a=a0+a1*t, t=sqrt(t2)
    b0, b1 = px, pr
    d0 = 1 + 4*pr - 4*(px**2 + pr**2*t2)
    d1 = -4*px - 8*pr*px

    def sign_of(first, second):
        if not second or not t2:
            return (first > 0) - (first < 0)
        if not first or (first > 0) == (second > 0):
            return 1 if second > 0 else -1
        difference = first**2 - second**2*t2
        return ((difference > 0) - (difference < 0)) * (1 if first > 0 else -1)

    discriminant_sign = sign_of(d0, d1)
    def voltage_squared_vs(bound):
        # Sign of 2(|u|²-bound) = c0+c1*t+sqrt(d). Square only
        # after establishing opposite signs, preserving exact threshold contact.
        c0, c1 = 1 + 2*a0 - 2*bound, 2*a1
        sign = sign_of(c0, c1)
        if sign >= 0:
            return 1 if sign or discriminant_sign else 0
        return sign_of(d0-c0*c0-c1*c1*t2, d1-2*c0*c1)

    def dec(f):
        return Decimal(f.numerator)/Decimal(f.denominator)

    def wire(value):
        result = float(value)
        if not math.isfinite(result) or (value and not result):
            raise ValueError("Feeder result outside the finite binary64 range")
        return result

    with localcontext() as ctx:
        ctx.prec = 90
        t = dec(t2).sqrt()
        def pair(first, second):
            a, b = dec(first), dec(second)*t
            if a*b < 0:
                return dec(first**2-second**2*t2)/(a-b)
            return a+b

        a, b = pair(a0,a1), pair(b0,b1)
        power, reactive = dec(p), -dec(p)*t
        apparent = abs(p)/pf
        result = {"Q_reactive_var": wire(reactive), "S_apparent_kVA": wire(dec(apparent)/1000),
                  "linearized_voltage_rise_pu": wire(a), "power_flow_feasible": discriminant_sign >= 0}
        if discriminant_sign < 0:
            return {**result, "voltage_root_count": 0, "thermal_and_voltage_ok": False,
                    "dual_verified": False, "reason": "No real two-bus constant-power voltage root"}
        d = pair(d0, d1).sqrt()
        re = (1+d)/2
        mag2 = re**2+b**2
        magnitude = mag2.sqrt()
        # Retain the signed magnitude change when a small voltage displacement
        # would round out of the published absolute voltage.
        c = pair(2*a0-1, 2*a1)
        if c < 0:
            numerator = pair(8*a0-4*(pr**2+px**2)*(1+t2),8*a1)
            delta = numerator/(2*(d-c)*(magnitude+1))
        else:
            delta = (d+c)/(2*(magnitude+1))

        phase = dec(v)/Decimal(3).sqrt()
        scale = 3*phase*mag2
        ir, ii = (power*re+reactive*b)/scale, (power*b-reactive*re)/scale
        i2 = ir*ir+ii*ii
        loss, line_q = 3*i2*dec(r), 3*i2*dec(x)
        pg, qg = 3*phase*ir, -3*phase*ii
        thermal = voltage_squared_vs((apparent/(1000*rating))**2) >= 0
        voltage = ((limit >= 1 or voltage_squared_vs((1-limit)**2) >= 0)
                   and voltage_squared_vs((1+limit)**2) <= 0)

        # Independently reconstruct V=PCC and both terminal powers from
        # the published voltage and current phasors, rather than the discriminant.
        vr, vi, cr, ci = map(lambda h: Decimal.from_float(wire(h)), (re,b,ir,ii))
        kvl = max(abs(phase*(vr-1) - dec(r)*cr + dec(x)*ci),
                  abs(phase*vi - dec(r)*ci - dec(x)*cr))
        replayed_p = 3*phase*(vr*cr+vi*ci)
        replayed_q = 3*phase*(vi*cr-vr*ci)
        residual = max(kvl/(phase*max(1,magnitude)),
                       max(abs(power-replayed_p),abs(reactive-replayed_q)) / (dec(apparent) or 1))
        verified = residual < Decimal("1e-10")
        count = 1 if not discriminant_sign or not p or not (r or x) else 2
        return {**result, "voltage_root_count":count,
                "selected_voltage_root":"larger_magnitude" if count == 2 else "unique",
                "pcc_voltage_pu":[float(vr),float(vi)], "pcc_voltage_magnitude_pu":wire(magnitude),
                "voltage_rise_pu":wire(delta), "line_current_A":[float(cr),float(ci)],
                "grid_received_W":wire(pg), "grid_received_var":wire(qg),
                "feeder_loss_W":wire(loss), "feeder_reactive_absorption_var":wire(line_q),
                "grid_terminal_kVA":wire(dec(apparent)/(1000*magnitude)),
                "thermal_ok":thermal, "voltage_ok":voltage,
                "thermal_and_voltage_ok":bool(verified and thermal and voltage),
                "phasor_residual_relative":float(residual), "dual_verified":bool(verified)}
