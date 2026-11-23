"""Balanced series feeder with specified receiving-end current magnitude and power factor."""
import math
from decimal import Decimal, localcontext
from fractions import Fraction


def _number(spec, key, default=None):
    value = spec.get(key, default)
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} must be finite") from None
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return Fraction(value)


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _linear_surd(r, t, d):
    """r + t sqrt(d); conjugate opposite-signed terms before rounding."""
    if not t or not d:
        return _decimal(r)
    term = _decimal(t) * _decimal(d).sqrt()
    if r*t < 0:
        return _decimal(r*r-t*t*d) / (_decimal(r)-term)
    return _decimal(r) + term


def _with_sqrt_three(r, u, v, d):
    """r + sqrt(3) (u + v sqrt(d)), retaining exact zeros and signs."""
    inner = _linear_surd(u, v, d)
    term = Decimal(3).sqrt() * inner
    if (r < 0 < inner) or (inner < 0 < r):
        numerator = _linear_surd(r*r-3*(u*u+v*v*d), -6*u*v, d)
        return numerator / (_decimal(r)-term)
    return _decimal(r) + term


def _report(value):
    shown = float(value)
    if not math.isfinite(shown) or (value and not shown):
        raise ValueError("feeder result exceeds the finite nonzero reporting range")
    return shown


def three_phase_feeder(spec):
    voltage, current, pf = (_number(spec,k) for k in ("V_ll_V","I_A","pf"))
    distance = _number(spec, "length_m") / 1000
    resistance, reactance = (_number(spec,k) for k in ("R_ohm_per_km","X_ohm_per_km"))
    limit = _number(spec, "max_vdrop_pct", 3) / 100
    reference = spec.get("voltage_reference", "sending")
    direction = spec.get("power_factor_direction", "lagging")
    branch = spec.get("receiving_voltage_branch", "high")
    if min(current, distance, resistance, limit) < 0 or voltage <= 0 or not 0 <= pf <= 1:
        raise ValueError("positive voltage, nonnegative current, length, resistance and limit, and PF in [0,1] required")
    if reference not in ("sending","receiving") or direction not in ("lagging","leading"):
        raise ValueError("specify sending or receiving voltage and lagging or leading receiving-end current")
    if branch not in ("high","low") or (reference == "receiving" and "receiving_voltage_branch" in spec):
        raise ValueError("high/low receiving_voltage_branch applies to a prescribed sending voltage")

    # In the receiving-voltage reference frame, Vs/Vbase = Vr/Vbase + a + jb.
    # Normalize by the specified line-line voltage. Current is a phase current.
    k = current * distance / voltage
    d = 1-pf*pf
    reactive_sign = 1 if direction == "lagging" else -1
    u, v = k*resistance*pf, k*reactance*reactive_sign
    squared_drop = 3*k*k*(resistance*resistance+reactance*reactance)

    with localcontext() as ctx:
        ctx.prec = 90
        def plus_a(r, coefficient=1):
            return _with_sqrt_three(r, coefficient*u, coefficient*v, d)

        a = plus_a(0)
        b = _with_sqrt_three(0, k*reactance*pf, -k*resistance*reactive_sign, d)
        unit = _decimal(voltage)
        if reference == "receiving":
            receiving = Decimal(1)
            sending = plus_a(1+squared_drop, 2).sqrt()
            real, imag = plus_a(1), b
            # Conjugate the difference of terminal magnitudes.
            drop = plus_a(squared_drop, 2) / (sending+1)
            passes = plus_a(squared_drop-2*limit-limit*limit, 2) <= 0
            candidates, selected = [receiving], "specified"
        else:
            # The two real projections on the sending-voltage circle are
            # +/-sqrt(1-b²). Classify tangency using exact input coefficients.
            circle = _linear_surd(1-squared_drop+3*(u*u+v*v*d), 6*u*v, d)
            infeasible = dict(dual_verified=False, feasible=False, passes_3pct=False,
                              positive_receiving_voltages_ll_V=[], voltage_reference=reference)
            if circle < 0:
                return dict(infeasible, state="no_real_receiving_voltage")
            root = circle.sqrt()

            def offsets(h):
                # h - Vr_high and h - Vr_low. Their product is the exact
                # receiving-voltage quadratic evaluated at h. Conjugation
                # recovers a small root, voltage change or limit margin.
                center = plus_a(h)
                product = plus_a(h*h+squared_drop-1, 2*h)
                if center >= 0:
                    low_offset = center + root
                    return (product/low_offset if low_offset else Decimal(0)), low_offset
                high_offset = center - root
                return high_offset, product/high_offset

            high_offset, low_offset = offsets(0)
            high, low = -high_offset, -low_offset
            if high <= 0:
                return dict(infeasible, state="no_positive_receiving_voltage")
            candidates = [low, high] if root and low > 0 else [high]
            if branch == "low" and (low <= 0 or not root):
                raise ValueError("no distinct positive low receiving-voltage branch")
            index = 0 if branch == "high" else 1
            receiving = (high, low)[index]
            sending = Decimal(1)
            real, imag = (root if index == 0 else -root), b
            drop, passes = offsets(1)[index], offsets(1-limit)[index] <= 0
            selected = "tangent" if not root else branch

        vs, vr = _report(sending*unit), _report(receiving*unit)
        sx, sy = _report(real*unit), _report(imag*unit)
        choices = [_report(value*unit) for value in candidates]
        if len(choices) == 2 and choices[0] == choices[1]:
            raise ValueError("the two positive feeder voltages are not separately reportable")
        # Check complex KVL and terminal magnitude again from the rounded wire
        # state. The separate signed drop is retained even for a sub-ULP change.
        sx_d, sy_d, vr_d, vs_d = map(Decimal.from_float,(sx,sy,vr,vs))
        scale = max(vr_d,vs_d)
        residual = max(abs(sx_d-vr_d-unit*a), abs(sy_d-unit*b),
                       abs((sx_d*sx_d+sy_d*sy_d).sqrt()-vs_d)) / scale
        if residual > Decimal("1e-10"):
            raise ValueError("rounded feeder terminal phasors fail complex KVL")
        return {
            "voltage_reference":reference, "power_factor_direction":direction,
            "selected_receiving_branch":selected, "positive_receiving_voltages_ll_V":choices,
            "V_sending_ll_V":vs, "V_receiving_ll_V":vr, "V_sending_phasor_ll_V":[sx,sy],
            "vdrop_ll_V":_report(drop*unit), "vdrop_pct":_report(100*drop),
            "longitudinal_projection_ll_V":_report(a*unit),
            "relative_kvl_error":float(residual), "passes_3pct":passes,
            "feasible":True, "dual_verified":True,
        }
