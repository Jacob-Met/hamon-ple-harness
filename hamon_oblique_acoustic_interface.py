"""Oblique pressure and normal power at an interface between two lossless fluid half-spaces."""
from fractions import Fraction
import math

from mpmath.ctx_iv import MPIntervalContext


def _direction(params):
    angle = "incidence_angle_deg" in params
    if angle == ("incidence_sine" in params):
        raise ValueError("Supply exactly one direction: incidence_angle_deg or incidence_sine")
    raw = params["incidence_angle_deg" if angle else "incidence_sine"]
    if type(raw) not in (int, float) or (type(raw) is float and not math.isfinite(raw)):
        raise ValueError("Incidence direction must be a finite JSON number")
    direction = Fraction(raw)
    limit = 90 if angle else 1
    if abs(direction) >= limit:
        raise ValueError("Use a direction strictly inside the incident hemisphere: grazing has zero incident normal power")
    if not angle:
        return direction, direction**2, False
    # For these exact degree observations sin² is rational, including critical contacts.
    exact = {0: Fraction(0), 30: Fraction(1,4), 45: Fraction(1,2), 60: Fraction(3,4)}
    return direction, exact.get(abs(direction)), True


def _endpoint(value):
    sign, mantissa, exponent, bits = value._mpi_[0]
    if bits < 0 or exponent + bits > 1025:
        raise ValueError("Acoustic field exceeds the finite output range")
    if not mantissa or exponent + bits < -1200:
        return 0.
    exact = Fraction(mantissa) * Fraction(2)**exponent
    return float(-exact if sign else exact)


def _publish(iv, values):
    shown = {}
    for name, value in values.items():
        low, high = _endpoint(value.a), _endpoint(value.b)
        if low != high:
            return None
        if not math.isfinite(low):
            raise ValueError(f"{name} exceeds the finite output range")
        if value != 0 and not (abs(iv.mpf(low)-value) <= abs(value)/10**12):
            return None
        shown[name] = low
    return shown


def oblique_interface(params, rho1, c1, rho2, c2):
    direction, exact_sine_squared, degrees = _direction(params)
    z1, z2 = rho1*c1, rho2*c2
    k1, k2 = 1/z1**2, 1/z2**2
    slope = 1/(c1*rho2)**2
    for digits in (70, 150, 330, 700, 1500):
        iv = MPIntervalContext()
        iv.dps = digits

        def number(value):
            return iv.mpf(value.numerator)/value.denominator

        if degrees:
            sine = iv.sin(number(direction/180)*iv.pi)
        else:
            sine = number(direction)
        if exact_sine_squared is not None:
            sine = iv.sqrt(number(exact_sine_squared)) * (-1 if direction < 0 else 1)

        def affine(constant, coefficient):
            # Combine common terms exactly before conversion, including equal media.
            if exact_sine_squared is not None:
                return number(constant + coefficient*exact_sine_squared)
            return number(constant) + number(coefficient)*sine**2

        incident_sq = affine(k1, -k1)
        transmitted_sq = affine(k2, -slope)
        if not (incident_sq > 0):
            continue
        incident = iv.sqrt(incident_sq)
        difference = affine(k1-k2, slope-k1)
        total = affine(k1+k2, -k1-slope)
        zero, one = iv.mpf(0), iv.mpf(1)
        if transmitted_sq > 0:
            regime = "propagating"
            transmitted = iv.sqrt(transmitted_sq)
            scale = incident + transmitted
            rr, ri = difference/scale**2, zero
            tr, ti = 2*incident/scale, zero
            ur, ui = 2*transmitted/scale, zero
            reflection = rr**2
            power = 4*incident*transmitted/scale**2
            loss = zero if difference == 0 else -10*iv.log(power)/iv.log(10)
            slowness_real, slowness_imag = transmitted*number(rho2), zero
        elif transmitted_sq < 0:
            regime = "evanescent"
            decay = iv.sqrt(-transmitted_sq)
            # With exp(i*k_n*x - i*omega*t), +imag(k_n) decays into medium 2.
            rr, ri = total/difference, -2*incident*decay/difference
            tr, ti = 2*incident_sq/difference, ri
            ur, ui = -2*transmitted_sq/difference, -ri
            reflection, power, loss = one, zero, None
            slowness_real, slowness_imag = zero, decay*number(rho2)
        elif transmitted_sq == 0:
            regime = "critical"
            rr, ri, tr, ti, ur, ui = one, zero, 2*one, zero, zero, zero
            reflection, power, loss = one, zero, None
            slowness_real = slowness_imag = zero
        else:
            continue
        values = dict(rr=rr, ri=ri, tr=tr, ti=ti, ur=ur, ui=ui,
                      intensity_reflection=reflection, intensity_transmission_TI=power,
                      Z1_rayl=number(z1), Z2_rayl=number(z2),
                      tangential_slowness_s_m=sine/number(c1),
                      incident_normal_slowness_s_m=incident*number(rho1),
                      transmitted_slowness_real=slowness_real, transmitted_slowness_imag=slowness_imag)
        if loss is not None:
            values["transmission_loss_dB"] = loss
        shown = _publish(iv, values)
        if shown is None:
            continue
        # Replay the public phasors. Pressure is normalized to the incident pressure
        # and velocity to the incident normal velocity. I_normal = Re(p * conj(u)).
        r = complex(shown.pop("rr"),shown.pop("ri"))
        t = complex(shown.pop("tr"),shown.pop("ti"))
        u = complex(shown.pop("ur"),shown.pop("ui"))
        nr = shown.pop("transmitted_slowness_real")
        ni = shown.pop("transmitted_slowness_imag")
        # Keep dimensional products in the exact range of the decoded binary64 fields.
        y1 = Fraction(shown["incident_normal_slowness_s_m"])/rho1
        yr, yi = Fraction(nr)/rho2, Fraction(ni)/rho2
        pr, pi = Fraction(t.real), Fraction(t.imag)
        vr, vi = y1*Fraction(u.real), y1*Fraction(u.imag)
        expected_r, expected_i = yr*pr-yi*pi, yr*pi+yi*pr
        scale = max(abs(vr), abs(vi), abs(expected_r), abs(expected_i))
        velocity_residual = float(max(abs(vr-expected_r),abs(vi-expected_i))/scale) if scale else 0.
        tangent = Fraction(shown["tangential_slowness_s_m"])
        normal1 = Fraction(shown["incident_normal_slowness_s_m"])
        dispersion1 = abs(c1**2*(tangent**2+normal1**2)-1)
        dispersion2 = abs(c2**2*(tangent**2+Fraction(nr)**2-Fraction(ni)**2)-1)
        residual = max(abs(1+r-t), abs(1-r-u), velocity_residual,
                       float(dispersion1), float(dispersion2),
                       abs((t*u.conjugate()).real-shown["intensity_transmission_TI"]),
                       abs(abs(r)**2+shown["intensity_transmission_TI"]-1))
        shown.update(
            regime=regime, pressure_reflection_R=r.real if r.imag == 0 else None,
            pressure_transmission_T=t.real if t.imag == 0 else None,
            pressure_reflection_complex=[r.real,r.imag], pressure_transmission_complex=[t.real,t.imag],
            transmitted_normal_velocity_ratio=[u.real,u.imag],
            transmitted_normal_slowness_s_m=[nr,ni], transmission_loss_infinite=loss is None,
            residual=residual, verified=residual < 1e-11)
        shown.setdefault("transmission_loss_dB", None)
        return shown
    raise ValueError("Directional state or public acoustic fields cannot be resolved at the required accuracy")
