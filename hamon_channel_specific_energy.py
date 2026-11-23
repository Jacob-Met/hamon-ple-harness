"""Alternate wet depths at fixed discharge and specific energy in a prismatic channel."""

import math
from fractions import Fraction

import mpmath
import sympy


def _input(spec, name, default=None):
    value = spec.get(name, default)
    if type(value) is int:
        return Fraction(value)
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return Fraction(value)


def _wire(value):
    reported = float(value)
    if not math.isfinite(reported) or (value and not reported):
        raise ValueError("Channel solution exceeds the finite nonzero reporting range")
    if value:
        difference = (abs(Fraction(reported)/value-1) if isinstance(value,Fraction)
                      else abs(reported/value-1))
        if difference >= 1e-10:
            raise ValueError("A channel quantity is outside the supported output resolution")
    return reported


def _coords(r, energy, width, side, discharge, critical):
    y = r*energy
    a = y*(width + side*y)
    velocity = discharge/a
    # At a root, E-y is the velocity head. This form also resolves the
    # signed departure from critical flow without subtracting rounded Froudes.
    h = energy-y
    fr2 = 2*h*(width+2*side*y)/a
    return y, a, velocity, h, fr2, Fraction(0) if critical else fr2-1


def _state(poly, interval, critical, energy, width, side, discharge, gravity, ctx):
    lo, hi = interval
    def coords(point):
        return _coords(Fraction(int(point.p), int(point.q)), energy, width, side,
                       discharge, critical)
    # Each observed coordinate is monotone in depth in the wet interval.
    # Refine relative to each coordinate, including the signed departure
    # from criticality and kinetic head near the upper energy bound.
    while lo != hi:
        if lo > 0 and hi < 1:
            lower, upper = coords(lo), coords(hi)
            if all(abs(a-b)*2**76 <= max(abs(a),abs(b)) for a,b in zip(lower,upper)):
                break
        lo, hi = poly.refine_root(lo, hi, steps=12)
    y, area, v, h, fr2, excess = coords((lo+hi)/2)
    froude = ctx.sqrt(ctx.mpf(fr2.numerator)/fr2.denominator)
    record = {"regime": "critical" if critical else "supercritical" if excess>0 else "subcritical",
              "depth_m": _wire(y), "area_m2": _wire(area), "velocity_m_s": _wire(v),
              "velocity_head_m": _wire(h), "froude": _wire(froude),
              "froude_squared_minus_one": _wire(excess)}
    # Replay continuity, geometry and kinetic energy from independent
    # published quantities. The energy balance uses the specified bed datum.
    y_out, a_out, v_out, h_out = map(Fraction, (record["depth_m"], record["area_m2"],
                                               record["velocity_m_s"], record["velocity_head_m"]))
    residual = max(abs(a_out*v_out/discharge-1), abs(a_out/(y_out*(width+side*y_out))-1),
                   abs(v_out*v_out/(2*gravity*h_out)-1), abs((y_out+h_out)/energy-1))
    if residual >= Fraction(1,10**10):
        raise ValueError("Rounded depth state does not conserve the supplied flow and energy")
    record["relative_balance_error"] = float(residual)
    return record


def specific_energy(spec):
    out = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
           "calculation": "channel_specific_energy", "ok": False, "conservation_verified": False}
    try:
        allowed = {"mode","calculation","Q_m3_s","bottom_width_m","side_slope_z",
                   "specific_energy_m","gravity_m_s2"}
        if spec.keys() - allowed:
            raise ValueError("Supply a prismatic section, signed flow, energy and gravity")
        e = _input(spec,"specific_energy_m")
        b = _input(spec,"bottom_width_m")
        z = _input(spec,"side_slope_z",0.)
        q = _input(spec,"Q_m3_s")
        g = _input(spec,"gravity_m_s2",9.80665)
        if min(e,g) <= 0 or min(b,z) < 0 or max(b,z) == 0:
            raise ValueError("Wet depth requires positive energy, gravity and a nonempty section")
        if q == 0:
            area = e*(b+z*e)
            shown_area = _wire(area)
            point = {"regime":"stagnant", "depth_m":_wire(e), "area_m2":shown_area,
                     "velocity_m_s":0., "velocity_head_m":0., "froude":0.,
                     "froude_squared_minus_one":-1., "relative_balance_error":float(abs(Fraction(shown_area)/area-1))}
            return dict(out,ok=True,conservation_verified=True,state="stagnant",depth_solutions=[point])

        # Set r=y/E. The original energy equation has a polynomial residual,
        # H=2g E^3 r^2(1-r)(b+zEr)^2-Q^2, in the open interval 0<r<1.
        # Its single interior maximum lets positive simple roots be paired,
        # while a double root is a true critical contact. Exact rational
        # isolation distinguishes either side without a criticality tolerance.
        r = sympy.Symbol("r")
        h = sympy.Poly(2*g*e**3*r**2*(1-r)*(b+z*e*r)**2-q*q, r, domain=sympy.QQ)
        bands = h.intervals(inf=0,sup=1)
        if not bands:
            return dict(out,ok=True,state="below_minimum",depth_solutions=[])
        simple = h.sqf_part()
        ctx = mpmath.mp.clone()
        ctx.dps = 90
        points = [_state(simple,interval,multiplicity==2,e,b,z,q,g,ctx)
                  for interval,multiplicity in bands]
        regimes = [p["regime"] for p in points]
        if regimes == ["critical"]:
            state = "critical"
        elif regimes == ["supercritical","subcritical"] and points[0]["depth_m"] < points[1]["depth_m"]:
            state = "two_depths"
        else:
            raise ValueError("The alternate depths are not separately representable")
        return dict(out,ok=True,conservation_verified=True,state=state,depth_solutions=points)
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return dict(out,reason=str(exc))
