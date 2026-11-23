"""Admissible compressive Hugoniot states for a calorically perfect gas."""

import math
from fractions import Fraction

import mpmath


def _compute(ctx, m1, g, gas, p1, t1):
    def real(q):
        return ctx.mpf(q.numerator) / q.denominator

    def wire(q):
        value = float(q)
        if not math.isfinite(value) or (q and (not value or abs(ctx.mpf(value)/q - 1) >= 1e-10)):
            raise ValueError("Shock state is outside the supported output resolution")
        return value

    # q = M1²−1 retains the contact with the sonic state. The ratios and
    # their positive increments are exact for the submitted binary64 values.
    q = (m1 - 1)*(m1 + 1)
    pressure_jump = 2*g*q/(g+1)
    density_jump = 2*q/(g+1+(g-1)*q)
    pressure = 1 + pressure_jump
    density = 1 + density_jump
    temperature = pressure/density
    m2sq = (g+1+(g-1)*q)/(g+1+2*g*q)

    # Δs/R = log(T2/T1)/(γ−1) − log(ρ2/ρ1). Evaluate both logarithms
    # from exact increments. Their difference is cubic near sonic contact.
    ds_over_r = ctx.log1p(real(temperature-1))/real(g-1) - ctx.log1p(real(density_jump))
    if q and not ds_over_r > 0:
        raise ValueError("Positive compressive shock entropy could not be resolved")
    result = {
        "M1": float(m1), "M2": wire(ctx.sqrt(real(m2sq))),
        "P2_over_P1": wire(real(pressure)), "rho2_over_rho1": wire(real(density)),
        "T2_over_T1": wire(real(temperature)),
        "P2_Pa": wire(real(p1*pressure)), "T2_K": wire(real(t1*temperature)),
        "P02_over_P01": wire(ctx.exp(-ds_over_r)),
        "stagnation_pressure_loss_fraction": wire(-ctx.expm1(-ds_over_r)),
        "entropy_increase_J_kgK": wire(real(gas)*ds_over_r),
    }
    # Replay conservation using the actual published pressure, temperature,
    # density ratio and downstream Mach number. Normalize by upstream fluxes.
    pr = ctx.mpf(result["P2_Pa"])/real(p1)
    tr = ctx.mpf(result["T2_K"])/real(t1)
    rr = ctx.mpf(result["rho2_over_rho1"])
    v_ratio = ctx.mpf(result["M2"])/real(m1)*ctx.sqrt(tr)
    kinetic = real(g*m1*m1)
    energy2 = tr*(2+real(g-1)*ctx.mpf(result["M2"])**2)
    residual = max(abs(rr*v_ratio-1),
                   abs((pr + kinetic*rr*v_ratio**2)/(1+kinetic)-1),
                   abs(energy2/real(2+(g-1)*m1*m1)-1),
                   abs(pr/(rr*tr)-1),
                   abs(pr/ctx.mpf(result["P2_over_P1"])-1),
                   abs(tr/ctx.mpf(result["T2_over_T1"])-1))
    result.update(residual=float(residual), verified=bool(residual < 1e-10),
                  shock_state="compression" if q else "sonic_limit", shock_present=bool(q))
    return result


def normal_shock(params, gamma, r_gas):
    out = {"domain": "compressible_acoustics", "subkind": "normal_shock", "verified": False}
    try:
        values = [float(params["M1"]), gamma, r_gas,
                  float(params.get("P1", 101325.0)), float(params.get("T1", 288.15))]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Normal shock inputs must be finite")
        m1, g, gas, p1, t1 = map(Fraction, values)
        if g <= 1 or min(gas,p1,t1) <= 0:
            raise ValueError("Require gamma > 1 and positive R, absolute pressure and temperature")
        if m1 < 1:
            raise ValueError("A compressive normal shock requires supersonic inflow")
        ctx = mpmath.mp.clone()
        ctx.dps = 110
        return {**out, **_compute(ctx, m1, g, gas, p1, t1)}
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return dict(out, error=str(exc))
