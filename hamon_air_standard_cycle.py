"""Resolve one heat-addition state of a calorically perfect Otto or Diesel cycle."""

import math
import struct
from fractions import Fraction

from mpmath import mp
from mpmath.ctx_iv import MPIntervalContext
from sympy import integer_nthroot


class _Unresolved(ArithmeticError):
    pass


def _number(raw):
    if isinstance(raw, bool):
        raise ValueError("Expected a finite physical scalar")
    if type(raw) is int:
        return Fraction(raw)
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Expected a finite physical scalar")
    return Fraction(value)


def _cell(raw):
    """Exact integer prescriptions and round-to-nearest cells for floats."""
    center = _number(raw)
    if type(raw) is int:
        return center, center, center, True
    value = float(center)
    left, right = math.nextafter(value,-math.inf), math.nextafter(value,math.inf)
    if not math.isfinite(left):
        left = 2*center-Fraction(right)
    if not math.isfinite(right):
        right = 2*center-Fraction(left)
    even = not (struct.unpack(">Q",struct.pack(">d",value))[0] & 1)
    return center, (center+Fraction(left))/2, (center+Fraction(right))/2, even


def _rational_compression(r, exponent):
    degree = exponent.denominator
    if degree > max(r.numerator.bit_length(),r.denominator.bit_length()):
        return None
    num, n_exact = integer_nthroot(r.numerator,degree)
    den, d_exact = integer_nthroot(r.denominator,degree)
    if not (n_exact and d_exact):
        return None
    if max(num.bit_length(),den.bit_length())*exponent.numerator > 16384:
        return None
    return Fraction(num,den)**exponent.numerator


def _cycle(params, kind, digits):
    fields = {"domain","subkind","cycle_type","r","T1","P1","gamma","cv","T3","q_in"}
    if kind == "diesel":
        fields.add("r_c")
    if params.keys()-fields:
        raise ValueError("Unsupported air-standard cycle constraints")
    r,t1,p1,g,cv = (_number(params[k]) if k in params else _number(default)
                     for k,default in (("r",None),("T1",300.),("P1",101325.),("gamma",1.4),("cv",718.)))
    if r <= 1 or g <= 1 or min(t1,p1,cv) <= 0:
        raise ValueError("Require compression ratio and gamma above one and positive T1, P1, cv")
    cp = g*cv
    c = cv if kind == "otto" else cp
    ctx = mp.clone()
    ctx.dps = digits
    iv = MPIntervalContext()
    iv.dps = digits

    def real(v, space=ctx):
        return space.mpf(v.numerator)/v.denominator

    tau_exact = _rational_compression(r,g-1)
    exact_t2 = t1*tau_exact if tau_exact is not None else None
    log_tau = real(g-1)*ctx.log(real(r))
    if log_tau+ctx.log(real(t1)) > 720:
        raise ValueError("Compression temperature is outside the reporting range")
    t2 = real(exact_t2) if exact_t2 is not None else real(t1)*ctx.exp(log_tau)
    bound_t2 = (real(exact_t2,iv) if exact_t2 is not None else
                real(t1,iv)*iv.exp(real(g-1,iv)*iv.log(real(r,iv))))

    # All prescribed heat, peak-temperature and cutoff cells give affine
    # bounds a+b*T2 on the same physical temperature rise. Compare the
    # affine expressions before evaluating their nearly equal endpoints.
    def compare(u,v):
        a,b = u[0]-v[0], u[1]-v[1]
        if exact_t2 is not None or not b:
            value = a+(b*exact_t2 if b else 0)
            return (value>0)-(value<0)
        if not a or (a>0)==(b>0):
            return 1 if b>0 else -1
        value = real(a,iv)+real(b,iv)*bound_t2
        if value.a>0:
            return 1
        if value.b<0:
            return -1
        raise _Unresolved("Could not qualify the compression state at this precision")

    zero=(Fraction(0),Fraction(0))
    lower=(zero,False)
    upper=(((Fraction(0),r-1),False) if kind=="diesel" else None)
    centers=[]
    for field in ("q_in","T3","r_c"):
        if field not in params:
            continue
        center,lo,hi,closed = _cell(params[field])
        if center <= 0:
            raise ValueError("Heat, peak temperature and cutoff prescriptions must be positive")
        def heat_coordinate(v):
            if field=="q_in":
                return (v/c,Fraction(0))
            if field=="T3":
                return (v,Fraction(-1))
            return (Fraction(0),v-1)
        centers.append(heat_coordinate(center))
        l,h = heat_coordinate(lo),heat_coordinate(hi)
        order=compare(l,lower[0])
        if order>0 or (order==0 and not closed):
            lower=(l,closed)
        order=-1 if upper is None else compare(h,upper[0])
        if order<0 or (order==0 and not closed):
            upper=(h,closed)
    if not centers:
        raise ValueError("Supply heat input, peak temperature, or a Diesel cutoff ratio")
    ordering = compare(lower[0],upper[0])
    if ordering>0 or (ordering==0 and not (lower[1] and upper[1])):
        raise ValueError("No single admissible heat-addition state satisfies all supplied coordinates")

    def contains(candidate):
        lo,hi=compare(candidate,lower[0]),compare(candidate,upper[0])
        return (lo>0 or (lo==0 and lower[1])) and (hi<0 or (hi==0 and upper[1]))
    chosen=next((x for x in centers if contains(x)),None)
    if chosen is None:
        if len(centers)==1:
            raise ValueError("The supplied heat-addition state is outside the cycle domain")
        chosen=tuple((a+b)/2 for a,b in zip(lower[0],upper[0]))
    dt = (real(chosen[0]+chosen[1]*exact_t2) if exact_t2 is not None else
          real(chosen[0])+real(chosen[1])*t2)
    if dt<=0:
        raise _Unresolved("Positive heat addition could not be resolved")
    T1,G,Cv=real(t1),real(g),real(cv)
    qin=real(c)*dt
    t3=t2+dt
    cutoff=1+dt/t2
    if kind=="otto":
        rise4 = dt*ctx.exp(-log_tau)
        efficiency = -ctx.expm1(-log_tau)
    else:
        rise4 = T1*ctx.expm1(G*ctx.log1p(dt/t2))
        efficiency = 1-ctx.exp(-log_tau)*ctx.expm1(G*ctx.log1p(dt/t2))/(G*(dt/t2))
    qout=Cv*rise4
    work=qin*efficiency
    if not (0<efficiency<1 and work>0):
        raise _Unresolved("Thermal efficiency could not be resolved")
    # Independently integrate p dV around the two isentropic strokes and
    # the Diesel constant-pressure stroke. The Otto heating stroke has dV=0.
    log_expansion = (-log_tau if kind=="otto" else
                     (G-1)*ctx.log1p(dt/t2)-log_tau)
    w_compression = -Cv*T1*ctx.expm1(log_tau)
    w_expansion = -Cv*t3*ctx.expm1(log_expansion)
    w_heating = 0 if kind=="otto" else real(cv*(g-1))*dt
    pv_work = ctx.fsum((w_compression,w_heating,w_expansion))
    heat_work = qin-qout
    if max(abs(pv_work/work-1),abs(heat_work/work-1)) > ctx.mpf("1e-24"):
        raise _Unresolved("Independent pressure-volume work did not resolve the net cycle")
    volume_drop=real(cv*(g-1)*t1*(r-1)/(r*p1))

    def wire(x):
        f=float(x)
        if not math.isfinite(f) or (x and (not f or abs(ctx.mpf(f)/x-1)>=1e-10)):
            raise ValueError("A cycle output is outside the supported reporting range")
        return f
    out={"domain":"thermo_cycles","subkind":kind,"T2_K":wire(t2),"T3_K":wire(t3),
         "T4_K":wire(T1+rise4),"heat_addition_delta_T_K":wire(dt),
         "w_net_J_kg":wire(pv_work),"q_in_J_kg":wire(qin),"q_out_J_kg":wire(qout),
         "eta_thermal":wire(heat_work/qin),"eta_closed_form":wire(efficiency),"MEP_Pa":wire(pv_work/volume_drop)}
    if kind=="diesel":
        out["cutoff_ratio"]=wire(cutoff)
    # First-law replay uses published heat and work, with an independent
    # constant-volume exhaust calculation. Exact cell feasibility determines
    # consistency of the input data even when two printed temperatures tie.
    residual=max(abs((qin-qout-work)/qin),
                 abs((ctx.mpf(out["q_in_J_kg"])-out["q_out_J_kg"]-out["w_net_J_kg"])/qin))
    out.update(residual=float(residual),verified=bool(residual<1e-10))
    return out


def air_standard_cycle(params,kind):
    previous=None
    try:
        for digits in (90,180,360,720,1440):
            try:
                result=_cycle(params,kind,digits)
            except _Unresolved:
                continue
            values={k:v for k,v in result.items() if k!="residual"}
            if values==previous:
                return result
            previous=values
        raise ValueError("Cycle state could not be qualified at the supported precision")
    except (KeyError,TypeError,ValueError,ArithmeticError) as exc:
        return {"domain":"thermo_cycles","subkind":kind,"verified":False,"error":str(exc)}
