"""Passive constant-capacity heat-exchanger rating and finite-area sizing."""
import math
from fractions import Fraction

from mpmath import mp


def passive_exchanger(spec, *, sizing):
    from hamon_wave7_engineering_verifiers import _hx_eps_from_ntu

    ctx = mp.clone()
    ctx.dps = 90
    arr = str(spec.get("arrangement", "counterflow")).strip().lower()
    if arr not in ({"parallel", "counterflow"} if sizing else {"parallel", "counterflow", "shell_and_tube_1_2"}):
        return {"dual_verified": False, "reason": "unsupported_flow_arrangement"}

    def read(key):
        raw = spec[key]
        v = float(raw)
        if isinstance(raw, bool) or not math.isfinite(v):
            raise ValueError(f"{key} must be finite")
        return Fraction(v)

    def real(f):
        return ctx.mpf(f.numerator)/f.denominator

    def wire(v):
        x = float(v)
        if not math.isfinite(x) or (v and x == 0):
            raise ValueError("exchanger result outside the binary64 reporting range")
        return x

    def log_mean(a, b):
        if a == b:
            return a
        # Order magnitudes so log1p keeps thin terminal differences and the
        # ratio stays positive even across very different terminal scales.
        sign = 1 if a > 0 else -1
        hi, lo = max(abs(a), abs(b)), min(abs(a), abs(b))
        return sign*(hi-lo)/ctx.log1p((hi-lo)/lo)

    try:
        mh, ph, mc, pc, hi, ci = map(read, ("m_hot_kg_s", "cp_hot_J_kgK", "m_cold_kg_s",
                                              "cp_cold_J_kgK", "Th_in_K", "Tc_in_K"))
        if min(mh, ph, mc, pc, hi, ci) <= 0:
            raise ValueError("Stream masses, heat capacities and absolute temperatures must be positive")
        ch, cc = mh*ph, mc*pc
        cmin, cmax = min(ch, cc), max(ch, cc)
        cr = real(cmin/cmax)
        delta = hi-ci
        span = real(delta)
        inlet_h, inlet_c = real(hi), real(ci)
        result = {"arrangement": arr}
        if sizing:
            ho, u = read("Th_out_K"), read("U_W_m2K")
            if min(ho, u) <= 0:
                raise ValueError("Outlet absolute temperature and sizing U must be positive")
            q = ch*(hi-ho)
            co = ci+q/cc
            if co <= 0 or q*delta < 0:
                raise ValueError("Infeasible passive heat-transfer direction or outlet temperature")
            if delta == 0:
                if q:
                    raise ValueError("Equal inlets cannot drive nonzero heat transfer")
                return dict(result, Q_duty_W=0., Tc_out_K=float(ci), LMTD_K=0.,
                            entropy_generation_W_K=0., thermal_state_feasible=True,
                            area_identifiable=False, dual_verified=False,
                            reason="Equal inlet temperatures and zero duty do not identify the area")
            d1, d2 = (hi-co, ho-ci) if arr == "counterflow" else (delta, ho-co)
            if min(d1/delta, d2/delta) <= 0:
                raise ValueError("Requested outlet reaches or crosses the finite-area terminal approach")
            mean = log_mean(real(d1), real(d2))
            duty = real(q)
            ua = duty/mean
            ntu = ua/real(cmin)
            eps = real(q/(cmin*delta))
            out_h, out_c = real(ho), real(co)
            gap1, gap2 = real(d1), real(d2)
            area = ua/real(u)
            result.update(LMTD_K=wire(mean), area_m2=wire(area), area_identifiable=True)
        else:
            stated_ua = read("UA_W_K")
            if stated_ua < 0:
                raise ValueError("A passive exchanger requires nonnegative UA")
            ua = real(stated_ua)
            ntu = real(stated_ua/cmin)
            if arr == "counterflow":
                if cmin == cmax:
                    eps = ntu/(1+ntu)
                    small = large = 1/(1+ntu)
                else:
                    diff = real((cmax-cmin)/cmax)
                    x = ntu*diff
                    removed = -ctx.expm1(-x)
                    den = diff+cr*removed
                    eps = removed/den
                    large, small = diff/den, diff*ctx.exp(-x)/den
                # Complementary approaches avoid subtracting effectiveness
                # from one when finite conductance nearly swaps the streams.
                gap1,gap2 = ((large*span,small*span) if ch <= cc else (small*span,large*span))
            elif arr == "parallel":
                x = ntu*(1+cr)
                eps = -ctx.expm1(-x)/(1+cr)
                gap1,gap2 = span,span*ctx.exp(-x)
            else:
                factor = ctx.sqrt(1+cr*cr)
                t = ctx.tanh(ntu*factor/2)
                den = factor+(1+cr)*t
                eps = 2*t/den
                decay = ctx.exp(-ntu*factor)
                small = (cr*cr/(1+factor) + 2*decay/(1+decay) + cr*t)/den
                large = (factor+(1-cr)*t)/den
                # Counterflow pairing evaluates the total shell entropy
                # integral; it does not assert a local shell-side LMTD.
                gap1,gap2 = ((large*span,small*span) if ch <= cc else (small*span,large*span))
            duty = real(cmin*delta)*eps
            # Add nonnegative changes to the cooler inlet. Subtracting a
            # nearly complete hot-stream drop can erase its finite outlet.
            if delta >= 0:
                out_c = inlet_c+duty/real(cc)
                out_h = (out_c if arr == "parallel" else inlet_c)+gap2
            else:
                out_h = inlet_h-duty/real(ch)
                out_c = out_h-gap2 if arr == "parallel" else inlet_h-gap1
            result.update(effectiveness=wire(eps), NTU=wire(ntu), Th_out_K=wire(out_h))

        # An independent effectiveness-NTU law checks the measured sizing
        # state or high-precision forward rating at its dimensionless scale.
        layout = "shell_tube_1_2" if arr == "shell_and_tube_1_2" else arr
        ref = _hx_eps_from_ntu(wire(ntu), float(cr), layout)
        error = abs(ctx.mpf(ref)-eps)/abs(eps) if eps else abs(ref)
        if not ctx.isfinite(error) or error > 1e-8:
            raise ValueError("Independent effectiveness check did not resolve the reported duty")

        if not duty:
            entropy = ctx.mpf(0)
        else:
            hot = real(ch)*ctx.log1p(-duty/(real(ch)*inlet_h))
            cold = real(cc)*ctx.log1p(duty/(real(cc)*inlet_c))
            entropy = hot+cold
            if not ctx.isfinite(entropy) or abs(entropy) < ctx.mpf("1e-24")*(abs(hot)+abs(cold)):
                # Integrate dQ*(Th-Tc)/(Th*Tc) with the physical terminal
                # gaps. This keeps positive entropy near reversible exchange.
                ca, cb = (inlet_c,out_c) if arr == "parallel" else (out_c,inlet_c)
                entropy = duty*ctx.quad(
                    lambda x: ((1-x)*gap1+x*gap2) /
                        (((1-x)*inlet_h+x*out_h)*((1-x)*ca+x*cb)), [0,1])
        if entropy < 0 or not ctx.isfinite(entropy):
            raise ValueError("Negative or unresolved passive entropy production")
        duty_wire = wire(duty)
        hot_drop, cold_rise = wire(duty/real(ch)), wire(duty/real(cc))
        duty_fraction = Fraction(duty_wire)
        # A small duty can leave the rounded absolute outlet at its inlet.
        # The signed temperature changes independently replay both streams.
        residual = max(abs(duty_fraction-ch*Fraction(hot_drop)),
                       abs(duty_fraction-cc*Fraction(cold_rise)))
        if sizing:
            residual = max(residual, abs(duty_fraction-u*Fraction(result["area_m2"])*Fraction(result["LMTD_K"])))
        if duty_wire:
            error = max(error, float(residual/abs(duty_fraction)))
        if error > 1e-8:
            raise ValueError("Published heat or temperature changes do not resolve the energy balance")
        # Preserve the original sizing diagnostic on rounded absolute outlets;
        # the signed changes remain the replay coordinates when an endpoint ties.
        absolute_cold_error = abs(duty_fraction-cc*(Fraction(wire(out_c))-ci))
        return dict(result, Q_duty_W=duty_wire, Tc_out_K=wire(out_c),
                    hot_temperature_drop_K=hot_drop, cold_temperature_rise_K=cold_rise,
                    energy_balance_residual_W=float(absolute_cold_error),
                    temperature_change_balance_residual_W=float(residual),
                    entropy_generation_W_K=wire(entropy),
                    thermal_state_feasible=True, dual_rel_discrepancy=float(error), dual_verified=True)
    except (ArithmeticError, KeyError, TypeError, ValueError) as e:
        return {"arrangement": arr, "dual_verified": False, "reason": str(e)}
