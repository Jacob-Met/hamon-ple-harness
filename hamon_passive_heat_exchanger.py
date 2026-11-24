"""Passive constant-capacity heat-exchanger rating and finite-area sizing."""
import math
from fractions import Fraction

from mpmath import mp


def _tube_number(value):
    if isinstance(value, bool):
        raise ValueError("Tube properties must be finite numbers")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("Tube properties must be finite numbers")
    return x


def _tube_wire(value):
    number = float(value)
    if not math.isfinite(number) or (value and not number):
        raise ValueError("Tube result outside the reporting range")
    return number


def tubular_resistance(model):
    """Radial film, fouling and wall resistances times total active tube length."""
    required = {"inner_diameter_m", "outer_diameter_m", "wall_k_W_mK",
                "inner_h_W_m2K", "outer_h_W_m2K", "inner_fouling_m2K_W", "outer_fouling_m2K_W"}
    if not isinstance(model, dict) or not required <= model.keys() or model.keys() - (required | {"total_length_m"}):
        raise ValueError("tube_resistance requires the two diameters, wall k, both films and both fouling factors")
    ctx = mp.clone()
    ctx.dps = 80
    values = {key: ctx.mpf(_tube_number(model[key])) for key in required}
    di, do = values["inner_diameter_m"], values["outer_diameter_m"]
    k, hi, ho = values["wall_k_W_mK"], values["inner_h_W_m2K"], values["outer_h_W_m2K"]
    fi, fo = values["inner_fouling_m2K_W"], values["outer_fouling_m2K_W"]
    if not 0 < di < do or min(k, hi, ho) <= 0 or min(fi, fo) < 0:
        raise ValueError("Tube requires 0 < Di < Do, positive k and film coefficients, and nonnegative fouling")
    pi_di, pi_do = ctx.pi * di, ctx.pi * do
    terms = {"inner_film": 1/(hi*pi_di), "inner_fouling": fi/pi_di,
             "wall": ctx.log1p((do-di)/di)/(2*ctx.pi*k),
             "outer_fouling": fo/pi_do, "outer_film": 1/(ho*pi_do)}
    clean = terms["inner_film"] + terms["wall"] + terms["outer_film"]
    service = sum(terms.values())
    result = {"inner_diameter_m": float(di), "outer_diameter_m": float(do),
              "resistance_times_length_mK_W": {key: _tube_wire(v) for key, v in terms.items()},
              "clean_UA_per_m_W_mK": _tube_wire(1/clean),
              "service_UA_per_m_W_mK": _tube_wire(1/service),
              "service_U_inner_W_m2K": _tube_wire(1/(service*pi_di)),
              "service_U_outer_W_m2K": _tube_wire(1/(service*pi_do))}
    if "total_length_m" in model:
        length = _tube_number(model["total_length_m"])
        if length <= 0:
            raise ValueError("total_length_m must be positive")
        result.update(total_length_m=length, rated_UA_W_K=_tube_wire(length/service),
                      clean_UA_W_K=_tube_wire(length/clean))
    return result


def tube_extent(tube, ua):
    """Required total active length and heat-transfer areas from a service UA."""
    ctx = mp.clone()
    ctx.dps = 80
    length = ctx.mpf(ua)/tube["service_UA_per_m_W_mK"]
    return {"required_total_length_m": _tube_wire(length),
            "required_inner_area_m2": _tube_wire(ctx.pi*tube["inner_diameter_m"]*length),
            "required_outer_area_m2": _tube_wire(ctx.pi*tube["outer_diameter_m"]*length)}


def passive_exchanger(spec, *, sizing):
    from hamon_wave7_engineering_verifiers import _hx_eps_from_ntu

    ctx = mp.clone()
    ctx.dps = 90
    arr = str(spec.get("arrangement", "counterflow")).strip().lower()
    if arr not in ({"parallel", "counterflow"} if sizing else {"parallel", "counterflow", "shell_and_tube_1_2"}):
        return {"dual_verified": False, "reason": "unsupported_flow_arrangement"}

    def read(key, source=spec):
        raw = source[key]
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
        effective_u = None
        if "fouling" in spec:
            if "tube_resistance" in spec:
                raise ValueError("Choose one resistance model: fouling or tube_resistance")
            model = spec["fouling"]
            if not isinstance(model, dict):
                raise ValueError("fouling must specify the clean coefficient and the two surface resistances")
            if "UA_W_K" in spec or "U_W_m2K" in spec:
                raise ValueError("Specify effective conductance or the fouling model, not both")
            basis = model["area_basis"]
            if basis not in ("inside", "outside"):
                raise ValueError("area_basis must be inside or outside")
            clean = read("U_clean_W_m2K", model)
            ratio = read("outside_to_inside_area_ratio", model)
            inside = read("R_inside_m2K_W", model)
            outside = read("R_outside_m2K_W", model)
            if min(clean, ratio) <= 0 or min(inside, outside) < 0:
                raise ValueError("Clean U and area ratio must be positive; fouling resistances must be nonnegative")
            # R_total [K/W] = 1/(U_clean*A_ref) + R_inside/A_inside + R_outside/A_outside.
            added = inside*ratio + outside if basis == "outside" else inside + outside/ratio
            effective_u = 1/(1/clean + added)
            result.update(area_basis=basis, U_effective_W_m2K=wire(effective_u),
                          fouling_on_basis_m2K_W=wire(added))
        tube = tubular_resistance(spec["tube_resistance"]) if "tube_resistance" in spec else None
        if tube is not None:
            result["tube_resistance"] = tube
            if "U_W_m2K" in spec and not math.isclose(read("U_W_m2K"), tube["service_U_outer_W_m2K"],
                                                       rel_tol=1e-9, abs_tol=0):
                raise ValueError("Supplied U disagrees with the service U on the outer area")
        if sizing:
            ho = read("Th_out_K")
            if tube is not None:
                u = Fraction(tube["service_U_outer_W_m2K"])
            elif effective_u is not None:
                u = effective_u
            else:
                u = read("U_W_m2K")
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
            if effective_u is not None:
                stated_ua = effective_u * read("area_m2")
            elif tube is None:
                stated_ua = read("UA_W_K")
            else:
                if "rated_UA_W_K" not in tube:
                    raise ValueError("Tube rating requires total_length_m")
                stated_ua = Fraction(tube["rated_UA_W_K"])
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

        if effective_u is not None:
            result["UA_W_K"] = wire(ua)
        if tube is not None:
            if "UA_W_K" in spec and not math.isclose(read("UA_W_K"), wire(ua), rel_tol=1e-9, abs_tol=0):
                raise ValueError("Supplied UA disagrees with the service duty or tube rating")
            if sizing and "rated_UA_W_K" in tube and not math.isclose(wire(ua), tube["rated_UA_W_K"],
                                                                    rel_tol=1e-9, abs_tol=0):
                raise ValueError("Supplied total_length_m conflicts with the measured sizing duty")
            tube.update(tube_extent(tube, wire(ua)))

        # An independent effectiveness-NTU law checks the measured sizing
        # state or high-precision forward rating at its dimensionless scale.
        layout = "shell_tube_1_2" if arr == "shell_and_tube_1_2" else arr
        ref = _hx_eps_from_ntu(wire(ntu), float(cr), layout)
        error = abs(ctx.mpf(ref)-eps)/abs(eps) if eps else abs(ref)
        if not ctx.isfinite(error) or error > 1e-8:
            raise ValueError("Independent effectiveness check did not resolve the reported duty")

        if arr == "parallel":
            # Convert the coflow terminal gaps to the two cross-paired gaps
            # H_in-C_out and H_out-C_in. Signed additions remain conditioned.
            cross_in = duty/real(ch) + gap2
            cross_out = duty/real(cc) + gap2
        else:
            cross_in, cross_out = gap1, gap2

        if not duty:
            entropy = ctx.mpf(0)
        elif ch == cc:
            # H_out*C_out - H_in*C_in = (Q/C)*(H_out-C_in). The product
            # form retains the true entropy arbitrarily close to reversibility.
            entropy = real(ch)*ctx.log1p((duty/real(ch))*cross_out/(inlet_h*inlet_c))
        else:
            # Two exact decompositions of the entropy difference. The first
            # preserves small heat inputs; the second preserves near-swapped
            # temperatures without subtracting two full inlet-log terms.
            def change_log(delta_t, initial, final):
                ratio = delta_t/initial
                return ctx.log1p(ratio) if ratio > -ctx.mpf(".75") else ctx.log(final/initial)

            terms = (real(ch)*change_log(-duty/real(ch), inlet_h, out_h),
                     real(cc)*change_log(duty/real(cc), inlet_c, out_c))
            complement = (real(cc-ch)*ctx.log(inlet_h/inlet_c),
                          real(ch)*change_log(cross_out, inlet_c, out_h),
                          real(cc)*change_log(-cross_in, inlet_h, out_c))

            def conditioning(parts):
                size = sum(map(abs,parts))
                return abs(sum(parts))/size if size else 0

            chosen = max((terms, complement), key=conditioning)
            if conditioning(chosen) < ctx.mpf("1e-55"):
                raise ValueError("Entropy difference is not resolved")
            entropy = sum(chosen)
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
