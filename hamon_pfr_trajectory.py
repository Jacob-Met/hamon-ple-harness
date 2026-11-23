"""Variable-rate plug-flow trajectories in complementary concentration coordinates."""
import math


def solve_variable_rate_pfr(params, concentration, feed, flow, order, inlet_t,
                            heat_rise, expansion, gas, reference_t):
    """Integrate the original mole balance using t = -log(unreacted fraction).

    Temperature is the existing prescribed adiabatic line. No pressure drop,
    reversibility, dynamic stability, or heat-transfer model is introduced.
    """
    from mpmath import mp

    ctx = mp.clone()
    ctx.dps = 100 + (max(0, -math.floor(math.log10(abs(order)))) if order else 0)
    n, c0, fa, v0, ti, dt, eps, tref_gas = map(
        ctx.mpf, (order, concentration, feed, flow, inlet_t, heat_rise, expansion, reference_t))
    if "A_preexp" in params and "E_a" in params:
        rate, ea, reference = ctx.mpf(float(params["A_preexp"])), ctx.mpf(float(params["E_a"])), None
    elif all(k in params for k in ("k_ref", "T_ref", "E_a")):
        rate, ea, reference = (ctx.mpf(float(params[k])) for k in ("k_ref", "E_a", "T_ref"))
    else:
        rate, ea, reference = ctx.mpf(float(params["k"])), ctx.zero, None
    ea /= ctx.mpf(8.31446261815324)
    log_rate, log_c0, log_feed = ctx.log(rate), ctx.log(c0), ctx.log(fa)

    def state(t):
        u, x = ctx.exp(-t), -ctx.expm1(-t)
        if x <= u:
            temp, bulk = ti + dt*x, 1 + eps*x
        else:
            temp, bulk = (ti+dt)-dt*u, (1+eps)-eps*u
        if temp <= 0 or bulk <= 0:
            raise ValueError("PFR temperature and volumetric flow must remain positive")
        if reference is not None:
            difference = ((ti-reference)+dt*x if x <= u else
                          (ti+dt-reference)-dt*u)
            log_k = log_rate + ea*difference/(temp*reference)
        else:
            log_k = log_rate - ea/temp
        return x, u, temp, bulk, log_k

    quadrature_error = ctx.zero

    def integrate(function, a, b):
        nonlocal quadrature_error
        if a == b:
            return ctx.zero
        value, error = ctx.quad(function, [a, b], method="tanh-sinh", error=True, maxdegree=7)
        if not ctx.isfinite(value) or value < 0:
            raise ValueError("PFR volume integral is not resolved")
        quadrature_error = max(quadrature_error, error/abs(value) if value else error)
        return value

    def derivative(t):
        _, _, temp, bulk, log_k = state(t)
        log_value = log_feed-log_k-n*log_c0+(n-1)*t+n*ctx.log(bulk)
        if gas:
            log_value += n*ctx.log(temp/tref_gas)
        return ctx.exp(log_value)

    def volume_at(t):
        if n == 1 and ea == 0:
            # Constant-rate first-order gas expansion is a polynomial divided
            # by U. Evaluate its moments without subtracting X from -log(U)
            # at tiny conversion. This avoids quadrature on the common path.
            x = -ctx.expm1(-t)
            moments = [t]
            for degree in (1, 2):
                if x < ctx.mpf("0.25"):
                    power = x**(degree+1)
                    total = power/(degree+1)
                    for k in range(degree+2, 1000):
                        power *= x
                        term = power/k
                        total += term
                        if abs(term) <= ctx.eps*abs(total):
                            break
                    moments.append(total)
                else:
                    moments.append(t-x if degree == 1 else t-x-x*x/2)
            a, b, c = moments
            integral = ((ti*a+(ti*eps+dt)*b+dt*eps*c)/tref_gas if gas
                        else a+eps*b)
            return fa/(rate*c0)*integral
        # Normalize a tiny interval to [0,1]; preserve an arbitrarily small
        # positive dose instead of applying an absolute conversion tolerance.
        return t*integrate(lambda z: derivative(t*z), ctx.zero, ctx.one) if t else ctx.zero

    limit = ctx.one
    if dt < 0:
        limit = min(limit, -ti/dt)
    if eps < 0:
        limit = min(limit, -1/eps)
    t_limit = -ctx.log1p(-limit) if limit < 1 else ctx.inf
    depletion = None
    # For n<1 and a regular endpoint, z=U**(1-n) removes the integrable
    # concentration singularity. This also distinguishes exhaustion from an
    # impossible continuation of zero- or negative-order kinetics.
    if n < 1 and ti+dt > 0 and 1+eps > 0:
        def terminal_integrand(z):
            t = -ctx.log(z)/(1-n) if z else ctx.inf
            _, _, temp, bulk, log_k = state(t)
            value = log_feed-log_k-n*log_c0+n*ctx.log(bulk)-ctx.log(1-n)
            if gas:
                value += n*ctx.log(temp/tref_gas)
            return ctx.exp(value)
        depletion = integrate(terminal_integrand, ctx.zero, ctx.one)

    sizing = params.get("conversion") is not None
    plateau = False
    if sizing:
        x_requested = ctx.mpf(float(params["conversion"]))
        if x_requested == 1:
            if depletion is None:
                raise ValueError("complete PFR conversion has no finite positive-temperature outlet")
            t, required = ctx.inf, depletion
        else:
            t = -ctx.log1p(-x_requested)
            required = volume_at(t)
        volume = float(required)
    else:
        volume = float(params["volume"])
        required = ctx.mpf(volume)
        if depletion is not None and required >= depletion:
            if required > depletion and n <= 0:
                raise ValueError("PFR volume exceeds the physical depletion limit")
            t, plateau = ctx.inf, True
        elif required == 0:
            t = ctx.zero
        else:
            lo = ctx.zero
            hi = min(required/derivative(ctx.zero)*2, ctx.mpf(750), t_limit/2)
            if hi <= 0:
                raise ValueError("PFR positive dose is not resolved")
            for _ in range(1100):
                vh = volume_at(hi)
                if vh >= required:
                    break
                lo = hi
                hi = min(2*hi, (hi+t_limit)/2, ctx.mpf(750))
                if hi == lo:
                    raise ValueError("PFR remaining fraction is outside the representable physical interval")
            else:
                raise ValueError("PFR physical outlet bracket did not resolve")
            t = (lo+hi)/2
            for _ in range(180):
                value = volume_at(t)
                if abs(value-required) <= ctx.mpf("1e-35")*required:
                    break
                if value < required:
                    lo = t
                else:
                    hi = t
                trial = t-(value-required)/derivative(t)
                t = trial if lo < trial < hi else (lo+hi)/2
            else:
                raise ValueError("PFR inverse volume balance did not resolve")

    x, u, temperature, bulk, log_k = state(t)
    conversion, remaining, exit_t, k_exit = map(float, (x, u, temperature, ctx.exp(log_k)))
    space = ctx.mpf(volume)/v0
    space_time = float(space)
    if not all(map(math.isfinite, (volume, conversion, remaining, exit_t, k_exit, space_time))):
        raise ValueError("PFR public state is outside the finite output range")
    if (x and conversion == 0) or (u and remaining == 0) or (required and volume == 0):
        raise ValueError("PFR state is not representable in complementary coordinates")
    if (space and space_time == 0) or exit_t <= 0 or k_exit <= 0:
        raise ValueError("PFR temperature, rate or space time is not representable")

    def relative(a, b):
        scale = max(abs(a), abs(b))
        return abs(a-b)/scale if scale else ctx.zero

    if remaining == 0:
        closure = ctx.zero if plateau else relative(depletion, ctx.mpf(volume))
    else:
        reported_t = (-ctx.log1p(-ctx.mpf(conversion)) if conversion <= remaining
                      else -ctx.log(ctx.mpf(remaining)))
        closure = relative(volume_at(reported_t), ctx.mpf(volume))
    residual = float(max(closure, relative(x, ctx.mpf(conversion)),
                         relative(u, ctx.mpf(remaining)), relative(space, ctx.mpf(space_time)),
                         relative(temperature, ctx.mpf(exit_t)),
                         relative(ctx.exp(log_k), ctx.mpf(k_exit)), quadrature_error))
    if not math.isfinite(residual):
        raise ValueError("PFR numerical balance is not resolved")
    return {"domain": "reactor_kinetics", "reactor_type": "pfr",
            "conversion_X": conversion, "unreacted_fraction": remaining,
            "volume_m3": volume, "space_time_s": space_time, "exit_T_K": exit_t,
            "k_exit": k_exit, "residual": residual, "verified": residual < 1e-9,
            "integration_coordinate": "negative_log_unreacted_fraction"}
