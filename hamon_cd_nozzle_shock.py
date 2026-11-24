"""Back-pressure-driven normal shock in a steady calorically perfect CD nozzle."""
import math

from mpmath import mp


def internal_shock(params, gamma):
    """Locate a shock by conservation of choked mass and downstream total pressure."""
    out = {"domain": "compressible_acoustics", "subkind": "cd_nozzle_shock", "verified": False}
    allowed = {"domain", "subkind", "gamma", "R", "cp", "exit_to_throat_area", "back_to_total_pressure"}
    if params.keys() - allowed:
        raise ValueError("Unexpected CD nozzle inputs; supply two dimensionless ratios and gas properties")

    def ratio(name):
        value = params.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite JSON number")
        return value

    for name in ("gamma", "R", "cp"):
        if name in params:
            ratio(name)
    area = ratio("exit_to_throat_area")
    back = ratio("back_to_total_pressure")
    if area <= 1 or back >= 1:
        raise ValueError("Require exit area greater than throat and back pressure below reservoir pressure")
    ctx = mp.clone()
    ctx.dps = 85
    g = ctx.mpf(gamma.numerator) / gamma.denominator
    a = (g - 1) / 2
    exp_t = g / (g - 1)
    exp_a = (g + 1) / (2 * (g - 1))
    ae, pb = ctx.mpf(area), ctx.mpf(back)
    sonic_flux = (1 + a) ** -exp_a

    def temperature(m):
        return 1 / (1 + a*m*m)

    def upstream_area(m):
        return ((1 + a*m*m) / (1 + a)) ** exp_a / m

    def jump(m):
        m2 = ctx.sqrt((1 + a*m*m) / (g*m*m - a))
        p1 = temperature(m) ** exp_t
        p2 = p1 * (1 + 2*g*(m*m-1)/(g+1))
        return m2, p1, p2, p2 / temperature(m2) ** exp_t

    # Pe and Ae determine exit Mach without imposing the upstream reservoir
    # stagnation pressure on the flow downstream of the shock.
    flux = sonic_flux / (ae*pb)
    me = ctx.sqrt(2*flux*flux / (1 + ctx.sqrt(1 + 4*a*flux*flux)))
    target_pt = pb / temperature(me) ** exp_t
    if me >= 1:
        return dict(out, regime="no_internal_shock_low_back_pressure")
    if target_pt >= 1:
        return dict(out, regime="no_internal_shock_high_back_pressure")

    # Downstream total pressure decreases monotonically with pre-shock Mach.
    lo, hi = ctx.one, ctx.mpf(2)
    while jump(hi)[3] > target_pt:
        if upstream_area(hi) >= ae:
            return dict(out, regime="no_internal_shock_low_back_pressure")
        lo, hi = hi, hi*2
    for _ in range(230):
        mid = (lo + hi) / 2
        if jump(mid)[3] > target_pt:
            lo = mid
        else:
            hi = mid
    m1 = (lo + hi) / 2
    shock_area = upstream_area(m1)
    if shock_area >= ae:
        return dict(out, regime="no_internal_shock_low_back_pressure")
    m2, p1, p2, pt2 = jump(m1)
    result = dict(shock_area_over_throat=float(shock_area), upstream_mach=float(m1),
                  downstream_mach=float(m2), exit_mach=float(me),
                  upstream_p_over_reservoir=float(p1), downstream_p_over_reservoir=float(p2),
                  exit_p_over_reservoir=float(pt2 * temperature(me)**exp_t),
                  downstream_total_over_reservoir=float(pt2))
    if not all(math.isfinite(v) and v > 0 for v in result.values()):
        return dict(out, regime="unrepresentable_state")

    # Replay the numbers actually returned. Physical mass flux is proportional
    # to p*M/sqrt(T). Momentum uses one common reservoir pressure reference.
    ar, u, d, e = map(ctx.mpf, (result["shock_area_over_throat"], result["upstream_mach"],
                                 result["downstream_mach"], result["exit_mach"]))
    pu, pd, pe, pt = map(ctx.mpf, (result["upstream_p_over_reservoir"],
                                   result["downstream_p_over_reservoir"],
                                   result["exit_p_over_reservoir"], result["downstream_total_over_reservoir"]))
    residuals = {
        "mass": float(max(abs(A*p*m/ctx.sqrt(temperature(m))/sonic_flux-1)
                          for A, p, m in ((ar, pu, u), (ar, pd, d), (ae, pe, e)))),
        "momentum_jump": float(abs(pd*(1+g*d*d)/(pu*(1+g*u*u))-1)),
        "upstream_stagnation": float(abs(pu/temperature(u)**exp_t-1)),
        "downstream_stagnation": float(max(abs(p/temperature(m)**exp_t/pt-1)
                                          for p, m in ((pd, d), (pe, e)))),
        "back_pressure": float(abs(pe/pb-1)),
    }
    residual = max(residuals.values())
    resolved = 1 < ar < ae and u > 1 and 0 < e < d < 1 and pu < pd and pt < 1
    return dict(out, **result, regime="internal_normal_shock" if resolved else "unresolved_boundary",
                residuals=residuals, residual=residual, verified=bool(resolved and residual < 1e-10))


def presolve_result(request, dispatch):
    """Replay a complete dimensionless case using mass flux and the shock jump."""
    spec = request["spec"]
    if (set(spec) != {"subkind", "exit_to_throat_area", "back_to_total_pressure", "gamma"}
            or spec["subkind"] != "cd_nozzle_shock"):
        return None
    if not all(type(spec[key]) in (int, float) and math.isfinite(spec[key])
               for key in ("exit_to_throat_area", "back_to_total_pressure", "gamma")):
        return None
    g, ae, back = (spec[key] for key in ("gamma", "exit_to_throat_area", "back_to_total_pressure"))
    if not (1.01 <= g <= 2 and 1 < ae <= 1000 and 0 < back < 1):
        return None
    result = dispatch("hamon_conservation_cad_verifier", request)
    if (result.get("verified") is not True or result.get("regime") != "internal_normal_shock"
            or result.get("subkind") != spec["subkind"] or result.get("domain") != request["mode"]):
        return None
    keys = ("shock_area_over_throat", "upstream_mach", "downstream_mach", "exit_mach",
            "upstream_p_over_reservoir", "downstream_p_over_reservoir", "exit_p_over_reservoir",
            "downstream_total_over_reservoir")
    if any(type(result.get(k)) not in (int, float) or not math.isfinite(result[k]) for k in keys):
        return None
    ar, u, d, e, pu, pd, pe, pt = (result[k] for k in keys)
    if not (1 < ar < ae and u > 1 and 0 < e < d < 1 and 0 < pu < pd < pe < pt < 1):
        return None
    residuals = result.get("residuals", {})
    if set(residuals) != {"mass", "momentum_jump", "upstream_stagnation", "downstream_stagnation", "back_pressure"}:
        return None
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 1e-10 for v in residuals.values()):
        return None
    if result.get("residual") != max(residuals.values()):
        return None

    # Use the density jump to determine downstream velocity, then the ideal
    # gas to recover its temperature. This is independent of the M2 solver.
    a = (g-1)/2
    tu = 1/(1+a*u*u)
    density_jump = (g+1)*u*u/((g-1)*u*u+2)
    pressure_jump = 1+2*g*(u*u-1)/(g+1)
    td = tu*pressure_jump/density_jump
    te = 1/(1+a*e*e)
    flux = (2/(g+1))**((g+1)/(2*(g-1)))
    checks = (ar*pu*u/math.sqrt(tu)/flux,
              ae*pe*e/math.sqrt(te)/flux, pe/back, pu/tu**(g/(g-1)),
              pd/(pu*pressure_jump), d/(u*math.sqrt(tu/td)/density_jump),
              pt/(pd/td**(g/(g-1))), pt/(pe/te**(g/(g-1))))
    return result if all(abs(v-1) < 1e-10 for v in checks) else None
