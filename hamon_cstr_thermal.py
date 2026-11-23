"""Constant-density CSTR with a fixed-temperature heat exchanger.

The full steady heat balance gives an exact affine temperature/conversion
relation. Existing certified CSTR root isolation then finds every material
balance branch. Stability uses the two-state concentration/temperature
Jacobian, with exact rational bounds over each isolated root enclosure.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from collections.abc import Mapping

R_GAS = 8.31446261815324


def _number(values, name, *, default=None, positive=False, nonnegative=False):
    value = values.get(name, default)
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite JSON number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its physical domain")
    return value


def _float(value, name):
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} is outside the finite output range") from None
    if not math.isfinite(result) or (result == 0 and value != 0):
        raise ValueError(f"{name} is outside the finite nonzero output range")
    return result


def _decimal(value):
    value = Fraction(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _bounds(values, name):
    """Serialize enclosures outward, preserving the internal sign proof."""
    result = []
    for index, value in enumerate(values):
        rounded = _float(value, name)
        if index == 0 and Fraction(rounded) > value:
            rounded = math.nextafter(rounded, -math.inf)
        elif index == 1 and Fraction(rounded) < value:
            rounded = math.nextafter(rounded, math.inf)
        if not math.isfinite(rounded):
            raise ValueError(f"{name} enclosure exceeds the finite output range")
        result.append(rounded)
    return result


def _stability(root, *, c0, inverse_tau, cooling_rate, rise, base, slope, activation, order):
    """Bound trace/determinant without subtracting rounded eigenvalues.

    At equilibrium r=C0*X/tau. Therefore r_C=n*X/(tau*(1-X))
    and beta*r_T=delta_T_ad*X*(Ea/R)/(tau*T**2). The latter has
    at most one stationary point on an affine positive T(X) interval.
    """
    lo, hi = root.lower, root.upper
    points = [lo, hi]
    if slope and lo < base/slope < hi:
        points.append(base/slope)
    response = [x/(base+slope*x)**2 for x in points]
    thermal = sorted((rise*activation*inverse_tau*min(response),
                      rise*activation*inverse_tau*max(response)))
    dc = [order*inverse_tau*x/(1-x) if order else Fraction(0) for x in (lo, hi)]
    trace = [-2*inverse_tau-cooling_rate-dc[1]+thermal[0],
             -2*inverse_tau-cooling_rate-dc[0]+thermal[1]]
    determinant = [(inverse_tau+dc[0])*(inverse_tau+cooling_rate)-inverse_tau*thermal[1],
                   (inverse_tau+dc[1])*(inverse_tau+cooling_rate)-inverse_tau*thermal[0]]
    if determinant[1] < 0:
        stability = "unstable_saddle"
    elif trace[0] > 0:
        stability = "unstable"
    elif determinant[0] > 0 and trace[1] < 0:
        stability = "asymptotically_stable"
    elif determinant == [0, 0] or (trace == [0, 0] and determinant[0] > 0):
        stability = "nonhyperbolic"
    else:
        stability = "unresolved_near_boundary"
    x = (lo+hi)/2
    t = base+slope*x
    r_c = order*inverse_tau*x/(1-x) if order else Fraction(0)
    r_t = c0*inverse_tau*x*activation/t**2
    beta = rise/c0
    j = [[-inverse_tau-r_c, -r_t], [beta*r_c, -inverse_tau-cooling_rate+beta*r_t]]
    tr = j[0][0]+j[1][1]
    det = j[0][0]*j[1][1]-j[0][1]*j[1][0]
    disc = tr*tr-4*det
    with localcontext() as context:
        context.prec = 100
        td, dd = _decimal(tr), _decimal(det)
        if disc < 0:
            real, imag = td/2, _decimal(-disc).sqrt()/2
            eigenvalues = [[_float(real, "eigenvalue"), _float(imag, "eigenvalue")],
                           [_float(real, "eigenvalue"), _float(-imag, "eigenvalue")]]
        else:
            sd = _decimal(disc).sqrt()
            first = (td+(sd if td >= 0 else -sd))/2
            second = dd/first if first else Decimal(0)
            eigenvalues = [[_float(first, "eigenvalue"), 0.], [_float(second, "eigenvalue"), 0.]]
    return {
        "linear_stability": stability,
        "state_coordinates": ["C_A_mol_m3", "T_K"],
        "jacobian": [[_float(v, "Jacobian entry") for v in row] for row in j],
        "eigenvalues_s_inv": eigenvalues,
        "trace_s_inv": _float(tr, "Jacobian trace"),
        "determinant_s_inv2": _float(det, "Jacobian determinant"),
        "trace_bounds_s_inv": _bounds(trace, "trace bound"),
        "determinant_bounds_s_inv2": _bounds(determinant, "determinant bound"),
    }


def solve_jacketed_cstr(params, *, root_solver):
    """Fixed-volume power-law CSTR; all rates and heat transfer use SI units."""
    if params.get("reactor_type") != "cstr":
        raise ValueError("heat_transfer currently requires one fixed-volume CSTR")
    heat = params.get("heat_transfer")
    fields = {"UA_W_K", "rho_kg_m3", "Cp_J_kg_K", "coolant_T_K"}
    if not isinstance(heat, Mapping) or set(heat) != fields:
        raise ValueError("heat_transfer requires exactly UA_W_K, rho_kg_m3, Cp_J_kg_K, coolant_T_K")
    if any(params.get(key) is not None for key in ("conversion", "T", "n_stages")):
        raise ValueError("jacketed CSTR requires volume and feed T0; conversion, T and n_stages are not independent inputs")
    if params.get("gas_phase", False) is not False or _number(params, "epsilon_A", default=0) != 0:
        raise ValueError("jacketed CSTR requires a constant-density liquid")
    c0 = Fraction(_number(params, "C_A0", positive=True))
    q = Fraction(_number(params, "v0", positive=True))
    volume = Fraction(_number(params, "volume", positive=True))
    feed_t = Fraction(_number(params, "T0", positive=True))
    rise = Fraction(_number(params, "delta_T_adiabatic"))
    order = Fraction(_number(params, "order", default=1, nonnegative=True))
    feed = c0*q
    if "F_A0" in params and Fraction(_number(params, "F_A0", positive=True)) != Fraction(float(feed)):
        raise ValueError("F_A0 must agree with C_A0*v0; otherwise omit the redundant flow")
    ua = Fraction(_number(heat, "UA_W_K", nonnegative=True))
    rho = Fraction(_number(heat, "rho_kg_m3", positive=True))
    cp = Fraction(_number(heat, "Cp_J_kg_K", positive=True))
    coolant = Fraction(_number(heat, "coolant_T_K", positive=True))
    kinetics = {key for key in ("k", "A_preexp", "k_ref", "T_ref", "E_a") if key in params}
    if kinetics == {"k"}:
        rate = Fraction(_number(params, "k", positive=True))
        activation, reference, rate_kind = Fraction(0), Fraction(1), "constant"
    elif kinetics == {"A_preexp", "E_a"}:
        rate = Fraction(_number(params, "A_preexp", positive=True))
        activation = Fraction(_number(params, "E_a"))/Fraction(R_GAS)
        reference, rate_kind = Fraction(1), "preexponential"
    elif kinetics == {"k_ref", "T_ref", "E_a"}:
        rate = Fraction(_number(params, "k_ref", positive=True))
        activation = Fraction(_number(params, "E_a"))/Fraction(R_GAS)
        reference = Fraction(_number(params, "T_ref", positive=True))
        rate_kind = "reference"
    else:
        raise ValueError("provide exactly one complete kinetic rate law")
    ratio = ua/(rho*cp*q)
    base = (feed_t+ratio*coolant)/(1+ratio)
    slope = rise/(1+ratio)
    if min(base, base+slope) <= 0:
        raise ValueError("thermal model must have positive temperature throughout 0 <= X <= 1")
    roots = root_solver(params, volume, c0, feed, base, base, slope,
                        0., order, False, Fraction(0), True, exact_inputs=True)
    if not roots:
        raise ValueError("no physical CSTR steady state for the supplied thermal model")
    # An exact reference-rate root is also useful away from a fold, notably
    # when proving a zero trace at a Hopf boundary. Snap only on algebraic identity.
    if rate_kind == "reference" and slope:
        exact = (reference-base)/slope
        if (0 < exact < 1 and order.denominator+abs(order.numerator) <= 64
                and (feed*exact/(volume*rate))**order.denominator
                    == (c0*(1-exact))**order.numerator):
            roots = [r._replace(lower=exact, upper=exact) if r.lower <= exact <= r.upper else r for r in roots]
    rows = []
    with localcontext() as context:
        context.prec = 400
        for root in roots:
            conversion, remaining = root.coordinates()
            x = Fraction(conversion) if conversion <= remaining else 1-Fraction(remaining)
            temperature = _float(base+slope*x, "temperature")
            xd, td, cd = _decimal(x), _decimal(Fraction(temperature)), _decimal(c0*(1-x))
            kd = _decimal(rate)
            if rate_kind == "preexponential":
                kd *= (-_decimal(activation)/td).exp()
            elif rate_kind == "reference":
                kd *= (-_decimal(activation)*(1/td-1/_decimal(reference))).exp()
            reaction = kd*(cd**_decimal(order) if order else Decimal(1))
            consumed, reacted = _decimal(feed)*xd, _decimal(volume)*reaction
            mass_scale = max(abs(consumed), abs(reacted))
            mass_error = abs(consumed-reacted)/mass_scale if mass_scale else Decimal(0)
            heat_in = _decimal(rho*cp*q)*(_decimal(feed_t)-td)
            heat_reaction = _decimal(rho*cp*rise/c0*volume)*reaction
            heat_jacket = _decimal(ua)*(_decimal(coolant)-td)
            heat_scale = max(abs(heat_in), abs(heat_reaction), abs(heat_jacket))
            heat_error = abs(heat_in+heat_reaction+heat_jacket)/heat_scale if heat_scale else Decimal(0)
            residual = float(max(mass_error, heat_error))
            row = {"conversion_X": conversion, "unreacted_fraction": remaining,
                   "C_A_mol_m3": _float(cd, "concentration"), "exit_T_K": temperature,
                   "k_exit": _float(kd, "kinetic rate"), "mass_residual": float(mass_error),
                   "energy_residual": float(heat_error), "residual": residual, "verified": residual < 1e-9}
            row.update(_stability(root, c0=c0, inverse_tau=q/volume,
                       cooling_rate=ua/(rho*cp*volume), rise=rise, base=base,
                       slope=slope, activation=activation, order=order))
            rows.append(row)
    branch = params.get("steady_state_branch")
    if branch is None:
        if len(rows) != 1:
            raise ValueError("multiple physical CSTR steady states; choose steady_state_branch='all' or an ascending conversion index")
        branch = 0
    if branch != "all" and (type(branch) is not int or not 0 <= branch < len(rows)):
        raise ValueError("steady_state_branch must be 'all' or a valid ascending conversion index")
    result = {"domain": "reactor_kinetics", "reactor_type": "cstr", "thermal_model": "fixed_temperature_jacket",
              "volume_m3": _float(volume, "volume"), "space_time_s": _float(volume/q, "space time"),
              "temperature_bounds_K": [_float(v, "temperature bound") for v in sorted((base, base+slope))],
              "steady_states": rows, "steady_state_count": len(rows), "steady_state_branch": branch,
              "residual": max(row["residual"] for row in rows), "verified": all(row["verified"] for row in rows)}
    if branch != "all":
        result.update(rows[branch])
    return result
