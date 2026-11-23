"""Ultimate primary settlement from depth-dependent Boussinesq stress and compression."""
import math

from hamon_consolidation import _area_integral, _corner, _number, _rectangles, _scaled_footprints, primary_settlement


def _precise_corner(x0, x1, y0, y1, z):
    """Retain small positive influences when opposite corner terms nearly cancel."""
    terms = (_corner(x1, y1, z), -_corner(x0, y1, z), -_corner(x1, y0, z), _corner(x0, y0, z))
    value = math.fsum(terms)
    if value > 1e-7 * sum(map(abs, terms)):
        return value
    from mpmath import mp
    ctx = mp.clone()
    radius = math.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1)), z)
    # A*z^3/r_max^5 bounds the integral from below up to the constant 3/(2*pi).
    log_lower = math.log2(x1-x0) + math.log2(y1-y0) + 3*math.log2(z) - 5*math.log2(radius)
    ctx.prec = 70 + max(0, math.ceil(-log_lower))
    z = ctx.mpf(z)

    def corner(x, y):
        x, y = ctx.mpf(x), ctx.mpf(y)
        xy, xx, yy = x*y, x*x + z*z, y*y + z*z
        radius = ctx.sqrt(xx + y*y)
        return (ctx.atan2(xy, z*radius) + xy*z/radius*(1/xx + 1/yy))/(2*ctx.pi)

    return float(corner(x1,y1) - corner(x0,y1) - corner(x1,y0) + corner(x0,y0))


def _depth_stress(rectangles, z, influence):
    if z == 0:
        def surface_fraction(lo, hi):
            if lo < 0 < hi:
                return 1.
            return .5 if lo == 0 or hi == 0 else 0.
        return math.fsum(q*surface_fraction(x0, x1)*surface_fraction(y0, y1)
                         for x0, x1, y0, y1, q in rectangles)
    values = []
    for x0, x1, y0, y1, depth, pressure in _scaled_footprints(rectangles, z):
        if pressure:
            fraction = influence(x0, x1, y0, y1, depth)
            if not 0 < fraction <= 1 + 2e-12:
                raise ValueError("Footprint influence is outside the resolved positive loading domain")
            values.append(pressure*fraction)
    return math.fsum(values)


def _depth_intervals(top, bottom, rectangles):
    """Resolve each footprint scale before using interior quadrature nodes.

    A thick layer may be thousands of footing widths deep. Both adaptive rules
    can otherwise miss the shallow stress bulb by sampling only the far tail.
    """
    scales = {abs(v) for r in rectangles for v in r[:4] if v}
    cuts = {top, bottom}
    cuts.update(z for z in scales if top < z < bottom)
    scale = max(top, min(scales))
    while scale < bottom:
        cuts.add(scale)
        scale *= 2
    edges = sorted(cuts)
    return zip(edges, edges[1:])


def _yield_intervals(top, bottom, rectangles, yield_increment):
    """Isolate compression transitions, including shallow and buried yield pockets.

    For the positive point-load kernel K, z²K''/K = 6-35u+35u² with
    u=z²/(r²+z²). Thus |K''| <= 6K/z². Near the surface subtract the
    local tangent sector, whose integral is constant: its difference from
    the rectangle vanishes inside the nearest nonzero edge distance d.
    Integrating the same bound over r>=d gives 6z/(d²+z²)^(3/2).
    """
    from functools import lru_cache
    from scipy.optimize import brentq

    active = [r for r in rectangles if r[-1] > 0]
    pressure = math.fsum(r[-1] for r in active)
    if yield_increment <= 0 or yield_increment >= pressure:
        yield from _depth_intervals(top, bottom, rectangles)
        return
    distance = min(abs(v) for r in active for v in r[:4] if v)
    # Footprints containing the column shrink monotonically in coordinates
    # normalized by depth, so their combined stress is nonincreasing.
    decreasing = all(x0 <= 0 <= x1 and y0 <= 0 <= y1 for x0,x1,y0,y1,_ in active)

    @lru_cache(maxsize=2048)
    def residual(z):
        return _depth_stress(active, z, _precise_corner) - yield_increment

    def isolate(a, b, cuts, remaining):
        fa, fb = residual(a), residual(b)
        width = b-a
        # Bound curvature times width² directly to avoid powers of physical depth.
        rel_surface = (width/distance) * (width/distance) * (b/distance)
        rel_depth = (width/a) * (width/a) if a else math.inf
        deviation = .75 * pressure * min(rel_surface, rel_depth)  # M*width²/8
        if not decreasing:
            if max(fa, fb) < -deviation or min(fa, fb) > deviation:
                return
            # The derivative differs from the secant slope by at most M*width/2.
            if abs(fb-fa) <= 4*deviation:
                middle = a+width/2
                if remaining == 0 or not a < middle < b:
                    raise ValueError("Cannot resolve a possible buried compression transition")
                isolate(a, middle, cuts, remaining-1)
                isolate(middle, b, cuts, remaining-1)
                return
        if fa == 0:
            cuts.add(a)
        if fb == 0:
            cuts.add(b)
        if (fa < 0 < fb) or (fb < 0 < fa):
            t = brentq(lambda t: residual(a+width*t), 0., 1., xtol=1e-14)
            cuts.add(a+width*t)

    for start, end in _depth_intervals(top, bottom, rectangles):
        cuts = {start, end}
        isolate(start, end, cuts, 48)
        ordered = sorted(cuts)
        yield from zip(ordered, ordered[1:])


def _gauss_integral(function, nodes, lo=0., hi=1.):
    """Independently refine a Gauss-Legendre rule on a normalized depth interval."""
    def rule(a, b):
        h = b-a
        return h * math.fsum(w*function(a+h*t) for t, w in nodes)

    def refine(a, b, whole, remaining):
        middle = (a+b)/2
        left, right = rule(a, middle), rule(middle, b)
        divided = left + right
        if math.isclose(whole, divided, rel_tol=2e-8):
            return divided
        if remaining == 0:
            raise ValueError("Depth quadrature did not converge around the compression transition")
        return refine(a, middle, left, remaining-1) + refine(middle, b, right, remaining-1)

    return refine(lo, hi, rule(lo, hi), 20)


def finite_layer_settlement(spec):
    """Return the integrated settlement of contiguous, piecewise-constant soil layers."""
    from numpy.polynomial.legendre import leggauss
    from scipy.integrate import quad

    if {"stress_point", "delta_sigma_v_Pa", "delta_sigma_Pa", "H_clay_m", "H_m", "e0", "Cc", "Cr",
        "sigma_v0_eff_Pa", "sigma_v0_Pa", "sigma_preconsolidation_Pa", "sigma_p_Pa",
        "load_history", "cv_m2_s", "time_s", "drainage", "mv_per_Pa"} & spec.keys():
        raise ValueError("Provide layer properties and ultimate rectangular loads without point, scalar or time data")
    column = spec["settlement_column"]
    layers = spec["settlement_layers"]
    if not isinstance(column, dict) or set(column) != {"x_m", "y_m"} or not isinstance(layers, list) or not layers:
        raise ValueError("Provide settlement_column with x_m and y_m and a nonempty settlement_layers list")
    rectangles = _rectangles(spec["rectangular_loads"], _number(column, "x_m"), _number(column, "y_m"))
    roots, weights = leggauss(8)
    nodes = [((float(t)+1)/2, float(w)/2) for t, w in zip(roots, weights)]
    rows = []
    for layer in layers:
        material_keys = {"top_depth_m", "bottom_depth_m", "e0", "Cc", "Cr",
                         "sigma_v0_eff_Pa", "sigma_preconsolidation_Pa"}
        if not isinstance(layer, dict) or set(layer) - material_keys:
            raise ValueError("Each layer accepts only its depth interval and compression properties")
        top, bottom, e0, cc, initial = (_number(layer, k) for k in (
            "top_depth_m", "bottom_depth_m", "e0", "Cc", "sigma_v0_eff_Pa"))
        if not 0 <= top < bottom or (rows and top != rows[-1]["bottom_depth_m"]):
            raise ValueError("Layer intervals must be positive, ordered, contiguous and below the loading plane")
        pc = _number(layer, "sigma_preconsolidation_Pa") if "sigma_preconsolidation_Pa" in layer else initial
        cr = _number(layer, "Cr") if "Cr" in layer or pc > initial else 0.
        primary_settlement(1., e0, cc, initial, 0., pc, cr)
        h = bottom - top
        area_parts, corner_parts = [], []
        for a, b in _yield_intervals(top, bottom, rectangles, pc-initial if cr != cc else 0.):
            def strain(t, influence):
                z = a + (b-a)*t
                if z <= 0:
                    raise ValueError("Layer thickness is below the representable integration depth")
                stress = _depth_stress(rectangles, z, influence)
                return primary_settlement(1., e0, cc, initial, stress, pc, cr)["primary_consolidation_settlement_m"]

            average, error = quad(strain, 0., 1., args=(_area_integral,), epsabs=0., epsrel=2e-8, limit=100)
            check = _gauss_integral(lambda t: strain(t, _precise_corner), nodes)
            if error > 3e-8 * abs(average) or not math.isclose(average, check, rel_tol=3e-7):
                raise ValueError("Depth integrations of the area kernel and corner stress do not agree")
            area_parts.append((b-a)*average)
            corner_parts.append((b-a)*check)
        settlement = math.fsum(area_parts)
        check = math.fsum(corner_parts)
        solid_height = h/(1+e0)
        void_height = h*(e0/(1+e0))
        rows.append(dict(top_depth_m=top, bottom_depth_m=bottom, initial_solid_height_m=solid_height,
                         initial_void_height_m=void_height, primary_consolidation_settlement_m=settlement,
                         corner_settlement_m=check, final_void_ratio=e0-settlement/solid_height))
    total = math.fsum(r["primary_consolidation_settlement_m"] for r in rows)
    independent = math.fsum(r["corner_settlement_m"] for r in rows)
    solid = math.fsum(r["initial_solid_height_m"] for r in rows)
    void = math.fsum(r["initial_void_height_m"] for r in rows)
    if not all(math.isfinite(v) for v in (total, independent, solid, void)) or solid <= 0:
        raise ValueError("Integrated layer volumes exceed finite numerical range")
    relative = abs(total-independent)/max(total, independent) if total or independent else 0.
    return dict(primary_consolidation_settlement_m=total, corner_settlement_m=independent,
                dual_solver_rel_residual=relative, initial_void_height_m=void,
                final_void_ratio=(void-total)/solid, settlement_by_layer=rows,
                stress_model="Boussinesq elastic half-space; ultimate depth-integrated primary consolidation")


def verify_finite_layer_settlement(spec):
    result = dict(ok=False, conservation_verified=False, mode="civil_geotech",
                  calculation="layered_footing_settlement", tool="hamon_conservation_cad_verifier")
    try:
        data = finite_layer_settlement(spec)
        proposed = _number(spec, "proposed_settlement_m") if "proposed_settlement_m" in spec else None
        total = data["primary_consolidation_settlement_m"]
        valid = proposed is None or (0 <= proposed <= data["initial_void_height_m"]
                                     and math.isclose(proposed, total, rel_tol=1e-6))
        result.update(data, ok=True, conservation_verified=valid)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result
