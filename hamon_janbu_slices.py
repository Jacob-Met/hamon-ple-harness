"""Uncorrected simplified Janbu force equilibrium for supplied slice data."""
import math
import sys
from fractions import Fraction

_EPS = sys.float_info.epsilon
_TOP = {"mode", "calculation", "slices", "c_prime_Pa", "phi_deg", "slice_order",
        "proposed_factor_of_safety", "required_factor_of_safety"}
_ROW = {"id", "width_m", "weight_N_m", "base_angle_deg", "pore_pressure_Pa",
        "c_prime_Pa", "phi_deg"}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    return _finite(float(value), name)


def _finite(value, name):
    if not math.isfinite(value):
        raise ValueError(f"{name} exceeds finite numeric range")
    return value


def _rows(spec):
    if not isinstance(spec, dict) or set(spec)-_TOP:
        raise ValueError("Unsupported Janbu input fields")
    if spec.get("calculation") != "janbu_slices" or spec.get("mode", "auto") not in (
            "auto", "civil_geotech", "civil", "geotech"):
        raise ValueError("Use civil_geotech calculation=janbu_slices")
    supplied = spec.get("slices")
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= 2048:
        raise ValueError("Supply 1 to 2048 slices")
    if "slice_order" in spec and spec["slice_order"] not in ("toe_to_crest", "crest_to_toe"):
        raise ValueError("slice_order must be toe_to_crest or crest_to_toe")
    for name in ("proposed_factor_of_safety", "required_factor_of_safety"):
        if name in spec and _number(spec[name], name) <= 0:
            raise ValueError(f"{name} must be positive")
    rows, ids = [], set()
    for i, row in enumerate(supplied, 1):
        if not isinstance(row, dict) or set(row)-_ROW:
            raise ValueError(f"Slice {i}: unsupported or malformed fields")
        sid = row.get("id", str(i))
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Slice ids must be unique nonempty strings")
        ids.add(sid)
        b = _number(row["width_m"], "width_m")
        w = _number(row["weight_N_m"], "weight_N_m")
        angle = _number(row["base_angle_deg"], "base_angle_deg")
        u = _number(row.get("pore_pressure_Pa", 0.), "pore_pressure_Pa")
        c = _number(row.get("c_prime_Pa", spec.get("c_prime_Pa")), "c_prime_Pa")
        phi = _number(row.get("phi_deg", spec.get("phi_deg")), "phi_deg")
        if b <= 0 or min(w, u, c) < 0 or not -90 < angle < 90 or not 0 <= phi < 90:
            raise ValueError(f"Slice {sid}: invalid width, weight, pressure, strength, or angle")
        alpha = math.radians(angle)
        rows.append((sid, b, w, _finite(c*b, "cohesive force"), _finite(u*b, "pore force"),
                     math.sin(alpha), math.cos(alpha), math.tan(math.radians(phi))))
    return rows


def _outward(value):
    """Export finite binary64 endpoints, rounded away from the interval."""
    lo, hi = float(value.a), float(value.b)
    if lo > value.a:
        lo = math.nextafter(lo, -math.inf)
    if hi < value.b:
        hi = math.nextafter(hi, math.inf)
    return [_finite(lo, "interval lower bound"), _finite(hi, "interval upper bound")]


def _raw_interval_forces(spec, seed):
    """Bound the dimensional free-body solution for the exact supplied numbers."""
    from mpmath.ctx_iv import MPIntervalContext
    iv = MPIntervalContext()
    iv.dps = 80
    zero = iv.mpf(0)
    degrees = iv.pi / 180
    data = []
    drive = zero
    largest_pole = zero
    for row in spec["slices"]:
        width = iv.mpf(row["width_m"])
        weight = iv.mpf(row["weight_N_m"])
        angle = iv.mpf(row["base_angle_deg"]) * degrees
        pressure = iv.mpf(row.get("pore_pressure_Pa", 0))
        cohesion = iv.mpf(row.get("c_prime_Pa", spec.get("c_prime_Pa")))
        friction = iv.mpf(row.get("phi_deg", spec.get("phi_deg"))) * degrees
        tangent = iv.tan(friction)
        slope = iv.tan(angle)
        cosine = iv.cos(angle)
        coefficient = (cohesion * width + (weight - pressure * width) * tangent) / cosine**2
        pole = -slope * tangent
        if coefficient.a < 0 or cosine.a <= 0:
            raise ValueError("Unresolved resistance or angle in the interval replay")
        driving = weight * slope
        data.append((coefficient, pole, driving))
        largest_pole = max(largest_pole, pole.b)
        drive += driving
    if drive.a <= 0 or not any(k.a > 0 for k, _, _ in data):
        raise ValueError("Signed drive or strict monotonicity is unresolved")

    def residual(factor):
        return sum((k / (factor - p) for k, p, _ in data), zero) - drive

    # The binary64 root only suggests where to search. Prove both signs anew.
    center = iv.mpf(seed)
    step = iv.mpf(math.ulp(seed))
    for _ in range(125):
        right = center + step
        if right.b <= largest_pole or residual(right.b).b >= 0:
            step *= 2
            continue
        left = center - step
        if left.a <= largest_pole:
            left = (largest_pole + right) / 2
            for _ in range(260):
                if left.a <= largest_pole or residual(left.a).a > 0:
                    break
                left = (largest_pole + left) / 2
            if left.a <= largest_pole or residual(left.a).a <= 0:
                raise ValueError("The root cannot be separated from its pole")
        if left.a > largest_pole and residual(left.a).a > 0:
            break
        step *= 2
    else:
        raise ValueError("A strictly admissible root bracket was not established")
    root = iv.mpf([left.a, right.b])
    for _ in range(16):
        midpoint = root.mid
        derivative = -sum((k / (root-p)**2 for k, p, _ in data), zero)
        if derivative.b >= 0:
            raise ValueError("Root monotonicity is unresolved")
        newton = midpoint - residual(midpoint) / derivative
        lower, upper = max(root.a, newton.a), min(root.b, newton.b)
        if lower > upper:
            raise ValueError("Root enclosures are disjoint")
        updated = iv.mpf([lower, upper])
        if updated.a == root.a and updated.b == root.b:
            break
        root = updated

    direction = 1 if spec["slice_order"] == "toe_to_crest" else -1
    increments = [direction * (k / (root - p) - driving) for k, p, driving in data]
    forward = []
    total = zero
    for delta in increments[:-1]:
        total += delta
        forward.append(total)
    total = zero
    bounds = []
    for left, delta in zip(reversed(forward), reversed(increments[1:])):
        total -= delta
        lower, upper = max(left.a, total.a), min(left.b, total.b)
        if lower > upper:
            raise ValueError("Opposite interval sweeps disagree at an interface")
        bounds.append(_outward(iv.mpf([lower, upper])))
    return _outward(root), list(reversed(bounds))


def _free_body_sums(components):
    """Prefix sums of projected binary64 forces and their cancellation scales."""
    total = magnitude = Fraction(0)
    values = []
    for first, second in components:
        a, b = Fraction(first), Fraction(second)
        total += a + b
        magnitude += abs(a) + abs(b)
        values.append((total, magnitude))
    return values


def _interslice_thrust(rows, details, spec, seed):
    order = spec["slice_order"]
    direction = 1 if order == "toe_to_crest" else -1
    root, intervals = _raw_interval_forces(spec, seed)
    forward, reverse = [], []
    for row, solved in zip(rows, details):
        _, _, w, _, _, si, co, _ = row
        normal, shear = solved["normal_force_N_m"], solved["mobilized_shear_N_m"]
        forward.append((direction * shear * co, -direction * normal * si))
        # Eliminate N using N*cos(alpha) + T*sin(alpha) = W and start at the other end.
        reverse.append((-direction * shear / co, direction * w * (si / co)))
    left = _free_body_sums(forward)
    right = list(reversed(_free_body_sums(reversed(reverse))))
    faces, tensile, indeterminate, max_error = [], [], [], 0.0
    subnormal = Fraction(math.ulp(0.0))
    secant = math.ceil(1 / min(row[6] for row in rows))
    for i, (lo, hi) in enumerate(intervals):
        fl, sl = left[i]
        fr, sr = right[i + 1]
        error = float(abs(fl - fr) / (sl + sr)) if sl + sr else 0.0
        max_error = max(max_error, error)
        force = _finite(float(fl if sl <= sr else fr), "interface thrust")
        shown = Fraction(force)
        radius = max(abs(fl-shown) + Fraction(256*_EPS)*sl + 8*(i+1)*subnormal,
                     abs(fr-shown) + Fraction(256*_EPS)*sr + 8*secant*(len(rows)-i-1)*subnormal,
                     abs(Fraction(lo)-shown), abs(Fraction(hi)-shown))
        tolerance = _finite(float(radius), "interface tolerance")
        if Fraction(tolerance) < radius:
            tolerance = _finite(math.nextafter(tolerance, math.inf), "interface tolerance")
        ids = [rows[i][0], rows[i + 1][0]]
        if force > tolerance and lo > 0:
            sign = "compression"
        elif force < -tolerance and hi < 0:
            sign = "tension"
            tensile.append(ids)
        else:
            sign = "indeterminate"
            indeterminate.append(ids)
        faces.append(dict(between_slice_ids=ids, total_thrust_N_m=force, sign=sign,
                          sign_tolerance_N_m=tolerance, total_thrust_interval_N_m=[lo, hi]))
    # A separately enclosed root also qualifies the published binary64 factor.
    # Fraction keeps this comparison meaningful for subnormal factors.
    stated = Fraction(seed)
    root_agrees = all(abs(stated-Fraction(bound)) <= Fraction(1e-10)*stated for bound in root)
    return dict(slice_order=order, factor_of_safety_interval=root,
                factor_of_safety_replay_verified=root_agrees, interfaces=faces,
                tensile_interfaces=tensile, indeterminate_interfaces=indeterminate,
                max_relative_interslice_replay_error=max_error,
                interslice_force_balance_verified=max_error <= 1e-10,
                interslice_compression_verified=max_error <= 1e-10 and not tensile and not indeterminate)


def _solve(spec):
    rows = _rows(spec)
    scale = max(max(w, cb, ub) for _, _, w, cb, ub, _, _, _ in rows)
    if scale == 0:
        raise ValueError("No positive driving load or available basal resistance")
    normalized, coefficients, poles, driving_terms = [], [], [], []
    for sid, b, w, cb, ub, si, co, t in rows:
        w, cb, ub = w/scale, cb/scale, ub/scale
        available = math.fsum((cb, (w-ub)*t))
        if available < 0:
            raise ValueError(f"Slice {sid}: negative available basal resistance")
        coefficients.append(_finite(available/co/co, "root coefficient"))
        poles.append(_finite(-si*t/co, "denominator pole"))
        driving_terms.append(_finite(w*si/co, "signed drive"))
        normalized.append((sid, b, w, cb, ub, si, co, t))
    drive = math.fsum(driving_terms)
    drive_scale = math.fsum(abs(d) for d in driving_terms)
    if drive <= 128*_EPS*drive_scale:
        raise ValueError("Signed driving resultant must be positive and numerically resolved")
    pole = max(0., max(poles))
    low = math.nextafter(pole, math.inf)

    def balance(factor):
        terms = []
        for k, p in zip(coefficients, poles):
            if k:
                term = k/(factor-p)
                if math.isinf(term):
                    return math.inf
                terms.append(term)
        try:
            return math.fsum(terms)-drive
        except OverflowError:
            return math.inf

    if balance(low) <= 0:
        raise ValueError("No resolved positive root above every denominator pole")
    high = _finite(math.nextafter(pole+math.fsum(coefficients)/drive, math.inf), "upper bracket")
    for _ in range(16):
        if balance(high) <= 0:
            break
        high = _finite(math.nextafter(high*2., math.inf), "upper bracket")
    else:
        raise ValueError("No representable upper safety-factor bracket")
    for iteration in range(1, 1101):
        mid = low+(high-low)/2
        if mid == low or mid == high or high-low <= 4*_EPS*high:
            break
        if balance(mid) > 0:
            low = mid
        else:
            high = mid
    factor = min((low, high, low+(high-low)/2), key=lambda f: abs(balance(f)))
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Safety factor is not representable")

    details, tension, horizontal, vertical_errors = [], [], [], []
    for (sid, b, w, cb, ub, si, co, t), p in zip(normalized, poles):
        water = ub/co
        # Total normal from vertical equilibrium and the mobilized shear law.
        q = co*(factor-p)
        normal = _finite((factor*w-(cb-ub*t)*si/co)/q, "base normal")
        effective = normal-water
        shear = _finite((cb/co+effective*t)/factor, "base shear")
        vertical = math.fsum((normal*co, shear*si, -w))
        horizontal.append(math.fsum((shear*co, -normal*si)))
        contact_scale = max(abs(normal), water, w, cb/co, sys.float_info.min)
        if effective < -256*_EPS*contact_scale:
            tension.append(sid)
        vertical_errors.append(abs(vertical)/max(w, abs(normal*co), abs(shear*si), sys.float_info.min))
        details.append(dict(id=sid, base_length_m=_finite(b/co, "base length"),
                            m_alpha=_finite(q/factor, "m_alpha"),
                            normal_force_N_m=_finite(normal*scale, "normal force"),
                            effective_normal_force_N_m=_finite(effective*scale, "effective normal"),
                            mobilized_shear_N_m=_finite(shear*scale, "mobilized shear"),
                            vertical_residual_N_m=_finite(vertical*scale, "vertical residual")))
    hres = math.fsum(horizontal)
    equation_error = abs(balance(factor))/drive
    equilibrium = max(vertical_errors) <= 1e-10 and abs(hres)/drive_scale <= 1e-10 and equation_error <= 1e-10
    out = dict(factor_of_safety=factor, solution_found=equilibrium, equilibrium_verified=equilibrium,
               base_contact_verified=not tension, tensile_slice_ids=tension,
               enforced_equilibrium=["slice_vertical_force", "overall_horizontal_force"],
               horizontal_force_residual_N_m=_finite(hres*scale, "horizontal residual"),
               max_relative_vertical_residual=max(vertical_errors), relative_equation_residual=equation_error,
               root_bracket=[low, high], iterations=iteration, slices=details,
               slice_geometry_verified=False, critical_surface_search_performed=False,
               scope="Specified slice table and uncorrected simplified Janbu; no interslice shear, "
                     "geometry reconstruction, critical-surface search or Janbu correction factor. "
                     "Global moment equilibrium, interslice thrust and thrust-line admissibility are not verified.")
    consistent = True
    if "proposed_factor_of_safety" in spec:
        consistent = math.isclose(spec["proposed_factor_of_safety"], factor, rel_tol=1e-7)
        out["proposed_factor_consistent"] = consistent
    verified = equilibrium and not tension and consistent
    if "slice_order" in spec:
        try:
            out.update(_interslice_thrust(rows, details, spec, factor))
        except (ValueError, ArithmeticError) as exc:
            out.update(slice_order=spec["slice_order"], interfaces=[], tensile_interfaces=[],
                       indeterminate_interfaces=[], interslice_force_balance_verified=False,
                       interslice_compression_verified=False, interval_replay_error=str(exc))
        if "proposed_factor_of_safety" in spec and "factor_of_safety_interval" in out:
            consistent = consistent and all(math.isclose(spec["proposed_factor_of_safety"], bound, rel_tol=1e-7)
                                            for bound in out["factor_of_safety_interval"])
            out["proposed_factor_consistent"] = consistent
        verified = (verified and consistent and out["interslice_compression_verified"]
                    and out.get("factor_of_safety_replay_verified", False))
        out["scope"] = ("Specified ordered slice table, free ends, total horizontal interslice forces, "
                        "and uncorrected simplified Janbu. Effective side pressure, thrust-line "
                        "location, moment equilibrium, geometry and critical surface remain unverified.")
    out.update(ok=verified, verified=verified, conservation_verified=verified)
    if "required_factor_of_safety" in spec:
        required = spec["required_factor_of_safety"]
        out.update(required_factor_of_safety=required, meets_required_factor=verified and out.get("factor_of_safety_interval", [factor])[0] >= required)
    if tension:
        out["reason"] = "Equilibrium requires tensile effective base contact; no contact correction applied"
    elif not equilibrium:
        out["reason"] = "Reconstructed force residuals did not qualify the root"
    elif not consistent:
        out["reason"] = "Proposed factor does not match the computed factor"
    elif "slice_order" in spec and not out["interslice_force_balance_verified"]:
        out["reason"] = out.get("interval_replay_error", "Opposite free-body reconstructions disagree")
    elif out.get("factor_of_safety_replay_verified") is False:
        out["reason"] = "The ordinary safety factor disagrees with the independently enclosed root"
    elif out.get("tensile_interfaces"):
        out["reason"] = "Ordered slices require tensile total interslice thrust"
    elif out.get("indeterminate_interfaces"):
        out["reason"] = "An interface thrust sign is numerically unresolved"
    return out


def verify_janbu_slices(spec):
    """Return a qualified safety factor, diagnostic root, or finite refusal."""
    result = dict(tool="hamon_conservation_cad_verifier", mode="civil_geotech",
                  calculation="janbu_slices", method="janbu_simplified_uncorrected")
    try:
        result.update(_solve(spec))
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        result.update(ok=False, verified=False, conservation_verified=False, solution_found=False, reason=str(exc))
    return result
