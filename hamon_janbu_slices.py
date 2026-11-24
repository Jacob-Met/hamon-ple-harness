"""Uncorrected simplified Janbu force equilibrium for supplied slice data."""
import math
import sys

_EPS = sys.float_info.epsilon
_TOP = {"mode", "calculation", "slices", "c_prime_Pa", "phi_deg",
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
    out.update(ok=verified, verified=verified, conservation_verified=verified)
    if "required_factor_of_safety" in spec:
        required = spec["required_factor_of_safety"]
        out.update(required_factor_of_safety=required, meets_required_factor=verified and factor >= required)
    if tension:
        out["reason"] = "Equilibrium requires tensile effective base contact; no contact correction applied"
    elif not equilibrium:
        out["reason"] = "Reconstructed force residuals did not qualify the root"
    elif not consistent:
        out["reason"] = "Proposed factor does not match the computed factor"
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
