"""Bishop simplified equilibrium for a supplied circular-surface slice table.

USACE EM 1110-2-1902, Appendix F, equations F-1, F-4 and F-6,
specialized to weight and base pore pressure, without external surface loads.
Horizontal interslice forces are assumed; horizontal equilibrium is not checked.
"""
from __future__ import annotations

import math


_TOP_KEYS = {"mode", "domain", "calculation", "slip_surface", "circle_radius_m", "slices",
             "proposed_factor_of_safety", "required_factor_of_safety"}
_SLICE_KEYS = {"id", "width_m", "weight_N_m", "base_angle_deg", "cohesion_Pa",
               "phi_deg", "pore_pressure_Pa"}


def _finite_product(*values):
    """Multiply dimensional scales without overflowing an intermediate product."""
    mantissa, exponent = 1., 0
    for value in values:
        fraction, power = math.frexp(value)
        mantissa *= fraction
        exponent += power
    return math.ldexp(mantissa, exponent)


def _number(source, key, *, default=None, positive=False, nonnegative=False):
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{key} is outside its finite physical range")
    return value


def _root(coefficients, driving):
    # With A_i >= 0 and cos(alpha_i) > 0, sum A_i/(F*C_i+B_i)
    # decreases strictly on the domain where all denominators are positive.
    # This excludes the poles that a fixed-point iteration can cross.
    pole = max(0., max(-b / c for a, b, c in coefficients))

    def residual(factor):
        terms = []
        for a, b, c in coefficients:
            if a == 0:
                continue
            denominator = math.fsum((factor * c, b))
            if denominator <= 0:
                return math.inf
            terms.append(a / denominator)
        return math.fsum(terms) - driving

    lower = math.nextafter(pole, math.inf)
    if not math.isfinite(lower):
        raise ValueError("No representable factor above the base-normal pole")
    if residual(lower) < 0:
        raise ValueError("No positive root with every base-normal factor positive")
    scale = math.fsum(a for a, _, _ in coefficients) / driving
    span = max(1., scale / min(c for _, _, c in coefficients))
    upper = pole + span
    for _ in range(1024):
        if not math.isfinite(upper):
            raise ValueError("The equilibrium factor exceeds the numerical range")
        if upper > lower and residual(upper) <= 0:
            break
        span *= 2
        upper = pole + span
    else:
        raise ValueError("Could not bracket the positive equilibrium factor")

    for iteration in range(1, 2201):
        middle = lower + (upper - lower) / 2
        if middle == lower or middle == upper or upper - lower <= 8 * math.ulp(middle):
            break
        if residual(middle) > 0:
            lower = middle
        else:
            upper = middle
    else:
        raise ValueError("Equilibrium root did not reach floating-point resolution")
    candidates = (lower, lower + (upper - lower) / 2, upper)
    factor = min(candidates, key=lambda value: abs(residual(value)))
    return factor, (lower, upper), iteration


def solve_bishop_simplified(spec):
    """Solve one user-supplied circular surface; no search or geometry inference."""
    if not isinstance(spec, dict) or set(spec) - _TOP_KEYS:
        raise ValueError("Unsupported Bishop inputs; surface loads, seismic forces and reinforcement are not modeled")
    if spec.get("mode", "civil_geotech") not in ("auto", "civil_geotech", "civil", "geotech"):
        raise ValueError("Bishop slices require the civil_geotech solver family")
    if spec.get("domain", "civil_geotech") != "civil_geotech":
        raise ValueError("Bishop slices require domain=civil_geotech when a domain is supplied")
    if spec.get("calculation") != "bishop_simplified" or spec.get("slip_surface") != "circular":
        raise ValueError("Use calculation=bishop_simplified and declare slip_surface=circular")
    radius = _number(spec, "circle_radius_m", positive=True)
    supplied = spec.get("slices")
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= 2048:
        raise ValueError("Provide between 1 and 2048 circular-surface slices")
    rows, identifiers = [], set()
    for index, row in enumerate(supplied):
        if not isinstance(row, dict) or set(row) - _SLICE_KEYS:
            raise ValueError(f"Slice {index + 1} contains unsupported fields")
        identifier = row.get("id", index + 1)
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or str(identifier) in identifiers:
            raise ValueError("Slice ids must be unique strings or integers")
        identifiers.add(str(identifier))
        width = _number(row, "width_m", positive=True)
        weight = _number(row, "weight_N_m", nonnegative=True)
        cohesion = _number(row, "cohesion_Pa", nonnegative=True)
        phi = _number(row, "phi_deg", nonnegative=True)
        alpha = _number(row, "base_angle_deg")
        pressure = _number(row, "pore_pressure_Pa", default=0., nonnegative=True)
        if not -90 < alpha < 90 or phi >= 90:
            raise ValueError("Require -90 < base_angle_deg < 90 and 0 <= phi_deg < 90")
        sine, cosine = math.sin(math.radians(alpha)), math.cos(math.radians(alpha))
        tangent = math.tan(math.radians(phi))
        if (alpha and not sine) or (phi and not tangent):
            raise ValueError("A nonzero slice angle underflowed during trigonometric conversion")
        cb, ub, length = cohesion * width, pressure * width, width / cosine
        if (cohesion and not cb) or (pressure and not ub):
            raise ValueError("A nonzero slice force underflowed during width multiplication")
        if not all(map(math.isfinite, (cb, ub, length))):
            raise ValueError("Slice force or base length exceeds the numerical range")
        rows.append(dict(id=identifier, width_m=width, weight_N_m=weight,
                         base_angle_deg=alpha, cohesion_Pa=cohesion, phi_deg=phi,
                         pore_pressure_Pa=pressure, base_length_m=length,
                         sine=sine, cosine=cosine, tangent=tangent, cb=cb, ub=ub))
    force_scale = max(max(r["weight_N_m"], r["cb"], r["ub"]) for r in rows)
    if force_scale == 0:
        raise ValueError("The slice table has no driving load or resistance")
    coefficients = []
    for row in rows:
        w, cb, ub = (row[key] / force_scale for key in ("weight_N_m", "cb", "ub"))
        if any(row[key] != 0 and scaled == 0 for key, scaled in
               (("weight_N_m", w), ("cb", cb), ("ub", ub))):
            raise ValueError("A nonzero slice force underflowed during normalization")
        a = math.fsum((cb, math.fsum((w, -ub)) * row["tangent"]))
        if not math.isfinite(a) or a < 0:
            raise ValueError(f"Slice {row['id']} cannot supply nonnegative resistance on the compressive branch")
        row.update(w=w, cwidth=cb, uwidth=ub)
        coefficients.append((a, row["sine"] * row["tangent"], row["cosine"]))
    driving = math.fsum(r["w"] * r["sine"] for r in rows)
    absolute_driving = math.fsum(abs(r["w"] * r["sine"]) for r in rows)
    if not math.isfinite(driving) or driving <= 1e-12 * absolute_driving:
        raise ValueError("The signed slice weights must give a resolved positive driving moment")
    if not any(a > 0 for a, _, _ in coefficients):
        raise ValueError("No positive shear resistance is available")
    factor, bracket, iterations = _root(coefficients, driving)
    reactions, shear_terms = [], []
    worst_vertical, worst_strength = 0., 0.
    for row, (a, b, c) in zip(rows, coefficients):
        denominator = math.fsum((factor * c, b))
        if denominator <= 128 * math.ulp(1.) * (abs(factor * c) + abs(b)):
            raise ValueError(f"Slice {row['id']} has an unresolved base-normal factor near a Bishop pole")
        shear = a / denominator
        pore_force = row["uwidth"] / c
        normal = math.fsum((row["w"], -shear * row["sine"])) / c
        effective_normal = math.fsum((normal, -pore_force))
        if effective_normal < 0:
            raise ValueError(f"Slice {row['id']} requires a tensile effective normal force")
        # Reconstruct the original force and strength equations, independently
        # of the rational scalar equation used to locate the root.
        vertical_terms = (normal * c, shear * row["sine"], -row["w"])
        strength_terms = (factor * shear, -row["cwidth"] / c,
                          -effective_normal * row["tangent"])
        vertical_scale = max(map(abs, vertical_terms))
        strength_scale = max(map(abs, strength_terms))
        vertical = abs(math.fsum(vertical_terms)) / vertical_scale if vertical_scale else 0.
        strength = abs(math.fsum(strength_terms)) / strength_scale if strength_scale else 0.
        worst_vertical, worst_strength = max(worst_vertical, vertical), max(worst_strength, strength)
        shear_terms.append(shear)
        reactions.append(dict(id=row["id"], base_length_m=row["base_length_m"],
                              m_alpha=denominator / factor,
                              total_normal_N_m=normal * force_scale,
                              pore_force_N_m=pore_force * force_scale,
                              effective_normal_N_m=effective_normal * force_scale,
                              mobilized_shear_N_m=shear * force_scale,
                              vertical_force_relative_residual=vertical,
                              shear_strength_relative_residual=strength))
    resisting = math.fsum(shear_terms)
    moment_error = abs(math.fsum((resisting, -driving))) / max(resisting, driving)
    if max(worst_vertical, worst_strength, moment_error) > 1e-10:
        raise ValueError("The reconstructed slice-force or overall-moment equations did not close")
    driving_moment, resisting_moment = (_finite_product(v, force_scale, radius) for v in (driving, resisting))
    if driving_moment <= 0 or resisting_moment <= 0:
        raise ValueError("The positive equilibrium moments underflow the numerical range")
    values = [factor, *bracket, driving_moment, resisting_moment,
              *(r[k] for r in reactions for k in ("m_alpha", "total_normal_N_m", "pore_force_N_m",
                                                "effective_normal_N_m", "mobilized_shear_N_m"))]
    if not all(map(math.isfinite, values)):
        raise ValueError("An equilibrium output exceeds the numerical range")
    result = dict(factor_of_safety=factor, factor_bracket=list(bracket), root_iterations=iterations,
                  circle_radius_m=radius, slice_reactions=reactions,
                  driving_moment_Nm_per_m=driving_moment,
                  mobilized_resisting_moment_Nm_per_m=resisting_moment,
                  moment_relative_residual=moment_error,
                  max_vertical_force_relative_residual=worst_vertical,
                  max_shear_strength_relative_residual=worst_strength,
                  equilibrium_verified=True, compressive_contact_verified=True,
                  horizontal_force_equilibrium_checked=False,
                  slice_geometry_verified=False, critical_surface_search_performed=False,
                  scope="Bishop simplified equilibrium on the supplied circular-surface slice table")
    if "required_factor_of_safety" in spec:
        required = _number(spec, "required_factor_of_safety", positive=True)
        result["required_factor_of_safety"] = required
        result["selected_surface_meets_required_factor"] = (
            True if bracket[0] >= required else False if bracket[1] < required else None)
    return result


def verify_bishop_simplified(spec):
    result = dict(ok=False, conservation_verified=False, equilibrium_verified=False,
                  mode="civil_geotech", calculation="bishop_simplified",
                  factor_of_safety=None, tool="hamon_conservation_cad_verifier")
    try:
        solved = solve_bishop_simplified(spec)
        proposed = (_number(spec, "proposed_factor_of_safety", positive=True)
                    if "proposed_factor_of_safety" in spec else None)
        matches = proposed is None or math.isclose(proposed, solved["factor_of_safety"], rel_tol=1e-6)
        result.update(solved, ok=True, conservation_verified=matches)
        if proposed is not None:
            result["proposed_factor_matches"] = matches
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result
