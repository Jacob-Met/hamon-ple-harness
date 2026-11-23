"""Vertical bearing on homogeneous soil beneath a horizontal shallow footing."""

import bisect
import math


# USACE EM 1110-1-1905 (1992), Table 4-1: intermediate N_gamma values
# use piecewise-linear interpolation between published general-shear factors.
TERZAGHI_ANGLES = (
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    35,
    36,
    38,
    40,
    42,
    44,
    45,
    46,
    48,
    50,
)
TERZAGHI_GAMMA = (
    0,
    0.2,
    0.4,
    0.6,
    0.9,
    1.2,
    1.7,
    2.3,
    3.0,
    3.9,
    4.9,
    5.8,
    7.8,
    11.7,
    15.7,
    19.7,
    27.9,
    36.0,
    42.4,
    52.0,
    80.0,
    100.4,
    180.0,
    257.0,
    297.5,
    420.0,
    780.1,
    1153.2,
)


def numeric(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def bearing_factors(method, degrees):
    if not 0 <= degrees < 90:
        raise ValueError("Effective friction angle must lie in [0, 90) degrees")
    phi = math.radians(degrees)
    tangent = math.tan(phi)
    if method == "terzaghi":
        if degrees > TERZAGHI_ANGLES[-1]:
            raise ValueError("Terzaghi N_gamma table ends at 50 degrees")
        k = min(
            bisect.bisect_right(TERZAGHI_ANGLES, degrees) - 1, len(TERZAGHI_ANGLES) - 2
        )
        fraction = (degrees - TERZAGHI_ANGLES[k]) / (
            TERZAGHI_ANGLES[k + 1] - TERZAGHI_ANGLES[k]
        )
        ng = TERZAGHI_GAMMA[k] + fraction * (TERZAGHI_GAMMA[k + 1] - TERZAGHI_GAMMA[k])
        log_nq = (1.5 * math.pi - phi) * tangent - math.log1p(-math.sin(phi))
        nc0 = 1 + 1.5 * math.pi
    else:
        log_nq = math.pi * tangent + 2 * math.atanh(math.sin(phi))
        margin = math.expm1(log_nq)
        if method == "meyerhof":
            if 1.4 * degrees >= 90:
                raise ValueError("Meyerhof N_gamma is undefined at this friction angle")
            ng = margin * math.tan(1.4 * phi)
        else:
            ng = 2 * (margin + 2) * tangent
        nc0 = 2 + math.pi
    nc = math.expm1(log_nq) / tangent if degrees else nc0
    return nc, math.exp(log_nq), ng


def geometry_factors(method, degrees, ratio, embedment, shape, nc, nq):
    shape_f, depth_f, _ = _geometry_factors_with_increments(method, degrees, ratio, embedment, shape, nc, nq)
    return shape_f, depth_f


def _geometry_factors_with_increments(method, degrees, ratio, embedment, shape, nc, nq):
    # Shape and embedment in USACE EM 1110-1-1905, Tables 4-2, 4-3 and 4-6.
    if method == "terzaghi":
        if embedment > 1:
            raise ValueError("Terzaghi shallow footing requires depth <= width")
        if shape not in ("strip", "square", "circular"):
            raise ValueError(
                "Terzaghi shape factors cover strip, square and circular footings"
            )
        sc, sg = {"strip": (1.0, 1.0), "square": (1.3, 0.8), "circular": (1.3, 0.6)}[
            shape
        ]
        return (sc, 1.0, sg), (1.0, 1.0, 1.0), (0.0, 0.0)
    if shape == "circular":
        raise ValueError(
            "Use the declared strip or rectangular geometry for this method"
        )
    phi = math.radians(degrees)
    if method == "meyerhof":
        # Table 4-3 gives the cohesion factors at the actual friction angle.
        root_kp = math.tan(math.pi / 4 + phi / 2) if degrees else 1.0
        sc = 1 + 0.2 * ratio * root_kp**2
        dc = 1 + 0.2 * embedment * root_kp
        # Only the wedge and surcharge columns interpolate from zero to 10°.
        reference_kp_root = math.tan(math.pi / 4 + math.radians(max(10, degrees)) / 2)
        mix = min(1.0, degrees / 10)
        sq_increment = 0.1 * ratio * reference_kp_root**2 * mix
        dq_increment = 0.1 * embedment * reference_kp_root * mix
        sq, dq = 1 + sq_increment, 1 + dq_increment
        return (sc, sq, sq), (dc, dq, dq), (sq_increment, dq_increment)
    # Historical unnamed routes use the original Vesic factor set with depth
    # represented by surcharge alone. Preserve their rectangular correction.
    sc = 1 + ratio * (nq / nc if degrees or method == "legacy_vesic" else 0.2)
    sq_increment = ratio * math.tan(phi)
    sq = 1 + sq_increment
    sg = 1 - 0.4 * ratio if degrees or method == "legacy_vesic" else 1.0
    if method == "legacy_vesic":
        return (sc, sq, sg), (1.0, 1.0, 1.0), (sq_increment, 0.0)
    k = embedment if embedment <= 1 else math.atan(embedment)
    dq_increment = 2 * math.tan(phi) * (1 - math.sin(phi)) ** 2 * k
    return (sc, sq, sg), (1 + 0.4 * k, 1 + dq_increment, 1.0), (sq_increment, dq_increment)


def calculate(spec, legacy_rectangle=False):
    method = spec.get("bearing_method", "legacy_vesic")
    if method not in ("legacy_vesic", "terzaghi", "meyerhof", "vesic"):
        raise ValueError("Unknown bearing_method")

    def get(k, default=None):
        return numeric(spec.get(k, default), k)

    b = get("footing_B_m")
    depth = get("depth_Df_m")
    gamma = get("gamma_N_m3")
    phi = get("phi_deg", 30)
    cohesion = get("c_prime_Pa", 0)
    fs = get("FS", 3)
    if min(b, gamma, fs) <= 0 or min(depth, cohesion) < 0:
        raise ValueError(
            "Invalid footing dimension, strength, unit weight or factor of safety"
        )
    if fs < 1:
        raise ValueError("Design factor of safety must be at least one")
    length = get("footing_L_m") if "footing_L_m" in spec else None
    if length is not None:
        if length <= 0:
            raise ValueError("Plan dimensions must be positive")
    shape = spec.get(
        "footing_shape",
        "strip" if length is None else ("square" if length == b else "rectangular"),
    )
    if shape == "strip" and length is not None:
        raise ValueError("Strip footing has no finite plan length")
    if shape == "square" and length is not None and length != b:
        raise ValueError("Square footing requires equal dimensions")
    if shape == "square" and length is None:
        length = b
    if shape == "rectangular" and length is None:
        raise ValueError("Rectangular footing requires both dimensions")
    if shape not in ("strip", "square", "rectangular", "circular"):
        raise ValueError("Unknown footing shape")
    if shape == "circular" and length is not None:
        raise ValueError("Circular footing uses its diameter alone")
    if any(
        get(k, 0) != 0
        for k in (
            "horizontal_load_N",
            "base_tilt_deg",
            "ground_slope_deg",
        )
    ):
        raise ValueError(
            "Only vertical loading, horizontal base and level ground are represented"
        )
    offset_b = get("eccentricity_B_m", 0)
    offset_l = get("eccentricity_L_m", 0)
    eccentric_method = spec.get("eccentricity_method")
    effective = {}
    if eccentric_method is not None or offset_b or offset_l:
        if eccentric_method != "meyerhof_effective_area" or method != "meyerhof":
            raise ValueError("Specify Meyerhof bearing and meyerhof_effective_area for eccentric loading")
        if shape not in ("rectangular", "square") or length is None:
            raise ValueError("Effective-area bearing requires a finite rectangular or square plan")
        if get("column_load_N") <= 0:
            raise ValueError("Effective-area bearing requires a positive vertical load")
        # FHWA GEC 6, eqs. 6-6 through 6-8. Reduce the axes named by the
        # caller before ordering the effective dimensions for the factors.
        b -= 2 * abs(offset_b)
        length -= 2 * abs(offset_l)
        area = b * length
        if min(b, length, area) <= 0 or not math.isfinite(area):
            raise ValueError("The resultant must be inside the plan with a finite positive effective area")
        effective = {"eccentricity_method": eccentric_method, "effective_B_m": b,
                     "effective_L_m": length, "effective_area_m2": area}
    if length is not None:
        b, length = sorted((b, length))
    ratio = 0 if shape == "strip" else b / (length or b)
    nc, nq, ng = bearing_factors(method, phi)
    # Nc = (Nq - 1)/tan(phi); recover the expm1-based excess before the
    # historical rectangle route substitutes its zero-angle cohesion factor.
    nq_increment = nc * math.tan(math.radians(phi))
    if method == "legacy_vesic" and legacy_rectangle and phi <= 1e-6:
        nc = 5.14
    shape_f, depth_f, (sq_increment, dq_increment) = _geometry_factors_with_increments(
        method, phi, ratio, depth / b, shape, nc, nq
    )
    wt = (
        max(0.0, get("water_table_depth_m"))
        if "water_table_depth_m" in spec
        else math.inf
    )
    submerged = get("gamma_sat_N_m3", gamma) - 9806.65
    if wt < depth + b and submerged <= 0:
        raise ValueError("Submerged soil requires positive effective unit weight")
    # Integrate effective weight above the base and over one footing width
    # below it, using the named input's conventional linear water-table correction.
    dry_top = min(wt, depth)
    dry_below = max(0.0, min(wt - depth, b))
    surcharge = gamma * dry_top + submerged * (depth - dry_top)
    weight = (gamma * dry_below + submerged * (b - dry_below)) / b

    def effective_stress(z):
        below = max(0.0, z - wt)
        return gamma * z + (submerged - gamma) * below

    second_top = effective_stress(depth)
    second_weight = (effective_stress(depth + b) - second_top) / b
    cterm = cohesion * nc * shape_f[0] * depth_f[0]
    qcoeff = nq * shape_f[1] * depth_f[1]
    gcoeff = 0.5 * b * ng * shape_f[2] * depth_f[2]
    total = math.fsum((cterm, qcoeff * surcharge, gcoeff * weight))
    # Subtract the overburden in its own term. Rounding gross capacity first
    # can discard the entire net resistance of a weak soil.
    # Nq*sq*dq - 1 = (Nq-1)*sq*dq + (sq-1)*dq + (dq-1).
    net_qcoeff = math.fsum((nq_increment * shape_f[1] * depth_f[1],
                            sq_increment * depth_f[1], dq_increment))
    net = math.fsum((cterm, net_qcoeff * surcharge, gcoeff * weight))
    other = math.fsum((cterm, qcoeff * second_top, gcoeff * second_weight))
    error = abs(total - other)
    out = {
        "ok": True,
        "calculation": "bearing_capacity",
        "bearing_method": method,
        **effective,
        "footing_shape": shape,
        "Nq": nq,
        "Nc": nc,
        "Ngamma": ng,
        "effective_overburden_Pa": surcharge,
        "effective_wedge_unit_weight_N_m3": weight,
        "cohesion_term_Pa": cterm,
        "surcharge_term_Pa": qcoeff * surcharge,
        "soil_weight_term_Pa": gcoeff * weight,
        "q_ult_Pa": total,
        "q_allow_Pa": total / fs,
        "q_net_ult_Pa": net,
        "q_net_allow_Pa": net / fs,
        "allowable_basis": "gross ultimate / FS; net ultimate / FS",
        "stress_integration_residual_Pa": error,
    }
    checks = [error < 1e-6]
    for name in ("q_ult_Pa", "q_allow_Pa", "q_net_ult_Pa", "q_net_allow_Pa"):
        key = "proposed_" + name
        if key in spec:
            checks.append(
                abs(get(key) - out[name])
                <= min(1e-6, 1e-7 * abs(out[name]) + 8 * math.ulp(out[name]))
            )
    if "column_load_N" in spec:
        if length is None or shape not in ("rectangular", "square"):
            raise ValueError("Column load requires a rectangular or square plan area")
        pressure = get("column_load_N") / (b * length)
        if "applied_gross_pressure_Pa" in spec and not math.isclose(
            pressure, get("applied_gross_pressure_Pa"), rel_tol=1e-10
        ):
            raise ValueError("Inconsistent applied pressure and column load")
        spec = {**spec, "applied_gross_pressure_Pa": pressure}
    if "applied_gross_pressure_Pa" in spec:
        pressure = get("applied_gross_pressure_Pa")
        if pressure < 0:
            raise ValueError("Gross contact pressure must be nonnegative")
        out["bearing_ok"] = pressure <= out["q_allow_Pa"]
        out["q_applied_Pa"] = pressure
        checks.append(out["bearing_ok"])
    if not all(math.isfinite(v) for v in out.values() if isinstance(v, float)):
        raise ValueError("Bearing calculation exceeds numerical range")
    out["conservation_verified"] = all(checks)
    return out


def verify_bearing_capacity(spec, *, legacy_rectangle=False):
    try:
        if legacy_rectangle:
            # Historical domain API supplies finite rectangular geometry and
            # total force. Keep the original names and field requirements.
            spec = {
                **spec,
                "footing_B_m": spec["B_m"],
                "footing_L_m": spec["L_m"],
                "depth_Df_m": spec["Df_m"],
                "c_prime_Pa": spec["c_Pa"],
                "phi_deg": spec["phi_deg"],
                "FS": spec["FS"],
                "column_load_N": spec["column_load_N"],
            }
        return calculate(spec, legacy_rectangle)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        return {
            "ok": False,
            "conservation_verified": False,
            "calculation": "bearing_capacity",
            "reason": str(exc),
        }


def _strip_quantity(value, field, unit):
    """Numbers are SI; text must carry the dimensions of the named input."""
    if isinstance(value, str):
        from hamon_dual_verifier import eval_quantity
        quantity = eval_quantity(value)
        if quantity.dim != eval_quantity(unit).dim:
            raise ValueError(f"{field} must have dimensions {unit}")
        value = quantity.val
    return numeric(value, field)


def verify_rigid_strip_contact(spec):
    """Rigid strip on a compression-only bed: p(x) = max(0, a + b*x).

    x is measured from the base center; integral(p)=N and integral(x*p)=M.
    This is contact equilibrium, not soil bearing capacity or settlement.
    """
    from fractions import Fraction

    calculation = "rigid_strip_contact"
    out = {"ok": False, "calculation": calculation, "conservation_verified": False}
    units = {
        "base_width_m": "m",
        "vertical_resultant_N_m": "N/m",
        "moment_Nm_m": "N*m/m",
        "contact_width_m": "m",
        "eccentricity_m": "m",
        "q_min_Pa": "Pa",
        "q_max_Pa": "Pa",
        "q_left_Pa": "Pa",
        "q_right_Pa": "Pa",
    }
    scalars = ("contact_width_m", "eccentricity_m", "q_min_Pa",
               "q_max_Pa", "q_left_Pa", "q_right_Pa")
    allowed = {"mode", "kind", "calculation", "pressure_model",
               "base_width_m", "vertical_resultant_N_m", "moment_Nm_m",
               "proposed_contact_interval_m", "proposed_contact_regime"}
    allowed.update("proposed_" + key for key in scalars)

    def rendered(value, name):
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result) or (value and not result):
            raise ValueError(f"{name} is outside the representable numeric range")
        return result

    def claim_matches(value, expected, key):
        if expected is None:
            return value is None
        supplied = Fraction(_strip_quantity(value, "proposed_" + key, units[key]))
        rounded = rendered(expected, key)
        tolerance = abs(expected) / 10**9 + 8 * Fraction(math.ulp(rounded))
        return abs(supplied - expected) <= tolerance

    try:
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError("Unsupported rigid strip inputs: " + ", ".join(sorted(unknown)))
        if spec.get("calculation", calculation) != calculation:
            raise ValueError("Expected calculation=rigid_strip_contact")
        if spec.get("pressure_model", "positive_part_affine") != "positive_part_affine":
            raise ValueError("Only pressure_model=positive_part_affine is represented")
        b, normal, moment = (
            Fraction(_strip_quantity(spec[key], key, units[key]))
            for key in ("base_width_m", "vertical_resultant_N_m", "moment_Nm_m")
        )
        if b <= 0:
            raise ValueError("base_width_m must be positive")
        if normal < 0 or (normal == 0 and moment != 0):
            return {**out, "equilibrium_status": "no_finite_equilibrium",
                    "reason": "Nonnegative contact pressure cannot supply this normal force and moment"}
        half = b / 2
        if normal > 0 and abs(moment) >= normal * half:
            return {**out, "equilibrium_status": "no_finite_equilibrium",
                    "reason": "The resultant is at or beyond a base edge; no finite contact pressure equilibrates it"}

        if normal == 0:
            regime = "unloaded"
            width = eccentricity = interval = None
            q_left = q_right = Fraction(0)
            contact_pressures = [Fraction(0), Fraction(0)]
            integrated_force = integrated_moment = Fraction(0)
        else:
            eccentricity = moment / normal
            if 6 * abs(moment) <= normal * b:
                regime = "full_contact"
                width = b
                left, right = -half, half
                q_left = normal / b - 6 * moment / b**2
                q_right = normal / b + 6 * moment / b**2
                contact_pressures = [q_left, q_right]
            else:
                regime = "partial_contact"
                width = 3 * (half - abs(eccentricity))
                peak = 2 * normal / width
                if moment > 0:
                    left, right = half - width, half
                    q_left, q_right = Fraction(0), peak
                    contact_pressures = [Fraction(0), peak]
                else:
                    left, right = -half, -half + width
                    q_left, q_right = peak, Fraction(0)
                    contact_pressures = [peak, Fraction(0)]
            interval = [left, right]
            pa, pb = contact_pressures
            # Integrate the actual returned affine segment independently of
            # the inverse formulas. Keep regime decisions on the parsed data
            # exact: a rounded M/N can otherwise hide a valid narrow contact.
            integrated_force = width * (pa + pb) / 2
            integrated_moment = width * ((2 * left + right) * pa
                                         + (left + 2 * right) * pb) / 6
            if integrated_force != normal or integrated_moment != moment:
                raise ArithmeticError("Contact pressure does not satisfy both equilibrium integrals")

        exact = {
            "contact_width_m": width, "eccentricity_m": eccentricity,
            "q_min_Pa": min(q_left, q_right), "q_max_Pa": max(q_left, q_right),
            "q_left_Pa": q_left, "q_right_Pa": q_right,
        }
        coordinates = None if interval is None else [
            rendered(value, "contact_interval_m") for value in interval
        ]
        if coordinates is not None and coordinates[0] >= coordinates[1]:
            raise ValueError("Contact interval is unresolved in the numeric output precision")
        out.update({
            "ok": True, "equilibrium_status": "finite",
            "pressure_model": "positive_part_affine", "contact_regime": regime,
            "base_width_m": rendered(b, "base_width_m"),
            "vertical_resultant_N_m": rendered(normal, "vertical_resultant_N_m"),
            "moment_Nm_m": rendered(moment, "moment_Nm_m"),
            **{key: rendered(value, key) for key, value in exact.items()},
            "contact_interval_m": coordinates,
            "contact_edge_pressure_Pa": [
                rendered(value, "contact_edge_pressure_Pa") for value in contact_pressures
            ],
            "integrated_vertical_resultant_N_m": rendered(integrated_force, "integrated_vertical_resultant_N_m"),
            "integrated_moment_Nm_m": rendered(integrated_moment, "integrated_moment_Nm_m"),
            "force_residual_N_m": 0.0, "moment_residual_Nm_m": 0.0,
        })

        output_geometry_verified = True
        if coordinates is not None:
            a, z = map(Fraction, coordinates)
            pa, pz = map(Fraction, out["contact_edge_pressure_Pa"])
            displayed_force = (z-a)*(pa+pz)/2
            displayed_moment = (z-a)*((2*a+z)*pa+(a+2*z)*pz)/6
            force_error = abs(displayed_force-normal)
            moment_error = abs(displayed_moment-moment)
            output_geometry_verified = (
                force_error <= abs(normal)/10**9
                and moment_error <= abs(normal*b)/10**9
            )
            out["displayed_force_residual_N_m"] = rendered(displayed_force-normal, "displayed_force_residual_N_m")
            out["displayed_moment_residual_Nm_m"] = rendered(displayed_moment-moment, "displayed_moment_residual_Nm_m")
        out["output_geometry_verified"] = output_geometry_verified
        checks = {}
        for key in scalars:
            proposed = "proposed_" + key
            if proposed in spec:
                checks[proposed] = claim_matches(spec[proposed], exact[key], key)
        if "proposed_contact_interval_m" in spec:
            value = spec["proposed_contact_interval_m"]
            if interval is None:
                checks["proposed_contact_interval_m"] = value is None
            else:
                checks["proposed_contact_interval_m"] = (
                    isinstance(value, (list, tuple)) and len(value) == 2
                    and all(claim_matches(v, expected, "contact_width_m")
                            for v, expected in zip(value, interval))
                )
        if "proposed_contact_regime" in spec:
            checks["proposed_contact_regime"] = spec["proposed_contact_regime"] == regime
        out["claim_checks"] = checks
        out["conservation_verified"] = output_geometry_verified and all(checks.values())
        return out
    except (KeyError, TypeError, ValueError, ArithmeticError, SyntaxError) as exc:
        return {"ok": False, "calculation": calculation,
                "conservation_verified": False, "reason": str(exc)}
