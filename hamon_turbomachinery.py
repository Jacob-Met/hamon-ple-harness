"""Adiabatic Euler rotor work from signed meridional/tangential velocity triangles."""

import math
from fractions import Fraction


def finite(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def positive(spec, field):
    value = finite(spec[field], field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def matches(observed, expected):
    tolerance = min(1e-6, max(8 * math.ulp(expected), 1e-7 * abs(expected)))
    return abs(observed - expected) <= tolerance


def station(data, omega):
    radius = positive(data, "radius_m")
    meridional = positive(data, "meridional_velocity_m_s")
    speed = float(omega * Fraction(radius))
    alternatives = []
    if "absolute_tangential_velocity_m_s" in data:
        direct = finite(
            data["absolute_tangential_velocity_m_s"], "absolute_tangential_velocity_m_s"
        )
        alternatives.append((direct, math.ulp(direct)))
    for angle, offset in (
        ("absolute_flow_angle_deg", 0),
        ("relative_flow_angle_deg", speed),
    ):
        if angle in data:
            degrees = finite(data[angle], angle)
            if abs(degrees) >= 90:
                raise ValueError(
                    f"{angle} must lie strictly between -90 and 90 degrees"
                )
            radians = math.radians(degrees)
            tangent = math.tan(radians)
            component = meridional * tangent
            value = offset + component
            # A relative velocity angle can nearly cancel a large blade speed.
            # Propagate angle resolution through d(tan θ)/dθ = 1 + tan²θ.
            angle_error = math.radians(math.ulp(degrees)) + math.ulp(radians)
            tangent_error = (1 + tangent * tangent) * angle_error + math.ulp(tangent)
            roundoff = 4 * (
                meridional * tangent_error
                + math.ulp(component)
                + math.ulp(offset)
                + math.ulp(value)
            )
            alternatives.append((value, roundoff))
    if not alternatives:
        raise ValueError(
            "Supply an absolute tangential velocity or a signed flow angle at each station"
        )
    tangential, reference_error = min(alternatives, key=lambda item: item[1])
    for value, roundoff in alternatives:
        tolerance = min(1e-6, 1e-7 * abs(tangential) + reference_error + roundoff)
        if abs(value - tangential) > tolerance:
            raise ValueError("Conflicting station velocity and flow-angle measurements")
    relative = tangential - speed
    result = {
        "blade_speed_m_s": speed,
        "meridional_velocity_m_s": meridional,
        "absolute_tangential_velocity_m_s": tangential,
        "relative_tangential_velocity_m_s": relative,
        "absolute_speed_m_s": math.hypot(meridional, tangential),
        "relative_speed_m_s": math.hypot(meridional, relative),
        "absolute_flow_angle_deg": math.degrees(math.atan2(tangential, meridional)),
        "relative_flow_angle_deg": math.degrees(math.atan2(relative, meridional)),
    }
    if "blade_metal_angle_deg" in data:
        metal = finite(data["blade_metal_angle_deg"], "blade_metal_angle_deg")
        if abs(metal) >= 90:
            raise ValueError(
                "Blade metal angle must lie strictly between -90 and 90 degrees"
            )
        result["flow_minus_metal_angle_deg"] = result["relative_flow_angle_deg"] - metal
    return result


def npsh_at_suction_flange(data):
    def quantity(field, default=None):
        value = data[field] if default is None else data.get(field, default)
        finite(value, field)
        # Integer tokens retain their full value. For a parsed float, use its
        # shortest round-trippable decimal spelling, as displayed to the caller.
        return Fraction(str(value))

    density = quantity("density_kg_m3")
    if density <= 0:
        raise ValueError("density_kg_m3 must be positive")
    gauge = "suction_gauge_pressure_Pa" in data
    atmospheric = "local_atmospheric_pressure_Pa" in data
    if gauge or atmospheric:
        if not gauge or not atmospheric or "suction_absolute_pressure_Pa" in data:
            raise ValueError("Supply absolute suction pressure or a gauge reading paired with local atmospheric pressure")
        atmosphere = quantity("local_atmospheric_pressure_Pa")
        if atmosphere <= 0:
            raise ValueError("local_atmospheric_pressure_Pa must be positive")
        pressure = quantity("suction_gauge_pressure_Pa") + atmosphere
    else:
        pressure = quantity("suction_absolute_pressure_Pa")
    if pressure <= 0:
        raise ValueError("Absolute suction pressure must be positive")
    vapor = quantity("vapor_pressure_Pa")
    velocity = quantity("suction_velocity_m_s")
    required = quantity("required_npsh_m")
    if vapor < 0 or velocity < 0 or required < 0:
        raise ValueError(
            "Vapor pressure, suction speed and required NPSH must be nonnegative"
        )
    elevation = quantity("gauge_elevation_above_npsh_datum_m", 0)
    loss = quantity("loss_to_npsh_datum_m", 0)
    if loss < 0:
        raise ValueError("Loss between the measurement point and NPSH datum must be nonnegative")
    # Compare the signed total-head margin before rounding the displayed head.
    # The suction flange and an elevated gauge use the same comparison datum.
    g = Fraction(980665, 100000)
    head = ((pressure - vapor) / (density * g)
            + velocity ** 2 / (2 * g) + elevation - loss)
    margin = head - required
    return {"available_npsh_m": float(head), "npsh_margin_m": float(margin),
            "meets_required_npsh": margin >= 0}


def rotor(spec):
    kind = spec["machine_kind"]
    if kind not in ("pump", "compressor", "turbine"):
        raise ValueError("machine_kind must be pump, compressor, or turbine")
    omega = Fraction(positive(spec, "rotation_rpm")) * Fraction(math.tau) / 60
    inlet, outlet = (station(spec[k], omega) for k in ("inlet", "outlet"))
    cm1, cm2 = (Fraction(s["meridional_velocity_m_s"]) for s in (inlet, outlet))
    vt1, vt2 = (Fraction(s["absolute_tangential_velocity_m_s"]) for s in (inlet, outlet))
    r1, r2 = (Fraction(float(spec[k]["radius_m"])) for k in ("inlet", "outlet"))
    # Rounding each r*Vtheta product before subtraction can turn a loaded
    # rotor into a zero-torque rotor. Round the signed change only at output.
    angular_momentum_change = r2 * vt2 - r1 * vt1
    specific_work = omega * angular_momentum_change
    work = float(specific_work)
    # Reconstruct energy independently from rothalpy using the components.
    # A rounded speed norm can conceal changes smaller than its own ulp.
    u1, u2 = omega * r1, omega * r2
    wt1, wt2 = vt1 - u1, vt2 - u2
    meridional_term = (cm2 - cm1) * (cm2 + cm1)
    relative_term = (wt2 - wt1) * (wt2 + wt1)
    static_change = float(((u2 - u1) * (u2 + u1) - relative_term - meridional_term) / 2)
    kinetic_change = float((meridional_term + (vt2 - vt1) * (vt2 + vt1)) / 2)
    energy_work = math.fsum((static_change, kinetic_change))
    residual = abs(work - energy_work)
    compatible = angular_momentum_change >= 0 if kind in ("pump", "compressor") else angular_momentum_change <= 0
    out = {
        "ok": True,
        "calculation": "euler_rotor",
        "inlet": inlet,
        "outlet": outlet,
        "fluid_specific_work_J_kg": work,
        "rotor_static_enthalpy_change_J_kg": static_change,
        "rothalpy_work_J_kg": energy_work,
        "energy_residual_J_kg": residual,
        "machine_direction_consistent": compatible,
    }
    checks = [compatible, residual < 1e-6]
    if "mass_flow_kg_s" in spec:
        mass = positive(spec, "mass_flow_kg_s")
        out.update(
            fluid_power_W=float(Fraction(mass) * specific_work),
            fluid_torque_N_m=float(Fraction(mass) * angular_momentum_change),
        )
    if "stage_boundary_speeds_m_s" in spec:
        speeds = spec["stage_boundary_speeds_m_s"]
        v1, v2 = (finite(speeds[k], k) for k in ("inlet", "outlet"))
        if min(v1, v2) < 0:
            raise ValueError("Stage boundary speeds must be nonnegative")
        # One rotor per adiabatic stage; its stationary row does no shaft work.
        stage_change = work - 0.5 * (v2 - v1) * (v2 + v1)
        if stage_change == 0:
            raise ValueError(
                "Stage static enthalpy change is zero; reaction is undefined"
            )
        out.update(
            stage_static_enthalpy_change_J_kg=stage_change,
            degree_of_reaction=static_change / stage_change,
        )
    if "isentropic_specific_work_J_kg" in spec:
        ideal = finite(
            spec["isentropic_specific_work_J_kg"], "isentropic_specific_work_J_kg"
        )
        if work == 0 or ideal * work <= 0:
            raise ValueError(
                "Isentropic stage work must have the same nonzero sign as fluid work"
            )
        efficiency = work / ideal if kind == "turbine" else ideal / work
        out["total_to_total_isentropic_efficiency"] = efficiency
        checks.append(0 < efficiency <= 1 + 1e-12)
    if "suction_flange" in spec:
        if kind != "pump":
            raise ValueError("Liquid suction-flange NPSH check requires a pump")
        out.update(npsh_at_suction_flange(spec["suction_flange"]))
        checks.append(out["meets_required_npsh"])
    if "measured_rotor_static_enthalpy_change_J_kg" in spec:
        measured = finite(
            spec["measured_rotor_static_enthalpy_change_J_kg"],
            "measured_rotor_static_enthalpy_change_J_kg",
        )
        out["measured_rothalpy_residual_J_kg"] = abs(measured - static_change)
        checks.append(matches(measured, static_change))
    independent = "measured_rotor_static_enthalpy_change_J_kg" in spec
    if "measured_fluid_torque_N_m" in spec:
        if "fluid_torque_N_m" not in out:
            raise ValueError("Measured fluid torque requires mass_flow_kg_s")
        observed = finite(
            spec["measured_fluid_torque_N_m"], "measured_fluid_torque_N_m"
        )
        error = abs(observed - out["fluid_torque_N_m"])
        out["measured_torque_residual_N_m"] = error
        checks.append(matches(observed, out["fluid_torque_N_m"]))
        independent = True
    if "claimed" in spec:
        claimed = spec["claimed"]
        if not isinstance(claimed, dict) or not claimed:
            raise ValueError("claimed requires at least one calculated scalar")
        for k, v in claimed.items():
            if k not in out or type(out[k]) is not float:
                raise ValueError(f"Unsupported scalar claim {k}")
            checks.append(matches(finite(v, k), out[k]))
    for record in (inlet, outlet, out):
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in record.values()
        ):
            raise ValueError("Inputs exceed the representable velocity or energy range")
    out["calculation_consistent"] = all(checks)
    out["independent_measurement_supplied"] = independent
    out["conservation_verified"] = independent and all(checks)
    return out


def verify_turbomachinery(spec):
    try:
        return rotor(spec)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "ok": False,
            "conservation_verified": False,
            "calculation": "euler_rotor",
            "reason": str(exc),
        }

