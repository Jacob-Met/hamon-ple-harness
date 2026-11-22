"""Rankine limit pressures on a smooth vertical plane in level, cohesionless layers."""
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _coefficients(angle_deg, passive):
    if not 0 <= angle_deg < 90:
        raise ValueError("Require 0 <= phi_deg < 90")
    if angle_deg == 0:
        return 1.0, 1.0
    a = math.radians(angle_deg)
    ka = math.tan(math.pi / 4 - a / 2) ** 2
    # Independent Mohr circle tangent criterion, solved for the minor/major
    # principal effective stress ratio on [0,1]; reciprocal in passive limit.
    sine = math.sin(a)
    if sine >= 1.0 or not ka:
        raise ValueError("Friction angle is too close to 90 degrees to resolve")
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if (1 - mid) / (1 + mid) > sine:
            lo = mid
        else:
            hi = mid
    numerical = (lo + hi) / 2
    return (1 / ka, 1 / numerical) if passive else (ka, numerical)


def verify_rankine_layers(spec):
    """Report separate effective-soil and water loads per metre of wall width."""
    result = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
              "calculation": "rankine_layers", "ok": False, "conservation_verified": False}
    try:
        state = spec["pressure_state"]
        if state not in ("active", "passive"):
            raise ValueError("pressure_state must be active or passive")
        for key in ("wall_friction_deg", "wall_batter_deg", "backfill_slope_deg", "cohesion_Pa"):
            if _finite(spec.get(key, 0), key) != 0:
                raise ValueError("Only a smooth, vertical plane in level, cohesionless soil is supported")
        layers = spec["layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError("layers must be a nonempty list ordered from the surface")
        water = _finite(spec["water_table_depth_m"], "water_table_depth_m") if "water_table_depth_m" in spec else math.inf
        gw = _finite(spec.get("gamma_water_N_m3", 9810), "gamma_water_N_m3")
        vertical = _finite(spec.get("surcharge_Pa", 0), "surcharge_Pa")
        if min(water, vertical) < 0 or gw <= 0:
            raise ValueError("Require nonnegative depth and surcharge, and positive water unit weight")
        height = math.fsum(_finite(layer["thickness_m"], "thickness_m") for layer in layers)
        z = 0.0
        soil_force = water_force = soil_moment = water_moment = 0.0
        quadrature_force = quadrature_moment = residual = 0.0
        profile = []
        for i, layer in enumerate(layers):
            end = z + _finite(layer["thickness_m"], "thickness_m")
            dry = _finite(layer["gamma_above_water_N_m3"], "gamma_above_water_N_m3")
            if end <= z or dry <= 0 or _finite(layer.get("cohesion_Pa", 0), "cohesion_Pa") != 0:
                raise ValueError("Each layer requires positive thickness and weight and zero cohesion")
            wet = _finite(layer["gamma_submerged_N_m3"], "gamma_submerged_N_m3") if end > water else None
            if wet is not None and wet <= 0:
                raise ValueError("Submerged effective unit weight must be positive")
            k, kn = _coefficients(_finite(layer["phi_deg"], "phi_deg"), state == "passive")
            residual = max(residual, abs(k-kn) / k)
            boundaries = (water, end) if z < water < end else (end,)
            for bottom in boundaries:
                dz = bottom - z
                gamma = wet if z >= water else dry
                sigma_end = vertical + gamma * dz
                p0, p1 = k * vertical, k * sigma_end
                u0, u1 = gw * max(0., z - water), gw * max(0., bottom - water)
                for soil, p, q in ((True, p0, p1), (False, u0, u1)):
                    force = dz * (p + q) / 2
                    moment = force * (height - bottom) + dz**2 * (2*p + q) / 6
                    if soil:
                        soil_force += force
                        soil_moment += moment
                    else:
                        water_force += force
                        water_moment += moment
                for t in ((1-1/math.sqrt(3))/2, (1+1/math.sqrt(3))/2):
                    depth = z + t * dz
                    pressure = kn * (vertical + t*dz*gamma) + gw * max(0., depth - water)
                    quadrature_force += dz * pressure / 2
                    quadrature_moment += dz * pressure * (height - depth) / 2
                profile.append({"layer": i, "top_m": z, "bottom_m": bottom, "coefficient": k,
                                "effective_top_Pa": p0, "effective_bottom_Pa": p1,
                                "water_top_Pa": u0, "water_bottom_Pa": u1})
                z, vertical = bottom, sigma_end
        thrust = soil_force + water_force
        moment = soil_moment + water_moment
        residual = max(residual, abs(thrust-quadrature_force)/max(1.,thrust),
                       abs(moment-quadrature_moment)/max(1.,moment))
        arm = moment / thrust if thrust else None
        if not all(math.isfinite(v) for v in (thrust, moment, soil_force,water_force,soil_moment,water_moment,
                                                quadrature_force,quadrature_moment,residual)) or (arm is not None and not math.isfinite(arm)):
            raise ValueError("Pressure or load exceeds finite numerical range")
        valid = residual < 1e-6
        for key, predicted in (("proposed_thrust_N_m", thrust), ("proposed_moment_Nm_m", moment)):
            if key in spec and abs(_finite(spec[key],key)-predicted) > 1e-6*max(1.,predicted):
                valid = False
        result.update(ok=True, conservation_verified=valid, pressure_state=state, profile=profile,
                      effective_thrust_N_m=soil_force, hydrostatic_thrust_N_m=water_force,
                      thrust_N_m=thrust, effective_moment_Nm_m=soil_moment, hydrostatic_moment_Nm_m=water_moment,
                      moment_Nm_m=moment, resultant_above_base_m=arm,
                      dual_solver_rel_residual=residual)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result
