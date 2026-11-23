"""Rankine limit pressures on a smooth vertical plane in level, cohesionless layers."""
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def verify_at_rest_plane_stress(spec):
    """Calculate stress at a horizontal plane using a prescribed Jaky coefficient.

    The dry/saturated unit weights belong to the materials above the plane.
    The friction angle belongs to the material in which stress is requested.
    Water pressure is hydrostatic. Omitting the water table means dry overburden.
    """
    result = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
              "calculation": "at_rest_plane_stress", "ok": False,
              "conservation_verified": False}
    allowed = {"mode", "calculation", "coefficient_model", "receiving_phi_deg",
               "overburden_layers", "water_table_depth_m", "gamma_water_N_m3",
               "surcharge_Pa", "proposed_horizontal_effective_stress_Pa"}
    try:
        if spec.keys() - allowed or spec.get("coefficient_model") != "jaky":
            raise ValueError("Specify the Jaky at-rest model and supported plane inputs")
        phi = _finite(spec["receiving_phi_deg"], "receiving_phi_deg")
        if not 0 <= phi < 90:
            raise ValueError("Require 0 <= receiving_phi_deg < 90")
        water = (_finite(spec["water_table_depth_m"], "water_table_depth_m")
                 if "water_table_depth_m" in spec else math.inf)
        gw = _finite(spec.get("gamma_water_N_m3", 9810.), "gamma_water_N_m3")
        surcharge = _finite(spec.get("surcharge_Pa", 0.), "surcharge_Pa")
        if water < 0 or gw <= 0 or surcharge < 0:
            raise ValueError("Invalid water table, water unit weight or surcharge")
        layers = spec["overburden_layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError("Provide overburden layers from the surface to the plane")

        depth = 0.
        effective = [surcharge]
        total = [surcharge]
        layer_keys = {"thickness_m", "gamma_above_water_N_m3", "gamma_saturated_N_m3"}
        for layer in layers:
            if not isinstance(layer, dict) or layer.keys() - layer_keys:
                raise ValueError("Unsupported overburden layer fields")
            h = _finite(layer["thickness_m"], "thickness_m")
            if h <= 0 or not math.isfinite(depth + h) or depth + h <= depth:
                raise ValueError("Layer must have positive resolvable thickness")
            dry = min(h, max(0., water - depth))
            submerged = h - dry
            if dry:
                gamma = _finite(layer["gamma_above_water_N_m3"], "gamma_above_water_N_m3")
                if gamma <= 0:
                    raise ValueError("Dry unit weight must be positive")
                effective.append(dry * gamma)
                total.append(dry * gamma)
            if submerged:
                gamma = _finite(layer["gamma_saturated_N_m3"], "gamma_saturated_N_m3")
                if gamma <= gw:
                    raise ValueError("Saturated soil must have positive submerged unit weight")
                effective.append(submerged * (gamma - gw))
                total.append(submerged * gamma)
            depth += h

        vertical = math.fsum(effective)
        vertical_total = math.fsum(total)
        pore = gw * max(0., depth - water)
        # 1 - sin(phi), in a form that retains precision near 90 degrees.
        k0 = 2. * math.sin(math.radians((90. - phi) / 2.)) ** 2
        horizontal = k0 * vertical
        horizontal_total = horizontal + pore
        if not all(math.isfinite(x) for x in (vertical, vertical_total, pore,
                                                horizontal, horizontal_total)):
            raise ValueError("Stress exceeds finite numerical range")
        # The reconstruction sums total loads first and removes pore pressure at
        # the plane. Its rounding scale is total stress when subtraction cancels.
        residual = abs(vertical - (vertical_total - pore))
        verified = residual <= 1.e-12 * vertical_total
        if "proposed_horizontal_effective_stress_Pa" in spec:
            claim = _finite(spec["proposed_horizontal_effective_stress_Pa"], "proposed stress")
            verified = verified and math.isclose(horizontal, claim, rel_tol=1.e-6)
        result.update(ok=True, conservation_verified=verified, coefficient_model="jaky",
                      coefficient=k0, depth_m=depth, pore_pressure_Pa=pore,
                      vertical_total_stress_Pa=vertical_total, vertical_effective_stress_Pa=vertical,
                      horizontal_total_stress_Pa=horizontal_total,
                      horizontal_effective_stress_Pa=horizontal, effective_balance_residual_Pa=residual)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result


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


def _net_water_head(depth, rear, front):
    """Difference of hydrostatic heads, including below two nearly coincident waterlines."""
    if depth <= min(rear, front) or rear == front:
        return 0.0
    if rear < front:
        return min(depth, front) - rear
    return -(min(depth, rear) - front)


def verify_rankine_layers(spec):
    """Integrate signed loads towards the exposed face, per metre of wall.

    Both water tables are hydrostatic and measured downwards from the common
    wall top. Front water acts on the exposed wall, without opposing soil.
    """
    result = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
              "calculation": "rankine_layers", "ok": False, "conservation_verified": False}
    try:
        allowed = {"calculation", "mode", "pressure_state", "layers", "surcharge_Pa",
                   "water_table_depth_m", "front_water_table_depth_m", "gamma_water_N_m3",
                   "wall_friction_deg", "wall_batter_deg", "backfill_slope_deg", "cohesion_Pa",
                   "proposed_thrust_N_m", "proposed_moment_Nm_m"}
        if spec.keys() - allowed:
            raise ValueError("Unsupported Rankine wall fields: " + ", ".join(sorted(spec.keys() - allowed)))
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
        front = (_finite(spec["front_water_table_depth_m"], "front_water_table_depth_m")
                 if "front_water_table_depth_m" in spec else math.inf)
        gw = _finite(spec.get("gamma_water_N_m3", 9810), "gamma_water_N_m3")
        vertical = _finite(spec.get("surcharge_Pa", 0), "surcharge_Pa")
        if min(water, front, vertical) < 0 or gw <= 0:
            raise ValueError("Require nonnegative depth and surcharge, and positive water unit weight")
        height = math.fsum(_finite(layer["thickness_m"], "thickness_m") for layer in layers)
        z = 0.0
        soil_force = water_force = soil_moment = water_moment = 0.0
        quadrature_force = quadrature_moment = residual = 0.0
        profile = []
        layer_keys = {"thickness_m", "phi_deg", "gamma_above_water_N_m3",
                      "gamma_submerged_N_m3", "gamma_saturated_N_m3", "cohesion_Pa"}
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict) or layer.keys() - layer_keys:
                raise ValueError("Unsupported Rankine soil layer fields")
            end = z + _finite(layer["thickness_m"], "thickness_m")
            if end <= z or _finite(layer.get("cohesion_Pa", 0), "cohesion_Pa") != 0:
                raise ValueError("Each layer requires positive thickness and zero cohesion")
            dry = None
            if z < water or "gamma_above_water_N_m3" in layer:
                dry = _finite(layer["gamma_above_water_N_m3"], "gamma_above_water_N_m3")
                if dry <= 0:
                    raise ValueError("Above-water unit weight must be positive")
            wet = None
            if end > water:
                if "gamma_saturated_N_m3" in layer:
                    saturated = _finite(layer["gamma_saturated_N_m3"], "gamma_saturated_N_m3")
                    wet = saturated - gw
                if "gamma_submerged_N_m3" in layer:
                    supplied = _finite(layer["gamma_submerged_N_m3"], "gamma_submerged_N_m3")
                    if wet is not None and not math.isclose(wet, supplied, rel_tol=1.e-6):
                        raise ValueError("Saturated and submerged weights disagree with water unit weight")
                    wet = supplied
                if wet is None or wet <= 0:
                    raise ValueError("Require positive submerged unit weight or saturated weight above water weight")
            k, kn = _coefficients(_finite(layer["phi_deg"], "phi_deg"), state == "passive")
            residual = max(residual, abs(k-kn) / k)
            boundaries = sorted({level for level in (water, front) if z < level < end}) + [end]
            for bottom in boundaries:
                dz = bottom - z
                gamma = wet if z >= water else dry
                sigma_end = vertical + gamma * dz
                p0, p1 = k * vertical, k * sigma_end
                u0, u1 = gw * max(0., z - water), gw * max(0., bottom - water)
                net_u0 = gw * _net_water_head(z, water, front)
                net_u1 = gw * _net_water_head(bottom, water, front)
                for soil, p, q in ((True, p0, p1), (False, net_u0, net_u1)):
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
                    rear_pressure = gw * max(0., depth - water)
                    front_pressure = gw * max(0., depth - front)
                    pressure = math.fsum((kn * (vertical + t*dz*gamma), rear_pressure, -front_pressure))
                    quadrature_force += dz * pressure / 2
                    quadrature_moment += dz * pressure * (height - depth) / 2
                profile.append({"layer": i, "top_m": z, "bottom_m": bottom, "coefficient": k,
                                "effective_top_Pa": p0, "effective_bottom_Pa": p1,
                                "water_top_Pa": u0, "water_bottom_Pa": u1})
                if "front_water_table_depth_m" in spec:
                    profile[-1].update(front_water_top_Pa=gw * max(0., z-front),
                                       front_water_bottom_Pa=gw * max(0., bottom-front),
                                       net_water_top_Pa=net_u0, net_water_bottom_Pa=net_u1)
                z, vertical = bottom, sigma_end
        thrust = soil_force + water_force
        moment = soil_moment + water_moment
        residual = max(residual, abs(thrust-quadrature_force)/max(1.,abs(thrust)),
                       abs(moment-quadrature_moment)/max(1.,abs(moment)))
        arm = moment / thrust if thrust else None
        if not all(math.isfinite(v) for v in (thrust, moment, soil_force,water_force,soil_moment,water_moment,
                                                quadrature_force,quadrature_moment,residual)) or (arm is not None and not math.isfinite(arm)):
            raise ValueError("Pressure or load exceeds finite numerical range")
        valid = residual < 1e-6
        for key, predicted in (("proposed_thrust_N_m", thrust), ("proposed_moment_Nm_m", moment)):
            if key in spec and abs(_finite(spec[key],key)-predicted) > 1e-6*max(1.,abs(predicted)):
                valid = False
        result.update(ok=True, conservation_verified=valid, pressure_state=state, profile=profile,
                      effective_thrust_N_m=soil_force, hydrostatic_thrust_N_m=water_force,
                      thrust_N_m=thrust, effective_moment_Nm_m=soil_moment, hydrostatic_moment_Nm_m=water_moment,
                      moment_Nm_m=moment, resultant_above_base_m=arm,
                      dual_solver_rel_residual=residual)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result
