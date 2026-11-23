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
    ka = math.tan(math.radians((90 - angle_deg) / 2)) ** 2
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


def _integrate_face(spec, grade=0.):
    """Integrate signed loads towards the exposed face, per metre of wall.

    Both water tables and the face grade use the common wall-top datum.
    Equal water tables integrate just the effective earth load on this face.
    """
    result = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
              "calculation": "rankine_layers", "ok": False, "conservation_verified": False}
    try:
        allowed = {"calculation", "mode", "pressure_state", "layers", "surcharge_Pa",
                   "water_table_depth_m", "front_water_table_depth_m", "gamma_water_N_m3",
                   "wall_friction_deg", "wall_batter_deg", "backfill_slope_deg", "cohesion_Pa"}
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
        height = grade + math.fsum(_finite(layer["thickness_m"], "thickness_m") for layer in layers)
        z = grade
        soil_force = water_force = soil_moment = water_moment = 0.0
        quadrature_force = quadrature_moment = residual = 0.0
        profile = []
        layer_keys = {"thickness_m", "phi_deg", "gamma_above_water_N_m3",
                      "gamma_submerged_N_m3", "gamma_saturated_N_m3", "cohesion_Pa"}
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict) or layer.keys() - layer_keys:
                raise ValueError("Unsupported Rankine soil layer fields")
            end = z + _finite(layer["thickness_m"], "thickness_m")
            if i == len(layers) - 1:
                end = height
            if not math.isfinite(end) or end <= z or _finite(layer.get("cohesion_Pa", 0), "cohesion_Pa") != 0:
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
                    pressure = kn * (vertical + t*dz*gamma) + gw * _net_water_head(depth, water, front)
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
        residual = max(residual, abs(thrust-quadrature_force)/max(1.,abs(soil_force),abs(water_force)),
                       abs(moment-quadrature_moment)/max(1.,abs(soil_moment),abs(water_moment)))
        if not all(math.isfinite(v) for v in (thrust, moment, soil_force,water_force,soil_moment,water_moment,
                                                quadrature_force,quadrature_moment,residual)):
            raise ValueError("Pressure or load exceeds finite numerical range")
        valid = residual < 1e-6
        result.update(ok=True, conservation_verified=valid, pressure_state=state, profile=profile,
                      effective_thrust_N_m=soil_force, hydrostatic_thrust_N_m=water_force,
                      thrust_N_m=thrust, effective_moment_Nm_m=soil_moment, hydrostatic_moment_Nm_m=water_moment,
                      moment_Nm_m=moment, quadrature=(quadrature_force, quadrature_moment),
                      dual_solver_rel_residual=residual)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        result["reason"] = str(exc)
    return result


def _resolve_opposed_loads(spec, rear_profile, front_profile):
    """Integrate opposed force and moment contributions before rounding each face."""
    from sympy import Rational, S, pi, tan
    from sympy.core.evalf import PrecisionExhausted

    def exact(value):
        return Rational(float(value))

    # Use the validated public interface coordinates on both faces. A water
    # table coincident with a published layer boundary has no wet sliver.
    height = exact(rear_profile[-1]["bottom_m"])
    gamma_water = exact(spec.get("gamma_water_N_m3", 9810))
    rear_water = exact(spec["water_table_depth_m"]) if "water_table_depth_m" in spec else height
    front_water = exact(spec["front_water_table_depth_m"]) if "front_water_table_depth_m" in spec else height
    # Group vertical-stress integrals by their Rankine coefficient before
    # subtracting opposed loads. A fictitious subdivision cannot introduce
    # a jump in stress or create a force or couple in a balanced diagram.
    force, moment = {}, {}
    faces = [(spec["layers"], rear_profile, spec.get("surcharge_Pa", 0),
              rear_water, spec["pressure_state"] == "passive", 1)]
    if front_profile:
        faces.append((spec["front_layers"], front_profile, spec.get("front_surcharge_Pa", 0),
                      front_water, spec["pressure_state"] == "active", -1))
    for layers, profile, surcharge, water, passive, sign in faces:
        stress = exact(surcharge)
        for section in profile:
            row = layers[section["layer"]]
            top, bottom = exact(section["top_m"]), exact(section["bottom_m"])
            coefficient = (exact(row["phi_deg"]), passive)
            if top < water:
                gamma = exact(row["gamma_above_water_N_m3"])
            elif "gamma_submerged_N_m3" in row:
                gamma = exact(row["gamma_submerged_N_m3"])
            else:
                gamma = exact(row["gamma_saturated_N_m3"]) - gamma_water
            dz = bottom - top
            change = gamma * dz
            area = dz * (stress + change / 2)
            toe_moment = area * (height-bottom) + dz**2 * (stress/2 + change/6)
            force[coefficient] = force.get(coefficient, 0) + sign * area
            moment[coefficient] = moment.get(coefficient, 0) + sign * toe_moment
            stress += change

    def rounded(value, resultant):
        try:
            number = float(value.evalf(25, maxn=768, strict=True))
            if resultant and number == 0 and value.is_zero is not True:
                raise ValueError("Nonzero Rankine resultant is below the representable range")
            return number
        except PrecisionExhausted as exc:
            raise ValueError("Could not resolve opposed Rankine loads") from exc

    soil_force = soil_moment = S.Zero
    for (phi, passive), area in force.items():
        k = tan(pi*(90 - phi)/360)**(-2 if passive else 2)
        soil_force += k * area
        soil_moment += k * moment[phi, passive]
    rear_depth = max(S.Zero, height - rear_water)
    front_depth = max(S.Zero, height - front_water)
    water_force = gamma_water * (rear_depth**2 - front_depth**2) / 2
    water_moment = gamma_water * (rear_depth**3 - front_depth**3) / 6
    return {key: rounded(value, key in ("thrust_N_m", "moment_Nm_m")) for key, value in (
        ("effective_thrust_N_m", soil_force), ("hydrostatic_thrust_N_m", water_force),
        ("thrust_N_m", soil_force + water_force), ("effective_moment_Nm_m", soil_moment),
        ("hydrostatic_moment_Nm_m", water_moment), ("moment_Nm_m", soil_moment + water_moment))}


def _classify_load(thrust, moment, force_scale):
    if thrust == 0.:
        return ("pure_couple" if moment else "balanced"), None
    if abs(thrust) <= 8 * math.ulp(force_scale):
        return "force_unresolved_at_roundoff", None
    arm = moment / thrust
    if not math.isfinite(arm):
        raise ValueError("Force line of action exceeds finite numerical range")
    return "force", arm


def verify_rankine_layers(spec):
    """Integrate rear earth minus front earth, plus the signed hydrostatic load.

    Face pressure is positive in compression. Net thrust is positive toward the
    exposed face. Its first moment uses height above the common toe as the arm.
    Each soil column starts at its grade; water depths use the common wall top.
    """
    invalid = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
               "calculation": "rankine_layers", "ok": False, "conservation_verified": False}
    try:
        rear_spec = dict(spec)
        claims = {k: rear_spec.pop(k) for k in ("proposed_thrust_N_m", "proposed_moment_Nm_m")
                  if k in rear_spec}
        front_keys = ("front_layers", "front_surface_depth_m", "front_surcharge_Pa")
        front = {k: rear_spec.pop(k) for k in front_keys if k in rear_spec}
        rear = _integrate_face(rear_spec)
        if not rear["ok"]:
            return rear
        components = [rear["effective_thrust_N_m"], rear["hydrostatic_thrust_N_m"]]
        quadrature = rear.pop("quadrature")
        if front:
            rows = front.get("front_layers")
            if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                raise ValueError("Provide nonempty front_layers from the exposed grade to the toe")
            height = rear["profile"][-1]["bottom_m"]
            thicknesses = [_finite(r["thickness_m"], "thickness_m") for r in rows]
            if min(thicknesses) <= 0:
                raise ValueError("Front soil layers must have positive thickness")
            grade = height - math.fsum(thicknesses)
            if grade < 0:
                raise ValueError("Exposed grade must be at or below the common wall top")
            if "front_surface_depth_m" in front:
                supplied = _finite(front["front_surface_depth_m"], "front_surface_depth_m")
                if supplied < 0 or abs(supplied - grade) > 4 * math.ulp(height):
                    raise ValueError("Front soil profile does not terminate at the common toe")
            state = "passive" if rear["pressure_state"] == "active" else "active"
            front_spec = {"pressure_state": state, "layers": rows,
                          "surcharge_Pa": front.get("front_surcharge_Pa", 0.),
                          "gamma_water_N_m3": rear_spec.get("gamma_water_N_m3", 9810.)}
            if "front_water_table_depth_m" in rear_spec:
                # Water on the whole wall has already been integrated on the rear
                # pass. Equal heads here leave only effective earth pressure.
                water = rear_spec["front_water_table_depth_m"]
                front_spec.update(water_table_depth_m=water, front_water_table_depth_m=water)
            exposed = _integrate_face(front_spec, grade)
            if not exposed["ok"]:
                return exposed
            front_quad = exposed.pop("quadrature")
            quadrature = tuple(math.fsum((a, -b)) for a, b in zip(quadrature, front_quad))
            rear.update(front_profile=exposed["profile"], front_pressure_state=state,
                        front_surface_depth_m=grade)
            for quantity in ("thrust_N_m", "moment_Nm_m"):
                earth = "effective_" + quantity
                rear["rear_" + earth] = rear[earth]
                rear["front_" + earth] = exposed[earth]
                rear[earth] = math.fsum((rear[earth], -exposed[earth]))
                rear[quantity] = math.fsum((rear["rear_" + earth], -exposed[earth],
                                           rear["hydrostatic_" + quantity]))
            components.append(exposed["effective_thrust_N_m"])
            rear["dual_solver_rel_residual"] = max(rear["dual_solver_rel_residual"],
                                                    exposed["dual_solver_rel_residual"])
            rear["conservation_verified"] &= exposed["conservation_verified"]
        force, moment = rear["thrust_N_m"], rear["moment_Nm_m"]
        moment_scale = max(abs(rear.get("rear_effective_moment_Nm_m", rear["effective_moment_Nm_m"])),
                           abs(rear.get("front_effective_moment_Nm_m", 0.)), abs(rear["hydrostatic_moment_Nm_m"]))
        force_scale = max(map(abs, components))
        guard = math.sqrt(math.ulp(1.))
        if abs(force) <= guard * force_scale or abs(moment) <= guard * moment_scale:
            rear.update(_resolve_opposed_loads(spec, rear["profile"], rear.get("front_profile")))
            force, moment = rear["thrust_N_m"], rear["moment_Nm_m"]
        residual = max(rear["dual_solver_rel_residual"],
                       abs(force-quadrature[0])/max(1., *map(abs, components)),
                       abs(moment-quadrature[1])/max(1., moment_scale))
        rear.update(dual_solver_rel_residual=residual,
                    conservation_verified=rear["conservation_verified"] and residual < 1.e-6)
        kind, arm = _classify_load(force, moment, force_scale)
        rear.update(resultant_kind=kind, resultant_above_base_m=arm)
        for key, value in claims.items():
            prediction = rear[key.removeprefix("proposed_")]
            if not math.isclose(_finite(value, key), prediction, rel_tol=1e-6, abs_tol=1e-6):
                rear["conservation_verified"] = False
        return rear
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        invalid["reason"] = str(exc)
        return invalid
