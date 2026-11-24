"""Simplified Bishop equilibrium for a specified circular, gravity-loaded assembly."""
import math

from hamon_bishop import verify_bishop_simplified, _number


def verify_bishop_circle(spec):
    result = dict(mode="civil_geotech", calculation="bishop_slip_circle",
                  tool="hamon_conservation_cad_verifier", ok=False, conservation_verified=False,
                  equilibrium_verified=False, compressive_contact_verified=False,
                  factor_of_safety=None)
    try:
        fields = {"mode", "calculation", "circle", "slide_direction", "slices", "proposed_factor_of_safety"}
        if not isinstance(spec, dict) or set(spec) - fields:
            raise ValueError("Additional loads or material models are unsupported")
        if spec.get("mode", "auto") not in ("auto", "civil", "geotech", "civil_geotech"):
            raise ValueError("The slice assembly requires civil_geotech mode")
        circle = spec["circle"]
        if not isinstance(circle, dict) or set(circle) != {"center_x_m", "center_y_m", "radius_m"}:
            raise ValueError("Specify the circular surface center and radius in metres")
        cx, cy, radius = (_number(circle, key) for key in ("center_x_m", "center_y_m", "radius_m"))
        direction = {"right": 1, "left": -1}.get(spec["slide_direction"])
        slices = spec["slices"]
        if radius <= 0 or direction is None or not isinstance(slices, list) or not 1 <= len(slices) <= 256:
            raise ValueError("Require a positive trial radius, left or right sliding, and 1 to 256 slices")
        keys = {"left_x_m", "right_x_m", "weight_N_per_m", "cohesion_Pa", "friction_angle_deg", "pore_pressure_Pa"}
        previous = None
        rows, geometry = [], []
        for row in slices:
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError("Every slice needs horizontal bounds, weight, effective strength and basal pore pressure")
            left, right, weight, cohesion, angle, pore = (_number(row, k) for k in (
                "left_x_m", "right_x_m", "weight_N_per_m", "cohesion_Pa", "friction_angle_deg", "pore_pressure_Pa"))
            if (left >= right or (previous is not None and previous != left)
                    or max(abs(left-cx), abs(right-cx)) > radius):
                raise ValueError("Slices must be contiguous, increasing and inside the trial circle")
            if weight <= 0 or cohesion < 0 or not 0 <= angle < 90 or pore < 0:
                raise ValueError("Specify positive weight and nonnegative effective strength and pore pressure")
            width = right - left
            x = .5 * left + .5 * right
            sine = direction * (cx - x) / radius
            cosine = math.sqrt((1-sine)*(1+sine))
            if cosine <= 0 or x == left or x == right:
                raise ValueError("The midpoint tangent is unresolved")
            alpha = math.degrees(math.asin(sine))
            rows.append(dict(width_m=width, weight_N_m=weight, base_angle_deg=alpha,
                             cohesion_Pa=cohesion, phi_deg=angle, pore_pressure_Pa=pore))
            geometry.append(dict(mid_x_m=x, base_y_m=cy-radius*cosine,
                                 base_length_m=width/cosine, downhill_base_angle_deg=alpha))
            previous = right
        arguments = dict(calculation="bishop_simplified", slip_surface="circular",
                         circle_radius_m=radius, slices=rows)
        if "proposed_factor_of_safety" in spec:
            arguments["proposed_factor_of_safety"] = spec["proposed_factor_of_safety"]
        solved = verify_bishop_simplified(arguments)
        if not solved["ok"]:
            result["reason"] = solved["reason"]
            return result
        controls = []
        for geo, reaction in zip(geometry, solved["slice_reactions"]):
            controls.append(dict(geo, effective_normal_N_per_m=reaction["effective_normal_N_m"],
                                 mobilized_shear_N_per_m=reaction["mobilized_shear_N_m"],
                                 total_normal_N_per_m=reaction["total_normal_N_m"],
                                 pore_force_N_per_m=reaction["pore_force_N_m"],
                                 vertical_force_relative_residual=reaction["vertical_force_relative_residual"],
                                 shear_strength_relative_residual=reaction["shear_strength_relative_residual"]))
        if not all(math.isfinite(value) for row in controls for value in row.values()):
            raise ValueError("Slice geometry or force exceeds the numerical range")
        result.update(solved)
        result.update(calculation="bishop_slip_circle", slices=controls,
                      resisting_moment_Nm_per_m=solved["mobilized_resisting_moment_Nm_per_m"],
                      equilibrium_rel_residual=max(solved["moment_relative_residual"],
                                                  solved["max_vertical_force_relative_residual"],
                                                  solved["max_shear_strength_relative_residual"]),
                      slice_geometry_verified=True,
                      geometry_scope="contiguous horizontal bounds and midpoint tangents on the specified lower circle",
                      scope="specified lower circular surface, midpoint slices, unit width, gravity and basal pore pressure")
    except (KeyError, TypeError, ValueError, ArithmeticError, RuntimeError) as err:
        result["reason"] = str(err)
    return result
