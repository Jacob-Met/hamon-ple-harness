"""Simplified Bishop equilibrium for one specified circular, gravity-loaded slice assembly."""
import math

from scipy.optimize import brentq
from hamon_consolidation import _number


def verify_bishop_circle(spec):
    result = dict(mode="civil_geotech", calculation="bishop_slip_circle",
                  tool="hamon_conservation_cad_verifier", ok=False, conservation_verified=False)
    try:
        fields = {"mode", "calculation", "circle", "slide_direction", "slices", "proposed_factor_of_safety"}
        if spec.keys() - fields:
            raise ValueError("Additional loads or material models are unsupported")
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
        rows, terms, poles, driving = [], [], [], []
        for row in slices:
            if not isinstance(row, dict) or row.keys() != keys:
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
            tangent = math.tan(math.radians(angle))
            length = width / cosine
            available = cohesion * width + (weight - pore * width) * tangent
            if available < 0:
                raise ValueError("A slice has negative available basal shear strength")
            terms.append(available / cosine)
            poles.append(-sine * tangent / cosine)
            driving.append(weight * sine)
            rows.append((x, sine, cosine, tangent, length, weight, cohesion, pore))
            previous = right
        drive = math.fsum(driving)
        scale = math.fsum(abs(d) for d in driving)
        if drive <= 1e-12 * scale:
            raise ValueError("The chosen sliding direction has zero, reverse or unresolved driving moment")
        span = math.fsum(terms) / drive
        if not 0 < span < math.inf:
            raise ValueError("Finite positive basal resistance is required")

        # Divide the fixed-point equation by F on the physically admissible
        # side of every pole. Each nonzero term is strictly decreasing.
        lower = math.nextafter(max(0., *poles), math.inf)
        upper = 2 * (lower + span)
        def balance(factor):
            return math.fsum(term / (factor - pole) for term, pole in zip(terms, poles)) - drive
        if not math.isfinite(upper) or balance(lower) <= 0 or balance(upper) >= 0:
            raise ValueError("No resolved positive root on the admissible Bishop branch")
        factor = brentq(balance, lower, upper, xtol=math.ulp(0.), rtol=8*math.ulp(1.), maxiter=512)
        # Independently recover effective base forces from vertical equilibrium
        # and the mobilized Mohr-Coulomb shear law, then sum signed moments.
        controls, mobilized = [], []
        residual = 0.
        for x, sine, cosine, tangent, length, weight, cohesion, pore in rows:
            water = pore * length
            normal = math.fsum((weight, -water*cosine, -cohesion*length*sine/factor)) / (
                cosine + sine*tangent/factor)
            if normal < 0:
                raise ValueError("An effective basal normal force is tensile")
            shear = (cohesion * length + normal * tangent) / factor
            vertical = math.fsum(((normal+water)*cosine, shear*sine, -weight))
            residual = max(residual, abs(vertical) / weight)
            mobilized.append(shear)
            controls.append(dict(mid_x_m=x, base_y_m=cy-radius*cosine, base_length_m=length,
                                 downhill_base_angle_deg=math.degrees(math.asin(sine)),
                                 effective_normal_N_per_m=normal, mobilized_shear_N_per_m=shear))
        resisting = math.fsum(mobilized)
        residual = max(residual, abs(resisting-drive)/drive)
        moments = (radius * drive, radius * resisting)
        if (not all(math.isfinite(v) for row in controls for v in row.values())
                or not all(math.isfinite(v) for v in (*moments, residual)) or residual >= 1e-8):
            raise ValueError("Slice forces or global moment balance did not resolve")
        accepted = "proposed_factor_of_safety" not in spec or math.isclose(
            _number(spec, "proposed_factor_of_safety"), factor, rel_tol=1e-6)
        result.update(ok=True, conservation_verified=accepted, factor_of_safety=factor,
                      driving_moment_Nm_per_m=moments[0], resisting_moment_Nm_per_m=moments[1],
                      equilibrium_rel_residual=residual, slices=controls,
                      scope="specified lower circular surface, midpoint slices, unit width, gravity and basal pore pressure")
    except (KeyError, TypeError, ValueError, ArithmeticError, RuntimeError) as err:
        result["reason"] = str(err)
    return result
