"""Finite-grid search for gravity-driven circular failure of a dry homogeneous slope."""
import bisect
import itertools
import math
from collections import Counter

from hamon_bishop import verify_bishop_simplified
from hamon_janbu_slices import verify_janbu_slices


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Coordinates and material parameters must be finite numbers")
    return float(value)


def crossings(profile, cx, cy, radius, depth):
    """Intersect each profile segment shifted down by depth with the lower circle."""
    roots = []
    for (x, y), (x1, y1) in zip(profile, profile[1:]):
        dx, dy = x1 - x, y1 - y
        length = math.hypot(dx, dy)
        tx, ty = dx / length, dy / length
        rx, ry = cx - x, cy - (y - depth)
        along = rx * tx + ry * ty
        distance = abs(rx * ty - ry * tx)
        if distance >= radius:
            continue
        half = math.sqrt((radius - distance) * (radius + distance))
        for along_segment in (along - half, along + half):
            tol = 16 * math.ulp(max(radius, length))
            if -tol <= along_segment <= length + tol:
                along_segment = min(length, max(0., along_segment))
                if y - depth + along_segment * ty <= cy:
                    roots.append(x + along_segment * tx)
    result = []
    for x in sorted(roots):
        if not result or x - result[-1] > 32 * max(math.ulp(x), math.ulp(radius)):
            result.append(x)
    return result


def make_slices(profile, xs, circle, direction, count, floor, crack, unit_weight):
    cx, cy, radius = circle
    intersections = crossings(profile, cx, cy, radius, 0.)
    if len(intersections) != 2:
        return None
    left, right = intersections

    def base(x):
        u = x - cx
        return cy - math.sqrt(max(0., (radius - u) * (radius + u)))

    def ground(x):
        i = min(len(xs) - 2, max(0, bisect.bisect_right(xs, x) - 1))
        (x0, y0), (x1, y1) = profile[i:i + 2]
        return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

    if ground(0.5 * (left + right)) <= base(0.5 * (left + right)):
        return None
    if crack:
        cuts = [x for x in crossings(profile, cx, cy, radius, crack) if left < x < right]
        if not cuts:
            return None
        if direction > 0:
            left = cuts[0]
        else:
            right = cuts[-1]
    if base(min(right, max(left, cx))) < floor:
        return None
    rows = []
    width = (right - left) / count
    for i in range(count):
        a, b = left + i * width, left + (i + 1) * width
        points = [a, *xs[bisect.bisect_right(xs, a):bisect.bisect_left(xs, b)], b]
        areas = []
        for x0, x1 in zip(points, points[1:]):
            midpoint, half = (x0 + x1) / 2, (x1 - x0) / 2
            offset = half / math.sqrt(3)
            depths = [ground(x) - base(x) for x in (midpoint - offset, midpoint + offset)]
            if min(depths) <= 0 or half <= 0:
                return None
            areas.append(half * sum(depths))
        angle = direction * (cx - (a + b) / 2) / radius
        if abs(angle) >= 1:
            return None
        rows.append({"width_m": b - a, "weight_N_m": unit_weight * math.fsum(areas),
                     "base_angle_deg": math.degrees(math.asin(angle))})
    return rows, [left, right]


def verify_critical_slip_grid(spec):
    out = {"tool": "hamon_conservation_cad_verifier", "mode": "civil_geotech",
           "calculation": "critical_slip_circle_grid", "ok": False, "conservation_verified": False}
    try:
        allowed = {"mode", "calculation", "ground_profile_m", "base_elevation_m", "slide_direction",
                   "unit_weight_N_m3", "cohesion_Pa", "friction_angle_deg", "circle_grid", "slices_per_surface",
                   "method", "dry_tension_crack_depth_m"}
        if set(spec) - allowed:
            raise ValueError("Additional loads, pore pressures or material fields are unsupported")
        source = spec["ground_profile_m"]
        if not isinstance(source, list) or not 2 <= len(source) <= 64:
            raise ValueError("Provide 2 to 64 ground-profile vertices")
        profile = [tuple(finite(v) for v in pair) for pair in source]
        if any(len(pair) != 2 for pair in profile):
            raise ValueError("Ground vertices must be [x_m, elevation_m]")
        xs = [pair[0] for pair in profile]
        floor = finite(spec["base_elevation_m"])
        if any(a >= b for a, b in zip(xs, xs[1:])) or any(y < floor for _, y in profile):
            raise ValueError("Ground must increase in x and lie above the basal cutoff")
        direction = {"left": -1, "right": 1}[spec["slide_direction"]]
        method = spec["method"]
        if method not in ("bishop", "janbu_uncorrected"):
            raise ValueError("Use bishop or janbu_uncorrected")
        weight, cohesion, phi = [finite(spec[key]) for key in ("unit_weight_N_m3", "cohesion_Pa", "friction_angle_deg")]
        crack = finite(spec.get("dry_tension_crack_depth_m", 0))
        if weight <= 0 or cohesion < 0 or not 0 <= phi < 90 or crack < 0:
            raise ValueError("Invalid strength, weight or dry rear crack depth")
        count = spec["slices_per_surface"]
        if type(count) is not int or not 2 <= count <= 256:
            raise ValueError("Use 2 to 256 slices per trial")
        grid = spec["circle_grid"]
        keys = ("center_x_m", "center_y_m", "radius_m")
        if not isinstance(grid, dict) or set(grid) != set(keys):
            raise ValueError("Specify explicit center_x_m, center_y_m and radius_m grid axes")
        axes = []
        for key in keys:
            values = grid[key]
            if not isinstance(values, list) or not 1 <= len(values) <= 256:
                raise ValueError("Every grid axis needs 1 to 256 distinct coordinates")
            axis = sorted(finite(v) for v in values)
            if len(set(axis)) != len(axis):
                raise ValueError("Grid axes must have distinct coordinates")
            axes.append(axis)
        trials = math.prod(map(len, axes))
        if axes[2][0] <= 0 or trials * count > 250000:
            raise ValueError("Require positive radii and at most 250000 trial slices")
        counters, best = Counter(), None
        for circle in itertools.product(*axes):
            generated = make_slices(profile, xs, circle, direction, count, floor, crack, weight)
            if generated is None:
                counters["excluded_geometry"] += 1
                continue
            rows, span = generated
            if method == "bishop":
                result = verify_bishop_simplified({"calculation": "bishop_simplified", "slip_surface": "circular",
                    "circle_radius_m": circle[2], "slices": [dict(row, cohesion_Pa=cohesion, phi_deg=phi) for row in rows]})
            else:
                result = verify_janbu_slices({"calculation": "janbu_slices", "slices": rows,
                                              "c_prime_Pa": cohesion, "phi_deg": phi})
            if not result.get("equilibrium_verified") and "requires a tensile effective normal" in result.get("reason", ""):
                counters["excluded_tensile_contact"] += 1
                continue
            if not result.get("equilibrium_verified"):
                counters["unresolved_equilibrium"] += 1
                continue
            if not result.get("compressive_contact_verified", result.get("base_contact_verified")):
                counters["excluded_tensile_contact"] += 1
                continue
            counters["qualified"] += 1
            if best is None or result["factor_of_safety"] < best["factor_of_safety"]:
                best = {"factor_of_safety": result["factor_of_safety"], "circle": dict(zip(keys, circle)),
                        "sliding_span_m": span, "slice_inputs": rows}
        out.update(method=method, trials=trials, counts=dict(counters), minimum=best,
                   scope="Least verified factor on the specified discrete grid of dry homogeneous lower-circle masses",
                   continuous_global_minimum_verified=False, dry_tension_crack_depth_m=crack,
                   grid_completely_evaluated=(counters["unresolved_equilibrium"] == 0))
        if best is None:
            raise ValueError("No candidate supplied a compressive equilibrium")
        out.update(ok=True, conservation_verified=True)
    except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError) as err:
        out["reason"] = str(err)
    return out
