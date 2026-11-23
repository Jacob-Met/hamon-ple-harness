"""McCabe–Thiele verification for a supplied piecewise-linear binary VLE curve."""
from bisect import bisect_right
from fractions import Fraction
from math import isfinite


def _number(value):
    if isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError("nonfinite_or_invalid_distillation_input")
    return Fraction(str(value))


def _interpolate(x, a, b):
    return a[1] + (b[1]-a[1]) * (x-a[0]) / (b[0]-a[0])


class _Equilibrium:
    def __init__(self, raw):
        if not isinstance(raw, (tuple, list)) or len(raw) < 2:
            raise ValueError("invalid_equilibrium_points")
        self.points = []
        for row in raw:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("invalid_equilibrium_points")
            x, y = map(_number, row)
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("invalid_equilibrium_points")
            if self.points and self.points[-1][0] >= x:
                raise ValueError("equilibrium_liquid_points_must_increase")
            self.points.append((x, y))
        self.xs = [x for x, _ in self.points]
        self.ys = [y for _, y in self.points]
        self.increasing = all(a < b for a, b in zip(self.ys, self.ys[1:]))

    def at(self, x):
        if not self.xs[0] <= x <= self.xs[-1]:
            raise ValueError("equilibrium_table_does_not_cover_products")
        i = min(bisect_right(self.xs, x)-1, len(self.points)-2)
        return _interpolate(x, self.points[i], self.points[i+1])

    def inverse(self, y):
        if self.increasing:
            if self.ys[0] <= y <= self.ys[-1]:
                i = min(bisect_right(self.ys, y)-1, len(self.ys)-2)
                return _interpolate(y, self.points[i][::-1], self.points[i+1][::-1])
        else:
            roots = set()
            for a, b in zip(self.points, self.points[1:]):
                if min(a[1], b[1]) <= y <= max(a[1], b[1]):
                    if a[1] == b[1]:
                        raise ValueError("ambiguous_equilibrium_inverse")
                    roots.add(_interpolate(y, a[::-1], b[::-1]))
                    if len(roots) > 1:
                        raise ValueError("ambiguous_equilibrium_inverse")
            if roots:
                return roots.pop()
        raise ValueError("stage_outside_equilibrium_table")


def _minimum(points, xb, z, xd, q, d, floor):
    # At composition x, either section may lie below the equilibrium line.
    # Its threshold in R is the smaller of the two line-specific ratios.
    # Both ratios have affine numerator and common positive affine denominator
    # on each table segment. Each is monotone there; their ordering changes
    # only where q*x + (1-q)*y = z. Include every such crossing, not just one.
    candidates = list(points)
    for a, b in zip(points, points[1:]):
        ga = q*a[0] + (1-q)*a[1] - z
        gb = q*b[0] + (1-q)*b[1] - z
        if ga*gb < 0:
            x = a[0] - ga*(b[0]-a[0])/(gb-ga)
            candidates.append((x, _interpolate(x, a, b)))
    bounds = []
    for x, y in candidates:
        head = y-x
        if head <= 0:
            raise ValueError("equilibrium_on_or_below_diagonal_in_product_interval")
        rect = (xd-x)/head - 1
        strip = ((1-d)*(x-xb)/head + (1-q))/d - 1
        bounds.append((min(rect, strip), x))
    pinch, location = max(bounds)
    return max(floor, pinch), location if pinch > floor else None


def tabulated_distillation(params):
    result = {"domain": "electrochem_separations", "subkind": "distillation",
              "equilibrium_model": "piecewise_linear_table", "verified": False}
    try:
        eq = _Equilibrium(params["equilibrium_points"])
        xb, z, xd = (_number(params[k]) for k in ("x_B", "z_F", "x_D"))
        q, feed = _number(params.get("q", 1)), _number(params.get("F_mol_s", 100))
        if not (0 < xb < z < xd < 1 and feed > 0):
            raise ValueError("invalid_distillation_compositions_or_flow")
        points = [(xb, eq.at(xb)), *((x,y) for x,y in eq.points if xb < x < xd),
                  (xd, eq.at(xd))]
        d = (z-xb)/(xd-xb)
        floor = max(Fraction(0), (1-q)/d-1)
        minimum, pinch = _minimum(points, xb, z, xd, q, d, floor)
        result.update(R_flow_floor=float(floor), R_min=float(minimum),
                      limiting_pinch_x=None if pinch is None else float(pinch))
        if params.get("R") is None:
            factor = _number(params.get("R_mult", 1.25))
            if factor < 0:
                raise ValueError("invalid_reflux_multiplier")
            reflux = factor*minimum
        else:
            reflux = _number(params["R"])
        vapor = (reflux+1)*d
        strip_vapor = vapor-(1-q)
        result.update(R_operating=float(reflux), D_mol_s=float(feed*d),
                      B_mol_s=float(feed*(1-d)), L_rect_mol_s=float(feed*reflux*d),
                      V_rect_mol_s=float(feed*vapor), V_strip_mol_s=float(feed*strip_vapor),
                      L_strip_mol_s=float(feed*(strip_vapor+1-d)))
        if reflux < 0 or strip_vapor < 0:
            raise ValueError("negative_internal_section_flow")
        if reflux == 0 and strip_vapor == 0:
            if eq.inverse(xd) != xb:
                raise ValueError("zero_circulation_flash_products_not_in_equilibrium")
            return {**result, "section_mode": "single_equilibrium_flash", "verified": True,
                    "N_stages_integer": 1, "N_theoretical": 1., "feed_stage": 1, "residual": 0.}
        # Each line is active on its side of the feed intersection. For
        # zero stripping vapour the rectifier extends down to the bottoms.
        switch = ((reflux+1)*z+(q-1)*xd)/(reflux+q)

        def operating(x):
            if x >= switch or strip_vapor == 0:
                return (reflux*x+xd)/(reflux+1)
            return x + (1-d)*(x-xb)/strip_vapor

        # On each interval partitioned by both table knots and the feed
        # switch the difference is affine: endpoint minima certify its entirety.
        locations = {x for x, _ in points}
        if xb < switch < xd:
            locations.add(switch)
        gap, where = min((eq.at(x)-operating(x), x) for x in locations)
        result.update(minimum_operating_gap=float(gap), minimum_gap_x=float(where))
        if gap <= 0:
            raise ValueError("operating_line_intersects_equilibrium")
        mode = ("stripping_only" if reflux == 0 else
                "rectifying_only" if strip_vapor == 0 else "two_section")
        result["section_mode"] = mode
        vapor_y, previous, feed_stage = xd, xd, None
        for stage in range(1, 2049):
            x = eq.inverse(vapor_y)
            if x >= previous:
                raise ValueError("non_descending_equilibrium_stage")
            if feed_stage is None and x <= switch:
                feed_stage = stage
            if x <= xb:
                fraction = (previous-xb)/(previous-x)
                return {**result, "verified": True, "N_stages_integer": stage,
                        "N_theoretical": float(stage-1+fraction),
                        "feed_stage": feed_stage or stage, "residual": 0.}
            if max(x.numerator.bit_length(), x.denominator.bit_length()) > 32768:
                break
            previous, vapor_y = x, operating(x)
        raise ValueError("tabulated_stage_resolution_limit")
    except (ValueError, OverflowError, ZeroDivisionError, KeyError, TypeError) as exc:
        return {**result, "error": str(exc)}
