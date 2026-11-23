"""Binary McCabe–Thiele construction for a tabulated, piecewise-linear equilibrium."""
from bisect import bisect_right
from fractions import Fraction
import math


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric inputs")
    return Fraction(str(value))


def _interpolate(x, xs, ys):
    i = min(bisect_right(xs, x) - 1, len(xs) - 2)
    return ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])


def solve_tabulated_distillation(spec):
    result = {"domain": "electrochem_separations", "subkind": "distillation",
              "equilibrium_model": "piecewise_linear_table", "verified": False}
    try:
        if "alpha" in spec:
            raise ValueError("supply either equilibrium_table or alpha")
        table = spec["equilibrium_table"]
        if not isinstance(table, (list, tuple)) or len(table) < 2:
            raise ValueError("equilibrium_table requires at least two [x,y] pairs")
        if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in table):
            raise ValueError("expected [liquid_fraction,vapor_fraction] pairs")
        xs, ys = zip(*[(_number(x), _number(y)) for x, y in table])
        if (xs[0], ys[0], xs[-1], ys[-1]) != (0, 0, 1, 1):
            raise ValueError("equilibrium_table must span (0,0) to (1,1)")
        if any(a >= b for axis in (xs, ys) for a, b in zip(axis, axis[1:])):
            raise ValueError("equilibrium coordinates must increase strictly on both axes")
        xb, z, xd = (_number(spec[k]) for k in ("x_B", "z_F", "x_D"))
        q = _number(spec.get("q", 1))
        feed = _number(spec.get("F_mol_s", 100))
        if not 0 < xb < z < xd < 1 or feed <= 0:
            raise ValueError("invalid feed or product compositions")
        d = feed * (z - xb) / (xd - xb)
        b = feed - d
        floor = max(Fraction(0), (1 - q) * feed / d - 1)
        result.update(D_mol_s=float(d), B_mol_s=float(b), R_flow_floor=float(floor))

        knots = [xb, *[x for x in xs if xb < x < xd], xd]
        points = [(x, _interpolate(x, xs, ys)) for x in knots]
        for x, y in points:
            if y <= x:
                result.update(limiting_composition=float(x), equilibrium_y=float(y))
                raise ValueError("separation crosses a total-reflux equilibrium barrier")

        # Both operating lines approach the diagonal as R grows. At any x the
        # active line is the lower of the two. Each of its two reflux thresholds
        # is a linear fraction on a table interval, hence monotone or constant.
        # Their ordering changes only where z = q*x + (1-q)*y.
        candidates = list(points)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            h0 = q*x0 + (1-q)*y0 - z
            h1 = q*x1 + (1-q)*y1 - z
            if h0*h1 < 0:
                t = -h0 / (h1-h0)
                candidates.append((x0+t*(x1-x0), y0+t*(y1-y0)))
        lower, limiting_x = floor, None
        for x, y in candidates:
            rect = (xd-x)/(y-x) - 1
            strip = b*(x-xb)/(d*(y-x)) - 1 + (1-q)*feed/d
            needed = min(rect, strip)
            if needed >= lower:
                lower, limiting_x = needed, x
        result.update(R_min=float(lower), limiting_composition=(
            float(limiting_x) if limiting_x is not None else None))
        if spec.get("R") is not None:
            r = _number(spec["R"])
        else:
            multiplier = _number(spec.get("R_mult", 1.25))
            if multiplier < 0:
                raise ValueError("negative reflux multiplier")
            r = multiplier * lower
        l, v = r*d, (r+1)*d
        vs = v - (1-q)*feed
        ls = vs + b
        result.update(R_operating=float(r), L_rect_mol_s=float(l), V_rect_mol_s=float(v),
                      L_strip_mol_s=float(ls), V_strip_mol_s=float(vs))
        if l < 0 or vs < 0:
            raise ValueError("negative internal section flow")

        # The flow floor guarantees r+q>0 for an interior feed composition.
        xi = ((r+1)*z + (q-1)*xd)/(r+q)
        yi = (l*xi+d*xd)/v
        if q*xi + (1-q)*yi != z or vs*yi != ls*xi-b*xb:
            raise ArithmeticError("feed intersection violates the material balances")
        result.update(feed_intersection_x=float(xi), feed_intersection_y=float(yi))
        if not l and not vs:
            if _interpolate(xb, xs, ys) != xd:
                raise ValueError("zero-circulation flash products are not in equilibrium")
            return dict(result, verified=True, section_mode="single_equilibrium_flash",
                        N_stages_integer=1, N_theoretical=1., feed_stage=1, residual=0.)

        def operating(x):
            return (l*x+d*xd)/v if x >= xi or not vs else (ls*x-b*xb)/vs

        # Equilibrium and the active line are affine between table knots and
        # the feed switch, so these values certify the entire composition span.
        active = points + [(xi, _interpolate(xi, xs, ys))]
        gap, worst = min((y-operating(x), x) for x, y in active)
        result.update(min_operating_gap=float(gap), min_gap_composition=float(worst))
        if gap <= 0:
            raise ValueError("operating line reaches or crosses equilibrium")
        mode = "stripping_only" if not l else "rectifying_only" if not vs else "two_section"
        result["section_mode"] = mode
        stage_y, previous, feed_stage = xd, xd, None
        for count in range(1, 4097):
            x = _interpolate(stage_y, ys, xs)
            if x >= previous:
                raise ArithmeticError("stage failed to descend")
            if feed_stage is None and x <= xi:
                feed_stage = count
            if x <= xb:
                fraction = (previous-xb)/(previous-x)
                return dict(result, verified=True, N_stages_integer=count,
                            N_theoretical=float(count-1+fraction), feed_stage=feed_stage or count,
                            residual=0.)
            if max(x.numerator.bit_length(), x.denominator.bit_length()) > 32768:
                break
            previous, stage_y = x, operating(x)
        raise ValueError("finite separation exceeds exact contact evaluation budget")
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return dict(result, error=str(exc))
