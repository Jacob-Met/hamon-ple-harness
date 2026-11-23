"""Flexural critical loads of a prismatic column under concentric compression."""

from decimal import Decimal, localcontext
from fractions import Fraction
import math


PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117")


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return Fraction(number)


def _decimal(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def _reported(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("A positive member result is outside the binary64 reporting range")
    return result


def _axis(stiffness, strength, area, length, inertia, factor, transition):
    length_squared = (factor * length) ** 2
    slender_squared = length_squared * area / inertia
    lam = _decimal(slender_squared).sqrt()
    relative = _decimal(slender_squared) / transition**2
    if relative >= 1:
        regime = "euler_elastic"
        stress = _decimal(strength) / (2 * relative)
    else:
        regime = "johnson_inelastic"
        stress = _decimal(strength) * (1 - relative / 2)

    pressure = _reported(stress)
    load = _reported(stress * _decimal(area))
    slenderness = _reported(lam)
    cc = _reported(transition)

    # Independent load-space equations at this member, using the actual wire values.
    euler_load = PI**2 * _decimal(stiffness * inertia / length_squared)
    yield_load = _decimal(strength * area)
    published = Decimal.from_float(load)
    if regime == "euler_elastic":
        equation_error = abs(published / euler_load - 1)
    else:
        equation_error = abs(published / yield_load + yield_load / (4 * euler_load) - 1)
    residual = max(equation_error,
                   abs(published / (Decimal.from_float(pressure) * _decimal(area)) - 1),
                   abs(Decimal.from_float(slenderness)**2 / _decimal(slender_squared) - 1),
                   abs(Decimal.from_float(cc)**2 * _decimal(strength / stiffness) / (2 * PI**2) - 1))
    return {"regime": regime, "slenderness_ratio": slenderness, "transition_Cc": cc,
            "sigma_cr": pressure, "P_cr_N": load, "residual": float(residual),
            "verified": residual < Decimal("1e-11")}


def column_buckling(params):
    result = {"domain": "solid_mohr_vessel_buckling", "subkind": "column_buckling", "verified": False}
    try:
        e, sy, area, length = (_positive(params[key], key) for key in ("E", "sigma_y", "A", "L"))
        multiple = "principal_axes" in params
        if multiple:
            axes = params["principal_axes"]
            if "I" in params or "K" in params:
                raise ValueError("Supply scalar I/K or principal_axes, without combining both forms")
            if not isinstance(axes, dict) or set(axes) != {"x", "y"}:
                raise ValueError("principal_axes must specify both x and y")
        else:
            axes = {"scalar": {"I": params["I"], "K": params.get("K", 1)}}
        ranks = {}
        outcomes = {}
        with localcontext() as ctx:
            ctx.prec = 90
            transition = (2 * PI**2 * _decimal(e / sy)).sqrt()
            for name in sorted(axes):
                axis = axes[name]
                if not isinstance(axis, dict) or set(axis) != {"I", "K"}:
                    raise ValueError(f"Axis {name} requires its own I and K")
                inertia = _positive(axis["I"], f"{name}.I")
                factor = _positive(axis["K"], f"{name}.K")
                # Both material branches decrease with squared effective slenderness.
                ranks[name] = factor**2 / inertia
                outcomes[name] = _axis(e, sy, area, length, inertia, factor, transition)
        governing = [name for name in outcomes if ranks[name] == max(ranks.values())]
        result.update(outcomes[governing[0]])
        if multiple:
            result.update(principal_axis_results=outcomes, governing_axes=governing,
                          residual=max(v["residual"] for v in outcomes.values()),
                          verified=all(v["verified"] for v in outcomes.values()))
    except (KeyError, TypeError, ArithmeticError, ValueError) as exc:
        result["error"] = str(exc)
    return result
