"""Equilibrium material split with supplied, fixed vapor/liquid K ratios."""

import math
from fractions import Fraction


def _publish(value):
    result = float(value)
    if value and not result:
        raise ValueError("A positive physical output is below the reporting range")
    return result


def _approx(value):
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _at(coordinate, beta):
    a, b, c, d = coordinate
    return (a + b*beta) / (c + d*beta)


def _resolve(residual, coordinates):
    """Round every monotone linear-fractional output at the same physical root."""
    low, high = Fraction(0), Fraction(1)
    resolved = [None] * len(coordinates)
    while True:
        middle = (low + high) / 2
        sign = residual(middle)
        if sign == 0:
            return [_publish(_at(q, middle)) for q in coordinates]
        if sign > 0:
            low = middle
        else:
            high = middle
        pending = False
        for i, coordinate in enumerate(coordinates):
            if resolved[i] is not None:
                continue
            left, right = (_approx(_at(coordinate, x)) for x in (low, high))
            small, large = sorted((left, right))
            if small == large:
                reported = small
            elif math.isfinite(large) and math.nextafter(small, math.inf) == large:
                # Exact midpoint comparison handles rational roots whose beta
                # is not dyadic, without waiting for a shrinking bracket to tie.
                midpoint = (Fraction(small) + Fraction(large)) / 2
                a, b, c, d = coordinate
                crossing = (midpoint*c-a) / (b-midpoint*d)
                sign_at_crossing = residual(crossing)
                if not sign_at_crossing:
                    reported = float(midpoint)
                else:
                    reported = right if sign_at_crossing > 0 else left
            else:
                pending = True
                continue
            if not math.isfinite(reported) or (not reported and any(coordinate[:2])):
                raise ValueError("A positive physical output is outside the reporting range")
            resolved[i] = reported
        if not pending:
            return resolved


def _rates(values, *, positive):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Supply nonempty feed_mol_s and K_values arrays")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Component rates and K values must be finite numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0 or (positive and not number):
            raise ValueError("Feed rates must be nonnegative and K values strictly positive")
        result.append(Fraction(number))
    return result


def fixed_k_flash(params):
    out = {"domain": "electrochem_separations", "subkind": "equilibrium_flash",
           "equilibrium_model": "supplied_fixed_K", "verified": False}
    try:
        if params.keys() - {"domain", "subkind", "feed_mol_s", "K_values"}:
            raise ValueError("The fixed-K model takes component feed rates and supplied K values")
        feed = _rates(params["feed_mol_s"], positive=False)
        k = _rates(params["K_values"], positive=True)
        total = sum(feed)
        if len(feed) != len(k) or not total:
            raise ValueError("Supply one K per component and positive total feed")
        out["feed_total_mol_s"] = _publish(total)
        z = [n/total for n in feed]
        terms = [(n, ratio-1) for n,ratio in zip(feed,k) if n and ratio != 1]

        def residual(beta):
            return sum(n*offset/(1+beta*offset) for n,offset in terms)

        if not terms:
            composition = [_publish(v) for v in z]
            return dict(out, verified=True, phase_state="indeterminate",
                        phase_fraction_identifiable=False, vapor_fraction_interval=[0.,1.],
                        vapor_fraction=None, liquid_fraction=None, vapor_mol_s=None, liquid_mol_s=None,
                        liquid_mole_fractions=composition, vapor_mole_fractions=composition,
                        liquid_component_mol_s=None, vapor_component_mol_s=None)

        bubble, dew = residual(0), residual(1)
        if bubble <= 0 or dew >= 0:
            vapor = dew >= 0
            state = ("dew_point" if dew == 0 else "vapor") if vapor else (
                     "bubble_point" if bubble == 0 else "liquid")
            zero = [0.] * len(feed)
            inventory, composition = [_publish(n) for n in feed], [_publish(v) for v in z]
            x = [_publish(v/r) for v,r in zip(z,k)] if dew == 0 else None
            y = [_publish(v*r) for v,r in zip(z,k)] if bubble == 0 else None
            return dict(out, verified=True, phase_state=state, phase_fraction_identifiable=True,
                        vapor_fraction=float(vapor), liquid_fraction=float(not vapor),
                        vapor_mol_s=_publish(total) if vapor else 0.,
                        liquid_mol_s=0. if vapor else _publish(total),
                        liquid_mole_fractions=x if vapor else composition,
                        vapor_mole_fractions=composition if vapor else y,
                        liquid_component_mol_s=zero if vapor else inventory,
                        vapor_component_mol_s=inventory if vapor else zero)

        coordinates = [(0,1,1,0), (1,-1,1,0), (0,total,1,0), (total,-total,1,0)]
        for n,v,ratio in zip(feed,z,k):
            offset = ratio - 1
            coordinates.extend(((v,0,1,offset), (ratio*v,0,1,offset),
                                (n,-n,1,offset), (0,ratio*n,1,offset)))
        results = _resolve(residual, coordinates)
        vapor, liquid, v_flow, l_flow = results[:4]
        return dict(out, verified=True, phase_state="two_phase", phase_fraction_identifiable=True,
                    vapor_fraction=vapor, liquid_fraction=liquid, vapor_mol_s=v_flow, liquid_mol_s=l_flow,
                    liquid_mole_fractions=results[4::4], vapor_mole_fractions=results[5::4],
                    liquid_component_mol_s=results[6::4], vapor_component_mol_s=results[7::4])
    except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
        return dict(out, error=str(exc))
