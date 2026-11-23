"""Balanced three-phase bolted fault at the transformer secondary, at nominal voltage."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math


def symmetrical_fault(spec):
    """Series positive-sequence utility and transformer impedances on the same base.

    Both component angles lie in [0, π/2]. Optional transformer_X_over_R and
    utility_X_over_R fix their angles. Missing angles span the complete quadrant.
    The breaker comparison covers symmetrical rms duty only.
    """
    def given(key):
        value = float(spec[key])
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite short-circuit input {key}")
        return Fraction(value)

    s, v, pct, utility_mva, breaker_ka = map(given, (
        "S_base_kVA", "V_ll_V", "Z_tx_pct", "S_utility_sc_MVA", "breaker_aic_kA"))
    if min(s, v, utility_mva) <= 0 or min(pct, breaker_ka) < 0:
        raise ValueError("Require positive bases and fault MVA, nonnegative impedance and breaker rating")
    ratios = [given(key) for key in ("transformer_X_over_R", "utility_X_over_R") if key in spec]
    if any(value < 0 for value in ratios):
        raise ValueError("Require passive inductive impedances with X/R >= 0")
    tx, utility = pct / 100, s / (1000 * utility_mva)
    # |Z1+Z2|² = z1² + z2² + 2 z1 z2 cos(angle1-angle2).
    # Keep squared cosines rational, including the two possible axis endpoints.
    lo, hi = Fraction(0), Fraction(1)
    if len(ratios) == 2:
        a, b = ratios
        lo = hi = (1 + a*b)**2 / ((1+a*a)*(1+b*b))
    elif ratios:
        a = ratios[0]
        lo = min(1, a*a) / (1+a*a)

    base_i2 = (1000*s/v)**2 / 3
    squares = tx*tx + utility*utility
    product = 2*tx*utility

    def compare_current(cos2, amps):
        """Exact sign of fault current minus an rms ampere threshold."""
        if not amps:
            return 1
        gap = base_i2 / amps**2 - squares
        if gap < 0:
            return -1
        difference = gap**2 - product**2*cos2
        return (difference > 0) - (difference < 0)

    def dec(f):
        return Decimal(f.numerator)/Decimal(f.denominator)

    def wire(d):
        value = float(d)
        if not math.isfinite(value) or (d and not value):
            raise ValueError("Short-circuit output outside finite binary64 range")
        return value

    with localcontext() as ctx:
        ctx.prec = 80
        i_base = dec(base_i2).sqrt()

        def impedance(cos2):
            return (dec(squares) + dec(product)*dec(cos2).sqrt()).sqrt()

        min_z, max_z = impedance(lo), impedance(hi)
        lower, upper = wire(i_base/max_z), wire(i_base/min_z)
        if compare_current(hi, Fraction(lower)) < 0:
            lower = math.nextafter(lower, 0.)
        if compare_current(lo, Fraction(upper)) > 0:
            upper = math.nextafter(upper, math.inf)
            if not math.isfinite(upper):
                raise ValueError("Fault current upper bound exceeds finite binary64 range")

        known = lo == hi or not tx
        rating = 1000*breaker_ka
        if compare_current(lo, rating) <= 0:
            status, decision = "pass", True
        elif compare_current(hi, rating) > 0:
            status, decision = "fail", False
        else:
            status, decision = "indeterminate", None
        return {"I_fla_A": wire(i_base), "Z_total_pu": wire(max_z) if known else None,
                "I_sc_sym_A": wire(i_base/max_z) if known else None,
                "I_sc_sym_min_A": lower, "I_sc_sym_max_A": upper,
                "co_phasal_I_sc_sym_A": wire(i_base/dec(tx+utility)),
                "aic_status": status, "aic_ok": decision, "dual_verified": True}
