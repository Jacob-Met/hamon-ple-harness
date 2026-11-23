"""Free-overflow rating and freeboard for an unsubmerged reservoir spillway."""
import math
import struct
from fractions import Fraction


def _quantity(value, name):
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite physical quantity")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return Fraction(number)


def _reported(compare, name):
    """Round an algebraic quantity using its exact comparison with a rational."""
    at_zero = compare(0)
    if at_zero == 0:
        return 0.
    sign = 1 if at_zero > 0 else -1

    def compare_magnitude(trial):
        return sign * compare(sign * trial)

    def from_bits(bits):
        return struct.unpack(">d", struct.pack(">Q", bits))[0]

    # Positive finite binary64 values have the same ordering as their bits.
    # The exponent and significand search needs at most 63 exact comparisons,
    # including when two large elevations cancel in a local clearance.
    low, high = 0, 0x7ff0000000000000
    while high - low > 1:
        middle = (low+high)//2
        trial = Fraction(from_bits(middle))
        order = compare_magnitude(trial)
        if order == 0:
            return sign * float(trial)
        if order > 0:
            low = middle
        else:
            high = middle
    left = Fraction(from_bits(low))
    right = Fraction(2**1024) if high == 0x7ff0000000000000 else Fraction(from_bits(high))
    side = compare_magnitude((left+right)/2)
    bits = high if side > 0 or (side == 0 and low % 2) else low
    shown = from_bits(bits)
    if not math.isfinite(shown) or shown == 0:
        raise ValueError(f"{name} is outside the finite reporting range")
    # The nearest subnormal can still have too few significant bits to certify.
    value = Fraction(shown)
    tolerance = Fraction(1, 10**10)
    if (compare_magnitude(value/(1+tolerance)) < 0 or
            compare_magnitude(value/(1-tolerance)) > 0):
        raise ValueError(f"{name} is outside the supported reporting resolution")
    return sign * shown


def spillway_stage(spec):
    names = ("Q_design_m3_s", "Cd_si", "weir_length_m", "crest_el_m",
             "dam_top_el_m", "min_freeboard_m")
    extra = spec.keys() - {*names, "water_surface_el_m", "mode", "kind"}
    if extra:
        raise ValueError("Unsupported spillway inputs: "+", ".join(sorted(extra)))
    q, coefficient, length, crest, top, minimum = (_quantity(spec[n], n) for n in names)
    if coefficient <= 0 or length <= 0 or q < 0 or minimum < 0:
        raise ValueError("Require positive coefficient and length, nonnegative forward flow and minimum freeboard")

    factor = coefficient*length
    clearance = top-crest
    available = clearance-minimum
    capacity_squared = factor**2 * max(available, 0)**3
    capacity = _reported(lambda x: 1 if x < 0 else capacity_squared-x*x, "capacity")
    observed = spec.get("water_surface_el_m")
    stage = None if observed is None else _quantity(observed, "water_surface_el_m")

    if q == 0:
        if stage is not None and stage > crest:
            raise ValueError("Zero overflow requires water surface at or below crest")
        lower = _reported(lambda x: clearance-x, "freeboard lower bound")
        result = {
            "flow_state": "no_overflow", "head_required_m": 0.,
            "water_surface_el_upper_bound_m": _reported(lambda x: crest-x, "crest"),
            "freeboard_lower_bound_m": lower,
            "water_surface_el_m": None, "freeboard_m": None,
            "passes_freeboard": True if available >= 0 else None,
        }
        if stage is not None:
            result.update(water_surface_el_m=_reported(lambda x: stage-x, "water surface"),
                          freeboard_m=_reported(lambda x: top-stage-x, "freeboard"),
                          passes_freeboard=top-stage >= minimum)
    else:
        # The real cube function is strictly increasing on the entire line.
        # For stage z, compare H^3 with (z-crest)^3. For freeboard b, the
        # corresponding signed cube is -H^3 at offset top-crest.
        head_cubed = (q/factor)**2
        def stage_comparison(x):
            return head_cubed-(x-crest)**3
        water_surface = _reported(stage_comparison, "water surface")
        if stage is not None:
            agrees = (stage_comparison(stage) == 0 if type(observed) is int
                      else stage == Fraction(water_surface))
            if not agrees:
                raise ValueError("Observed stage is inconsistent with the specified free overflow")
        result = {
            "flow_state": "free_overflow",
            "head_required_m": _reported(lambda x: head_cubed-x**3, "overflow head"),
            "water_surface_el_m": water_surface,
            "freeboard_m": _reported(lambda x: -head_cubed-(x-clearance)**3, "freeboard"),
            "passes_freeboard": available >= 0 and q*q <= capacity_squared,
        }
    return {**result, "Q_capacity_with_freeboard_m3_s": capacity, "dual_verified": True}
