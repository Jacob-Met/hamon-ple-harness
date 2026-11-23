"""NFPA 70 (2017) ordinary branch-circuit conductor ampacity checks."""
import math
from fractions import Fraction


def _quantity(value, name):
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return Fraction(parsed)


def _reported(current):
    try:
        shown = float(current)
    except OverflowError as exc:
        raise ValueError("ampacity exceeds the reporting range") from exc
    if current and abs(Fraction(shown)/current-1) > Fraction(1,10**10):
        raise ValueError("ampacity is outside the supported reporting resolution")
    return shown


def branch_ampacity(spec):
    """Apply §210.19(A)(1)(a),(b) and §110.14(C) to caller-supplied ratings."""
    continuous = _quantity(spec["I_continuous_A"], "I_continuous_A")
    other = _quantity(spec.get("I_noncontinuous_A", 0), "I_noncontinuous_A")
    table = _quantity(spec["base_ampacity_A"], "base_ampacity_A")
    terminal = (_quantity(spec["termination_ampacity_A"], "termination_ampacity_A")
                if "termination_ampacity_A" in spec else table)
    temperature = _quantity(spec["temp_factor"], "temp_factor")
    bundling = _quantity(spec["bundling_factor"], "bundling_factor")
    listed = spec.get("listed_100_percent_assembly", False)
    if type(listed) is not bool:
        raise ValueError("listed_100_percent_assembly must be an explicit boolean")
    if min(continuous, other) < 0 or min(table, terminal, temperature) <= 0 or not 0 < bundling <= 1:
        raise ValueError("currents must be nonnegative; ratings and corrections positive; bundling factor at most one")

    load = continuous + other
    design = load if listed else Fraction(5,4)*continuous + other
    unadjusted = min(table, terminal)
    factor = temperature * bundling
    adjusted = table * factor
    sizing_ok, adjusted_ok = unadjusted >= design, adjusted >= load
    passed = sizing_ok and adjusted_ok
    # Independently invert both conditions to recover the minimum tabular
    # rating for these loads, and test the termination rating separately.
    required_table = max(design, load/factor)
    if passed != (table >= required_table and terminal >= design):
        raise ValueError("forward branch-circuit checks disagree with required ampacity")

    return {
        "I_design_required_A": _reported(design),
        "I_derated_allowable_A": _reported(adjusted),
        "I_actual_load_A": _reported(load),
        "I_tabular_insulation_A": _reported(table),
        "I_termination_ampacity_A": _reported(terminal),
        "I_unadjusted_available_A": _reported(unadjusted),
        "I_installed_allowable_A": _reported(min(terminal, adjusted)),
        "I_required_table_A": _reported(required_table),
        "listed_100_percent_assembly": listed,
        "termination_rating_source": "provided" if "termination_ampacity_A" in spec else "common_base_rating",
        "unadjusted_sizing_ok": sizing_ok, "adjusted_load_ok": adjusted_ok,
        "ampacity_ok": passed, "dual_verified": True,
    }
