"""Signed electrical power of a fixed-current, lumped electrochemical port."""
import math
from fractions import Fraction

from mpmath.ctx_iv import MPIntervalContext


def _wire(value):
    shown = float(value)
    if not math.isfinite(shown) or (value and abs(Fraction(shown)/value-1) >= Fraction(1,10**11)):
        raise ValueError("Electrochemical port quantity exceeds finite reporting resolution")
    return shown


def _endpoint(interval):
    negative,mantissa,exponent,bits = interval._mpi_[0]
    if bits < 0:
        raise ValueError("Nonfinite electrochemical interval")
    value = Fraction(mantissa)*Fraction(2)**exponent
    return -value if negative else value


def _log_fields(quotient, expressions):
    """Publish a+b*ln(Q), retaining exact short-circuit terms in a."""
    result = {key:_wire(a) for key,(a,b) in expressions.items() if not b or quotient==1}
    needed = {key:pair for key,pair in expressions.items() if key not in result}
    if not needed:
        return result
    for digits in (70,150,330,700,1500):
        iv = MPIntervalContext()
        iv.dps = digits
        def number(v):
            return iv.mpf(v.numerator)/v.denominator
        logarithm = iv.log(number(quotient))
        candidate = {}
        for key,(a,b) in needed.items():
            interval = number(a)+number(b)*logarithm
            lo,hi = _endpoint(interval.a),_endpoint(interval.b)
            if lo <= 0 <= hi or float(lo) != float(hi):
                break
            candidate[key] = _wire(lo)
            _wire(hi)
        if len(candidate)==len(needed):
            return {**result,**candidate}
    raise ValueError("Nernst potential or port-power sign could not be resolved")


def electrical_state(*, potential, electrons, temperature, quotient, current, resistance,
                     activation, efficiency, mode, circuit, faraday, gas):
    if circuit not in ("prescribed_current", "passive_load"):
        raise ValueError("external_circuit must be prescribed_current or passive_load")
    s = 1 if mode=="galvanic" else -1
    signed_current = s*current
    assigned = signed_current*efficiency
    nf = electrons*faraday
    rt = gas*temperature
    log_slope = -rt/nf
    loss_voltage = activation+current*resistance
    terminal_base = potential-s*loss_voltage
    expressions = {
        "E_nernst_V": (potential,log_slope), "V_cell_V": (terminal_base,log_slope),
        "delta_G_J_mol": (-nf*potential,rt),
        "reversible_port_power_out_W": (signed_current*potential,signed_current*log_slope),
        "terminal_power_out_W": (signed_current*terminal_base,signed_current*log_slope),
        "assigned_reaction_reversible_power_out_W": (assigned*potential,assigned*log_slope),
    }
    fields = _log_fields(quotient,expressions)
    fields.update(delta_G0_J_mol=_wire(-nf*potential),ln_K_eq=_wire(nf*potential/rt),
                  reference_current_out_A=_wire(signed_current),
                  assigned_reference_extent_rate_mol_s=_wire(assigned/nf),
                  unassigned_current_magnitude_A=_wire(current*(1-efficiency)),
                  ohmic_dissipation_W=_wire(current**2*resistance),
                  activation_dissipation_W=_wire(current*activation))
    # Replay the first-law port relation from its public values, and the
    # individual reaction power from the signed extent and chemical potential.
    def relative_closure(values):
        values = tuple(map(Fraction,values))
        scale = max(map(abs,values))
        return abs(sum(values))/scale if scale else 0

    current_power = fields["terminal_power_out_W"]
    reversible = fields["reversible_port_power_out_W"]
    reaction = fields["assigned_reaction_reversible_power_out_W"]
    gibbs = Fraction(fields["delta_G_J_mol"])
    residual = max(relative_closure((reversible,-current_power,-fields["ohmic_dissipation_W"],
                                     -fields["activation_dissipation_W"])),
                   relative_closure((current_power,-signed_current*Fraction(fields["V_cell_V"]))),
                   relative_closure((reaction,gibbs*Fraction(fields["assigned_reference_extent_rate_mol_s"]))),
                   relative_closure((gibbs,nf*Fraction(fields["E_nernst_V"]))))
    if residual >= Fraction(1,10**11):
        raise ValueError("Public electrical power or reaction fields do not satisfy the port balance")
    passive = current_power >= 0
    state = ("open_circuit" if not current else "short_circuit" if not current_power else
             "delivering_power" if passive else "absorbing_power")
    fields.update(external_circuit=circuit,electrical_operating_state=state,
                  passive_load_feasible=passive,operating_point_feasible=circuit!="passive_load" or passive,
                  power_balance_residual=float(residual))
    return fields
