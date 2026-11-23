"""Verify steady-flow mass, energy and entropy balances on declared SI stream properties."""

import math
from fractions import Fraction


def _number(value, field):
    if type(value) is int:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(str(value))
    raise ValueError(f"{field} must be a finite JSON number")


def _wire(value, field):
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} exceeds the finite output range") from exc
    if abs(Fraction(rounded) - value) > abs(value) / 10**12:
        raise ValueError(f"{field} cannot be reported to the required accuracy")
    return rounded


def _fields(record, allowed, required, kind):
    if not isinstance(record, dict) or set(record) - allowed or required - record.keys():
        raise ValueError(f"invalid or incomplete {kind} fields")


def _stream_fluxes(streams, reference, gravity):
    if not isinstance(streams, list) or not streams:
        raise ValueError("inlets and outlets each require a nonempty list of streams")
    mass = enthalpy = kinetic = potential = entropy = Fraction(0)
    required = {"mdot_kg_s", "h_J_kg", "s_J_kgK"}
    for stream in streams:
        _fields(stream, required | {"property_reference", "speed_m_s", "elevation_m"},
                required, "stream")
        if stream.get("property_reference", reference) != reference:
            raise ValueError("stream property_reference conflicts with the common basis")
        flow = _number(stream["mdot_kg_s"], "mdot_kg_s")
        speed = _number(stream.get("speed_m_s", 0), "speed_m_s")
        if flow < 0 or speed < 0:
            raise ValueError("stream mass flow and speed must be nonnegative")
        mass += flow
        enthalpy += flow * _number(stream["h_J_kg"], "h_J_kg")
        kinetic += flow * speed**2 / 2
        potential += flow * gravity * _number(stream.get("elevation_m", 0), "elevation_m")
        entropy += flow * _number(stream["s_J_kgK"], "s_J_kgK")
    if mass == 0:
        raise ValueError("each side requires a positive total mass flow")
    return mass, enthalpy, kinetic, potential, entropy


def steady_flow_control_volume(spec):
    result = {"ok": False, "tool": "hamon_conservation_cad_verifier", "mode": "thermal_fluid",
              "calculation": "steady_flow_control_volume", "conservation_verified": False,
              "thermodynamically_feasible": False}
    try:
        required = {"inlets", "outlets", "heat_reservoirs", "shaft_work_out_W", "property_reference"}
        allowed = required | {"mode", "calculation", "gravity_m_s2", "entropy_generation_W_K"}
        _fields(spec, allowed, required, "control volume")
        reference = spec["property_reference"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("declare one common property_reference for all streams")
        gravity = _number(spec.get("gravity_m_s2", 9.80665), "gravity_m_s2")
        if gravity <= 0:
            raise ValueError("gravity_m_s2 must be positive")
        inlet = _stream_fluxes(spec["inlets"], reference, gravity)
        outlet = _stream_fluxes(spec["outlets"], reference, gravity)
        dm, dh, dk, dp, ds = (b - a for a, b in zip(inlet, outlet))
        heat = entropy_heat = Fraction(0)
        if not isinstance(spec["heat_reservoirs"], list):
            raise ValueError("heat_reservoirs must be a list")
        for reservoir in spec["heat_reservoirs"]:
            fields = {"heat_into_W", "temperature_K"}
            _fields(reservoir, fields, fields, "heat reservoir")
            q = _number(reservoir["heat_into_W"], "heat_into_W")
            temperature = _number(reservoir["temperature_K"], "temperature_K")
            if temperature <= 0:
                raise ValueError("reservoir temperature_K must be positive")
            heat += q
            entropy_heat += q / temperature
        work = _number(spec["shaft_work_out_W"], "shaft_work_out_W")
        energy_residual = heat - work - dh - dk - dp
        generation = ds - entropy_heat
        energy_scale = max(map(abs, (heat, work, dh, dk, dp)))
        energy_ok = abs(energy_residual) <= energy_scale / 10**10
        generation_ok = True
        if "entropy_generation_W_K" in spec:
            observed = _number(spec["entropy_generation_W_K"], "entropy_generation_W_K")
            generation_ok = observed >= 0 and abs(observed - generation) <= abs(generation) / 10**10
            result["entropy_generation_observation_ok"] = generation_ok
        ledger = {
            "mass_in_kg_s": inlet[0], "mass_out_kg_s": outlet[0], "mass_out_minus_in_kg_s": dm,
            "net_heat_into_W": heat, "shaft_work_out_W": work, "enthalpy_flux_change_W": dh,
            "kinetic_energy_flux_change_W": dk, "potential_energy_flux_change_W": dp,
            "energy_residual_W": energy_residual, "entropy_flux_change_W_K": ds,
            "entropy_heat_transfer_W_K": entropy_heat, "entropy_generation_W_K": generation,
        }
        published = {name: _wire(value, name) for name, value in ledger.items()}
        feasible = dm == 0 and energy_ok and generation >= 0 and generation_ok
        result.update(published, ok=True, property_reference=reference,
                      mass_balance_ok=dm == 0, first_law_ok=energy_ok,
                      second_law_ok=generation >= 0, thermodynamically_feasible=feasible,
                      conservation_verified=feasible)
    except ValueError as exc:
        result["error"] = str(exc)
    return result
