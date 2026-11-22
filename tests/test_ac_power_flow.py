"""Terminal-power and voltage controls for steady balanced AC load flow."""

import cmath
import math

import pytest

from hamon_ac_power_flow import verify_ac_power_flow


def generator_case():
    return {
        "calculation": "ac_power_flow",
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1},
            {"id": "G", "type": "PV", "p_injection_pu": 0.35, "vm_pu": 1.04},
            {"id": "L", "type": "PQ", "p_injection_pu": -0.85, "q_injection_pu": -0.35},
        ],
        "lines": [
            {"id": "a", "from": "S", "to": "G", "r_pu": 0.02, "x_pu": 0.12},
            {"id": "b", "from": "G", "to": "L", "r_pu": 0.03, "x_pu": 0.15},
            {"id": "c", "from": "S", "to": "L", "r_pu": 0.02, "x_pu": 0.10},
        ],
    }


def test_pv_regulation_and_reactive_saturation():
    spec = generator_case()
    r = verify_ac_power_flow(spec)
    assert r["conservation_verified"], r
    assert r["voltages"]["G"]["vm_pu"] == pytest.approx(1.04)
    gen = spec["buses"][1]
    gen.update(q_min_injection_pu=-0.1, q_max_injection_pu=0.12)
    limited = verify_ac_power_flow(spec)
    assert limited["conservation_verified"], limited
    assert limited["clamped_pv_q_pu"] == {"G": 0.12}
    assert limited["voltages"]["G"]["vm_pu"] < 1.04
    assert limited["bus_injections"]["G"] == pytest.approx({"p_pu": 0.35, "q_pu": 0.12})
    assert limited["tellegen_residual_pu"] < 1e-12
    bus = spec["buses"][1]
    bus.clear()
    bus.update(id="G", type="PQ", p_injection_pu=0.35, q_injection_pu=0.12)
    fixed = verify_ac_power_flow(spec)
    assert fixed["voltages"]["L"] == pytest.approx(limited["voltages"]["L"])


def test_tapped_charged_line_against_prescribed_voltage_and_signed_terminal_powers():
    vs = cmath.rect(1.03, math.radians(7))
    vt = cmath.rect(0.985, math.radians(-3))
    tap = cmath.rect(1.05, math.radians(8))
    y = 1 / complex(0.02, 0.17)
    charged = 0.025j
    series = y * (vs / tap - vt)
    sf = vs * ((series + charged * (vs / tap)) / tap.conjugate()).conjugate()
    st = vt * (-series + charged * vt).conjugate()
    shunt = complex(0.006, 0.014)
    pq = st + abs(vt) ** 2 * shunt.conjugate()
    spec = {
        "buses": [
            {"id": "A", "type": "slack", "vm_pu": 1.03, "va_deg": 7},
            {
                "id": "B",
                "type": "PQ",
                "p_injection_pu": pq.real,
                "q_injection_pu": pq.imag,
                "shunt_g_pu": shunt.real,
                "shunt_b_pu": shunt.imag,
            },
        ],
        "lines": [
            {
                "id": "t",
                "from": "A",
                "to": "B",
                "r_pu": 0.02,
                "x_pu": 0.17,
                "line_charging_b_pu": 0.05,
                "tap_ratio": 1.05,
                "phase_shift_deg": 8,
            }
        ],
    }
    r = verify_ac_power_flow(spec)
    assert r["conservation_verified"], r
    assert r["voltages"]["B"] == pytest.approx({"vm_pu": 0.985, "va_deg": -3})
    assert r["lines"]["t"] == pytest.approx(
        {
            "p_from_pu": sf.real,
            "q_from_pu": sf.imag,
            "p_to_pu": st.real,
            "q_to_pu": st.imag,
        }
    )


def test_tiny_load_and_disconnected_network():
    spec = {
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1},
            {"id": "L", "type": "PQ", "p_injection_pu": -1e-9, "q_injection_pu": 0},
        ],
        "lines": [{"id": "p", "from": "S", "to": "L", "r_pu": 0.2, "x_pu": 0.5}],
    }
    solved = verify_ac_power_flow(spec)
    spec["proposed_voltages"] = solved["voltages"]
    assert verify_ac_power_flow(spec)["conservation_verified"]
    spec["proposed_voltages"]["L"] = {"vm_pu": 1, "va_deg": 0}
    assert not verify_ac_power_flow(spec)["conservation_verified"]
    spec["buses"].append(
        {"id": "isolated", "type": "PQ", "p_injection_pu": 0, "q_injection_pu": 0}
    )
    assert not verify_ac_power_flow(spec)["ok"]


def test_auto_and_legacy_public_routes_accept_the_ac_case():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    from hamon_dual_verifier import run_dual_verify_spec

    spec = generator_case()
    assert run_router_conservation_cad_verifier({"spec":spec})["conservation_verified"]
    assert run_dual_verify_spec({"mode":"domain","kind":"ac_power_flow","spec":spec})["dual_verified"]


@pytest.mark.parametrize("reactance", [.001, .01, .1])
@pytest.mark.parametrize("direction", [-1,1])
def test_proposed_and_native_generator_reactive_limits(reactance, direction):
    g = {"id":"G", "type":"PV", "p_injection_pu":0, "vm_pu":1,
         "q_min_injection_pu":-1 if direction>0 else 0,
         "q_max_injection_pu":0 if direction>0 else 1}
    s = {"buses":[{"id":"S", "type":"slack", "vm_pu":1},g],
         "lines":[{"id":"line","from":"S","to":"G","r_pu":0,"x_pu":reactance}]}
    s["proposed_voltages"] = {k:{"vm_pu":1, "va_deg":0} for k in ("S","G")}
    assert verify_ac_power_flow(s)["conservation_verified"]
    s["proposed_voltages"]["G"]["vm_pu"] += direction*5e-7
    bad = verify_ac_power_flow(s)
    assert not bad["conservation_verified"]
    assert bad["proposed_pv_limits_consistent"] is False
    del s["proposed_voltages"]
    g["vm_pu"] += direction*5e-11
    limited = verify_ac_power_flow(s)
    assert limited["conservation_verified"], limited
    assert limited["clamped_pv_q_pu"] == {"G":0}
    assert limited["voltages"]["G"]["vm_pu"] == pytest.approx(1,abs=1e-13)


@pytest.mark.parametrize("setpoint_a", [0.96, 0.92])
@pytest.mark.parametrize("slack_susceptance", [0.2, 0.65])
def test_coupled_regulators_can_release_a_temporary_reactive_limit(setpoint_a, slack_susceptance):
    # With zero active power all angles vanish. Q_i = V_i sum_j (V_i-V_j)/X_ij.
    va, vb = 0.94, 1.02
    qa = va * ((va - 1) * slack_susceptance + (va - vb) / 0.1)
    qb = vb * ((vb - 1) * slack_susceptance + (vb - va) / 0.1)
    if setpoint_a > va:
        bounds_a, bounds_b = (qa - 2, qa), (qb - 0.0004, qb + 2)
    else:
        bounds_a, bounds_b = (qa, qa + 2), (qb - 2, qb + 0.0004)
    spec = {
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1},
            {"id": "A", "type": "PV", "vm_pu": setpoint_a, "p_injection_pu": 0,
             "q_min_injection_pu": bounds_a[0], "q_max_injection_pu": bounds_a[1]},
            {"id": "B", "type": "PV", "vm_pu": vb, "p_injection_pu": 0,
             "q_min_injection_pu": bounds_b[0], "q_max_injection_pu": bounds_b[1]},
        ],
        "lines": [
            {"id": "SA", "from": "S", "to": "A", "r_pu": 0, "x_pu": 1 / slack_susceptance},
            {"id": "AB", "from": "A", "to": "B", "r_pu": 0, "x_pu": 0.1},
            {"id": "SB", "from": "S", "to": "B", "r_pu": 0, "x_pu": 1 / slack_susceptance},
        ],
    }
    result = verify_ac_power_flow(spec)
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == {"A": qa}
    assert result["voltages"]["A"]["vm_pu"] == pytest.approx(va, abs=1e-10)
    assert result["voltages"]["B"]["vm_pu"] == pytest.approx(vb, abs=1e-10)
    assert result["bus_injections"]["B"]["q_pu"] == pytest.approx(qb, abs=1e-10)


@pytest.mark.parametrize("vb, angle, slack_x, upper", [(0.98, 5, 3, True), (1.04, 2, 5, False)])
def test_alternative_reactive_pivot_after_inconsistent_intermediate(vb, angle, slack_x, upper):
    magnitudes = [1.0, 0.94, vb]
    angles = [0, angle, -angle]
    voltages = [cmath.rect(vm, math.radians(va)) for vm, va in zip(magnitudes, angles)]
    currents = [0j, 0j, 0j]
    ids = "SAB"
    lines = []
    for f, t, x in [(0, 1, slack_x), (1, 2, 0.1), (2, 0, slack_x)]:
        branch = (voltages[f] - voltages[t]) / (1j*x)
        currents[f] += branch
        currents[t] -= branch
        lines.append({"id": ids[f] + ids[t], "from": ids[f], "to": ids[t], "r_pu": 0, "x_pu": x})
    power = [v * i.conjugate() for v, i in zip(voltages, currents)]
    qa, qb = power[1].imag, power[2].imag
    a_bounds = (qa - 1.5, qa) if upper else (qa, qa + 1.5)
    b_bounds = (qb - 0.001, qb + 1.5) if upper else (qb - 1.5, qb + 0.001)
    spec = {
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1},
            {"id": "A", "type": "PV", "vm_pu": 0.94 + (0.07 if upper else -0.07),
             "p_injection_pu": power[1].real,
             "q_min_injection_pu": a_bounds[0], "q_max_injection_pu": a_bounds[1]},
            {"id": "B", "type": "PV", "vm_pu": vb, "p_injection_pu": power[2].real,
             "q_min_injection_pu": b_bounds[0], "q_max_injection_pu": b_bounds[1]},
        ],
        "lines": lines,
        "proposed_voltages": {name: {"vm_pu": vm, "va_deg": va}
                              for name, vm, va in zip(ids, magnitudes, angles)},
    }
    result = verify_ac_power_flow(spec)
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == {"A": qa}


@pytest.mark.parametrize("injections", [(1.2,), (-1.2,), (1.2, -1.4)])
def test_reactive_bound_can_solve_unreachable_pv_setpoint(injections):
    # A unit slack and unit reactance give P=Im(V), Q=|V|²-Re(V).
    # |P|>1 rules out the PV setpoint |V|=1. V=1+jP satisfies Qmin=P².
    buses = [{"id": "S", "type": "slack", "vm_pu": 1}]
    lines = []
    expected = {}
    proposed = {"S": {"vm_pu": 1, "va_deg": 0}}
    for i, p in enumerate(injections):
        name = f"G{i}"
        buses.append({"id": name, "type": "PV", "p_injection_pu": p, "vm_pu": 1,
                      "q_min_injection_pu": p * p, "q_max_injection_pu": 3})
        lines.append({"id": name, "from": "S", "to": name, "r_pu": 0, "x_pu": 1})
        expected[name] = p * p
        proposed[name] = {"vm_pu": math.hypot(1, p), "va_deg": math.degrees(math.atan(p))}
    spec = {"buses": buses, "lines": lines, "proposed_voltages": proposed}
    answer = verify_ac_power_flow(spec)
    assert answer["conservation_verified"], answer
    assert answer["clamped_pv_q_pu"] == expected
    proposed["G0"]["va_deg"] += 0.1
    assert not verify_ac_power_flow(spec)["conservation_verified"]
    # Q=Re(V)^2-Re(V)+P^2 >= P^2-1/4, so neither bound is attainable.
    del spec["proposed_voltages"]
    buses[1].update(q_min_injection_pu=-1, q_max_injection_pu=0)
    assert not verify_ac_power_flow(spec)["conservation_verified"]
