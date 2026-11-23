"""Terminal-power and voltage controls for steady balanced AC load flow."""

import cmath
import math

import pytest

from hamon_ac_power_flow import verify_ac_power_flow


def coupled_generator_groups(angle=0):
    buses = [{"id": "S", "type": "slack", "vm_pu": 1, "va_deg": angle,
              "shunt_g_pu": 0.02, "shunt_b_pu": 0.03}]
    lines = []
    proposed = {"S": {"vm_pu": 1, "va_deg": angle}}
    for i in range(5):
        name = f"G{i}"
        high = i < 3
        buses.append({"id": name, "type": "PV", "p_injection_pu": 6 if high else 0,
                      "vm_pu": 1 if high else 2, "q_min_injection_pu": 0 if high else -1,
                      "q_max_injection_pu": 3 if high else 0})
        lines.append({"id": name, "from": "S", "to": name, "r_pu": 1, "x_pu": 0})
        proposed[name] = {"vm_pu": 3 if high else 1, "va_deg": angle}
    for a, b in ((0, 1), (1, 2), (3, 4)):
        lines.append({"id": f"tie{a}{b}", "from": f"G{a}", "to": f"G{b}",
                      "r_pu": 100, "x_pu": 0})
    return {"buses": buses, "lines": lines, "proposed_voltages": proposed}


@pytest.mark.parametrize("angle,reverse", [(0, False), (17, False), (17, True)])
def test_independent_generator_groups_share_one_fixed_slack(angle, reverse):
    # Internal ties carry zero at the exact solution, but prevent a single-bus
    # shortcut. The two coupled groups require different joint Q-limit choices.
    spec = coupled_generator_groups(angle)
    if reverse:
        spec["buses"].reverse()
        spec["lines"].reverse()
    answer = verify_ac_power_flow(spec)
    assert answer["conservation_verified"], answer
    assert answer["clamped_pv_q_pu"] == {f"G{i}": 0 for i in range(5)}
    # Recompute on the original network: duplicating the slack shunt in local
    # bookkeeping must never duplicate its reported physical power.
    assert answer["bus_injections"]["S"] == pytest.approx({"p_pu": -5.98, "q_pu": -0.03})
    for name, expected in spec["proposed_voltages"].items():
        assert answer["voltages"][name] == pytest.approx(expected, abs=1e-10)
    spec["proposed_voltages"]["G3"]["va_deg"] += 0.1
    assert not verify_ac_power_flow(spec)["conservation_verified"]


@pytest.mark.parametrize("resistance", [10, 100])
@pytest.mark.parametrize("angle,reverse", [(0, False), (17, False), (17, True)])
def test_cross_tie_requires_joint_mixed_reactive_limits(resistance, angle, reverse):
    spec = coupled_generator_groups(angle)
    spec["lines"].append({"id": "cross", "from": "G2", "to": "G3",
                          "r_pu": resistance, "x_pu": 0})
    # At V_G2=3 and V_G3=1 the new branch injects 6/R and -2/R.
    # The old claims must fail until these physical terminal powers are supplied.
    assert not verify_ac_power_flow(spec)["conservation_verified"]
    spec["buses"][3]["p_injection_pu"] += 6 / resistance
    spec["buses"][4]["p_injection_pu"] -= 2 / resistance
    if reverse:
        spec["buses"].reverse()
        spec["lines"].reverse()
    answer = verify_ac_power_flow(spec)
    assert answer["conservation_verified"], answer
    assert answer["clamped_pv_q_pu"] == {f"G{i}": 0 for i in range(5)}
    for name, expected in spec["proposed_voltages"].items():
        assert answer["voltages"][name] == pytest.approx(expected, abs=1e-10)
    assert answer["lines"]["cross"] == pytest.approx({
        "p_from_pu": 6 / resistance, "q_from_pu": 0,
        "p_to_pu": -2 / resistance, "q_to_pu": 0}, abs=1e-12)
    spec["proposed_voltages"]["G3"]["va_deg"] += 0.1
    assert not verify_ac_power_flow(spec)["conservation_verified"]


def test_pq_branches_cannot_exhaust_regulator_search_allowance():
    # Many already-flat PQ branches must not consume the attempts needed for
    # the three bounded regulators. Their all-lower state solves exactly.
    buses = [{"id": "S", "type": "slack", "vm_pu": 1}]
    lines = []
    for i in range(3):
        name = f"G{i}"
        buses.append({"id": name, "type": "PV", "p_injection_pu": 1.2, "vm_pu": 1,
                      "q_min_injection_pu": 1.44, "q_max_injection_pu": 3})
        lines.append({"id": name, "from": "S", "to": name, "r_pu": 0, "x_pu": 1})
    lines.append({"id": "tie", "from": "G0", "to": "G1", "r_pu": 100, "x_pu": 0})
    for i in range(77):
        name = f"L{i}"
        buses.append({"id": name, "type": "PQ", "p_injection_pu": 0, "q_injection_pu": 0})
        lines.append({"id": name, "from": "S", "to": name, "r_pu": 0, "x_pu": 1})
    answer = verify_ac_power_flow({"buses": buses, "lines": lines})
    assert answer["conservation_verified"], answer
    assert answer["clamped_pv_q_pu"] == {f"G{i}": 1.44 for i in range(3)}


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
@pytest.mark.parametrize("copies", [1, 3, 6])
def test_alternative_reactive_pivot_after_inconsistent_intermediate(vb, angle, slack_x, upper, copies):
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
    # A-bus ties join each pair of regulators to the next; equal phasors
    # carry zero tie current, so the terminal-power solution above still holds.
    buses, lines, claims = [spec["buses"][0]], [], {"S": spec["proposed_voltages"]["S"]}
    for i in range(copies):
        labels = {"S": "S", "A": f"A{i}", "B": f"B{i}"}
        for bus in spec["buses"][1:]:
            name = labels[bus["id"]]
            buses.append({**bus, "id": name})
            claims[name] = spec["proposed_voltages"][bus["id"]].copy()
        for line in spec["lines"]:
            lines.append({**line, "id": f'{line["id"]}{i}', "from": labels[line["from"]],
                          "to": labels[line["to"]]})
        if i:
            lines.append({"id": f"tie{i}", "from": f"A{i-1}", "to": f"A{i}",
                          "r_pu": 0.01, "x_pu": 0.4})
    network = {"buses": buses, "lines": lines}
    if copies == 6:
        buses[0]["va_deg"] = 17.0
        for phasor in claims.values():
            phasor["va_deg"] += 17.0
        buses.reverse()
        lines.reverse()
    result = verify_ac_power_flow(network)
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == {f"A{i}": qa for i in range(copies)}
    for name, claim in claims.items():
        assert result["voltages"][name]["vm_pu"] == pytest.approx(claim["vm_pu"], abs=1e-9)
        assert result["voltages"][name]["va_deg"] == pytest.approx(claim["va_deg"], abs=1e-8)
    assert verify_ac_power_flow({**network, "proposed_voltages": claims})["conservation_verified"]
    claims["A0"]["vm_pu"] += 5e-7
    assert not verify_ac_power_flow({**network, "proposed_voltages": claims})["conservation_verified"]


def test_small_network_preserves_independently_verified_equilibrium():
    # Both generators at their upper bounds admit a second physical root. The
    # flat-start rectangular check instead agrees with the single-bound root.
    spec = {
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1},
            {"id": "A", "type": "PV", "vm_pu": 1.24, "p_injection_pu": 1.91,
             "q_min_injection_pu": 0.20, "q_max_injection_pu": 1.45},
            {"id": "B", "type": "PV", "vm_pu": 1.09, "p_injection_pu": -1.84,
             "q_min_injection_pu": -0.83, "q_max_injection_pu": -0.78},
        ],
        "lines": [
            {"id": "SA", "from": "S", "to": "A", "r_pu": 0, "x_pu": 2.62},
            {"id": "AB", "from": "A", "to": "B", "r_pu": 0, "x_pu": 0.15},
            {"id": "BS", "from": "B", "to": "S", "r_pu": 0, "x_pu": 5.45},
        ],
    }
    result = verify_ac_power_flow(spec)
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == {"A": 1.45}


@pytest.mark.parametrize("injections", [(1.2,), (-1.2,), (1.2, -1.4), (1.2,) * 5])
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
    rejected = verify_ac_power_flow(spec)
    assert not rejected["conservation_verified"]
    if len(injections) == 5:
        assert "feasibility undetermined" in rejected["reason"]


@pytest.mark.parametrize("transformers", [False, True])
def test_mixed_limits_on_five_independent_resistive_spokes(transformers):
    # A unit resistor gives P=u²+v²-u and Q=-v. The first three buses
    # require V=3 at Qmin, the last two require V=1 at Qmax. The wrong
    # bounds are impossible because Q² cannot exceed P+1/4.
    buses = [{"id": "S", "type": "slack", "vm_pu": 1}]
    lines = []
    expected_voltages, expected_bounds = {}, {}
    for i in range(5):
        name = f"G{i}"
        lower_limit = i < 3
        vm, actual = (1, 3) if lower_limit else (2, 1)
        degrees = (7 if lower_limit else -4) if transformers else 0
        voltage = cmath.rect(actual, math.radians(degrees))
        from_leaf = bool(i % 2)
        tap = cmath.rect(1.05, math.radians(9)) if transformers else 1
        charging = 0.06 if transformers else 0
        shunt = 0.03j if transformers else 0j
        vf, vt = (voltage, 1) if from_leaf else (1, voltage)
        series_current = vf / tap - vt  # unit series resistance
        current = ((series_current + 0.5j * charging * vf / tap) / tap.conjugate()
                   if from_leaf else -series_current + 0.5j * charging * vt)
        power = voltage * (current + shunt * voltage).conjugate()
        lower, upper = ((power.imag, power.imag + 3) if lower_limit
                        else (power.imag - 1, power.imag))
        buses.append({"id": name, "type": "PV", "p_injection_pu": power.real, "vm_pu": vm,
                      "q_min_injection_pu": lower, "q_max_injection_pu": upper,
                      "shunt_b_pu": shunt.imag})
        f, t = (name, "S") if from_leaf else ("S", name)
        lines.append({"id": name, "from": f, "to": t, "r_pu": 1, "x_pu": 0,
                      "tap_ratio": abs(tap), "phase_shift_deg": math.degrees(cmath.phase(tap)),
                      "line_charging_b_pu": charging})
        expected_voltages[name] = {"vm_pu": actual, "va_deg": degrees}
        expected_bounds[name] = power.imag
    result = verify_ac_power_flow({"buses": buses, "lines": lines})
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == expected_bounds
    for name, expectation in expected_voltages.items():
        assert result["voltages"][name] == pytest.approx(expectation)

@pytest.mark.parametrize("count", [5, 12, 24])
def test_mixed_limits_on_one_connected_generator_chain(count):
    # Each unit resistor to slack supplies V*(V-1). The chain adds the exact
    # signed terminal powers V_i*(V_i-V_j)/100, including every cross-group tie.
    magnitudes = [3.0 if i < (3 * count // 5) else 1.0 for i in range(count)]
    injections = [v * (v - 1) for v in magnitudes]
    lines = [{"id": f"S{i}", "from": "S", "to": f"G{i}", "r_pu": 1, "x_pu": 0}
             for i in range(count)]
    for i in range(count - 1):
        flow = (magnitudes[i] - magnitudes[i + 1]) / 100
        injections[i] += magnitudes[i] * flow
        injections[i + 1] -= magnitudes[i + 1] * flow
        lines.append({"id": f"T{i}", "from": f"G{i}", "to": f"G{i + 1}",
                      "r_pu": 100, "x_pu": 0})
    buses = [{"id": "S", "type": "slack", "vm_pu": 1}]
    proposed = {"S": {"vm_pu": 1, "va_deg": 0}}
    for i, (magnitude, power) in enumerate(zip(magnitudes, injections)):
        high = magnitude > 1
        buses.append({"id": f"G{i}", "type": "PV", "p_injection_pu": power,
                      "vm_pu": 1 if high else 2,
                      "q_min_injection_pu": 0 if high else -1,
                      "q_max_injection_pu": 3 if high else 0})
        proposed[f"G{i}"] = {"vm_pu": magnitude, "va_deg": 0}
    result = verify_ac_power_flow({"buses": buses, "lines": lines, "proposed_voltages": proposed})
    assert result["conservation_verified"], result
    assert result["clamped_pv_q_pu"] == {f"G{i}": 0 for i in range(count)}
    for name, expected in proposed.items():
        assert result["voltages"][name] == pytest.approx(expected, abs=1e-10)

@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("angle", [0, 17])
def test_saturated_claim_requires_reactive_power_at_its_bound(upper, angle):
    bound = 0.1 if upper else -0.1
    spec = {
        "buses": [
            {"id": "S", "type": "slack", "vm_pu": 1, "va_deg": angle},
            {"id": "G", "type": "PV", "vm_pu": 1.2 if upper else 0.8,
             "p_injection_pu": 0,
             "q_min_injection_pu": -1 if upper else bound,
             "q_max_injection_pu": bound if upper else 1},
        ],
        "lines": [{"id": "SG", "from": "S", "to": "G", "r_pu": 0, "x_pu": 1}],
    }
    solved = verify_ac_power_flow(spec)
    assert solved["conservation_verified"], solved
    assert solved["clamped_pv_q_pu"] == {"G": bound}
    # Q=V*(V-1). This tiny movement makes Q strictly interior while V is still
    # far from the regulator target, so it cannot describe a regulated state.
    magnitude = (1 + math.sqrt(1 + 4 * bound)) / 2
    spec["proposed_voltages"] = {
        "S": {"vm_pu": 1, "va_deg": angle},
        "G": {"vm_pu": magnitude + (-5e-9 if upper else 5e-9), "va_deg": angle},
    }
    rejected = verify_ac_power_flow(spec)
    assert not rejected["conservation_verified"]
    assert rejected["proposed_pv_limits_consistent"] is False


@pytest.mark.parametrize("p,x", [(-1e-9, 1e-6), (1e-12, 1e-6), (-1e-7, 1e-8)])
@pytest.mark.parametrize("kind,reverse", [("PQ", False), ("PV", True)])
def test_stiff_line_preserves_signed_active_injection_in_the_published_map(p, x, kind, reverse):
    # For a lossless line and a unit slack S=V*conj((V-1)/(jX)),
    # so P=Im(V)/X and Q=(|V|²-Re(V))/X, independent of line orientation.
    load = {"id": "L", "type": kind, "p_injection_pu": p}
    load.update({"q_injection_pu": 0} if kind == "PQ" else {"vm_pu": 1})
    start, end = ("L", "S") if reverse else ("S", "L")
    spec = {
        "buses": [{"id": "S", "type": "slack", "vm_pu": 1}, load],
        "lines": [{"id": "SL", "from": start, "to": end, "r_pu": 0, "x_pu": x}],
    }
    result = verify_ac_power_flow(spec)
    assert result["conservation_verified"], result
    leaf = result["voltages"]["L"]
    assert leaf["vm_pu"] * math.sin(math.radians(leaf["va_deg"])) / x == pytest.approx(
        p, rel=1e-7, abs=0)
    assert result["bus_injections"]["L"]["p_pu"] == pytest.approx(p, rel=1e-7, abs=0)

    # Both high-voltage branches have vm=1 to binary64 precision here.
    # The separating angle is representable even for the smallest loading.
    angle = math.degrees(math.asin(p * x))
    for multiplier, valid in [(1, True), (0, False), (-1, False)]:
        proposed = {"S": {"vm_pu": 1, "va_deg": 0},
                    "L": {"vm_pu": 1, "va_deg": multiplier * angle}}
        assert verify_ac_power_flow({**spec, "proposed_voltages": proposed})["conservation_verified"] is valid


def test_native_flat_map_must_replay_its_terminal_injections():
    # No shunts or charging, so equal bus voltages produce zero current in
    # both lines. Summing the B row of Ybus can manufacture precisely 2^-39j.
    spec = {
        "buses": [
            {"id": "A", "type": "slack", "vm_pu": 1},
            {"id": "B", "type": "PQ", "p_injection_pu": 0, "q_injection_pu": 2**-39},
            {"id": "C", "type": "PQ", "p_injection_pu": 0, "q_injection_pu": 0},
        ],
        "lines": [
            {"id": "AB", "from": "A", "to": "B", "r_pu": 0, "x_pu": 1.7e-5},
            {"id": "BC", "from": "B", "to": "C", "r_pu": 0, "x_pu": 1.3e-4},
        ],
    }
    result = verify_ac_power_flow(spec)
    if result["conservation_verified"]:
        volts = {name: cmath.rect(v["vm_pu"], math.radians(v["va_deg"]))
                 for name, v in result["voltages"].items()}
        current = ((volts["B"] - volts["A"]) / 1.7e-5j
                   + (volts["B"] - volts["C"]) / 1.3e-4j)
        assert (volts["B"] * current.conjugate()).imag == pytest.approx(2**-39, rel=1e-7, abs=0)
    assert not verify_ac_power_flow({
        **spec, "proposed_voltages": {key: {"vm_pu": 1, "va_deg": 0} for key in "ABC"},
    })["conservation_verified"]


@pytest.mark.parametrize("winding", [-360, 360])
def test_whole_turn_cannot_create_active_flow(winding):
    reactance = 1e-6
    prescribed = math.sin(math.radians(winding)) / reactance
    spec = {
        "buses": [{"id": "S", "type": "slack", "vm_pu": 1},
                  {"id": "L", "type": "PQ", "p_injection_pu": prescribed, "q_injection_pu": 0}],
        "lines": [{"id": "SL", "from": "S", "to": "L", "r_pu": 0, "x_pu": reactance}],
    }
    assert verify_ac_power_flow(spec)["conservation_verified"]
    for angle in (0, winding):
        # V_L and V_S are physically identical, and the branch current is zero.
        claim = {"S": {"vm_pu": 1, "va_deg": 0}, "L": {"vm_pu": 1, "va_deg": angle}}
        assert not verify_ac_power_flow({**spec, "proposed_voltages": claim})["conservation_verified"]


def test_one_ulp_at_rotated_pv_bus_has_its_own_signed_transfer():
    reference = 17.0
    angle = math.nextafter(reference, math.inf)
    reactance = 2**-20
    prescribed = 5.5664262965032565e-11
    # For two unit magnitudes the exact terminal transfer depends on delta,
    # not the difference of two separately rounded rectangular phasors.
    physical = math.sin(math.radians(angle - reference)) / reactance
    assert abs(physical - prescribed) > 0.15 * prescribed
    spec = {
        "buses": [{"id": "S", "type": "slack", "vm_pu": 1, "va_deg": reference},
                  {"id": "L", "type": "PV", "vm_pu": 1, "p_injection_pu": prescribed}],
        "lines": [{"id": "SL", "from": "S", "to": "L", "r_pu": 0, "x_pu": reactance}],
        "proposed_voltages": {"S": {"vm_pu": 1, "va_deg": reference},
                              "L": {"vm_pu": 1, "va_deg": angle}},
    }
    assert not verify_ac_power_flow(spec)["conservation_verified"]
    del spec["proposed_voltages"]
    result = verify_ac_power_flow(spec)
    if result["conservation_verified"]:
        v = result["voltages"]
        actual = (v["S"]["vm_pu"] * v["L"]["vm_pu"]
                  * math.sin(math.radians(v["L"]["va_deg"] - v["S"]["va_deg"])) / reactance)
        assert actual == pytest.approx(prescribed, rel=1e-7, abs=0)


def test_stiff_line_large_transfer_still_meets_absolute_power_tolerance():
    target = 149922522.2315777
    reactance = 2**-36
    assert abs(math.sin(math.radians(0.125)) / reactance - target) > 6e-6
    spec = {
        "buses": [{"id": "S", "type": "slack", "vm_pu": 1, "va_deg": 45},
                  {"id": "L", "type": "PV", "vm_pu": 1, "p_injection_pu": target}],
        "lines": [{"id": "SL", "from": "S", "to": "L", "r_pu": 0, "x_pu": reactance}],
        "proposed_voltages": {"S": {"vm_pu": 1, "va_deg": 45},
                              "L": {"vm_pu": 1, "va_deg": 45.125}},
    }
    assert not verify_ac_power_flow(spec)["conservation_verified"]
