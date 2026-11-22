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
