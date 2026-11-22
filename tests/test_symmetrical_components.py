import math

import pytest

from hamon_ac_power_flow import verify_symmetrical_fault


def z(real, imag):
    return {"real": real, "imag": imag}


def as_complex(record):
    return complex(record["real"], record["imag"])


def source(fault):
    # MIT 6.061 Spring 2011, Ch. 4, §7.4. At the fault terminals the
    # positive/negative Thevenin impedance is j(.15 || .12); zero is j.2.
    # https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/
    return {
        "calculation": "symmetrical_fault",
        "fault_type": fault,
        "prefault_positive_voltage_pu": z(1, 0),
        "sequence_impedances_pu": {
            "positive": z(0, 1 / 15),
            "negative": z(0, 1 / 15),
            "zero": z(0, 0.2),
        },
    }


@pytest.mark.parametrize(
    "fault,expected_i1", [("3PH", -15j), ("SLG", -3j), ("LL", -7.5j)]
)
def test_positive_sequence_current_at_fault_terminals(fault, expected_i1):
    got = verify_symmetrical_fault(source(fault))
    assert got["conservation_verified"], got
    assert as_complex(got["sequence_currents_pu"]["positive"]) == pytest.approx(
        expected_i1
    )
    assert got["max_terminal_boundary_residual_pu"] < 1e-12


@pytest.mark.parametrize("phases", [["A", "B"], ["B", "C"], ["C", "A"]])
def test_double_line_shared_ground_impedance_and_arbitrary_phases(phases):
    spec = source("DLG")
    spec["faulted_phases"] = phases
    spec["fault_impedance_pu"] = z(0.02, 0.04)
    ans = verify_symmetrical_fault(spec)
    assert ans["conservation_verified"], ans
    terminal = [as_complex(ans["phase_voltages_pu"][p]) for p in phases]
    ground = as_complex(ans["ground_return_current_pu"])
    assert terminal == pytest.approx([(0.02 + 0.04j) * ground] * 2)
    healthy = next(p for p in "ABC" if p not in phases)
    assert abs(as_complex(ans["phase_currents_pu"][healthy])) < 1e-12
    spec["proposed_phase_currents_pu"] = ans["phase_currents_pu"].copy()
    assert verify_symmetrical_fault(spec)["conservation_verified"]
    spec["proposed_phase_currents_pu"][phases[0]] = z(0, 0)
    assert not verify_symmetrical_fault(spec)["conservation_verified"]


def test_rotated_slg_and_ll_boundary_conditions():
    spec = source("SLG")
    spec["faulted_phases"] = ["B"]
    ans = verify_symmetrical_fault(spec)
    assert ans["conservation_verified"]
    assert as_complex(ans["phase_currents_pu"]["B"]) == pytest.approx(
        -9j * complex(-0.5, -math.sqrt(3) / 2)
    )
    spec.update(
        fault_type="LL", faulted_phases=["A", "C"], fault_impedance_pu=z(0.02, 0.015)
    )
    ans = verify_symmetrical_fault(spec)
    v = [as_complex(ans["phase_voltages_pu"][p]) for p in "AC"]
    i = as_complex(ans["phase_currents_pu"]["A"])
    assert ans["conservation_verified"]
    assert v[0] - v[1] == pytest.approx((0.02 + 0.015j) * i)
    assert abs(as_complex(ans["ground_return_current_pu"])) < 1e-12


def test_ideal_short_with_no_source_impedance_and_wrong_phase_count():
    spec = source("3PH")
    spec["sequence_impedances_pu"]["positive"] = z(0, 0)
    assert not verify_symmetrical_fault(spec)["ok"]
    spec = source("LL")
    spec["faulted_phases"] = ["A"]
    assert not verify_symmetrical_fault(spec)["ok"]


def test_normal_conservation_and_domain_routes_reject_a_missing_phase_current():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    from hamon_dual_verifier import run_dual_verify_spec

    spec = source("SLG")
    for expected in (True,False):
        if not expected:
            spec["proposed_phase_currents_pu"] = {k: z(0,0) for k in "ABC"}
        assert run_router_conservation_cad_verifier({"spec":spec})["conservation_verified"] is expected
        assert run_dual_verify_spec({"mode":"domain","kind":"symmetrical_fault","spec":spec})["dual_verified"] is expected
