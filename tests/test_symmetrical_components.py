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


@pytest.mark.parametrize("zero_scale", [1e12, 1e19])
@pytest.mark.parametrize("fault_z", [0j, 0.01 + 0.02j])
@pytest.mark.parametrize("fault,phases", [
    ("3PH", "ABC"), ("SLG", "A"), ("SLG", "B"), ("SLG", "C"),
    ("LL", "AB"), ("LL", "BC"), ("LL", "CA"),
    ("DLG", "AB"), ("DLG", "BC"), ("DLG", "CA"),
])
def test_weak_ground_against_shared_neutral_circuit(fault, phases, fault_z, zero_scale):
    # Equal positive/negative series impedances are three independent
    # impedances plus a shared neutral. Solve that physical star directly.
    series = 0.05 + 0.125j
    neutral = (0.25 + 0.5j) * zero_scale
    zero = series + 3 * neutral
    source_a = 0.94 + 0.17j
    rotate = complex(-0.5, -math.sqrt(3) / 2)
    emf = dict(zip("ABC", (source_a, source_a * rotate, source_a * rotate.conjugate())))
    spec = source(fault)
    spec.update(faulted_phases=list(phases),
                prefault_positive_voltage_pu=z(source_a.real, source_a.imag),
                fault_impedance_pu=z(fault_z.real, fault_z.imag))
    spec["sequence_impedances_pu"] = {
        key: z(value.real, value.imag)
        for key, value in (("positive", series), ("negative", series), ("zero", zero))
    }
    currents = dict.fromkeys("ABC", 0j)
    ground = 0j
    if fault == "SLG":
        ground = emf[phases] / (series + neutral + fault_z)
        currents[phases] = ground
    elif fault == "3PH":
        currents = {name: e / (series + fault_z) for name, e in emf.items()}
    else:
        b, c = phases
        if fault == "LL":
            currents[b] = (emf[b] - emf[c]) / (2 * series + fault_z)
            currents[c] = -currents[b]
        else:
            ground = (emf[b] + emf[c]) / (series + 2 * (neutral + fault_z))
            difference = (emf[b] - emf[c]) / series
            currents[b], currents[c] = (ground + difference) / 2, (ground - difference) / 2

    got = verify_symmetrical_fault(spec)
    assert got["conservation_verified"], got
    assert as_complex(got["ground_return_current_pu"]) == pytest.approx(ground, rel=1e-12, abs=0)
    for name, i in currents.items():
        actual_i = as_complex(got["phase_currents_pu"][name])
        actual_v = as_complex(got["phase_voltages_pu"][name])
        assert actual_i == pytest.approx(i, rel=1e-12, abs=1e-30)
        assert actual_v == pytest.approx(emf[name] - series * i - neutral * ground, abs=1e-13)
        if fault in ("DLG", "SLG") and name in phases:
            assert actual_v == pytest.approx(fault_z * ground, rel=1e-12, abs=1e-38)


@pytest.mark.parametrize("mode", ["zero", "positive", "negative"])
def test_three_phase_fault_rejects_a_singular_terminal_mode(mode):
    spec = source("3PH")
    spec["fault_impedance_pu"] = z(0, 0.25)
    spec["sequence_impedances_pu"][mode] = z(0, -0.25)
    assert not verify_symmetrical_fault(spec)["conservation_verified"]


@pytest.mark.parametrize("fault,phases", [
    ("3PH","ABC"), ("SLG","A"), ("SLG","B"), ("SLG","C"),
    ("LL","AB"), ("LL","CA"), ("LL","BC"),
    ("DLG","AB"), ("DLG","CA"), ("DLG","BC"),
])
def test_unbalanced_prefault_source_against_independent_phase_branches(fault, phases):
    # Equal sequence impedances give three independent phase branches. Solve
    # each circuit directly at its terminals, including a shared DLG junction.
    phase_emf = [1+.3j,-.6+.4j,.15-.7j]
    alpha = complex(-.5,math.sqrt(3)/2)
    e0 = sum(phase_emf)/3
    e1 = (phase_emf[0]+alpha*phase_emf[1]+alpha.conjugate()*phase_emf[2])/3
    e2 = (phase_emf[0]+alpha.conjugate()*phase_emf[1]+alpha*phase_emf[2])/3
    def pack(x):
        return z(x.real,x.imag)
    impedance, ground = .2+.4j, .07+.13j
    spec = dict(source(fault), faulted_phases=list(phases),fault_impedance_pu=pack(ground),
                prefault_positive_voltage_pu=pack(e1),prefault_negative_voltage_pu=pack(e2),
                prefault_zero_voltage_pu=pack(e0),
                sequence_impedances_pu={k:pack(impedance) for k in ("zero","positive","negative")})
    current = [0j,0j,0j]
    selected = ["ABC".index(p) for p in phases]
    if fault in ("SLG","3PH"):
        for i in selected:
            current[i]=phase_emf[i]/(impedance+ground)
    elif fault=="LL":
        b,c = selected
        current[b]=(phase_emf[b]-phase_emf[c])/(2*impedance+ground)
        current[c]=-current[b]
    else:
        b,c=selected
        junction=ground*(phase_emf[b]+phase_emf[c])/(impedance+2*ground)
        for i in selected:
            current[i]=(phase_emf[i]-junction)/impedance
    spec["proposed_phase_currents_pu"]=dict(zip("ABC",map(pack,current)))
    got=verify_symmetrical_fault(spec)
    assert got["conservation_verified"], got
    for i,p in enumerate("ABC"):
        assert as_complex(got["phase_currents_pu"][p])==pytest.approx(current[i],abs=1e-12)
        assert as_complex(got["phase_voltages_pu"][p])==pytest.approx(phase_emf[i]-impedance*current[i])


def test_cancelling_prefault_sequences_reject_the_old_one_pu_claim():
    spec=dict(source("SLG"),sequence_impedances_pu={k:z(1,0) for k in ("zero","positive","negative")},
              prefault_negative_voltage_pu=z(-1,0))
    claim={"A":z(1,0),"B":z(0,0),"C":z(0,0)}
    assert not verify_symmetrical_fault(dict(spec,proposed_phase_currents_pu=claim))["conservation_verified"]
    answer=verify_symmetrical_fault(spec)
    assert answer["conservation_verified"]
    assert all(as_complex(value)==0 for value in answer["phase_currents_pu"].values())
    assert not verify_symmetrical_fault(dict(spec,prefault_other_voltage_pu=z(1,0)))["ok"]
    assert not verify_symmetrical_fault(dict(spec,prefault_zero_voltage_pu=3))["ok"]


def test_negative_sequence_only_solid_three_phase_fault():
    spec=dict(source("3PH"),prefault_positive_voltage_pu=z(0,0),prefault_negative_voltage_pu=z(.3,-.2))
    ans=verify_symmetrical_fault(spec)
    assert ans["conservation_verified"]
    assert as_complex(ans["sequence_currents_pu"]["negative"])==pytest.approx((.3-.2j)/(1j/15))
    assert as_complex(ans["sequence_currents_pu"]["zero"])==0
    assert as_complex(ans["sequence_currents_pu"]["positive"])==0
