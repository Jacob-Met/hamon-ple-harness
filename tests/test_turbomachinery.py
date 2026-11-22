"""Reference velocity triangles and independent shaft/enthalpy observations."""

import math

import pytest

from hamon_turbomachinery import verify_turbomachinery


def lecture_case():
    # Georgia Tech AE4451, Compressor Example, slides 31–33.
    # https://seitzman.gatech.edu/classes/ae4451/turbomachinery_compressors.pdf
    return {
        "machine_kind": "compressor",
        "rotation_rpm": 8000,
        "inlet": {
            "radius_m": 0.30,
            "meridional_velocity_m_s": 100,
            "absolute_tangential_velocity_m_s": 36.4,
        },
        "outlet": {
            "radius_m": 0.30,
            "meridional_velocity_m_s": 100,
            "relative_flow_angle_deg": -50,
            "blade_metal_angle_deg": -50,
        },
    }


def test_published_axial_compressor_and_observation_boundary():
    spec = lecture_case()
    result = verify_turbomachinery(spec)
    assert result["calculation_consistent"]
    assert not result["conservation_verified"]
    assert result["inlet"]["blade_speed_m_s"] == pytest.approx(251, abs=0.5)
    assert result["fluid_specific_work_J_kg"] / 1000 == pytest.approx(24, abs=0.1)
    assert result["outlet"]["flow_minus_metal_angle_deg"] == pytest.approx(0)
    assert result["energy_residual_J_kg"] < 1e-8


def radial_case():
    # omega = 100 rad/s. u1 = 10, u2 = 20; independently given shaft torque = 3 N m.
    return {
        "machine_kind": "pump",
        "rotation_rpm": 6000 / (2 * math.pi),
        "inlet": {
            "radius_m": 0.1,
            "meridional_velocity_m_s": 4,
            "absolute_tangential_velocity_m_s": 2,
        },
        "outlet": {
            "radius_m": 0.2,
            "meridional_velocity_m_s": 3,
            "absolute_tangential_velocity_m_s": 8.5,
        },
        "mass_flow_kg_s": 2,
        "measured_fluid_torque_N_m": 3,
        "stage_boundary_speeds_m_s": {"inlet": 5, "outlet": 5},
        "isentropic_specific_work_J_kg": 120,
        "suction_flange": {
            "density_kg_m3": 1000,
            "suction_absolute_pressure_Pa": 100000,
            "vapor_pressure_Pa": 2500,
            "suction_velocity_m_s": 2,
            "required_npsh_m": 4,
        },
    }


def test_radial_pump_reaction_efficiency_suction_and_bad_claims():
    spec = radial_case()
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"], result
    assert result["fluid_specific_work_J_kg"] == pytest.approx(150)
    assert result["rotor_static_enthalpy_change_J_kg"] == pytest.approx(119.375)
    assert result["degree_of_reaction"] == pytest.approx(119.375 / 150)
    assert result["total_to_total_isentropic_efficiency"] == pytest.approx(0.8)
    assert result["npsh_margin_m"] == pytest.approx(99.5 / 9.80665 - 4)
    spec["measured_rotor_static_enthalpy_change_J_kg"] = 119.375
    assert verify_turbomachinery(spec)["conservation_verified"]
    spec["measured_rotor_static_enthalpy_change_J_kg"] = 150
    assert not verify_turbomachinery(spec)["conservation_verified"]
    del spec["measured_rotor_static_enthalpy_change_J_kg"]
    spec["suction_flange"]["required_npsh_m"] = 30
    assert not verify_turbomachinery(spec)["conservation_verified"]


def test_turbine_signs_and_mechanical_to_thermal_efficiency():
    spec = radial_case()
    spec["inlet"], spec["outlet"] = spec["outlet"], spec["inlet"]
    spec["machine_kind"] = "turbine"
    spec["measured_fluid_torque_N_m"] = -3
    spec["isentropic_specific_work_J_kg"] = -200
    del spec["suction_flange"]
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"], result
    assert result["fluid_power_W"] == pytest.approx(-300)
    assert result["total_to_total_isentropic_efficiency"] == pytest.approx(0.75)
    spec["isentropic_specific_work_J_kg"] = -100
    assert not verify_turbomachinery(spec)["conservation_verified"]


def test_conflicting_angles_and_metal_angle_alone():
    spec = lecture_case()
    spec["outlet"]["absolute_flow_angle_deg"] = 0
    assert not verify_turbomachinery(spec)["ok"]
    del (
        spec["outlet"]["absolute_flow_angle_deg"],
        spec["outlet"]["relative_flow_angle_deg"],
    )
    assert not verify_turbomachinery(spec)["ok"]


def test_registered_conservation_and_domain_solvers_require_independent_measurement():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    from hamon_dual_verifier import run_dual_verify_spec

    case = radial_case()
    case["calculation"] = "euler_rotor"
    for sign in (1,-1):
        case["measured_fluid_torque_N_m"] = sign*3
        assert run_router_conservation_cad_verifier({"spec":case})["conservation_verified"] is (sign == 1)
        assert run_dual_verify_spec({"mode":"domain","kind":"euler_rotor","spec":case})["dual_verified"] is (sign == 1)


@pytest.mark.parametrize("swirl", [5e-6,5e-7,5e-8,5e-9,5e-11,5e-13,5e-15])
def test_redundant_angles_use_local_swirl_and_preserve_relative_cancellation(swirl):
    spec = {
        "machine_kind": "pump", "rotation_rpm":6000/(2*math.pi), "mass_flow_kg_s":2,
        "inlet":{"radius_m":.1,"meridional_velocity_m_s":4,"absolute_tangential_velocity_m_s":0},
        "outlet":{"radius_m":.2,"meridional_velocity_m_s":3,"absolute_tangential_velocity_m_s":swirl},
        "measured_fluid_torque_N_m":.4*swirl,
    }
    station = spec["outlet"]
    station["relative_flow_angle_deg"] = math.degrees(math.atan2(swirl-20,3))
    station["absolute_flow_angle_deg"] = math.degrees(math.atan2(swirl,3))
    assert verify_turbomachinery(spec)["conservation_verified"]
    station["absolute_flow_angle_deg"] *= -1
    assert not verify_turbomachinery(spec)["conservation_verified"]
