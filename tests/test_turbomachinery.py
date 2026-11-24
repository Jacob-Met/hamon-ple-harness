"""Reference velocity triangles and independent shaft/enthalpy observations."""

import math
from fractions import Fraction

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


@pytest.mark.parametrize("direction,kind", [(1, "pump"), (-1, "turbine")])
def test_small_angular_momentum_change_preserves_torque_and_rothalpy(direction, kind):
    # One binary64 increment of swirl at fixed radius. At 900 rpm both U*Vtheta
    # products round to the same value before subtraction.
    omega = 900 * 2 * math.pi / 60
    radius = 0.125
    low, high = 15.0, math.nextafter(15.0, math.inf)
    inlet, outlet = (low, high) if direction == 1 else (high, low)
    delta = outlet - inlet
    torque = 2 * radius * delta
    # Unchanged meridional velocity and blade speed give dh=(U-mean(Vtheta))*dVtheta.
    enthalpy = (omega * radius - (inlet + outlet) / 2) * delta
    spec = {
        "machine_kind": kind, "rotation_rpm": 900, "mass_flow_kg_s": 2,
        "inlet": {"radius_m": radius, "meridional_velocity_m_s": 5,
                  "absolute_tangential_velocity_m_s": inlet},
        "outlet": {"radius_m": radius, "meridional_velocity_m_s": 5,
                   "absolute_tangential_velocity_m_s": outlet},
        "measured_fluid_torque_N_m": torque,
        "measured_rotor_static_enthalpy_change_J_kg": enthalpy,
    }
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["fluid_torque_N_m"] == torque
    assert result["fluid_power_W"] == pytest.approx(omega * torque, abs=0)
    assert result["rothalpy_work_J_kg"] == pytest.approx(omega * torque / 2, abs=0)
    assert not verify_turbomachinery(dict(spec, measured_fluid_torque_N_m=0))["conservation_verified"]
    spec["machine_kind"] = "turbine" if kind == "pump" else "pump"
    wrong = verify_turbomachinery(spec)
    assert not wrong["machine_direction_consistent"]
    assert not wrong["conservation_verified"]


def test_zero_angular_momentum_change_with_unequal_radii():
    # r1*Vtheta1 = r2*Vtheta2 = 3/8 m²/s, while the static enthalpy changes.
    spec = {
        "machine_kind": "pump", "rotation_rpm": 900, "mass_flow_kg_s": 2,
        "inlet": {"radius_m": 0.125, "meridional_velocity_m_s": 5,
                  "absolute_tangential_velocity_m_s": 3},
        "outlet": {"radius_m": 0.375, "meridional_velocity_m_s": 5,
                   "absolute_tangential_velocity_m_s": 1},
        "measured_fluid_torque_N_m": 0, "measured_rotor_static_enthalpy_change_J_kg": 4,
    }
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["fluid_torque_N_m"] == result["fluid_specific_work_J_kg"] == 0
    assert result["rotor_static_enthalpy_change_J_kg"] == pytest.approx(4)


def test_zero_rothalpy_change_with_reoriented_velocity_triangles():
    # Equal r*Vtheta = 15/4 and equal C² = 169 give zero work and zero dh.
    spec = {
        "machine_kind": "pump", "rotation_rpm": 900, "mass_flow_kg_s": 3,
        "inlet": {"radius_m": 0.75, "meridional_velocity_m_s": 12,
                  "absolute_tangential_velocity_m_s": 5},
        "outlet": {"radius_m": 0.3125, "meridional_velocity_m_s": 5,
                   "absolute_tangential_velocity_m_s": 12},
        "measured_rotor_static_enthalpy_change_J_kg": 0,
    }
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["fluid_specific_work_J_kg"] == result["rotor_static_enthalpy_change_J_kg"] == 0
    spec["measured_rotor_static_enthalpy_change_J_kg"] = -(2 ** -42)
    assert not verify_turbomachinery(spec)["conservation_verified"]


def test_direction_and_power_survive_underflow_of_reported_specific_work():
    spec = {
        "machine_kind": "pump", "rotation_rpm": 1e-30, "mass_flow_kg_s": 1e300,
        "inlet": {"radius_m": 1, "meridional_velocity_m_s": 1,
                  "absolute_tangential_velocity_m_s": 0},
        "outlet": {"radius_m": 1, "meridional_velocity_m_s": 1,
                   "absolute_tangential_velocity_m_s": 1e-300},
        "measured_fluid_torque_N_m": 1,
    }
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["fluid_specific_work_J_kg"] == 0
    assert result["fluid_power_W"] == pytest.approx(1e-30 * math.tau / 60, abs=0)
    assert not verify_turbomachinery(dict(spec, machine_kind="turbine"))["conservation_verified"]


def test_minimum_positive_rpm_retains_representable_power_and_blade_speed():
    rpm = math.ulp(0.0)
    mass = 1e300
    # The rad/s conversion underflows on its own, while torque * omega is representable.
    omega = Fraction(rpm) * Fraction(math.tau) / 60
    spec = {
        "machine_kind": "pump", "rotation_rpm": rpm, "mass_flow_kg_s": mass,
        "inlet": {"radius_m": 1, "meridional_velocity_m_s": 1,
                  "absolute_tangential_velocity_m_s": 0},
        "outlet": {"radius_m": 1, "meridional_velocity_m_s": 1,
                   "absolute_tangential_velocity_m_s": 1},
        "measured_fluid_torque_N_m": mass,
    }
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["fluid_power_W"] == float(omega * Fraction(mass)) > 0
    assert not verify_turbomachinery(dict(spec, machine_kind="turbine"))["conservation_verified"]
    for station in ("inlet", "outlet"):
        spec[station]["radius_m"] = mass
    spec["mass_flow_kg_s"] = 1
    result = verify_turbomachinery(spec)
    assert result["conservation_verified"]
    assert result["outlet"]["blade_speed_m_s"] == float(omega * Fraction(mass)) > 0


@pytest.mark.parametrize("z,loss", [(-8, 0), (1, 9), (3, 0.75), (0, 0)])
def test_pump_suction_measurement_referred_to_comparison_datum(z, loss):
    spec = radial_case()
    inlet = spec["suction_flange"]
    inlet.update(gauge_elevation_above_npsh_datum_m=z, loss_to_npsh_datum_m=loss)
    result = verify_turbomachinery(spec)
    # Compute in pressure units first: stagnation, hydrostatic shift, loss, vapor.
    g, rho = 9.80665, inlet["density_kg_m3"]
    pa = (inlet["suction_absolute_pressure_Pa"] + rho * inlet["suction_velocity_m_s"]**2 / 2
          + rho*g*(z-loss) - inlet["vapor_pressure_Pa"])
    threshold = rho*g*inlet["required_npsh_m"]
    assert result["available_npsh_m"] == pytest.approx(pa/(rho*g))
    assert result["npsh_margin_m"] == pytest.approx((pa-threshold)/(rho*g))
    assert result["meets_required_npsh"] is (pa >= threshold)
    assert result["conservation_verified"] is (pa >= threshold)


def test_negative_intervening_head_loss_is_refused():
    spec = radial_case()
    spec["suction_flange"]["loss_to_npsh_datum_m"] = -0.1
    assert not verify_turbomachinery(spec)["ok"]


@pytest.mark.parametrize("sg,absolute_psi,vapor_psi,speed_fps,height_ft,loss_ft,published_ft,tol_ft", [
    (0.914, 70 + 12.69, 77.68, 6.28, 5, 0.1, 18.17, 0.05),
    (0.500, 152 + 14, 163, 4.54, -2, 0, 12, 0.5),
])
def test_published_suction_gauge_examples(sg, absolute_psi, vapor_psi, speed_fps,
                                         height_ft, loss_ft, published_ft, tol_ft):
    # EPRI 1026475, Appendix B, example 5; Henshaw, What Is NPSH?, problem 2.
    # Both publications round their customary-unit conversion factors and answers.
    ft, psi = 0.3048, 6894.757293168
    spec = radial_case()
    spec["suction_flange"] = {
        "density_kg_m3": 1000 * sg, "suction_absolute_pressure_Pa": absolute_psi * psi,
        "vapor_pressure_Pa": vapor_psi * psi, "suction_velocity_m_s": speed_fps * ft,
        "gauge_elevation_above_npsh_datum_m": height_ft * ft,
        "loss_to_npsh_datum_m": loss_ft * ft, "required_npsh_m": 0,
    }
    original = verify_turbomachinery(spec)
    assert original["available_npsh_m"] / ft == pytest.approx(published_ft, abs=tol_ft)


def test_measurements_at_two_stations_recover_same_datum_head():
    spec = radial_case()
    inlet = spec["suction_flange"]
    reference = verify_turbomachinery(spec)
    # Moving upstream by 2 m in elevation and 0.5 m in dissipated head reduces
    # the static pressure by rho*g*1.5 when the measured velocities agree.
    inlet["suction_absolute_pressure_Pa"] -= inlet["density_kg_m3"] * 9.80665 * 1.5
    inlet.update(gauge_elevation_above_npsh_datum_m=2, loss_to_npsh_datum_m=0.5)
    upstream = verify_turbomachinery(spec)
    assert upstream["available_npsh_m"] == pytest.approx(reference["available_npsh_m"], abs=1e-12)
