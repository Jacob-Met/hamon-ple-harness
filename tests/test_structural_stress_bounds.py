import math

from hamon_conservation_cad_verifier import verify_structural_mechanics


def tube(**changes):
    return {
        "span_m": 2.0, "E_Pa": 200e9, "boundary": "cantilever_point",
        "P_N": 750, "safety_factor": 1, "yield_strength_Pa": 250e6,
        "section": {"shape": "thin_walled_circular_tube",
                    "mean_radius_m": 0.08, "wall_thickness_m": 0.002},
        **changes,
    }


def test_unsupported_boundary_has_no_reactions_or_demands():
    result = verify_structural_mechanics(tube(boundary="unspecified"))
    assert result["ok"] is False and result["conservation_verified"] is False
    assert "max_shear_N" not in result and "max_moment_Nm" not in result


def test_undeclared_combined_section_cannot_pass_by_dropping_transverse_shear():
    spec = {"span_m": 1, "E_Pa": 2e11, "boundary": "simply_supported_point",
            "I_m4": 0.01, "area_m2": 0.001, "c_m": 0.1, "P_N": 1000,
            "safety_factor": 1, "yield_strength_Pa": 1e6}
    assert verify_structural_mechanics(spec)["passes_stress"] is False
    for torque in (1e-30, -1e-30, 1, -1):
        result = verify_structural_mechanics({**spec, "torque_Nm": torque})
        assert result["passes_stress"] is None
        assert result["von_mises_stress_Pa"] is None
        assert result["max_shear_N"] == 500


def test_tube_stress_recovers_the_pure_torque_and_bending_limits():
    radius, thickness = 0.08, 0.002
    inertia = math.pi * radius**3 * thickness
    pure_torque = verify_structural_mechanics(tube(P_N=0, torque_Nm=-40))
    assert math.isclose(pure_torque["von_mises_stress_Pa"],
                        math.sqrt(3) * 40 * radius / (2 * inertia))
    # The largest bending and transverse stresses sit 90 degrees apart.
    exact = max(750 * 2 * radius / inertia,
                math.sqrt(3) * 2 * 750 / (2 * math.pi * radius * thickness))
    for torque in (0, 1e-30, -1e-30):
        result = verify_structural_mechanics(tube(torque_Nm=torque))
        bounds = result["von_mises_bounds_Pa"]
        assert math.isclose(bounds["lower"], exact)
        assert exact <= bounds["upper"] <= 1.001 * exact


def test_tube_rejects_contradictions_and_reports_an_unresolved_limit():
    assert verify_structural_mechanics(tube(I_m4=1))["ok"] is False
    assert verify_structural_mechanics(tube(w_N_m=1))["ok"] is False
    result = verify_structural_mechanics(tube(torque_Nm=40))
    bounds = result["von_mises_bounds_Pa"]
    limit = (bounds["lower"] + bounds["upper"]) / 2
    assert bounds["lower"] < limit < bounds["upper"]
    assert verify_structural_mechanics(tube(torque_Nm=40, yield_strength_Pa=limit))["passes_stress"] is None
