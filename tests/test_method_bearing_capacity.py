import math

import pytest

from hamon_bearing_capacity import verify_bearing_capacity


def base(method):
    return {
        "calculation": "bearing_capacity",
        "bearing_method": method,
        "footing_B_m": 2.0,
        "depth_Df_m": 1.0,
        "gamma_N_m3": 18000.0,
        "c_prime_Pa": 10000.0,
        "phi_deg": 30.0,
        "FS": 3.0,
    }


@pytest.mark.parametrize(
    "method,nc,nq,ng",
    [
        ("terzaghi", 37.16, 22.46, 19.7),
        ("meyerhof", 30.14, 18.40, 15.67),
        ("vesic", 30.14, 18.40, 22.40),
    ],
)
def test_usace_tables_and_surface_strip_capacity(method, nc, nq, ng):
    spec = base(method)
    spec["depth_Df_m"] = 0
    result = verify_bearing_capacity(spec)
    assert result["conservation_verified"], result
    assert [result[k] for k in ("Nc", "Nq", "Ngamma")] == pytest.approx(
        [nc, nq, ng], rel=0.0004
    )
    assert result["q_ult_Pa"] == pytest.approx(10000 * nc + 18000 * ng, rel=0.0004)


def test_ponding_does_not_reduce_effective_soil_stress_below_submerged_weight():
    spec = base("vesic")
    spec.update(gamma_sat_N_m3=19806.65, water_table_depth_m=-4)
    result = verify_bearing_capacity(spec)
    assert result["conservation_verified"]
    assert result["effective_overburden_Pa"] == pytest.approx(10000)
    assert result["effective_wedge_unit_weight_N_m3"] == pytest.approx(10000)
    spec["water_table_depth_m"] = 0
    assert result["q_ult_Pa"] == pytest.approx(
        verify_bearing_capacity(spec)["q_ult_Pa"]
    )
    spec["water_table_depth_m"] = 2
    result = verify_bearing_capacity(spec)
    assert result["effective_overburden_Pa"] == pytest.approx(18000)
    assert result["effective_wedge_unit_weight_N_m3"] == pytest.approx(14000)


def test_embedment_and_finite_rectangle_depth_factors():
    spec = base("meyerhof")
    spec.update(footing_L_m=4)
    a = verify_bearing_capacity(spec)
    assert a["conservation_verified"]
    kp = 3.0
    ratio = 0.5
    depth_ratio = 0.5
    nc, nq, ng = (a[k] for k in ("Nc", "Nq", "Ngamma"))
    expected = 10000 * nc * (1 + 0.2 * kp * ratio) * (
        1 + 0.2 * math.sqrt(kp) * depth_ratio
    ) + (18000 * nq + 18000 * ng) * (1 + 0.1 * kp * ratio) * (
        1 + 0.1 * math.sqrt(kp) * depth_ratio
    )
    assert a["q_ult_Pa"] == pytest.approx(expected)
    spec["applied_gross_pressure_Pa"] = a["q_allow_Pa"] * 1.1
    bad = verify_bearing_capacity(spec)
    assert not bad["conservation_verified"] and not bad["bearing_ok"]
    spec.update(bearing_method="terzaghi")
    assert not verify_bearing_capacity(spec)["ok"]


def test_invalid_weight_slope_claim_and_historical_rectangle():
    spec = base("vesic")
    spec.update(water_table_depth_m=0.1, gamma_sat_N_m3=8000)
    assert not verify_bearing_capacity(spec)["ok"]
    spec = base("vesic")
    spec["ground_slope_deg"] = 10
    assert not verify_bearing_capacity(spec)["ok"]
    spec = base("vesic")
    spec["proposed_q_ult_Pa"] = 123
    assert not verify_bearing_capacity(spec)["conservation_verified"]
    historical = {
        "B_m": 2,
        "L_m": 3,
        "Df_m": 1,
        "gamma_N_m3": 18000,
        "phi_deg": 0,
        "c_Pa": 10000,
        "column_load_N": 12000,
        "FS": 3,
    }
    result = verify_bearing_capacity(historical, legacy_rectangle=True)
    assert (
        result["Nc"] == 5.14 and result["q_applied_Pa"] == 2000 and result["bearing_ok"]
    )

def test_domain_and_auto_routes_reject_wrong_capacity():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    from hamon_dual_verifier import run_dual_verify_spec

    spec = base("terzaghi")
    exact = verify_bearing_capacity(spec)["q_ult_Pa"]
    for claim, expected in ((exact, True), (-exact, False)):
        spec["proposed_q_ult_Pa"] = claim
        assert run_router_conservation_cad_verifier({"mode": "auto", "spec": spec})["conservation_verified"] is expected
        assert run_dual_verify_spec({"mode": "domain", "kind": "bearing_capacity", "spec": spec})["dual_verified"] is expected


@pytest.mark.parametrize("method", ["legacy_vesic", "terzaghi", "meyerhof", "vesic"])
def test_design_factor_cannot_certify_loading_above_ultimate(method):
    spec = base(method)
    spec["FS"] = 1
    ultimate = verify_bearing_capacity(spec)["q_ult_Pa"]
    spec["applied_gross_pressure_Pa"] = ultimate
    assert verify_bearing_capacity(spec)["bearing_ok"]
    spec["applied_gross_pressure_Pa"] = 1.5 * ultimate
    assert not verify_bearing_capacity(spec)["conservation_verified"]
    spec["FS"] = 0.5
    assert not verify_bearing_capacity(spec)["ok"]


def eccentric_spec(**changes):
    return {
        **base("meyerhof"), "footing_B_m": 4, "footing_L_m": 6, "c_prime_Pa": 0,
        "column_load_N": 6_300_000, "eccentricity_method": "meyerhof_effective_area",
        "eccentricity_B_m": 0.5, "eccentricity_L_m": -0.25, **changes,
    }


def test_biaxial_eccentric_bearing_at_registered_entry_points():
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    from hamon_dual_verifier import run_dual_verify_spec

    # The 4 x 6 m footprint becomes 3 x 5.5 m. Use independently expanded
    # Meyerhof factors at 30 degrees and a dry 1 m embedment.
    nq = 3 * math.exp(math.pi / math.sqrt(3))
    ng = (nq - 1) * math.tan(math.radians(42))
    factor = (1 + 0.3 * 3 / 5.5) * (1 + 0.1 * math.sqrt(3) / 3)
    expected = (18_000 * nq + 27_000 * ng) * factor
    for force, safe in ((6_300_000, False), (2_500_000, True)):
        spec = eccentric_spec(column_load_N=force)
        result = run_router_conservation_cad_verifier({"mode": "auto", "spec": spec})
        assert result["effective_area_m2"] == 16.5
        assert result["q_ult_Pa"] == pytest.approx(expected)
        assert result["q_applied_Pa"] == pytest.approx(force / 16.5)
        assert result["bearing_ok"] is safe
        dual = run_dual_verify_spec({"mode": "domain", "kind": "bearing_capacity", "spec": spec})
        assert dual["dual_verified"] is safe


def test_effective_dimensions_follow_named_axes_and_ignore_offset_signs():
    original = verify_bearing_capacity(eccentric_spec())
    swapped = verify_bearing_capacity(eccentric_spec(
        footing_B_m=6, footing_L_m=4, eccentricity_B_m=0.25, eccentricity_L_m=-0.5))
    assert (swapped["effective_B_m"], swapped["effective_L_m"]) == (5.5, 3)
    assert swapped["q_ult_Pa"] == original["q_ult_Pa"]
    assert swapped["q_applied_Pa"] == original["q_applied_Pa"]


def test_fhwa_prestressed_girder_example_effective_geometry():
    # FHWA PSC girder worked example, §7.2.4.5. Only the published contact
    # geometry is compared: its subsequent LRFD resistance uses other factors.
    foot = 0.3048
    result = verify_bearing_capacity(eccentric_spec(
        footing_B_m=12 * foot, footing_L_m=12 * foot,
        eccentricity_B_m=121 / 1374 * foot, eccentricity_L_m=626 / 1374 * foot))
    assert round(result["effective_B_m"] / foot, 2) == 11.82
    assert round(result["effective_L_m"] / foot, 2) == 11.09


@pytest.mark.parametrize("changes", [
    {"eccentricity_method": None}, {"eccentricity_method": "other"},
    {"bearing_method": "vesic"}, {"eccentricity_B_m": -2}, {"eccentricity_L_m": 4},
    {"column_load_N": 0}, {"horizontal_load_N": 1}, {"ground_slope_deg": 1},
    {"base_tilt_deg": 1}, {"footing_shape": "circular"}, {"applied_gross_pressure_Pa": 100},
])
def test_unsupported_eccentric_design_refused(changes):
    result = verify_bearing_capacity(eccentric_spec(**changes))
    assert not result["ok"]
    assert not result["conservation_verified"]


@pytest.mark.parametrize(
    "phi, allowable",
    [(0, 77123.8898038469), (5, 99943.45328687997), (10, 134014.95855960564)],
)
def test_meyerhof_cohesion_shape_uses_the_actual_friction_angle(phi, allowable):
    # The 3 × 5 m footing has a 2.5 × 4 m effective rectangle. At the
    # surface, its surcharge and all embedment corrections vanish.
    spec = eccentric_spec(
        footing_B_m=3, footing_L_m=5, eccentricity_B_m=0.25, eccentricity_L_m=0.5,
        depth_Df_m=0, phi_deg=phi, c_prime_Pa=40000, column_load_N=1_000_000,
    )
    result = verify_bearing_capacity(spec)
    assert result["q_allow_Pa"] == pytest.approx(allowable, rel=1e-12)
    assert result["q_applied_Pa"] == 100000
    assert result["bearing_ok"] is (allowable >= 100000)


def test_meyerhof_cohesion_depth_uses_the_actual_friction_angle():
    spec = {**base("meyerhof"), "footing_B_m": 2.5, "footing_L_m": 4,
            "phi_deg": 5, "c_prime_Pa": 40000}
    surface = verify_bearing_capacity({**spec, "depth_Df_m": 0})
    embedded = verify_bearing_capacity({**spec, "depth_Df_m": 1.5})
    # Dividing the terms cancels both Nc and the shape factor.
    depth_factor = 1 + 0.2 * (1.5 / 2.5) * math.tan(math.radians(45 + 5 / 2))
    assert embedded["cohesion_term_Pa"] / surface["cohesion_term_Pa"] == pytest.approx(
        depth_factor, rel=1e-12
    )
