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
