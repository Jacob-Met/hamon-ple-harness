import pytest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
from hamon_dual_verifier import run_dual_verify_spec


def lot(**changes):
    return {
        "calculation": "compaction_acceptance",
        "gamma_d_max": 120,
        "w_opt_pct": 10,
        "min_rc": 0.95,
        "w_tol_below_pct": 1,
        "w_tol_above_pct": 3,
        "tests": [
            {"id": "A", "gamma_dry": 114, "w_pct": 9},
            {"id": "B", "gamma_wet": 128.82, "w_pct": 13},
        ],
        **changes,
    }


def civil(spec):
    return run_router_conservation_cad_verifier({"mode": "auto", "spec": spec})


def test_independent_edges_and_equivalent_density_bases():
    original = civil(lot())
    assert original["lot_accepted"] and original["conservation_verified"]
    assert original["A_rc"] == original["B_rc"] == 0.95
    assert original["moisture_range_pct"] == [9, 13]
    assert original["dry_density_minimum"] == 114
    assert all(row["dry_density"] == 114 for row in original["test_results"])
    for moisture, density, density_ok, moisture_ok in [
        (8.999, 115, True, False),
        (13.001, 115, True, False),
        (10, 113.999, False, True),
    ]:
        result = civil(lot(tests=[{"id": 1, "w_pct": moisture, "gamma_dry": density}]))
        assert (
            result["ok"]
            and not result["lot_accepted"]
            and not result["conservation_verified"]
        )
        (row,) = result["test_results"]
        assert (row["density_passes"], row["moisture_passes"]) == (
            density_ok,
            moisture_ok,
        )


def test_legacy_wet_density_and_symmetric_domain_contract():
    spec = lot(w_tol_pct=2, tests=[{"id": 8, "w_pct": 10, "gamma_wet": 125.4}])
    del spec["w_tol_below_pct"], spec["w_tol_above_pct"]
    result = run_dual_verify_spec(
        {"mode": "domain", "kind": "compaction_acceptance", "spec": spec}
    )
    assert result["dual_verified"] and result["lot_accepted"]
    assert result["passed_ids"] == [8] and result["8_rc"] == 0.95


@pytest.mark.parametrize(
    "changes",
    [
        {"tests": []},
        {"tests": [{"id": "", "w_pct": 9, "gamma_dry": 120}]},
        {"tests": [{"id": "A", "w_pct": 9, "gamma_dry": 120}] * 2},
        {"tests": [{"id": "A", "w_pct": 9, "gamma_wet": 130, "gamma_dry": 120}]},
        {"tests": [{"id": "A", "w_pct": 9}]},
        {"tests": [{"id": "A", "w_pct": -1, "gamma_dry": 120}]},
        {"w_tol_pct": 4},
        {"w_tol_above_pct": None},
        {"w_tol_below_pct": -1},
        {"gamma_d_max": 0},
        {"min_rc": True},
    ],
)
def test_incomplete_or_ambiguous_lot_refuses(changes):
    result = civil(lot(**changes))
    assert (
        not result["ok"]
        and not result["lot_accepted"]
        and not result["conservation_verified"]
    )
    assert result["reason"]
