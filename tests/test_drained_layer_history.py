import json
import math

import pytest

from test_wave8_certified_budget import route


HISTORY = [{"time_s": 0, "surcharge_Pa": 1000}]
LAYER = {"H_clay_m": 1, "cv_m2_s": 1, "mv_per_Pa": 1e-7}
COLUMN = {
    "calculation": "drained_layer_consolidation_history",
    "interface_condition": "fully_drained", "top_boundary": "sealed", "bottom_boundary": "drained",
    "drained_layers": [LAYER, {**LAYER, "H_clay_m": 2, "mv_per_Pa": 2e-7}],
    "time_s": 1, "load_history": HISTORY,
}


def certified(spec):
    return json.loads(route(spec)["presolve_result_json"])


def test_column_uses_each_drainage_path_and_compressibility():
    result = certified(COLUMN)
    # Both layers have a 1 m drainage path. The uniform step's analytical
    # mean pore pressure follows the mixed-boundary diffusion eigenmodes.
    mean = 1000 * math.fsum(8 / (math.pi * (2*n+1))**2 * math.exp(-((2*n+1)*math.pi/2)**2)
                            for n in range(12))
    assert result["mean_excess_pore_pressure_Pa"] == pytest.approx(mean, rel=1e-8)
    assert result["settlement_at_time_m"] == pytest.approx((1000-mean)*(1e-7 + 2*2e-7), rel=1e-8)
    assert [row["drainage"] for row in result["layers"]] == ["single", "double"]
    assert result["dual_solver_rel_residual"] < 1e-6
    assert certified({**COLUMN, "proposed_settlement_m": result["settlement_at_time_m"]})
    changed = certified({**COLUMN, "top_boundary": "drained"})
    assert changed["settlement_at_time_m"] > result["settlement_at_time_m"]


@pytest.mark.parametrize('changes', [
    {"interface_condition": "hydraulically_connected"}, {"top_boundary": "unknown"},
    {"drained_layers": [LAYER]}, {"drained_layers": [LAYER] * 9},
    {"drained_layers": [LAYER, {**LAYER, "cv_m2_s": 0}]},
    {"rectangular_loads": []}, {"drained_layers": [LAYER, {**LAYER, "outside_head_m": 2}]},
    {"proposed_settlement_m": 0.01}, {"proposed_mean_excess_pore_pressure_Pa": 0},
])
def test_column_abstains_on_incomplete_or_inconsistent_physics(changes):
    assert "presolve_result_json" not in route({**COLUMN, **changes})


def test_small_absolute_error_cannot_certify_a_wrong_relative_claim():
    history = {**LAYER, "mv_per_Pa": 1e-18, "time_s": 1, "load_history": HISTORY,
               "drainage": "double", "calculation": "consolidation_history_1d"}
    exact = certified(history)["settlement_at_time_m"]
    assert certified({**history, "proposed_settlement_m": exact})
    assert "presolve_result_json" not in route({**history, "proposed_settlement_m": 9e-13})
    tail = {**history, "time_s": 8}
    assert 0 < certified(tail)["mean_excess_pore_pressure_Pa"] < 1e-30
    assert "presolve_result_json" not in route({**tail, "proposed_mean_excess_pore_pressure_Pa": 8e-7})


def test_reporting_increment_is_checked_in_the_quantity_units():
    history = {**LAYER, "mv_per_Pa": 1e-18, "time_s": 1, "load_history": HISTORY,
               "drainage": "double", "calculation": "consolidation_history_1d"}
    rounded = certified({**history, "proposed_settlement_m": 0, "settlement_reporting_increment_m": 1e-14,
                         "proposed_mean_excess_pore_pressure_Pa": 0, "pressure_reporting_increment_Pa": 0.1})
    assert rounded["verified_claims"]["proposed_settlement_m"] == {"value": 0, "unit": "m", "reporting_increment": 1e-14}
    assert rounded["verified_claims"]["proposed_mean_excess_pore_pressure_Pa"]["unit"] == "Pa"
    for changes in ({"proposed_settlement_m": 0}, {"proposed_settlement_m": 0, "settlement_reporting_increment_m": 1e-16},
                    {"proposed_settlement_m": 9e-13, "settlement_reporting_increment_m": 1e-14},
                    {"settlement_reporting_increment_m": 1e-12},
                    {"proposed_settlement_m": 0, "settlement_reporting_increment_m": 0},
                    {"proposed_settlement_m": 0, "settlement_reporting_increment_m": True},
                    {"proposed_settlement_m": 0, "pressure_reporting_increment_Pa": 2}):
        assert "presolve_result_json" not in route({**history, **changes})
    unloaded = {**history, "load_history": [{"time_s": 0, "surcharge_Pa": 0}]}
    assert certified({**unloaded, "proposed_settlement_m": 0, "proposed_mean_excess_pore_pressure_Pa": 0})


def test_a_reporting_bin_must_contain_both_computed_values():
    from hamon_consolidation_certificate import history_certificate

    def disagree(tool, request):
        return {"calculation": "consolidation_history_1d", "ok": True, "conservation_verified": True,
                "dual_solver_rel_residual": 1e-9, "settlement_at_time_m": .00499,
                "finite_volume_mean_effective_increment_Pa": .00501 / LAYER["mv_per_Pa"],
                "mean_excess_pore_pressure_Pa": 0, "finite_volume_mean_excess_pore_pressure_Pa": 0}

    request = {**LAYER, "calculation": "consolidation_history_1d", "proposed_settlement_m": 0,
               "settlement_reporting_increment_m": .01}
    assert history_certificate(request, disagree) is None


def test_column_reports_worst_absolute_layer_residual():
    from hamon_consolidation_certificate import history_certificate

    residuals = iter([-6e-7, 2e-8])

    def checked_layer(tool, request):
        return {"ok": True, "conservation_verified": True, "calculation": "consolidation_history_1d",
                "dual_solver_rel_residual": next(residuals), "settlement_at_time_m": 0,
                "finite_volume_mean_effective_increment_Pa": 0, "mean_excess_pore_pressure_Pa": 0,
                "finite_volume_mean_excess_pore_pressure_Pa": 0}

    assert history_certificate(COLUMN, checked_layer)["dual_solver_rel_residual"] == 6e-7
