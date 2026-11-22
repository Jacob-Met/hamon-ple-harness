"""Redundant independent duty and conductance observations must survive routing."""
import math

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


BASE = dict(Th_in_K=400.0, Th_out_K=360.0, Tc_in_K=300.0, Tc_out_K=340.0,
            Ch_W_K=2000.0, Cc_W_K=2000.0)
UA = 80000.0 / 60.0


def course():
    yield "rated", BASE, True
    yield "all_observations", dict(BASE, UA_W_K=UA, Q_W=80000.0), True
    yield "duty_conflicts", dict(BASE, Q_W=-1000.0), False
    yield "conductance_conflicts", dict(BASE, UA_W_K=1.0), False
    yield "duty_ok_conductance_wrong", dict(BASE, Q_W=80000.0, UA_W_K=1.0), False
    yield "conductance_ok_duty_wrong", dict(BASE, UA_W_K=UA, Q_W=60.0), False
    yield "reordered", dict(reversed(list(dict(BASE, Q_W=60.0, UA_W_K=UA).items()))), False
    yield "zero_capacity", dict(BASE, Ch_W_K=0, Cc_W_K=0), False
    yield "invalid_K", dict(BASE, Th_out_K=0), False
    yield "unusable_conductance", dict(BASE, UA_W_K="NaN"), False
    yield "negative_conductance", dict(BASE, UA_W_K=-UA), False
    yield "bad_arrangement", dict(BASE, arrangement="undefined"), False
    yield "counterflow_outlets_cross", dict(BASE, Th_out_K=330, Tc_out_K=370, Q_W=140000.0), True
    yield "parallel_outlets_cross", dict(BASE, arrangement="parallel", Th_out_K=330, Tc_out_K=370), False
    yield "parallel_lmtd", dict(BASE, arrangement="parallel", UA_W_K=80000*math.log(5)/80), True
    yield "zero_duty", dict(BASE, Th_out_K=400, Tc_out_K=300, UA_W_K=0, Q_W=0), True
    yield "tolerance_pass", dict(BASE, Q_W=80000.1), True
    yield "tolerance_fail", dict(BASE, Q_W=80010), False
    yield "partial", dict(Th_in_K=400), False
    yield "empty", {"mode": "thermal_fluid"}, False
    yield "empty_streams", {"inlets": [], "outlets": []}, False
    streams = {"inlets": [{"mdot_kg_s": 1}], "outlets": [{"mdot_kg_s": 1}]}
    yield "mass_only", streams, True
    yield "partial_plus_mass", dict(streams, Tc_out_K=340), False
    yield "bad_mass_good_exchanger", dict(BASE, inlets=[{"mdot_kg_s":2}], outlets=[{"mdot_kg_s":1}]), False
    yield "concordant_mixed", dict(BASE, **streams, Q_W=80000), True


def test_independent_exchanger_measurements_and_routing():
    for name, spec, expected in course():
        result = run_router_conservation_cad_verifier({"spec": spec})
        assert result["conservation_verified"] is expected, (name, result)
        if name == "all_observations":
            assert result["hx_duties_W"] == dict(hot_stream=80000, cold_stream=80000, Q_W=80000, UA_times_lmtd=80000)
        if name == "conductance_conflicts":
            assert result["hx_duty_checks"]["UA_times_lmtd"] is False

