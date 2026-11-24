"""External detention-basin hydrograph and adversarial routed continuity cases."""
from bisect import bisect_right
import json
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route


FT = .3048
CF = FT**3
PUBLISHED_CFS = [0,5.2,15.2,25.3,35,137.9,190.7,200.6,179.4,148.1,118.2,
                 83.7,73.5,64.3,56.1,46.3,39.7,38.3,36.2,34.7,32.8,30.8,28.8,
                 26.7,24.7,22.7,20.7,18.6,16.7,15.1,13.7,12.5,11,8.7,6.9,5.5,
                 4.3,3.4,2.7,2.2,1.7,1.4,1.1]


def problem(curve, inflows, start=0):
    return {"calculation": "level_pool_routing", "initial_stage_m": start,
            "rating_curve": [dict(stage_m=z, storage_m3=s, outflow_m3_s=q) for z,s,q in curve],
            "inflow_hydrograph": [dict(time_s=t, inflow_m3_s=q) for t,q in inflows]}


def county_case():
    # Los Angeles County Hydrology Manual (Jan 2006), Figure 8.2 and Table 8.1.
    storage = [0,30000,60000,80000,100000,120000,140000,160000,180000,
               200000,220000,230000,240000,250000,260000,300000,340000]
    flow = [0,11.6,16.4,20.1,23.2,25.9,28.4,30.7,32.8,34.8,36.6,38.4,40.1,
            66.5,113.4,173.5,244.3]
    inflow = [0,50,100,150,200,220,220,190,150,110,90,80,70,60,50,40,30,20,10,5] + [0]*23
    return problem([(i*.5*FT, s*CF, q*CF) for i,(s,q) in enumerate(zip(storage,flow))],
                   [(600*i,q*CF) for i,q in enumerate(inflow)])


def solve(spec, mode="civil_geotech"):
    return route({"mode": mode, "spec": spec})


def county_oracle(result):
    assert result["ok"] and result["conservation_verified"], result
    states = result["hydrograph"]
    assert len(states) == len(PUBLISHED_CFS)
    assert states[0] == {"stage_m": 0, "storage_m3": 0, "outflow_m3_s": 0, "time_s": 0}
    source = county_case()
    rating = source["rating_curve"]
    stages = [p["stage_m"] for p in rating]
    incoming = source["inflow_hydrograph"]
    step_errors = []
    for i, row in enumerate(states):
        assert row["time_s"] == incoming[i]["time_s"]
        j = min(len(rating)-2, max(0, bisect_right(stages, row["stage_m"])-1))
        left, right = rating[j:j+2]
        offset = (row["stage_m"]-left["stage_m"])/(right["stage_m"]-left["stage_m"])
        assert 0 <= offset <= 1
        for field in ("storage_m3","outflow_m3_s"):
            rated = left[field]+offset*(right[field]-left[field])
            assert math.isclose(row[field], rated, rel_tol=1e-12, abs_tol=1e-12)
        if i:
            prev = states[i-1]
            dt = row["time_s"]-prev["time_s"]
            net = ((incoming[i-1]["inflow_m3_s"]-prev["outflow_m3_s"])
                   + (incoming[i]["inflow_m3_s"]-row["outflow_m3_s"])) * dt/2
            step_errors.append(abs(row["storage_m3"]-prev["storage_m3"]-net))
    assert max(step_errors) < 1e-10
    # Both the manual's storage/outflow curve and outflow ordinates are rounded
    # in print to 0.1 cfs. Compare every ordinate to that resolution.
    error = max(abs(state["outflow_m3_s"]/CF-q) for state,q in zip(states,PUBLISHED_CFS))
    assert error <= .1, error
    assert result["peak_outflow_time_s"] == 4200 and result["peak_lag_s"] == 1200
    assert result["residual"] < 1e-12, result["residual"]
    return {"matched_ordinates": len(states), "peak_outflow_cfs": result["peak_outflow_m3_s"]/CF,
            "peak_stage_ft": result["peak_stage_m"]/FT, "peak_time_min": result["peak_outflow_time_s"]/60,
            "max_tabular_error_cfs": error, "max_relative_residual": result["residual"],
            "max_independent_step_error_m3": max(step_errors)}


class LevelPoolRouting(unittest.TestCase):
    def test_county_reference(self):
        summary = county_oracle(solve(county_case()))
        self.assertEqual(summary, county_oracle(solve(county_case(), mode="auto")))
        print("County level-pool benchmark " + json.dumps(summary, sort_keys=True))

    def test_variable_steps_closed_outlet_and_shifted_storage(self):
        spec = problem([(10,300,0),(12,500,0),(13,600,0)], [(5,2),(8,6),(13,4)], 11)
        r = solve(spec)
        self.assertTrue(r["conservation_verified"], r)
        self.assertEqual([v["storage_m3"] for v in r["hydrograph"]], [400,412,437])
        self.assertEqual(r["inflow_volume_m3"], 37)
        self.assertEqual(r["outflow_volume_m3"], 0)
        self.assertEqual(r["peak_stage_time_s"], 13)
        self.assertEqual(r["peak_outflow_time_s"], 5)

    def test_linear_storage_outflow_converges_to_continuous_solution(self):
        # S=20Q, constant inflow=4 and empty initial state give S(t)=80*(1-exp(-t/20)).
        exact = 80*(1-math.exp(-2))
        errors = []
        for step in (8,4,2,1):
            r = solve(problem([(0,0,0),(10,100,5)], [(t,4) for t in range(0,41,step)]))
            self.assertTrue(r["ok"], r)
            errors.append(abs(r["hydrograph"][-1]["storage_m3"]-exact))
        self.assertTrue(all(3.9 < a/b < 4.1 for a,b in zip(errors,errors[1:])), errors)

    def test_no_extrapolation_or_ignored_hydraulic_conditions(self):
        base = problem([(0,0,0),(1,10,1)], [(0,0),(1,1)])
        for spec in (dict(base, initial_stage_m=2), dict(base, tailwater_stage_m=.4),
                     problem([(0,0,0),(1,10,1)],[(0,0),(5,80)]),
                     problem([(0,0,9),(1,10,10)],[(0,0),(1,0)]),
                     problem([(0,0,0),(1,0,1)],[(0,0),(1,1)]),
                     problem([(0,0,1),(1,10,0)],[(0,0),(1,1)]),
                     problem([(0,0,0),(1,10,1)],[(1,0),(1,1)]),
                     problem([(0,0,0),(1,10,1)],[(0,False),(1,1)])):
            with self.subTest(spec=spec):
                r = solve(spec)
                self.assertFalse(r["conservation_verified"])
                self.assertNotIn("peak_stage_m", r)
                self.assertTrue(r["reason"])

    def test_published_datum_must_resolve_the_routed_excursion(self):
        h = 2**53
        r = solve(problem([(h,0,0),(h+2,2,0)],[(0,1),(1,1)], h))
        self.assertFalse(r["conservation_verified"])
        self.assertGreater(r["max_rating_relative_error"], .1)
