"""Independent discharge-law inversion and signed survey/freeboard controls."""
import math
import unittest
from decimal import Decimal as D, localcontext
from hamon_dual_verifier import run_dual_verify_spec


def run(q,crest,top,minimum,coefficient=1.75,length=4.,**other):
    spec=dict(Q_design_m3_s=q,Cd_si=coefficient,weir_length_m=length,
              crest_el_m=crest,dam_top_el_m=top,min_freeboard_m=minimum,**other)
    return run_dual_verify_spec(dict(mode="domain",kind="spillway_weir",spec=spec))


def invert(q, factor):
    def rating(h):
        return factor*h*h.sqrt()
    lo,hi=D(0),D(1)
    while rating(hi)<q:
        hi*=2
    while rating(hi/2)>=q:
        hi/=2
    if rating(hi)==q:
        return hi
    for _ in range(450):
        mid=(lo+hi)/2
        if mid in (lo,hi):
            break
        if rating(mid)<q:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


class FreeOverflowSurveyOracle(unittest.TestCase):
    def test_original_rating_near_capacity_at_shifted_survey_datums(self):
        with localcontext() as ctx:
            ctx.prec=140
            factor=D(7)
            limit=D("2.25")
            capacity=factor*limit*limit.sqrt()
            for crest in (0,2**60,-2**60):
                for flow in (3.125,math.nextafter(float(capacity),0.),float(capacity),
                             math.nextafter(float(capacity),math.inf)):
                    with self.subTest(crest=crest,flow=flow):
                        h=invert(D.from_float(flow),factor)
                        required_stage=D(crest)+h
                        clearance=D(3)-h
                        out=run(flow,crest,crest+3,.75,
                                water_surface_el_m=float(required_stage))
                        self.assertTrue(out["dual_verified"],out)
                        self.assertEqual(out["head_required_m"],float(h))
                        self.assertEqual(out["water_surface_el_m"],float(required_stage))
                        self.assertEqual(out["freeboard_m"],float(clearance))
                        self.assertEqual(out["Q_capacity_with_freeboard_m3_s"],float(capacity))
                        self.assertEqual(out["passes_freeboard"],D.from_float(flow)<=capacity)
                        if crest:
                            with self.assertRaises(ValueError):
                                run(flow,crest,crest+3,.75,water_surface_el_m=crest)

    def test_no_overflow_set_and_exact_integer_observation(self):
        uncertain=run(0,-12,-14,.5)
        self.assertIsNone(uncertain["passes_freeboard"])
        self.assertEqual(uncertain["water_surface_el_upper_bound_m"],-12)
        self.assertEqual(uncertain["freeboard_lower_bound_m"],-2)
        for stage, passes in ((-200,True),(-15,True),(-14.5,True),(-14.,False),(-12,False)):
            with self.subTest(stage=stage):
                out=run(0,-12,-14,.5,water_surface_el_m=stage)
                self.assertEqual(out["freeboard_m"],-14-stage)
                self.assertEqual(out["passes_freeboard"],passes)
        self.assertTrue(run(0,-12,-11,1.)["passes_freeboard"])
        with self.assertRaises(ValueError):
            run(0,-12,-14,.5,water_surface_el_m=-11)
        for crest in (-2**60,2**60):
            self.assertTrue(run(56.,crest,crest+6,2.,water_surface_el_m=crest+4)["passes_freeboard"])
            with self.assertRaises(ValueError):
                run(56.,crest,crest+6,2.,water_surface_el_m=crest+5)

    def test_finite_rating_across_extreme_head_scales(self):
        u=math.ulp(0.)
        a=run(2.**-511,0,u,0,2.**600,2.**500)
        self.assertEqual(a["head_required_m"],u)
        self.assertEqual(a["water_surface_el_m"],u)
        self.assertEqual(a["freeboard_m"],0.)
        self.assertTrue(a["passes_freeboard"])
        with self.assertRaises(ValueError):
            run(1.75*2.**-511,0,u,0,2.**600,2.**500)
        b=run(2.**100,0,2.**601,2.**600,2.**-400,2.**-400)
        self.assertEqual(b["head_required_m"],2.**600)
        self.assertEqual(b["freeboard_m"],2.**600)
        self.assertEqual(b["Q_capacity_with_freeboard_m3_s"],2.**100)
        self.assertTrue(b["passes_freeboard"])
