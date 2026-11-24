"""Analytical contact transitions on independent two-, three-, and four-span courses."""
from fractions import Fraction as Q
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route
from hamon_two_span_beam import two_span_beam


def course(q, **overrides):
    n = len(q)
    spec = dict(calculation="continuous_beam_multispan", spans_m=[1]*n, udl_N_m=q,
                span_EI_Nm2=[1]*n, support_conditions={k:"compression_only" for k in ("A",chr(65+n))})
    spec.update(overrides)
    return route({"mode":"structural","spec":spec})


class MultiSpanContact(unittest.TestCase):
    def test_middle_load_frees_both_overhangs(self):
        result = course([0,2,0])
        self.assertTrue(result["conservation_verified"])
        self.assertEqual(list(result["support_reactions_N"].values()),[0,1,1,0])
        self.assertEqual(list(result["support_gaps_m"].values()),[1/12,0,0,1/12])
        self.assertEqual(result["M_max_Nm"],.25)
        self.assertEqual(list(result["support_states"].values()),["lifted","bilateral","bilateral","lifted"])

    def test_three_moment_solution_and_load_pattern_switches(self):
        # Equal spans under uniform load give M_B=M_C=-qL^2/10.
        full = course([1]*3)
        self.assertEqual(list(full["support_moments_Nm"].values()),[0,-.1,-.1,0])
        self.assertEqual(list(full["support_reactions_N"].values()),[.4,1.1,1.1,.4])
        partial = course([1,2,0])
        self.assertEqual(list(partial["support_reactions_N"].values()),[5/16,15/8,13/16,0])
        self.assertEqual(partial["support_states"]["D"],"lifted")
        self.assertEqual(partial["support_states"]["A"],"bearing")

    def test_four_span_subdivision_and_rigid_seat_tilt(self):
        # Split one loaded physical span into two elements with a low,
        # noncontacting seat. Neither the force nor moment field may change.
        spec = dict(spans_m=[1,.5,.5,1], support_conditions=["compression_only","bilateral",
                    "compression_only","bilateral","compression_only"],support_settlements_m=[0,0,1,0,0])
        row = course([0,2,2,0],**spec)
        self.assertEqual(list(row["support_reactions_N"].values()),[0,1,0,1,0])
        self.assertEqual(row["support_gaps_m"]["C"],1-5/192)
        self.assertEqual(row["M_max_Nm"],.25)
        x=[0,1,1.5,2,3]
        tilt = [level+.5*p-2 for p,level in zip(x,spec["support_settlements_m"])]
        moved=course([0,2,2,0],**dict(spec,support_settlements_m=tilt))
        self.assertEqual(moved["support_gaps_m"],row["support_gaps_m"])
        self.assertEqual(moved["support_reactions_N"],row["support_reactions_N"])

    def test_unequal_span_controls_against_separate_closed_form(self):
        from itertools import product
        for loads in product((-2,0,3),repeat=2):
            x=(Q(2),Q(3))
            rigid=(Q(5),Q(11))
            levels=(Q(1,9),Q(-1,7),Q(0))
            conditions={"A":"compression_only"}
            got=course(list(loads),spans_m=[2,3],span_EI_Nm2=[5,11],support_conditions=conditions,
                       support_settlements_m=["-1/9","1/7",0])
            stock=two_span_beam(*x,*map(Q,loads),rigid,conditions,levels)
            self.assertEqual(got["support_gaps_m"],stock["support_gaps_m"])
            self.assertEqual(got["support_reactions_N"],{k:stock["R_"+k+"_N"] for k in "ABC"})
            self.assertEqual(got["M_max_Nm"],stock["M_max_Nm"])

    def test_contact_boundary_and_no_equilibrium(self):
        # The free A overhang sits at 1/12. Moving its seat through that
        # ordinate crosses lifted -> touching -> compressive contact.
        states=[course([0,2,0],support_settlements_m={"A":v})["support_states"]["A"]
                for v in ("0","-1/12","-1/11")]
        self.assertEqual(states,["lifted","touching","bearing"])
        all_seats=["compression_only"]*4
        self.assertFalse(course([-1]*3,support_conditions=all_seats)["conservation_verified"])
        self.assertFalse(course([0]*3,support_conditions=all_seats)["conservation_verified"])

    def test_refuse_ambiguous_and_unsupported_structural_inputs(self):
        self.assertFalse(course([0,2,0],span_m=3,E_Pa=1,I_m4=1,w_N_m=2)["ok"])
        self.assertFalse(course([0,2,0],support_conditions={"Z":"bilateral"})["ok"])
        self.assertFalse(course([0,2,0],support_conditions=["bilateral"])["ok"])
        for key, value in (("support_conditions",["compression_only"]*2),
                           ("spans_m",[1,1,1]),("supports",["A","B","C"])):
            bad=route({"mode":"structural","spec":{"span_m":3,"E_Pa":1,"I_m4":1,"w_N_m":2,key:value}})
            self.assertFalse(bad["ok"])

    def test_midspan_point_load_and_far_bearing_lift(self):
        # Shi Lab CSA 03: equal spans, one central force. The independent
        # force-method and OpenSees results are 13P/32, 22P/32, -3P/32.
        # https://www.shitaoshi.web.illinois.edu/bootcamp/csa/03-indeterminate-beam.html
        points = [{"x_m": 1, "downward_N": 32}]
        fixed = course([0,0], spans_m=[2,2], support_conditions={}, point_loads=points)
        self.assertTrue(fixed["conservation_verified"])
        self.assertEqual(list(fixed["support_reactions_N"].values()), [13,22,-3])
        self.assertEqual(fixed["support_moments_Nm"]["B"], -6)
        self.assertEqual(fixed["M_max_sagging_Nm"], 13)
        free = course([0,0], spans_m=[2,2], support_conditions={"C":"compression_only"},
                      span_EI_Nm2=[16,16], point_loads=points)
        self.assertEqual(list(free["support_reactions_N"].values()), [16,16,0])
        # End slope of a simply supported central point load times the overhang.
        self.assertEqual(free["support_gaps_m"]["C"], 1)
        self.assertEqual(free["M_max_sagging_Nm"], 16)

    def test_arbitrary_point_and_distributed_load_against_three_moment(self):
        # For two bilateral spans with level seats, integrate each simple-span
        # bending moment diagram against the linear unit-moment diagram.
        from itertools import product
        for a, p, b, q in product((Q(1,4), Q(5,4)), (-2,3), (Q(1,2), Q(5,2)), (-3,4)):
            L, R = Q(2), Q(3)
            E1, E2 = Q(7), Q(11)
            wl, wr = Q(-1), Q(2)
            loading = (wl*L**3/4 + p*a*(L**2-a**2)/L)/E1
            loading += (wr*R**3/4 + q*(R-b)*(R**2-(R-b)**2)/R)/E2
            mb = -loading/(2*(L/E1+R/E2))
            ra = wl*L/2 + p*(L-a)/L + mb/L
            rc = wr*R/2 + q*b/R + mb/R
            rb = wl*L + wr*R + p + q - ra - rc
            row = course([-1,2], spans_m=[2,3], span_EI_Nm2=[7,11], support_conditions={},
                         point_loads=[{"x_m":str(a),"downward_N":p},{"x_m":str(L+b),"downward_N":q}])
            self.assertEqual(row["support_moments_Nm"]["B"], float(mb))
            self.assertEqual(list(row["support_reactions_N"].values()), list(map(float,(ra,rb,rc))))

    def test_coincident_loads_and_invalid_coordinates(self):
        loads = [{"x_m":1, "downward_N":7}, {"x_m":1, "downward_N":-2},
                 {"x_m":0, "downward_N":3}, {"x_m":2, "downward_N":4}]
        row = course([0,0], support_conditions={}, point_loads=loads)
        self.assertEqual(list(row["support_reactions_N"].values()), [3,5,4])
        self.assertEqual(row["M_max_Nm"], 0)
        for invalid in (None, {}, [{"x_m":3,"downward_N":2}],
                        [{"x_m":1,"upward_N":2}], [{"x_m":True,"downward_N":1}]):
            self.assertFalse(course([0,0], point_loads=invalid)["ok"])
