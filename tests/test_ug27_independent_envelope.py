"""Independent pressure/remaining-metal contact checks for supplied cylindrical shells."""
import math
import unittest
from fractions import Fraction as F

from hamon_dual_verifier import run_dual_verify_spec


class ShellEnvelope(unittest.TestCase):
    def test_corroded_wall_and_design_pressure_around_their_separate_envelopes(self):
        # In p=P/(SE), u=t/R coordinates, the hoop-stress wall relation is
        # p=u/(1+3u/5). At u=1/2 this gives p=5/13 < 77/200. Thus the union
        # of the two applicability alternatives ends at (385/769,77/200).
        radius=769/1024
        allowance=1/128
        supply=385/1024+allowance
        stress=56e6
        efficiency=.5
        joint=F(stress)*F(efficiency)
        boundary=float(joint*F(77,200))
        for wall in (math.nextafter(supply,0.),supply,math.nextafter(supply,1.),.25,.5):
            for p in (0.,math.nextafter(boundary,0.),boundary,math.nextafter(boundary,math.inf),50e6):
                for scale in (1.,2.**-16):
                    r=run_dual_verify_spec(dict(mode="domain",kind="asme_ug27_shell",spec={
                        "R_inner_m":radius*scale, "P_Pa":p,"S_allow_Pa":stress,
                        "weld_efficiency_E":efficiency,"supplied_thickness_m":wall*scale,
                        "corrosion_allowance_m":allowance*scale}))
                    u=(F(wall)-F(allowance))/F(radius)
                    x=F(p)/joint
                    sizing=x<=F(77,200)
                    rating=u<=F(385,769)
                    self.assertIs(r["dual_verified"],sizing or rating)
                    self.assertIs(r["passes_ug27"],u*(1-F(3,5)*x)>=x if sizing or rating else None)
                    expected=(F(radius)*x/(1-F(3,5)*x)+F(allowance))*F(scale) if sizing else None
                    self.assertEqual(r["t_required_m"],float(expected) if sizing else None)
                    expected_capacity=joint*u/(1+F(3,5)*u)
                    self.assertEqual(r["circumferential_pressure_capacity_Pa"],
                                     float(expected_capacity) if rating else None)
