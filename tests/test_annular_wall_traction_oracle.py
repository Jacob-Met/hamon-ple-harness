"""Annulus verification via boundary interpolation in inverse squared radius and cut traction."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import unittest

import numpy as np

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def vessel(**spec):
    return run_router_conservation_cad_verifier(
        {"mode": "solid_mohr_vessel_buckling", "spec": dict(subkind="pressure_vessel", **spec)})


class WallTractionOracle(unittest.TestCase):
    def test_radial_traction_interpolation_and_original_equivalent_stress(self):
        ri, ro = 3, 4
        inner, outer = Fraction(1, ri**2), Fraction(1, ro**2)
        for base in (0, 2**190):
            for delta in (-53, 47):
                pi, po = base + delta, base
                for closed in (False, True):
                    for radius in (ri, 3.25, 3.75, ro):
                        with self.subTest(base=base, delta=delta, closed=closed, r=radius):
                            x = Fraction(1) / Fraction(radius)**2
                            radial = -po + (po-pi)*(x-outer)/(inner-outer)
                            # Local force balance d(r*sigma_r)/dr = sigma_hoop.
                            hoop = radial - 2*x*(po-pi)/(inner-outer)
                            axial = Fraction(pi*ri**2-po*ro**2, ro**2-ri**2) if closed else Fraction(0)
                            mean = (radial + hoop + axial)/3
                            sq = ((hoop-radial)**2 + (radial-axial)**2 + (axial-hoop)**2)/2
                            with localcontext() as ctx:
                                ctx.prec = 150
                                vm = (Decimal(sq.numerator)/Decimal(sq.denominator)).sqrt()
                            ans = vessel(r_i=ri,r_o=ro,t=1,r_eval=radius,p_i=pi,p_o=po,
                                         closed_ends=closed)
                            self.assertTrue(ans["verified"], ans)
                            for name, actual in (("sigma_hoop_lame",hoop),("sigma_radial_lame",radial),
                                                 ("sigma_axial_lame",axial),("sigma_von_mises",vm)):
                                self.assertEqual(ans[name], float(actual))
                            for name, actual in (("deviatoric_radial",radial-mean),
                                                 ("deviatoric_hoop",hoop-mean),
                                                 ("deviatoric_axial",axial-mean)):
                                self.assertEqual(ans["stress_split"][name],float(actual))

    def test_integrated_diametral_cut_and_end_cap(self):
        nodes, weights = np.polynomial.legendre.leggauss(12)
        for pi,po in ((-9,24),(31,-8)):
            for closed in (False,True):
                terms = []
                for xi, weight in zip(nodes, weights):
                    r = 3.5 + float(xi)/2
                    ans = vessel(r_i=3,r_o=4,r_eval=r,p_i=pi,p_o=po,closed_ends=closed)
                    self.assertTrue(ans["verified"], ans)
                    terms.append(float(weight)*ans["sigma_hoop_lame"]/2)
                    self.assertAlmostEqual(7*ans["sigma_axial_lame"],
                                           pi*9-po*16 if closed else 0, places=10)
                self.assertAlmostEqual(math.fsum(terms), pi*3-po*4, places=10)

    def test_requires_off_center_geometry_and_respects_sampling_cells(self):
        # All three center-pair reconstructions miss the third measurement.
        # The upper halves of both ri and t cells intersect the lower half of ro's cell.
        ro = math.nextafter(.375, math.inf)
        for sample in (.125, .25, ro):
            ans = vessel(r_i=.125,t=.25,r_o=ro,r_eval=sample,p_i=3,p_o=1)
            self.assertTrue(ans["verified"], ans)
        for rbad in (math.nextafter(.125, 0), math.nextafter(ro, math.inf)):
            self.assertFalse(vessel(r_i=.125,t=.25,r_o=ro,r_eval=rbad,
                                    p_i=3,p_o=1)["verified"])
        self.assertFalse(vessel(r_i=.125,t=.25,r_o=math.nextafter(ro,math.inf),
                                p_i=3,p_o=1)["verified"])
