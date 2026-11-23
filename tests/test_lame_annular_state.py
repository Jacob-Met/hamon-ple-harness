"""Lamé wall geometry and radial/hoop stress invariants on the registered route."""
import math
import unittest
from fractions import Fraction

from hamon_wave6_physics_verifiers import dispatch_wave6_verifier


def vessel(**values):
    spec = dict(subkind="pressure_vessel", r_i=1, r_o=2, p_i=3e6, p_o=1e6)
    return dispatch_wave6_verifier("solid_mohr_vessel_buckling", dict(spec, **values))


class LameAnnularState(unittest.TestCase):
    def test_every_supplied_wall_dimension_belongs_to_one_annulus(self):
        for thickness in (-1, 0, .02, 1.01):
            with self.subTest(thickness=thickness):
                r = vessel(t=thickness)
                self.assertFalse(r["verified"], r)
        self.assertTrue(vessel(t=1)["verified"])

    def test_closed_end_equivalent_stress_survives_hydrostatic_shift(self):
        ri, ro = .001, 1000.
        geometry = Fraction(ri)**2 / (Fraction(ro)**2-Fraction(ri)**2)
        for pressure in (0, 100000):
            for excess in (-1, 1):
                with self.subTest(pressure=pressure, excess=excess):
                    r = vessel(r_i=ri, r_o=ro, r_eval=ro, p_i=pressure+excess,
                               p_o=pressure, closed_ends=True)
                    self.assertTrue(r["verified"], r)
                    expected = math.sqrt(3)*float(geometry)
                    self.assertTrue(math.isclose(r["sigma_von_mises"], expected,
                                                 rel_tol=3e-15), r)
                    if pressure:
                        self.assertEqual(r["sigma_radial_lame"], -pressure)
                        self.assertEqual(r["sigma_hoop_lame"], -pressure)
                        self.assertEqual(r["sigma_axial_lame"], -pressure)

    def test_joint_pressures_and_ends_have_distinct_equivalent_stress(self):
        # Identical pressure on both faces is a hydrostatic loading only when
        # the same pressure resultant acts on closed end caps.
        for closed, expected in ((True,0.), (False,100000.)):
            with self.subTest(closed=closed):
                result = vessel(p_i=100000, p_o=100000, closed_ends=closed, t=1)
                self.assertTrue(result["verified"], result)
                self.assertEqual(result["sigma_von_mises"], expected)

    def test_independent_end_area_and_diametral_force_balances(self):
        ri,ro=2,5
        for pi,po in ((5000,1000),(1000,5000),(-30,10),(350,350)):
            for closed in (False,True):
                for radius in (ri,3,ro):
                    with self.subTest(pi=pi,po=po,closed=closed,r=radius):
                        out=vessel(r_i=ri,r_o=ro,t=ro-ri,r_eval=radius,
                                   p_i=pi,p_o=po,closed_ends=closed)
                        self.assertTrue(out["verified"],out)
                        a=Fraction(pi*ri**2-po*ro**2,ro**2-ri**2)
                        b=Fraction((pi-po)*ri**2*ro**2,ro**2-ri**2)
                        radial,hoop,axial=a-b/radius**2,a+b/radius**2,a if closed else 0
                        mean=(radial+hoop+axial)/3
                        for name,exact in (("sigma_radial_lame",radial),("sigma_hoop_lame",hoop),
                                          ("sigma_axial_lame",axial)):
                            self.assertEqual(out[name],float(exact))
                        for name,exact in (("radial_hoop_amplitude",b/radius**2),
                                           ("uniform_in_plane",a),("mean_normal_stress",mean),
                                           ("deviatoric_radial",radial-mean),
                                           ("deviatoric_hoop",hoop-mean),
                                           ("deviatoric_axial",axial-mean)):
                            self.assertEqual(out["stress_split"][name],float(exact))
                        self.assertEqual(a*(ro-ri)+b*(Fraction(1,ri)-Fraction(1,ro)),pi*ri-po*ro)

    def test_integer_wall_and_float_measurements_and_sampling(self):
        # An exact integer unit of wall thickness remains distinct at this datum.
        datum=2**54
        for spec in (dict(r_i=datum,r_o=datum+1,t=1),
                     dict(r_i=datum,r_o=None,t=1),
                     dict(r_i=float(datum),r_o=float(datum),t=1)):
            out=vessel(p_i=100,p_o=99,**spec)
            self.assertTrue(out["verified"],out)
            self.assertEqual(out["wall_thickness"],1.)
        for spec in (dict(r_i=.1,r_o=.3,t=.2),dict(r_i=.1,r_o=.3,t=.2,r_eval=.3)):
            self.assertTrue(vessel(**spec)["verified"])
        self.assertFalse(vessel(r_i=datum,r_o=datum,t=1)["verified"])
        self.assertFalse(vessel(r_eval=math.nextafter(2., math.inf))["verified"])
        self.assertFalse(vessel(closed_ends="false")["verified"])

    def test_public_split_recovers_small_signed_outer_face_deviation(self):
        for excess in (-1,1):
            out=vessel(r_i=1,r_o=2**20,r_eval=2**20,p_o=10**6,p_i=10**6+excess)
            split=out["stress_split"]
            shear=Fraction(excess,2**40-1)
            self.assertEqual(split["radial_hoop_amplitude"],float(shear))
            self.assertEqual(split["deviatoric_hoop"],float(shear))
            self.assertEqual(split["deviatoric_radial"],-float(shear))
            self.assertEqual(split["deviatoric_axial"],0.)
            self.assertEqual(out["sigma_hoop_lame"],out["sigma_radial_lame"])
            self.assertGreater(out["sigma_von_mises"],0.)
