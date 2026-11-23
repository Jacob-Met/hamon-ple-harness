"""Independent complex pressure/normal-velocity boundary-value solution."""
import math
import unittest

from mpmath import mp
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(a, b, rho_a, rho_b, field, value):
    spec=dict(subkind="acoustics",c1=a,c2=b,rho1=rho_a,rho2=rho_b)
    spec[field]=value
    return run_router_conservation_cad_verifier(dict(mode="compressible_acoustics",spec=spec))


class AcousticBoundaryMatrix(unittest.TestCase):
    def test_pressure_velocity_and_outgoing_partial_waves(self):
        with mp.workdps(130):
            contact=.375
            samples=[(3,8,1.25,3.5,"incidence_sine",x)
                     for x in (0,.125,-.125,math.nextafter(contact,0.),contact,
                               math.nextafter(contact,1.),-.5,.75)]
            samples += [(8,3,5,2,"incidence_angle_deg",x) for x in (-80.,-11.,0.,37.,80.)]
            # At 45 degrees for these two dissimilar fluids, normal slowness
            # increases by a factor seven and density increases by the same factor.
            samples += [(5,1,1,7,"incidence_angle_deg",x) for x in
                        (math.nextafter(45.,0.),45.,math.nextafter(45.,90.))]
            for a,b,ra,rb,field,value in samples:
                with self.subTest(a=a,b=b,field=field,value=value):
                    out=solve(a,b,ra,rb,field,value)
                    self.assertTrue(out["verified"],out)
                    sine=mp.sinpi(mp.mpf(value)/180) if field=="incidence_angle_deg" else mp.mpf(value)
                    tangent=sine/a
                    ka=mp.sqrt(1/mp.mpf(a)**2-tangent**2)
                    kb=mp.sqrt(1/mp.mpf(b)**2-tangent**2)
                    ya,yb=ka/ra,kb/rb
                    r,t=mp.lu_solve([[1,-1],[ya,yb]],[-1,ya])
                    if (a,b,value)==(5,1,45.):
                        r,t=mp.mpf(0),mp.mpf(1)
                    u=yb*t/ya
                    reflected=mp.mpc(*out["pressure_reflection_complex"])
                    transmitted=mp.mpc(*out["pressure_transmission_complex"])
                    velocity=mp.mpc(*out["transmitted_normal_velocity_ratio"])
                    normal=mp.mpc(*out["transmitted_normal_slowness_s_m"])
                    for got,exact in ((reflected,r),(transmitted,t),(velocity,u),(normal,kb),
                                      (mp.mpf(out["tangential_slowness_s_m"]),tangent),
                                      (mp.mpf(out["incident_normal_slowness_s_m"]),ka)):
                        for component,expected in ((mp.re(got),mp.re(exact)),(mp.im(got),mp.im(exact))):
                            if expected:
                                self.assertLess(abs(component/expected-1),mp.mpf("3e-14"))
                            else:
                                self.assertEqual(component,0)
                    net_incident=mp.re((1+reflected)*mp.conj(1-reflected))
                    terminal=mp.re(transmitted*mp.conj(velocity))
                    self.assertLess(abs(net_incident-terminal),mp.mpf("8e-16"))
                    transmitted_flux=out["intensity_transmission_TI"]
                    self.assertLess(abs(transmitted_flux-terminal),mp.mpf("8e-16"))
                    if mp.im(kb):
                        self.assertGreater(normal.imag,0)
                        self.assertLess(abs(mp.exp(mp.j*normal)),1)
                        self.assertEqual(transmitted_flux,0)
                    elif not kb:
                        self.assertEqual(out["regime"],"critical")
                        self.assertEqual(transmitted_flux,0)
                    else:
                        self.assertGreater(normal.real,0)
                        self.assertGreater(transmitted_flux,0)

    def test_grazing_has_no_nonzero_incident_normal_flux(self):
        for field,edge in (("incidence_angle_deg",90),("incidence_sine",1)):
            for s in (-1,1):
                with self.subTest(field=field,s=s):
                    self.assertFalse(solve(8,3,5,2,field,s*edge)["verified"])
                    near=solve(8,3,5,2,field,math.nextafter(float(s*edge),0.))
                    self.assertTrue(near["verified"],near)
                    self.assertGreater(near["intensity_transmission_TI"],0)
