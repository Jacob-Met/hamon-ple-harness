import math
import unittest
from decimal import Decimal,localcontext

from hamon_dual_verifier import run_dual_verify_spec


def solve(**changes):
    spec = dict(b="300*mm",d="500*mm",As="6000*mm2",fc="30*MPa",fy="420*MPa")
    return run_dual_verify_spec(dict(mode="domain",kind="rc_beam_aci318",spec=dict(spec,**changes)))


def reference(b,d,area,fc,fy,es):
    """Independently bisect net longitudinal force using the plane strain diagram."""
    with localcontext() as ctx:
        ctx.prec=120
        b,d,area,fc,fy,es=map(Decimal.from_float,(b,d,area,fc,fy,es))
        beta=max(Decimal(".65"),min(Decimal(".85"),Decimal(".85")-(fc-28000000)/140000000))
        low,high=Decimal(0),d
        for _ in range(325):
            c=(low+high)/2
            strain=Decimal(".003")*(d/c-1)
            fs=min(fy,es*strain)
            a=beta*c
            C=Decimal(".85")*fc*b*a
            T=area*fs
            if C>T:
                high=c
            else:
                low=c
        phi=max(Decimal(".65"),min(Decimal(".9"),Decimal(".65")+Decimal(".25")*(strain-fy/es)/Decimal(".003")))
        Mn=T*d-Decimal(".85")*fc*b*a*a/2
        return [float(x) for x in (c,a,strain,fs,T,C,phi,Mn,phi*Mn)]


class RCSectionTests(unittest.TestCase):
    def test_yielded_and_elastic_sections_replay_force_and_moment(self):
        for area in (2000,6000,9000):
            r=solve(As=f"{area}*mm2")
            self.assertTrue(r["dual_verified"],r)
            self.assertIs(r["steel_yielded"],area==2000)
            self.assertGreater(r["tensile_strain"],0)
            ref=reference(.3,.5,area*1e-6,30e6,420e6,200e9)
            keys=("neutral_axis_c_m","whitney_a_m","tensile_strain","steel_stress_Pa",
                  "steel_tension_N","concrete_compression_N","phi","Mn_Nm","phi_Mn_Nm")
            for key,expected in zip(keys,ref):
                self.assertAlmostEqual(r[key]/expected,1.,places=13)

    def test_exact_yield_and_tension_controlled_neighbors(self):
        # Both balanced steel areas below are exactly 289/32768 m² for their
        # respective width. The adjacent wire areas lie on opposite branches.
        a0=289/32768
        for width,classes,yielded in [
            (45/128,("transition","compression_controlled","compression_controlled"),(True,True,False)),
            (75/128,("tension_controlled","tension_controlled","transition"),(True,True,True)),
        ]:
            for k,area in enumerate([math.nextafter(a0,0),a0,math.nextafter(a0,math.inf)]):
                r=solve(b=width,d=5/8,As=area,fc=25e6,fy=300e6,Es=200e9)
                self.assertTrue(r["dual_verified"])
                self.assertEqual(r["strain_classification"],classes[k])
                self.assertIs(r["steel_yielded"],yielded[k])

    def test_material_grade_and_geometric_similarity(self):
        r=solve(b=.25,d=.5,As=.005,fc=40e6,fy=505e6,Es=175e9)
        self.assertTrue(r["dual_verified"])
        for n in (-15,12):
            s=2.**n
            z=solve(b=.25*s,d=.5*s,As=.005*s*s,fc="40*MPa",fy="505*MPa",Es="175*GPa")
            self.assertTrue(z["dual_verified"])
            self.assertEqual(z["steel_yielded"],r["steel_yielded"])
            self.assertEqual(z["phi_Mn_Nm"],r["phi_Mn_Nm"]*s**3)
            self.assertEqual(z["steel_tension_N"],r["steel_tension_N"]*s**2)
        self.assertAlmostEqual(r["steel_yield_strain"],505e6/175e9)

    def test_section_inputs_and_scope(self):
        r=solve()
        self.assertEqual(r["analysis_scope"],"rectangular_section_pure_flexure")
        self.assertIn("21.2.2",r["strength_factor_basis"])
        for changes in [dict(As="4*m"),dict(d=0),dict(Es=-1),dict(fc="12*MPa")]:
            with self.assertRaises(ValueError):
                solve(**changes)
