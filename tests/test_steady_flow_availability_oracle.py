"""Independent mechanical availability and entropy checks on the public steady-flow route."""
from fractions import Fraction
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def stream(m, h, s, v=0, z=0):
    return dict(mdot_kg_s=m, h_J_kg=h, s_J_kgK=s, speed_m_s=v, elevation_m=z)


def verify(inlets, outlets, heat, work, **overrides):
    spec = dict(calculation="steady_flow_control_volume", property_reference="common-table",
                gravity_m_s2=10, inlets=inlets, outlets=outlets,
                heat_reservoirs=[dict(heat_into_W=q, temperature_K=t) for q,t in heat],
                shaft_work_out_W=work)
    return run_router_conservation_cad_verifier({"mode":"thermal_fluid", "spec":dict(spec,**overrides)})


class SteadyFlowAvailabilityOracle(unittest.TestCase):
    def test_reservoir_work_and_stream_availability_at_two_reference_temperatures(self):
        # The separately derived shaft-work ceiling uses h - T_ref*s, along
        # with the Carnot fraction of every boundary heat transfer.
        def flux(rows, ambient):
            return sum(r["mdot_kg_s"]*(r["h_J_kg"]-ambient*r["s_J_kgK"]+
                       Fraction(r["speed_m_s"]**2,2)+10*r["elevation_m"]) for r in rows)
        for dh, ds, dz in ((0,0,0),(10**95,-10**80,10**72)):
            inlet = [stream(1,dh+2000,ds+4,20,dz+1),stream(2,dh+1500,ds+1,5,dz)]
            for out_entropy in (1,2,3):
                outlet = [stream(3,dh+1400,ds+out_entropy,10,dz+2)]
                for heat in ([(750,750),(-250,250)], [(750,250),(-250,750)]):
                    energy_delta = flux(outlet,0)-flux(inlet,0)
                    work = sum(q for q,t in heat)-energy_delta
                    self.assertEqual(work.denominator,1)
                    ans = verify(inlet,outlet,heat,int(work))
                    self.assertTrue(ans["first_law_ok"],ans)
                    self.assertTrue(ans["mass_balance_ok"],ans)
                    for ambient in (250,425):
                        availability = sum(q*(1-Fraction(ambient,t)) for q,t in heat)
                        max_work = availability - (flux(outlet,ambient)-flux(inlet,ambient))
                        generated = (max_work-work)/ambient
                        with self.subTest(dh=dh,s=out_entropy,heat=heat,ambient=ambient):
                            self.assertEqual(ans["entropy_generation_W_K"],float(generated))
                            self.assertEqual(ans["thermodynamically_feasible"],work<=max_work)
                    wrong = verify(inlet,outlet,heat,int(work)+2)
                    self.assertFalse(wrong["first_law_ok"])
                    self.assertFalse(wrong["conservation_verified"])
                    self.assertEqual(wrong["energy_residual_W"],-2)

    def test_turbine_and_nozzle_and_common_material_tag(self):
        # Inlet and outlet specific enthalpy difference equals either shaft
        # work per kg or the increase of kinetic energy per kg.
        for h,v,work in ((1600,0,800),(1600,20,400),(1200,40,0)):
            ins,out=[stream(2,2000,-30)], [stream(2,h,-30,v)]
            ans=verify(ins,out,[],work,entropy_generation_W_K=0)
            self.assertTrue(ans["thermodynamically_feasible"],ans)
            self.assertEqual(ans["entropy_generation_W_K"],0)
            out[0]["property_reference"]="unrelated-datum"
            self.assertFalse(verify(ins,out,[],work)["conservation_verified"])
