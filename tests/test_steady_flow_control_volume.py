import unittest
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def flow(inlets,outlets,**options):
    spec=dict(calculation="steady_flow_control_volume",property_reference="documented common basis",
              inlets=inlets,outlets=outlets,heat_reservoirs=[],shaft_work_out_W=0)
    return run_router_conservation_cad_verifier({"mode":"thermal_fluid","spec":dict(spec,**options)})


def stream(m,h,s,**extra):
    return dict(mdot_kg_s=m,h_J_kg=h,s_J_kgK=s,**extra)


class SteadyFlowThermodynamics(unittest.TestCase):
    def test_adiabatic_turbine_needs_nonnegative_entropy_and_first_law(self):
        inlet=[stream(1,1000000,4000)]
        valid=flow(inlet,[stream(1,800000,4000)],shaft_work_out_W=200000)
        self.assertTrue(valid["conservation_verified"],valid)
        for h,s,w in ((800000,3999,200000),(800000,4000,250000)):
            with self.subTest(h=h,s=s,w=w):
                r=flow(inlet,[stream(1,h,s)],shaft_work_out_W=w)
                self.assertFalse(r["conservation_verified"],r)

    def test_conflicting_property_basis_cannot_be_certified_by_mass_only(self):
        r=flow([stream(1,0,0)],[stream(1,0,0,property_reference="unrelated material table")])
        self.assertFalse(r["conservation_verified"],r)
