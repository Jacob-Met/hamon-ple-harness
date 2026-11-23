import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def rotor(inlet, outlet, **options):
    spec = dict(calculation="euler_rotor", machine_kind="compressor", rotation_rpm=120,
                inlet=dict(radius_m=.5, meridional_velocity_m_s=5,
                           absolute_tangential_velocity_m_s=0, **inlet),
                outlet=dict(radius_m=.5, meridional_velocity_m_s=10,
                            absolute_tangential_velocity_m_s=4, **outlet),
                mass_flow_kg_s=500, measured_fluid_torque_N_m=1000)
    return run_router_conservation_cad_verifier(
        {"mode":"thermal_fluid", "spec":dict(spec, **options)})


class EulerRotorStreamContinuity(unittest.TestCase):
    def test_measured_torque_cannot_certify_unequal_steady_throughputs(self):
        inlet = dict(density_kg_m3=1000, flow_area_m2=.1)
        matched = rotor(inlet, dict(density_kg_m3=500,flow_area_m2=.1))
        self.assertTrue(matched["conservation_verified"], matched)
        # 500 kg/s enters, but the supplied outlet passes 1000 kg/s.
        wrong = rotor(inlet, dict(density_kg_m3=1000,flow_area_m2=.1))
        self.assertFalse(wrong["conservation_verified"], wrong)

    def test_agreeing_stations_cannot_discard_conflicting_total_mass_flow(self):
        # Both stations now carry 250 kg/s. The supplied 500 kg/s and torque
        # still match angular momentum if continuity is silently omitted.
        wrong = rotor(dict(density_kg_m3=500,flow_area_m2=.1),
                      dict(density_kg_m3=250,flow_area_m2=.1))
        self.assertFalse(wrong["conservation_verified"], wrong)

    def test_partial_or_nonpositive_flow_section_has_no_admitted_mass(self):
        for section in (dict(flow_area_m2=.1),
                        dict(density_kg_m3=1000,flow_area_m2=0),
                        dict(density_kg_m3=-1000,flow_area_m2=.1)):
            with self.subTest(section=section):
                wrong = rotor(section, {})
                self.assertFalse(wrong["conservation_verified"], wrong)
