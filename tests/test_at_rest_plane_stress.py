"""FHWA-HRT-13-046, Chapter 7, figures 128–130 and closed-form controls."""
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route


FT = 0.3048
LBF = 4.4482216152605
PSF = LBF / FT**2
PCF = LBF / FT**3


def example(**changes):
    spec = dict(calculation="at_rest_plane_stress", coefficient_model="jaky",
                receiving_phi_deg=30.1, water_table_depth_m=3*FT, gamma_water_N_m3=62.4*PCF,
                overburden_layers=[dict(thickness_m=25*FT, gamma_above_water_N_m3=90*PCF,
                                        gamma_saturated_N_m3=90*PCF)])
    spec.update(changes)
    return spec


class AtRestPlaneStressTest(unittest.TestCase):
    def test_published_sand_stress_below_clay_overburden(self):
        # The receiving friction angle is the mobilized sand value. The weight
        # above that point is from the overlying clay, independent of its phi.
        for mode in ("auto", "civil_geotech"):
            with self.subTest(mode=mode):
                r = route({"mode": mode, "spec": example()})
                self.assertTrue(r["conservation_verified"])
                self.assertAlmostEqual(r["coefficient"], .499, delta=.001)
                self.assertAlmostEqual(r["vertical_effective_stress_Pa"] / PSF, 877, delta=.5)
                self.assertAlmostEqual(r["horizontal_effective_stress_Pa"] / PSF, 437, delta=.5)
                self.assertAlmostEqual(r["pore_pressure_Pa"] / PSF, 22*62.4)
                self.assertAlmostEqual(r["horizontal_total_stress_Pa"],
                                       r["horizontal_effective_stress_Pa"] + r["pore_pressure_Pa"])

    def test_wrong_pressure_state_or_inclusion_of_water_fails_claim(self):
        for stress_psf in (877/3, 877*3, 437 + 22*62.4, 0):
            with self.subTest(stress_psf=stress_psf):
                r = route({"mode": "civil_geotech", "spec": example(
                    proposed_horizontal_effective_stress_Pa=stress_psf*PSF)})
                self.assertFalse(r["conservation_verified"])

    def test_water_boundary_and_surcharge_at_separate_planes(self):
        layers = [{"thickness_m": 1.5, "gamma_above_water_N_m3": 18000},
                  {"thickness_m": 2.0, "gamma_saturated_N_m3": 20500}]
        r = route({"spec": example(receiving_phi_deg=0, overburden_layers=layers,
                                   water_table_depth_m=1.5, gamma_water_N_m3=10000,
                                   surcharge_Pa=2000, proposed_horizontal_effective_stress_Pa=50000)})
        self.assertTrue(r["conservation_verified"])
        self.assertAlmostEqual(r["vertical_total_stress_Pa"], 70000)
        self.assertAlmostEqual(r["horizontal_effective_stress_Pa"], 50000)
        for water, dry, wet in ((5., 2., 0.), (0., 0., 2.), (.75, .75, 1.25)):
            with self.subTest(water=water):
                r = route({"spec": example(receiving_phi_deg=30, water_table_depth_m=water,
                    gamma_water_N_m3=10000, overburden_layers=[dict(thickness_m=2.,
                    gamma_above_water_N_m3=17000, gamma_saturated_N_m3=20000)])})
                self.assertTrue(r["conservation_verified"])
                self.assertAlmostEqual(r["horizontal_effective_stress_Pa"],
                                       .5*(dry*17000 + wet*10000))

    def test_unsupported_and_empty_cases_refuse_certification(self):
        for spec in ({}, {"calculation": "unimplemented_plane",
                          "proposed_horizontal_effective_stress_Pa": -1},
                     {"calculation": "unimplemented_plane", "cut_bank_m3": 10,
                      "proposed_horizontal_effective_stress_Pa": -1},
                     example(coefficient_model="active"), example(receiving_phi_deg=90),
                     example(overburden_layers=[]), example(overconsolidation_ratio=2),
                     example(overburden_layers=[{"thickness_m": 2, "gamma_above_water_N_m3": 15000}])):
            with self.subTest(spec=spec):
                r = route({"mode": "civil_geotech", "spec": spec})
                self.assertFalse(r["ok"])
                self.assertFalse(r["conservation_verified"])

    def test_legacy_unnamed_civil_calculations_still_run(self):
        r = route({"mode": "civil", "spec": {"cut_bank_m3": 10, "fill_compacted_m3": 3}})
        self.assertTrue(r["conservation_verified"])
        self.assertEqual(r["net_compacted_surplus_m3"], 6.)
        r = route({"mode": "civil", "spec": {"channel_B_m": 2, "Q_m3_s": 1,
                                             "y1_m": 1, "y2_m": 1}})
        self.assertTrue(r["conservation_verified"])
