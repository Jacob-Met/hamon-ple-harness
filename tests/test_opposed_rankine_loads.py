"""Signed resultants of subdivided and excavated Rankine pressure diagrams."""
import math
import unittest
from fractions import Fraction

from hamon_rankine import verify_rankine_layers


def column(widths, dry, wet, phi=0):
    return [{"thickness_m": width, "gamma_above_water_N_m3": dry,
             "gamma_submerged_N_m3": wet, "phi_deg": phi} for width in widths]


class OpposedRankineLoads(unittest.TestCase):
    def test_identical_and_nearly_identical_pressure_diagrams(self):
        for scale in (1, 64):
            for widths in ((.25, .75, 2), (.125, .625, .75, 1.5)):
                for direction in (-1, 0, 1):
                    surcharge = 1050. * scale
                    front = (math.nextafter(surcharge, -math.inf if direction < 0 else math.inf)
                             if direction else surcharge)
                    height = 3 * scale
                    spec = {"pressure_state": "active", "layers": column([height], 18000.1, 9500.25),
                            "front_layers": column([w*scale for w in widths], 18000.1, 9500.25),
                            "water_table_depth_m": .375*scale, "front_water_table_depth_m": .375*scale,
                            "surcharge_Pa": surcharge, "front_surcharge_Pa": front}
                    # Overburden and water cancel identically at each depth,
                    # leaving a uniform pressure block from the surcharge.
                    pressure = Fraction(surcharge) - Fraction(front)
                    force, moment = float(pressure*height), float(pressure*height**2/2)
                    with self.subTest(scale=scale, widths=widths, direction=direction):
                        out = verify_rankine_layers({**spec, "proposed_thrust_N_m": force,
                                                     "proposed_moment_Nm_m": moment})
                        self.assertTrue(out["conservation_verified"], out)
                        self.assertEqual((out["effective_thrust_N_m"], out["thrust_N_m"],
                                          out["moment_Nm_m"]), (force, force, moment))
                        self.assertEqual(out["resultant_kind"], "balanced" if not direction
                                         else "force_unresolved_at_roundoff")

                        if scale == 64 and not direction and widths == (.25, .75, 2):
                            wrong = verify_rankine_layers({**spec, "proposed_moment_Nm_m": -1.9073486328125e-6})
                            self.assertFalse(wrong["conservation_verified"], wrong)

    def test_active_and_passive_coefficients_give_identical_wet_diagrams(self):
        # At 30 degrees, Ka=1/3 and Kp=3. A factor of nine in surcharge
        # and unit weights makes the lateral earth pressures identical.
        spec = {"pressure_state": "active", "layers": column((1, 3), 18000, 9000, 30),
                "front_layers": column((.25, 1.25, 2.5), 2000, 1000, 30),
                "water_table_depth_m": 1.125, "front_water_table_depth_m": 1.125,
                "surcharge_Pa": 1080., "front_surcharge_Pa": 120.}
        out = verify_rankine_layers(spec)
        self.assertTrue(out["conservation_verified"], out)
        self.assertEqual((out["thrust_N_m"], out["moment_Nm_m"], out["resultant_kind"]),
                         (0., 0., "balanced"))

    def test_excavated_face_cancels_only_below_its_own_grade(self):
        # Matching vertical stress at the excavated grade leaves only the
        # top metre of rear loading. Independently integrate the rectangles
        # and triangles through the water-table break, about the 3 m toe.
        height, water, grade = 3, Fraction(3, 8), 1
        p0 = 1050
        pw = p0 + 18000 * water
        pg = pw + 9000 * (grade-water)
        force = moment = Fraction(0)
        for top, bottom, a, b in ((0, water, p0, pw), (water, grade, pw, pg)):
            rectangle = (bottom-top)*a
            triangle = (bottom-top)*(b-a)/2
            force += rectangle + triangle
            moment += rectangle*(height-(top+bottom)/2) + triangle*(height-(top+2*bottom)/3)
        spec = {"pressure_state": "active", "layers": column((.25, .75, 2), 18000, 9000),
                "front_layers": column((.5, 1.5), 18000, 9000), "front_surface_depth_m": grade,
                "water_table_depth_m": float(water), "front_water_table_depth_m": float(water),
                "surcharge_Pa": p0, "front_surcharge_Pa": float(pg),
                "proposed_thrust_N_m": float(force), "proposed_moment_Nm_m": float(moment)}
        out = verify_rankine_layers(spec)
        self.assertTrue(out["conservation_verified"], out)
        self.assertTrue(math.isclose(out["thrust_N_m"], float(force), rel_tol=1e-14))
        self.assertTrue(math.isclose(out["moment_Nm_m"], float(moment), rel_tol=1e-14))

    def test_water_boundary_uses_the_admitted_dry_profile(self):
        # The exact binary64 values for .1 and .4 add to slightly more than
        # .5, but the public toe and water table coincide at the same depth.
        for table in (.5, 1.):
            layers = [{"thickness_m": h, "gamma_above_water_N_m3": 18000, "phi_deg": 0}
                      for h in (.1, .4)]
            out = verify_rankine_layers({"pressure_state": "active", "layers": layers,
                                        "front_layers": layers, "water_table_depth_m": table,
                                        "front_water_table_depth_m": table})
            self.assertTrue(out["conservation_verified"], out)
            self.assertEqual(out["resultant_kind"], "balanced")

    def test_unrepresentable_nonzero_load_has_no_balanced_certificate(self):
        rows = column([.25], 18000, 9000)
        for rear, front in ((math.ulp(0.), 0.), (0., math.ulp(0.))):
            out = verify_rankine_layers({"pressure_state": "active", "layers": rows,
                                        "front_layers": rows, "surcharge_Pa": rear,
                                        "front_surcharge_Pa": front})
            self.assertFalse(out["conservation_verified"], out)
            self.assertIn("below the representable range", out["reason"])
