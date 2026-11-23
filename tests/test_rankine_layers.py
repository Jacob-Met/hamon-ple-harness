"""Public FHWA NHI-06-089 Appendix A.5 pressure diagrams, in coherent SI units."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route

FT = 0.3048
LBF = 4.4482216152605
PCF = LBF / FT**3
KSF = 1000 * LBF / FT**2
KIP_PER_FT = 1000 * LBF / FT


def layer(feet, pcf, phi, submerged=None):
    fields = {"thickness_m": feet*FT, "gamma_above_water_N_m3": pcf*PCF, "phi_deg": phi}
    if submerged is not None:
        fields["gamma_submerged_N_m3"] = submerged*PCF
    return fields


def run(state, layers, **fields):
    return route({"spec": {"calculation": "rankine_layers", "pressure_state": state,
                           "layers": layers, "gamma_water_N_m3": 62.4*PCF, **fields}})


class TestRankineLayers(unittest.TestCase):
    def assert_water(self, response, depth):
        water = 0.5*62.4*PCF*depth**2
        self.assertAlmostEqual(response["hydrostatic_thrust_N_m"], water, places=7)
        self.assertAlmostEqual(response["hydrostatic_moment_Nm_m"], water*depth/3, places=7)
        self.assertAlmostEqual(response["thrust_N_m"], response["effective_thrust_N_m"] + water, places=7)
        self.assertLess(response["dual_solver_rel_residual"], 1e-6)
        self.assertTrue(response["conservation_verified"])

    def test_fhwa_active_effective_pressure_jump_and_force(self):
        r=run("active", [layer(33,130,40),layer(7,110,36,50)], water_table_depth_m=35*FT)
        p=r["profile"]
        self.assertEqual(len(p),3)
        points = [p[0]["effective_bottom_Pa"],p[1]["effective_top_Pa"],p[1]["effective_bottom_Pa"],p[2]["effective_bottom_Pa"]]
        for actual, printed in zip(points,[.94,1.11,1.17,1.24]):
            self.assertAlmostEqual(actual/KSF,printed,delta=.035)
        self.assertAlmostEqual(r["effective_thrust_N_m"]/KIP_PER_FT,24,delta=.65)
        self.assert_water(r,5*FT)

    def test_fhwa_passive_effective_pressure_and_force(self):
        r=run("passive",[layer(10,110,36,50)],water_table_depth_m=5*FT)
        for section,printed in zip(r["profile"],[2.1,3.1]):
            self.assertAlmostEqual(section["effective_bottom_Pa"]/KSF,printed,delta=.1)
        self.assertAlmostEqual(r["effective_thrust_N_m"]/KIP_PER_FT,18,delta=.65)
        self.assert_water(r,5*FT)

    def test_dry_surcharge_and_zero_friction(self):
        for state in ("active", "passive"):
            r=run(state,[layer(10,120,0)],surcharge_Pa=2300)
            h=10*FT
            self.assert_water(r,0)
            self.assertAlmostEqual(r["thrust_N_m"],2300*h+120*PCF*h*h/2)
            self.assertAlmostEqual(r["moment_Nm_m"],2300*h*h/2+120*PCF*h**3/6)

    def test_submerged_unit_weight_forms_and_independent_pressure_diagram(self):
        water, height, saturated = 9810., 6., 19810.
        submerged = saturated - water
        for state, coeff in (("active", 1/3), ("passive", 3)):
            for weights in ({"gamma_submerged_N_m3": submerged},
                            {"gamma_saturated_N_m3": saturated},
                            {"gamma_saturated_N_m3": saturated, "gamma_submerged_N_m3": submerged}):
                with self.subTest(state=state, weights=weights):
                    row = {"thickness_m": height, "phi_deg": 30, **weights}
                    r = run(state, [row], water_table_depth_m=0, gamma_water_N_m3=water)
                    self.assertTrue(r["conservation_verified"], r)
                    gradient = coeff * submerged + water
                    self.assertAlmostEqual(r["thrust_N_m"], gradient * height**2 / 2)
                    self.assertAlmostEqual(r["moment_Nm_m"], gradient * height**3 / 6)

    def test_water_table_reconciles_only_the_occupied_phases(self):
        dry = {"thickness_m": 2, "phi_deg": 0, "gamma_above_water_N_m3": 16000}
        wet = {"thickness_m": 3, "phi_deg": 0, "gamma_saturated_N_m3": 19000}
        r = run("active", [dry, wet], water_table_depth_m=2, gamma_water_N_m3=10000)
        self.assertTrue(r["conservation_verified"], r)
        self.assertEqual(r["profile"][1]["effective_bottom_Pa"], 2*16000 + 3*9000)
        self.assertEqual(r["profile"][1]["water_bottom_Pa"], 30000)
        for table, weights in ((0, {"gamma_submerged_N_m3": 14000}), (0, {"gamma_saturated_N_m3": 10000}),
                               (0, {"gamma_saturated_N_m3": True}), (1, {})):
            row = {**wet, **weights}
            with self.subTest(table=table, row=row):
                self.assertFalse(run("active", [row], water_table_depth_m=table,
                                     gamma_water_N_m3=10000)["conservation_verified"])

    def test_two_water_faces_cross_material_and_water_boundaries(self):
        rows = [{"thickness_m": 2, "phi_deg": 30, "gamma_above_water_N_m3": 18000,
                 "gamma_saturated_N_m3": 20000},
                {"thickness_m": 4, "phi_deg": 30, "gamma_above_water_N_m3": 18000,
                 "gamma_saturated_N_m3": 20000}]
        for state, k in (("active", 1/3), ("passive", 3)):
            for rear in (0, 1, 2, 4, 6, 8):
                for front in (0, 1, 2, 4, 6, 8):
                    with self.subTest(state=state, rear=rear, front=front):
                        h, gw = 6, 10000
                        # Superpose a dry soil triangle, submerged slope correction,
                        # and the opposed hydrostatic triangles about the wall base.
                        wet, flooded = max(0, h-rear), max(0, h-front)
                        effective_f = k * (18000*h*h - 8000*wet**2) / 2
                        effective_m = k * (18000*h**3 - 8000*wet**3) / 6
                        net_f = gw * (wet**2 - flooded**2) / 2
                        net_m = gw * (wet**3 - flooded**3) / 6
                        r = run(state, rows, water_table_depth_m=rear,
                                front_water_table_depth_m=front, gamma_water_N_m3=gw)
                        self.assertTrue(r["conservation_verified"], r)
                        self.assertAlmostEqual(r["thrust_N_m"], effective_f + net_f)
                        self.assertAlmostEqual(r["moment_Nm_m"], effective_m + net_m)
                        self.assertAlmostEqual(r["hydrostatic_thrust_N_m"], net_f)
                        self.assertAlmostEqual(r["hydrostatic_moment_Nm_m"], net_m)
                        self.assertEqual(r["profile"][-1]["net_water_bottom_Pa"], gw*(wet-flooded))

    def test_opposed_water_can_reverse_the_resultant_or_leave_a_pure_couple(self):
        soil = [{"thickness_m": 4, "phi_deg": 0, "gamma_above_water_N_m3": 2500}]
        def solve(front, **extra):
            return run("active", soil, front_water_table_depth_m=front,
                       gamma_water_N_m3=10000, **extra)
        reversed_load = solve(0)
        self.assertTrue(reversed_load["conservation_verified"])
        self.assertEqual(reversed_load["thrust_N_m"], -60000)
        self.assertAlmostEqual(reversed_load["moment_Nm_m"], -80000)
        self.assertAlmostEqual(reversed_load["resultant_above_base_m"], 4/3)
        self.assertFalse(solve(0, proposed_thrust_N_m=60000)["conservation_verified"])
        couple = solve(2)
        self.assertTrue(couple["conservation_verified"])
        self.assertEqual(couple["thrust_N_m"], 0)
        self.assertAlmostEqual(couple["moment_Nm_m"], 40000/3)
        self.assertIsNone(couple["resultant_above_base_m"])
        self.assertFalse(solve(-1)["conservation_verified"])
        self.assertFalse(solve(0, wall_acceleration_horizontal_g=0.15)["conservation_verified"])
        self.assertFalse(run("active", [dict(soil[0], pore_pressure_coefficient=0.5)])["conservation_verified"])

    def test_out_of_scope_or_false_claims(self):
        for kw in [{"backfill_slope_deg":2},{"wall_friction_deg":1},{"surcharge_Pa":-10},
                   {"water_table_depth_m":0},{"proposed_thrust_N_m":0}]:
            self.assertFalse(run("active",[layer(1,120,30)],**kw)["conservation_verified"])
        for invalid in (layer(1,120,90),layer(-1,120,30),dict(layer(1,120,30),cohesion_Pa=900)):
            self.assertFalse(run("passive",[invalid])["conservation_verified"])


    def test_opposing_dry_earth_force_and_moment_about_common_toe(self):
        def dry(h, phi=30):
            return {"thickness_m": h, "phi_deg": phi, "gamma_above_water_N_m3": 18000}
        for h in (1.9, 2., 2.1, 2.0000001):
            with self.subTest(h=h):
                # Independent centroids of rear active and front passive triangles.
                front_force = 27000*h**2
                force = 108000 - front_force
                moment = 216000 - front_force*h/3
                r = run("active", [dry(6)], front_layers=[dry(h)], front_surface_depth_m=6-h)
                self.assertTrue(r["conservation_verified"], r)
                self.assertEqual(r["front_pressure_state"], "passive")
                self.assertAlmostEqual(r["rear_effective_thrust_N_m"], 108000)
                self.assertAlmostEqual(r["front_effective_thrust_N_m"], front_force)
                self.assertAlmostEqual(r["thrust_N_m"], force)
                self.assertAlmostEqual(r["moment_Nm_m"], moment)
                if h == 2:
                    self.assertEqual(r["resultant_kind"], "pure_couple")
                    self.assertIsNone(r["resultant_above_base_m"])
                    self.assertFalse(run("active", [dry(6)], front_layers=[dry(h)],
                                         proposed_thrust_N_m=0, proposed_moment_Nm_m=0)["conservation_verified"])
                else:
                    self.assertEqual(r["resultant_kind"], "force")
                    self.assertAlmostEqual(r["resultant_above_base_m"]/(moment/force), 1., places=8)
                    self.assertTrue(r["resultant_above_base_m"] > 6 if h < 2 else r["resultant_above_base_m"] < 0)
        matched = run("passive", [dry(2, 0)], front_layers=[dry(2, 0)])
        self.assertTrue(matched["conservation_verified"])
        self.assertEqual((matched["thrust_N_m"], matched["moment_Nm_m"]), (0., 0.))
        self.assertEqual(matched["resultant_kind"], "balanced")
        self.assertIsNone(matched["resultant_above_base_m"])
        reversed_state = run("passive", [dry(6)], front_layers=[dry(2)])
        self.assertTrue(reversed_state["conservation_verified"])
        self.assertEqual(reversed_state["front_pressure_state"], "active")
        self.assertAlmostEqual(reversed_state["thrust_N_m"], 27000*36 - 3000*4)

    def test_layered_faces_referenced_to_distinct_grades_and_common_water_datum(self):
        def material(h, phi, submerged):
            return {"thickness_m": h, "phi_deg": phi, "gamma_above_water_N_m3": 18000,
                    "gamma_submerged_N_m3": submerged}
        def diagram(rows):
            # Rectangle plus triangle and their geometric centroids above the toe.
            f = m = 0.
            for top, bottom, p, q in rows:
                rectangle = (bottom-top)*p
                triangle = (bottom-top)*(q-p)/2
                f += rectangle + triangle
                m += rectangle*(6-(top+bottom)/2) + triangle*(6-(top+2*bottom)/3)
            return f, m

        back = [material(2, 0, 10000), material(4, 30, 10000)]
        front = [material(1, 30, 8000), material(2, 0, 8000)]
        rear_force, rear_moment = diagram([(0, 1, 2000, 20000), (1, 2, 20000, 30000),
                                          (2, 6, 10000, 70000/3)])
        for front_table, front_diagram in (
            (2, [(3, 4, 3000, 27000), (4, 6, 9000, 25000)]),
            (4.5, [(3, 4, 3000, 57000), (4, 4.5, 19000, 28000),
                   (4.5, 6, 28000, 40000)]),
        ):
            with self.subTest(front_table=front_table):
                exposed_force, exposed_moment = diagram(front_diagram)
                water_f = 5000*(5**2-(6-front_table)**2)
                water_m = 10000*(5**3-(6-front_table)**3)/6
                r = run("active", back, front_layers=front, front_surface_depth_m=3,
                        water_table_depth_m=1, front_water_table_depth_m=front_table,
                        surcharge_Pa=2000, front_surcharge_Pa=1000, gamma_water_N_m3=10000)
                self.assertTrue(r["conservation_verified"], r)
                self.assertEqual(r["front_profile"][0]["top_m"], 3.)
                self.assertEqual(r["front_profile"][0]["water_top_Pa"], max(0,3-front_table)*10000)
                self.assertAlmostEqual(r["thrust_N_m"], rear_force - exposed_force + water_f)
                self.assertAlmostEqual(r["moment_Nm_m"], rear_moment - exposed_moment + water_m)
                self.assertAlmostEqual(r["front_effective_moment_Nm_m"], exposed_moment)

    def test_front_grade_and_material_validation(self):
        row = {"thickness_m": 6, "phi_deg": 30, "gamma_above_water_N_m3": 18000}
        for extra in ({"front_surcharge_Pa": 5}, {"front_layers": []}, {"front_layers": [None]},
                      {"front_layers": [dict(row, thickness_m=7)]},
                      {"front_layers": [row], "front_surface_depth_m": 1},
                      {"front_layers": [dict(row, cohesion_Pa=10)]}):
            with self.subTest(extra=extra):
                self.assertFalse(run("active", [row], **extra)["conservation_verified"])


if __name__ == "__main__":
    unittest.main()
