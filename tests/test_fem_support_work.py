import unittest

from hamon_pde_control_psychro_fem import verify_fem_structure


def bar(displacement, load=0.0, temperature=0.0):
    return {
        "mode": "truss_2d", "nodes": [[0, 0], [2, 0]],
        "elements": [{"i": 0, "j": 1, "E_pa": 2e8, "A_m2": .01,
                      "alpha_per_K": 1e-5, "dT_K": temperature}],
        "loads": [{"node": 1, "fx_N": load}],
        "supports": [{"node": 0, "fix_x": True, "fix_y": True},
                     {"node": 1, "fix_x": True, "fix_y": True, "ux_m": displacement}],
    }


def beam(displacement, rotation=0.0):
    return {"mode": "euler_bernoulli_beam_1d", "x_nodes_m": [0, 2], "EI_Nm2": 4e6,
            "supports": [{"node": 0, "fix_w": True, "fix_theta": True},
                         {"node": 1, "fix_w": True, "w_m": displacement,
                          "fix_theta": True, "theta_rad": rotation}]}


class SupportWorkTests(unittest.TestCase):
    def test_bar_thermal_compatibility_and_support_work(self):
        for prescribed, load, temperature in ((.004, 0, 0), (.004, 3000, 0),
                (.004, 3000, 100), (0, 0, 100), (.002, 0, 100), (-.004, -3000, -100)):
            with self.subTest(u=prescribed, f=load, t=temperature):
                result = verify_fem_structure(bar(prescribed, load, temperature))
                force = 1e6 * (prescribed - 2e-5 * temperature)
                self.assertTrue(result["passed"])
                self.assertAlmostEqual(result["reactions_xy_N"][0][0], -force)
                self.assertAlmostEqual(result["reactions_xy_N"][1][0], force-load)
                self.assertAlmostEqual(result["axial_forces_N"][0], force)
                self.assertAlmostEqual(result["external_work_J"], .5*force*prescribed)
                self.assertAlmostEqual(result["support_work_J"], .5*(force-load)*prescribed)
                self.assertAlmostEqual(result["applied_load_work_J"], .5*load*prescribed)
                self.assertAlmostEqual(result["thermal_strain_work_J"], -force*1e-5*temperature)
                self.assertAlmostEqual(result["strain_energy_J"], force**2/2e6)

    def test_beam_settlement_and_rotation_against_cubic_curvature(self):
        for delta, theta in ((-.01,0),(.003,.009),(-.003,-.009),(0,.005)):
            with self.subTest(delta=delta, theta=theta):
                result = verify_fem_structure(beam(delta,theta))
                # Unloaded span: u(x) = a*x**3 + b*x**2, fixed at x=0.
                length, ei = 2.0, 4e6
                a = (theta*length-2*delta)/length**3
                b = (3*delta-theta*length)/length**2
                end_force, end_moment = -6*ei*a, ei*(6*a*length+2*b)
                energy = ei*(6*a*a*length**3+6*a*b*length**2+2*b*b*length)
                self.assertTrue(result["passed"])
                self.assertAlmostEqual(result["reactions_V_M"][1][0], end_force)
                self.assertAlmostEqual(result["reactions_V_M"][1][1], end_moment)
                self.assertAlmostEqual(result["support_work_J"], (end_force*delta+end_moment*theta)/2)
                self.assertAlmostEqual(result["external_work_J"], energy)
                self.assertAlmostEqual(result["strain_energy_J"], energy)
                claim = {**beam(delta,theta),"claimed_field":"external_work_J","claimed_value":0}
                self.assertFalse(verify_fem_structure(claim)["passed"])

    def test_repeated_constraints_are_order_independent(self):
        for source,field in ((bar(.004),"ux_m"),(beam(.004),"w_m"),(beam(0,.004),"theta_rad")):
            original = source["supports"]
            for repeated in (original[-1], {**original[-1],field:.008}):
                for supports in (original+[repeated],list(reversed(original+[repeated]))):
                    result=verify_fem_structure({**source,"supports":supports})
                    self.assertEqual(result["passed"],repeated==original[-1])
                    if not result["passed"]:
                        self.assertIn("Conflicting support",result["reason"])

    def test_rigid_motion_and_unsupported_structures(self):
        translated = bar(.01)
        translated["supports"][0]["ux_m"] = .01
        rotated = beam(.2, .1)
        rotated["supports"][0]["theta_rad"] = .1
        for source in (translated, rotated):
            solved = verify_fem_structure(source)
            self.assertTrue(solved["passed"])
            self.assertAlmostEqual(solved["strain_energy_J"], 0)
            self.assertAlmostEqual(solved["external_work_J"], 0)
            unrestrained = verify_fem_structure({**source, "supports": []})
            self.assertFalse(unrestrained["passed"])
            self.assertNotIn("external_work_J", unrestrained)
