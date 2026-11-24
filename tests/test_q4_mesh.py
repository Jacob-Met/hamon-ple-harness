"""Full-integration Q4 controls: affine patch, center-blind mode, distorted quadrature."""
import math
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(spec):
    return run_router_conservation_cad_verifier({"mode":"fem", "spec":{"mode":"q4_2d", **spec}})


class BilinearMeshTests(unittest.TestCase):
    def test_distorted_patch_free_center_and_clockwise_rows(self):
        nodes = [[0,0],[0.5,0],[1,0],[0,0.5],[0.375,0.625],[1,0.5],[0,1],[0.5,1],[1,1]]
        mesh = {"nodes":nodes, "elements":[[0,1,4,3],[1,4,5,2],[3,4,7,6],[4,5,8,7]],
                "E_Pa":1000, "nu":0.25, "thickness_m":2,
                "loads":[{"node":2,"fx_N":50},{"node":5,"fx_N":100},{"node":8,"fx_N":50}],
                "supports":[{"node":0,"ux_m":0,"uy_m":0},{"node":3,"ux_m":0},{"node":6,"ux_m":0}]}
        for kin, ex, ey, zz, energy in (("plane_stress",0.1,-0.025,0,10),
                                        ("plane_strain",0.09375,-0.03125,25,9.375)):
            with self.subTest(kin=kin):
                out = solve({**mesh, "kinematics":kin})
                self.assertTrue(out["passed"],out)
                for (x,y),(ux,uy) in zip(nodes,out["displacements_xy_m"]):
                    self.assertAlmostEqual(ux,x*ex)
                    self.assertAlmostEqual(uy,y*ey)
                self.assertEqual(out["element_stress_xx_yy_xy_Pa"],[[100,0,0]]*4)
                self.assertEqual(out["element_sigma_zz_Pa"],[zz]*4)
                self.assertEqual(out["strain_energy_J"],energy)
                self.assertEqual(out["nodal_load_work_J"],energy)
                self.assertEqual(out["reactions_xy_N"],[[-50,0],[0,0],[0,0],[-100,0],[0,0],[0,0],[-50,0],[0,0],[0,0]])

    def test_center_blind_alternating_nodal_motion_has_energy(self):
        # On this square, ux=xi*eta, uy=0 so ex=eta and gamma=xi.
        # Integral of (2*eta^2+xi^2)/2 over [-1,1]^2 is exactly 2 J.
        out=solve({"kinematics":"plane_stress", "nodes":[[-1,-1],[1,-1],[1,1],[-1,1]],
                   "elements":[[0,1,2,3]], "E_Pa":2,"nu":0,"thickness_m":1,
                   "supports":[{"node":i,"ux_m":u,"uy_m":0} for i,u in enumerate((1,-1,1,-1))]})
        self.assertTrue(out["passed"],out)
        self.assertEqual(out["integration"],"2x2_gauss")
        self.assertEqual(out["element_stress_strain_location"],"natural_center")
        self.assertEqual(out["element_strain_xx_yy_gamma_xy"],[[0,0,0]])
        self.assertEqual(out["strain_energy_J"],2)
        self.assertEqual(out["prescribed_support_work_J"],2)

    def test_distorted_nonaffine_direct_independent_gauss_energy(self):
        # Map x=(1+xi)*(5+eta)/4, y=(1+eta)*(3-xi)/4 from the square.
        # Prescribe ux=xi*eta, uy=0. Differentiate both fields analytically,
        # then integrate at the four individual irrational sample locations.
        nodes=[[0,0],[2,0],[3,1],[0,2]]
        r=1/math.sqrt(3)
        energy=0
        for xi in (-r,r):
            for eta in (-r,r):
                xx,xe=(5+eta)/4,(1+xi)/4
                yx,ye=-(1+eta)/4,(3-xi)/4
                det=xx*ye-xe*yx
                ex=(ye*eta-yx*xi)/det
                gamma=(xx*xi-xe*eta)/det
                energy += ((16/5)*ex**2+(6/5)*gamma**2)*det/2
        # The four-point integrals of ex^2 and gamma^2 are 28/33 and 76/33.
        # This trapezoid therefore also has an independent rational reference.
        self.assertAlmostEqual(energy, 452/165, places=14)
        common={"kinematics":"plane_stress","nodes":nodes,"E_Pa":3,"nu":0.25,"thickness_m":1,
                "supports":[{"node":i,"ux_m":u,"uy_m":0} for i,u in enumerate((1,-1,1,-1))]}
        for order in ([0,1,2,3],[0,3,2,1]):
            out=solve({**common,"elements":[order]})
            self.assertTrue(out["passed"],out)
            self.assertAlmostEqual(out["strain_energy_J"],energy,places=13)
        for bad in ([[0,1,3,2]], [[0,1,2,2]], [[0,1,2,3],[1,2,3,0]]):
            self.assertFalse(solve({**common,"elements":bad})["passed"])
        inside=[[0,0],[2,0],[0.25,0.25],[0,2]]
        self.assertFalse(solve({**common,"nodes":inside,"elements":[[0,1,2,3]]})["passed"])


if __name__ == "__main__":
    unittest.main()
