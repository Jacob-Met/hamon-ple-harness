import itertools
import math
import unittest

import numpy as np
from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier


def solve(**physical):
    return run_router_conservation_cad_verifier({
        "mode":"solid_mohr_vessel_buckling",
        "spec":{"subkind":"mohr_stress", **physical},
    })


class MohrTractions(unittest.TestCase):
    def test_uniaxial_octahedral_and_maximum_shear_differ(self):
        result=solve(sigma_x=120,plane_normal=[1,1,1])
        self.assertTrue(result["verified"])
        self.assertEqual(result["sigma_oct"],40)
        self.assertEqual(result["tau_max_abs"],60)
        self.assertAlmostEqual(result["tau_oct"],40*math.sqrt(2))
        self.assertAlmostEqual(result["plane_traction"]["normal_stress"],40)
        self.assertAlmostEqual(result["plane_traction"]["shear_magnitude"],result["tau_oct"])

    def test_actual_tractions_on_all_rotated_principal_body_diagonals(self):
        axes=np.array([[1,2,2],[2,1,-2],[-2,2,-1]])
        np.testing.assert_array_equal(axes.T@axes,9*np.eye(3))
        for stresses in ([90,18,-36],[90,90,-36],[-9,-9,-9]):
            tensor=(axes@np.diag(stresses)@axes.T)//9
            for signs in itertools.product((-1,1),repeat=3):
                normal=axes@signs
                unit=normal/np.linalg.norm(normal)
                traction=tensor@unit
                axial=float(unit@traction)
                tangential=traction-axial*unit
                result=solve(stress_tensor=tensor.tolist(),plane_normal=normal.tolist())
                self.assertTrue(result["verified"],result)
                plane=result["plane_traction"]
                np.testing.assert_allclose(plane["traction"],traction,atol=9e-14)
                np.testing.assert_allclose(plane["shear_traction"],tangential,atol=9e-14)
                self.assertAlmostEqual(result["sigma_oct"],axial,places=12)
                self.assertAlmostEqual(result["tau_oct"],np.linalg.norm(tangential),places=12)
                self.assertEqual(plane["normal_stress"],result["sigma_oct"])
                self.assertEqual(plane["shear_magnitude"],result["tau_oct"])

    def test_oblique_plane_and_hydrostatic_translation(self):
        normal=[2,3,-4]
        for pressure in (0,2**53):
            result=solve(sigma_x=pressure+1,sigma_y=pressure,sigma_z=pressure,tau_xy=2,plane_normal=normal)
            self.assertTrue(result["verified"])
            plane=result["plane_traction"]
            shear=np.array(plane["shear_traction"])
            self.assertAlmostEqual(float(shear@normal),0.0)
            if not pressure:
                expected=(result["tau_oct"],plane["shear_magnitude"],plane["shear_traction"])
            else:
                self.assertEqual((result["tau_oct"],plane["shear_magnitude"],plane["shear_traction"]),expected)
        reverse=solve(sigma_x=1,tau_xy=2,plane_normal=[-6,-9,12])["plane_traction"]
        np.testing.assert_array_equal(reverse["shear_traction"],-np.array(expected[2]))
        self.assertEqual(reverse["shear_magnitude"],expected[1])

    def test_invalid_tensor_normal_and_inconsistent_duplicate(self):
        for bad in ([[1,2],[2,1]], [[0,1,0],[0,0,0],[0,0,0]],
                    [[False,0,0],[0,1,0],[0,0,1]], [[float('inf'),0,0],[0,1,0],[0,0,1]]):
            self.assertFalse(solve(stress_tensor=bad)["verified"])
        for normal in ([0,0,0],[1,0],[0,None,1],[True,0,1],["1",0,0]):
            self.assertFalse(solve(sigma_x=1,plane_normal=normal)["verified"])
        self.assertFalse(solve(stress_tensor=[["9007199254740993",0,0],[0,0,0],[0,0,0]],
                               sigma_x=9007199254740992)["verified"])
        self.assertFalse(solve(stress_tensor=[[2**53,0,0],[0,0,0],[0,0,0]],
                               sigma_x="9007199254740993")["verified"])
        self.assertTrue(solve(sigma_x="120")["verified"])
        matching=dict(stress_tensor=[[1,2,0],[2,3,0],[0,0,4]],tau_xy=2)
        self.assertTrue(solve(**matching)["verified"])
        matching["tau_xy"]=3
        self.assertFalse(solve(**matching)["verified"])
