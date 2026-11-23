"""Schema-correct STEP primitives and analytic mass properties."""
import math
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry

def cylinder_text(height='4.',radius='2.',axis=('0.','0.','1.'),origin=('0.','0.','0.'),rooted=False):
    records=[f"#1=CARTESIAN_POINT('',({','.join(origin)}));"]
    if axis is not None:records.append(f"#2=DIRECTION('',({','.join(axis)}));")
    ref='$' if axis is None else '#2'
    records.extend([f"#3=AXIS1_PLACEMENT('',#1,{ref});",
                    f"#4=RIGHT_CIRCULAR_CYLINDER('',#3,{height},{radius});"])
    if rooted:records.append("#5=CSG_SOLID('',#4);")
    return 'DATA;'+''.join(records)+'ENDSEC;'

def sphere_text(radius='3.',origin=('7.','-11.','13.'),rooted=False):
    records=[f"#1=CARTESIAN_POINT('',({','.join(origin)}));",f"#2=SPHERE('',{radius},#1);"]
    if rooted:records.append("#3=CSG_SOLID('',#2);")
    return 'DATA;'+''.join(records)+'ENDSEC;'

class StepAnalyticCurveTests(unittest.TestCase):
    def verify(self,text,**kw):
        return verify_step_cad_geometry({'step_text':text,'density_kg_m3':1.,**kw})

    def assert_withheld(self,text):
        try:result=self.verify(text)
        except ValueError:return
        self.assertFalse(result['conservation_verified'],result)

    def test_axis1_cylinder_closed_form_properties(self):
        for rooted in (False,True):
            with self.subTest(rooted=rooted):
                r=self.verify(cylinder_text(rooted=rooted))
                self.assertTrue(r['conservation_verified'],r)
                np.testing.assert_allclose(r['volume_m3'],16*math.pi,rtol=3e-15)
                np.testing.assert_allclose(r['surface_area_m2'],24*math.pi,rtol=3e-15)
                np.testing.assert_array_equal(r['centroid_m'],[0.,0.,2.])
                np.testing.assert_allclose(r['inertia_tensor_centroid_kg_m2'],
                    np.diag([112*math.pi/3,112*math.pi/3,32*math.pi]),rtol=3e-15,atol=0)

    def test_sphere_uses_its_typed_centre_and_full_inertia(self):
        for rooted in (False,True):
            with self.subTest(rooted=rooted):
                r=self.verify(sphere_text(rooted=rooted))
                self.assertTrue(r['conservation_verified'],r)
                np.testing.assert_allclose(r['volume_m3'],36*math.pi,rtol=3e-15)
                np.testing.assert_allclose(r['surface_area_m2'],36*math.pi,rtol=3e-15)
                np.testing.assert_array_equal(r['centroid_m'],[7.,-11.,13.])
                np.testing.assert_allclose(r['inertia_tensor_centroid_kg_m2'],
                    np.eye(3)*(648*math.pi/5),rtol=3e-15,atol=0)

    def test_wrong_placement_type_and_negative_block_lengths_withhold(self):
        wrong=cylinder_text().replace("AXIS1_PLACEMENT('',#1,#2)",
                                      "AXIS2_PLACEMENT_3D('',#1,#2,$)")
        with self.subTest(case='axis2 is not a cylinder position'):self.assert_withheld(wrong)
        for dims in ('-2.,3.,4.','2.,-3.,4.','2.,3.,-4.'):
            text=("DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));"
                  "#2=AXIS2_PLACEMENT_3D('',#1,$,$);"
                  f"#3=BLOCK('',#2,{dims});ENDSEC;")
            with self.subTest(dimensions=dims):self.assert_withheld(text)

if __name__=='__main__':unittest.main()
