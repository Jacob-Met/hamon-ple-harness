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

    def test_exact_source_axis_ratios_and_scale_covariance(self):
        axis=np.array([3.,4.,12.])/13
        origin=np.array([7.,-11.,13.])
        tensor=(112*math.pi/3)*np.eye(3)+(32*math.pi-112*math.pi/3)*np.outer(axis,axis)
        for exponent in (-400,0,400):
            direction=tuple(f'{v}e{exponent}' for v in (3,4,12))
            for scale in (1.,.001,.0254):
                with self.subTest(exponent=exponent,scale=scale):
                    r=self.verify(cylinder_text(axis=direction,origin=('7.','-11.','13.')),
                                  unit_scale_m=scale)
                    self.assertTrue(r['conservation_verified'],r)
                    np.testing.assert_allclose(np.array(r['centroid_m'])/scale,origin+2*axis,rtol=3e-15)
                    np.testing.assert_allclose(np.array(r['inertia_tensor_centroid_kg_m2'])/scale**5,
                                               tensor,rtol=3e-15)
                    expected=4*np.abs(axis)+4*np.sqrt(1-axis**2)
                    np.testing.assert_allclose(np.array(r['aabb_extents_m'])/scale,expected,rtol=3e-15)

    def test_axis_defaults_reversal_and_exact_centroid_cancellation(self):
        default=self.verify(cylinder_text(axis=None))
        np.testing.assert_array_equal(default['centroid_m'],[0.,0.,2.])
        reversed_result=self.verify(cylinder_text(axis=('0.','0.','-1.')))
        np.testing.assert_array_equal(reversed_result['centroid_m'],[0.,0.,-2.])
        cancelled=self.verify(cylinder_text(height='26.',axis=('3.','4.','12.'),
                                             origin=('-3.','-4.','-12.')))
        np.testing.assert_array_equal(cancelled['centroid_m'],[0.,0.,0.])
        self.assertTrue(cancelled['output_rounding_verified'])

    def test_large_aspect_ratio_retains_small_principal_moment(self):
        r=self.verify(cylinder_text(height='1e60',radius='1e-30',axis=('1.','1.','1.')))
        self.assertTrue(r['conservation_verified'],r)
        np.testing.assert_allclose(r['volume_m3'],math.pi,rtol=3e-15)
        np.testing.assert_allclose(r['principal_moments_kg_m2'],
                                   [math.pi*5e-61,math.pi*1e120/12,math.pi*1e120/12],rtol=3e-15)
        self.assertTrue(r['analytic_invariants_verified'])

    def test_invalid_geometry_and_lost_positive_outputs_withhold(self):
        cases=[cylinder_text(height='0.'),cylinder_text(radius='-2.'),
               cylinder_text(axis=('0.','0.','0.')),sphere_text(radius='0.'),
               sphere_text(radius='-1.'),cylinder_text(height='1e-400',radius='1e100'),
               sphere_text(radius='1e-100'),
               cylinder_text().replace("DIRECTION('',(0.,0.,1.))","CARTESIAN_POINT('',(0.,0.,1.))"),
               sphere_text().replace("CARTESIAN_POINT('',(7.,-11.,13.))","DIRECTION('',(7.,-11.,13.))"),
               sphere_text().replace("(7.,-11.,13.)","(7.,-11.)")]
        for text in cases:
            with self.subTest(text=text):self.assert_withheld(text)
        zero_density=verify_step_cad_geometry({'step_text':sphere_text(radius='1e-100'),
                                               'density_kg_m3':0.})
        self.assertTrue(zero_density['conservation_verified'],zero_density)
        self.assertEqual(zero_density['mass_kg'],0.)
        self.assertEqual(zero_density['principal_moments_kg_m2'],[0.,0.,0.])

    def test_ambiguous_or_partial_material_is_not_a_single_primitive(self):
        for addition in ("#6=SPHERE('',1.,#1);",
                         "#6=RIGHT_CIRCULAR_CONE('',#3,4.,2.,0.5);",
                         "#6=EXTRUDED_AREA_SOLID('',#20,#3,#2,4.);",
                         "#6=MAPPED_ITEM('',#20,#21);",
                         "#6=SHAPE_REPRESENTATION('',(#4,#20),#21);"):
            with self.subTest(addition=addition):
                self.assert_withheld(cylinder_text().replace("ENDSEC;",addition+"ENDSEC;"))
        both=sphere_text(rooted=True).replace("ENDSEC;","#6=CSG_SOLID('',#2);ENDSEC;")
        self.assert_withheld(both)
        selected=self.verify(both,solid_entity_id=3)
        self.assertTrue(selected['conservation_verified'],selected)
        curved_boolean=cylinder_text().replace("ENDSEC;",
            "#6=BOOLEAN_RESULT('',.UNION.,#4,#4);#7=CSG_SOLID('',#6);ENDSEC;")
        self.assert_withheld(curved_boolean)

if __name__=='__main__':unittest.main()
