"""A positive primitive is independent of fixed mesh tolerances and global origin."""
import math
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry

def block_text(dimensions, origin=("0.","0.","0."), axis=("0.","0.","1."), reference=("1.","0.","0.")):
    return ("DATA;#1=CARTESIAN_POINT('',("+",".join(origin)+"));"
            "#2=DIRECTION('',("+",".join(axis)+"));"
            "#3=DIRECTION('',("+",".join(reference)+"));"
            "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);"
            "#5=BLOCK('',#4,"+",".join(dimensions)+");ENDSEC;")

class ExactBlockScaleTests(unittest.TestCase):
    def verify(self, dimensions, **kwargs):
        return verify_step_cad_geometry({"step_text":block_text(dimensions,**kwargs),
                                         "density_kg_m3":1.,"unit_scale_m":1.})

    def assert_properties(self,result,lengths,origin=(0.,0.,0.)):
        volume=math.prod(lengths)
        inertia=[volume*sum(lengths[j]**2 for j in range(3) if j!=i)/12 for i in range(3)]
        self.assertTrue(result["conservation_verified"],result)
        np.testing.assert_allclose(result["volume_m3"],volume,rtol=3e-15)
        np.testing.assert_allclose(result["surface_area_m2"],
            2*sum(lengths[i]*lengths[j] for i,j in ((0,1),(0,2),(1,2))),rtol=3e-15)
        np.testing.assert_allclose(result["aabb_extents_m"],lengths,rtol=3e-15)
        np.testing.assert_allclose(result["principal_moments_kg_m2"],sorted(inertia),rtol=3e-15)
        np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"],np.diag(inertia),rtol=3e-15,atol=0)
        np.testing.assert_array_equal(result["centroid_m"],[origin[i]+lengths[i]/2 for i in range(3)])

    def test_small_positive_volume_has_no_fixed_mesh_floor(self):
        result=self.verify(("1e-7","1e-7","1e-7"))
        self.assert_properties(result,(1e-7,1e-7,1e-7))

    def test_thin_material_has_no_fixed_vertex_welding_grid(self):
        thickness=math.ldexp(1.,-40)
        result=self.verify(("1.","1.",repr(thickness)))
        self.assert_properties(result,(1.,1.,thickness))

    def test_source_origin_does_not_erase_local_material(self):
        result=self.verify(("1.","1.","1."),origin=("10000000000000000.","0.","0."))
        self.assert_properties(result,(1.,1.,1.),(1e16,0.,0.))

    def test_long_block_keeps_a_positive_weak_principal_moment(self):
        result=self.verify(("1e16","1.","1."))
        self.assertTrue(result["conservation_verified"],result)
        np.testing.assert_allclose(result["principal_moments_kg_m2"],
                                   [1e16/6,1e48/12,1e48/12],rtol=3e-15,atol=0)

    def test_decimal_local_centroid_survives_large_opposing_source_coordinates(self):
        length="20000000000000000."+"0"*59+"2"
        result=self.verify((length,"1.","1."),origin=("-10000000000000000.","0.","0."))
        self.assertTrue(result["conservation_verified"],result)
        self.assertEqual(result["centroid_m"][0],1e-60)

    def test_irrational_frame_terms_cancel_exactly(self):
        result=self.verify(("2.","3.","2."),axis=("1.","1.","0."),reference=("1.","-1.","0."))
        self.assertTrue(result["conservation_verified"],result)
        self.assertEqual(result["centroid_m"][1],0.)
        np.testing.assert_allclose(result["centroid_m"],[math.sqrt(2),0.,-1.5],rtol=3e-15,atol=0)
        np.testing.assert_allclose(result["aabb_extents_m"],[2*math.sqrt(2),2*math.sqrt(2),3.],rtol=3e-15)
        np.testing.assert_array_equal(result["inertia_tensor_centroid_kg_m2"],np.diag([13.,13.,8.]))

    def test_rotated_slender_material_retains_weak_inertia(self):
        result=self.verify(("1e60","1e-30","1e-30"),
                           axis=("3.","4.","12."),reference=("10.","5.","24."))
        self.assertTrue(result["conservation_verified"],result)
        weak,strong=1e-60/6,1e120/12
        np.testing.assert_allclose(result["principal_moments_kg_m2"],[weak,strong,strong],rtol=3e-15,atol=0)
        unit=np.array([.8,-.6,0.])
        expected=strong*(np.eye(3)-np.outer(unit,unit))+weak*np.outer(unit,unit)
        np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"],expected,rtol=3e-15,atol=0)

    def test_rational_centroid_midpoint_uses_nearest_even(self):
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec=100
            for k in (1,3):
                middle=Decimal(1)+Decimal(k)/(Decimal(2)**53)
                length=2*(middle-Decimal(".1"))
                result=self.verify((str(length),"1.","1."),origin=(".1","0.","0."))
                self.assertEqual(result["centroid_m"][0],float(middle))

    def test_nonrepresentable_local_properties_and_multiple_material_withhold(self):
        length="20000000000000000."+"0"*399+"2"
        cases=[block_text((length,"1.","1."),origin=("-10000000000000000.","0.","0.")),
               block_text(("1e-400","1e100","1e100")),
               block_text(("1.","1.","1.")).replace("ENDSEC;","#6=BLOCK('',#4,1.,1.,1.);ENDSEC;"),
               block_text(("1.","1.","1.")).replace("ENDSEC;","#6=EXTRUDED_AREA_SOLID('',#10,#4,#2,1.);ENDSEC;")]
        for text in cases:
            with self.subTest(text=text):
                try:
                    result=verify_step_cad_geometry({"step_text":text,"density_kg_m3":1.})
                except ValueError:
                    continue
                self.assertFalse(result["conservation_verified"],result)

    def test_subresolution_cross_moment_retains_material_orientation(self):
        text=block_text(("2.","3.","4."),axis=("1.","1e-400","0."))
        text=text.replace("AXIS2_PLACEMENT_3D('',#1,#2,#3)","AXIS2_PLACEMENT_3D('',#1,#2,$)")
        result=verify_step_cad_geometry({"step_text":text,"unit_scale_m":1.,"density_kg_m3":1.})
        self.assertTrue(result["conservation_verified"],result)
        np.testing.assert_array_equal(result["centroid_m"],[2.,-1.,-1.5])
        np.testing.assert_array_equal(result["inertia_tensor_centroid_kg_m2"],np.diag([26.,50.,40.]))
        np.testing.assert_array_equal(result["principal_moments_kg_m2"],[26.,40.,50.])
        self.assertEqual(result["output_error_model"]["inertia_off_diagonal"],
                         "relative to sqrt(I_ii * I_jj)")

if __name__=="__main__":unittest.main()
