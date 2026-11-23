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

if __name__=="__main__":unittest.main()
