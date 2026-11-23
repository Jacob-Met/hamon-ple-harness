"""Closed rectangular tubes checked against independent outer-minus-inner moments."""
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry
from test_ifc_mapped_mass import Scene, scene, expected


def hollow_scene(wall=.25, **options):
    model,parts=scene(**options)
    model.rows=[row.replace('IFCRECTANGLEPROFILEDEF(.AREA.,$', 'IFCRECTANGLEHOLLOWPROFILEDEF(.AREA.,$').replace(',2.,4.);',f',2.,4.,{wall},0.,0.);') if '=IFCRECTANGLEPROFILEDEF(' in row else row for row in model.rows]
    inner=[]
    for center,edges in parts:
        e=edges.copy();e[:,0]*=1-wall;e[:,1]*=1-wall/2
        inner.append((center,e))
    return model,parts,inner


class HollowMass(unittest.TestCase):
    def check(self, model, outer, inner):
        density=7850.
        ov,om,oc,oi,oa,oe=expected(outer,density)
        iv,im,ic,ii,ia,ie=expected(inner,density)
        caps=sum(np.linalg.norm(np.cross(edges[:,0],edges[:,1])) for _,edges in inner)
        result=verify_step_cad_geometry({'step_text':model.text(),'density_kg_m3':density})
        self.assertTrue(result['conservation_verified'],result)
        self.assertTrue(result['watertight_manifold'])
        self.assertEqual(result['triangle_count'],32*len(outer))
        self.assertAlmostEqual(result['volume_m3']/(ov-iv),1.,places=12)
        self.assertAlmostEqual(result['mass_kg']/(om-im),1.,places=12)
        self.assertAlmostEqual(result['surface_area_m2']/(oa+ia-4*caps),1.,places=12)
        np.testing.assert_allclose(result['centroid_m'],oc,rtol=0,atol=2e-9)
        np.testing.assert_allclose(result['inertia_tensor_centroid_kg_m2'],oi-ii,rtol=2e-12,atol=2e-14*np.max(abs(oi-ii)))
        np.testing.assert_allclose(result['aabb_extents_m'],oe,rtol=0,atol=2e-9)
    def test_signed_nested_reflected_hollow_occurrences(self):
        for sign in [-1,1]:
            for wall in [.125,.25,.75]:
                with self.subTest(sign=sign,wall=wall):
                    self.check(*hollow_scene(wall,signed=sign,scale=2.,nested=True,mirror=True,shift=729011.25))
    def test_direct_rectangular_tube(self):
        s=Scene()
        p=s.add('IFCRECTANGLEHOLLOWPROFILEDEF(.AREA.,$,$,2.,4.,.25,$,$)')
        d=s.vector('IFCDIRECTION',[0,0,1])
        solid=s.add(f'IFCEXTRUDEDAREASOLID(#{p},#{s.identity},#{d},5.)')
        s.product(solid,np.eye(3),[0,0,0])
        center=np.array([0.,0.,2.5])
        self.check(s,[(center,np.diag([2.,4.,5.]))],[(center,np.diag([1.5,3.5,5.]))])
    def test_fillet_and_invalid_wall_are_not_sharp_tubes(self):
        for values in ('0.25,0.1,0.1','0.25,0.,0.2','1.,0.,0.','1.1,0.,0.','-0.1,0.,0.','0.,0.,0.'):
            with self.subTest(values=values):
                model,_,_=hollow_scene()
                model.rows=[row.replace(',0.25,0.,0.);',f',{values});') for row in model.rows]
                with self.assertRaises(ValueError):
                    verify_step_cad_geometry({'step_text':model.text()})


if __name__=='__main__':unittest.main()
