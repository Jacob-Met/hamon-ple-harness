"""STEP analytic solids must use the orthonormal frame of their placement."""
import math
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry


def block_text(axis=(0.,0.,1.), reference=(1.,0.,0.), origin=(0.,0.,0.)):
    records = [f"#1=CARTESIAN_POINT('',({','.join(map(repr, origin))}));"]
    for eid, vector in ((2,axis),(3,reference)):
        if vector is not None:
            records.append(f"#{eid}=DIRECTION('',({','.join(map(repr, vector))}));")
    z, x = ("$" if axis is None else "#2"), ("$" if reference is None else "#3")
    records.extend((f"#4=AXIS2_PLACEMENT_3D('',#1,{z},{x});", "#5=BLOCK('',#4,2.,3.,4.);"))
    return "DATA;\n" + "\n".join(records) + "\nENDSEC;"


class StepAnalyticFrameTests(unittest.TestCase):
    def verify(self, text, **options):
        return verify_step_cad_geometry({"step_text":text,"density_kg_m3":1.,**options})

    def assert_frame(self, text, matrix, origin=(0.,0.,0.), scale=1.):
        r = self.verify(text, unit_scale_m=scale)
        self.assertTrue(r['conservation_verified'],r)
        matrix = np.asarray(matrix)
        self.assertAlmostEqual(r['volume_m3']/scale**3,24.,places=11)
        self.assertAlmostEqual(r['mass_kg']/scale**3,24.,places=11)
        self.assertAlmostEqual(r['surface_area_m2']/scale**2,52.,places=11)
        np.testing.assert_allclose(np.array(r['centroid_m'])/scale,
                                   np.array(origin)+matrix@[1.,1.5,2.],rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(np.array(r['inertia_tensor_centroid_kg_m2'])/scale**5,
                                   matrix@np.diag([50.,40.,26.])@matrix.T,rtol=2e-13,atol=2e-12)
        np.testing.assert_allclose(np.array(r['principal_moments_kg_m2'])/scale**5,
                                   [26.,40.,50.],rtol=2e-13,atol=2e-12)

    def test_positive_direction_ratio_scaling_cannot_rotate_the_solid(self):
        for exponent in (0,-80,-400):
            factor = math.ldexp(1.,exponent)
            with self.subTest(exponent=exponent):
                r = self.verify(block_text((factor,0.,0.),(0.,factor,0.)))
                self.assertTrue(r['conservation_verified'],r)
                np.testing.assert_allclose(r['centroid_m'],[2.,1.,1.5],rtol=1e-13)
                np.testing.assert_allclose(r['inertia_tensor_centroid_kg_m2'],
                                           np.diag([26.,50.,40.]),rtol=1e-13,atol=1e-13)

    def test_omitted_reference_axis_is_projected_onto_the_normal_plane(self):
        r = self.verify(block_text((1.,0.,100.),None))
        self.assertTrue(r['conservation_verified'],r)
        self.assertAlmostEqual(r['volume_m3'],24.,places=12)

    def test_oblique_frames_survive_independent_direction_scales_and_unit_changes(self):
        matrix = np.array([[4/5,36/65,3/13],[-3/5,48/65,4/13],[0.,-5/13,12/13]])
        origin = (7.,-11.,13.)
        for exponent in (-1000,-500,0,500,1000):
            z = tuple(math.ldexp(v,exponent) for v in (3.,4.,12.))
            x = tuple(math.ldexp(v,-exponent) for v in (10.,5.,24.))
            for scale in (1.,.001,.0254):
                with self.subTest(exponent=exponent,scale=scale):
                    self.assert_frame(block_text(z,x,origin),matrix,origin,scale)

    def test_one_ulp_nonparallel_directions_keep_their_projection(self):
        z = np.ones(3)/math.sqrt(3)
        for toward,sign in ((math.inf,1.),(0.,-1.)):
            y = sign*np.array([-1.,0.,1.])/math.sqrt(2)
            x = sign*np.array([-1.,2.,-1.])/math.sqrt(6)
            self.assert_frame(block_text((1.,1.,1.),(1.,math.nextafter(1.,toward),1.)),
                              np.column_stack((x,y,z)))

    def test_default_axes_and_exactly_axial_default_reference(self):
        self.assert_frame(block_text(None,None),np.eye(3))
        for sign in (1.,-1.):
            self.assert_frame(block_text((sign,0.,0.),None),
                              [[0.,0.,sign],[1.,0.,0.],[0.,sign,0.]])
        norm = math.sqrt(10001.)
        self.assert_frame(block_text((1.,0.,100.),None),
                          np.array([[100.,0.,1.],[0.,norm,0.],[-1.,0.,100.]])/norm)

    def test_decimal_text_keeps_parallelism_and_small_nonzero_angles(self):
        with self.subTest(case='exact decimal parallel'),self.assertRaises(ValueError):
            self.verify(block_text((1.,2.,3.),(.1,.2,.3)))
        with self.subTest(case='decimal angle below binary64 resolution'):
            tiny = '1.'+'0'*59+'1'
            text = block_text((1.,1.,1.),(1.,1.,1.)).replace(
                "#3=DIRECTION('',(1.0,1.0,1.0));",f"#3=DIRECTION('',(1.,{tiny},1.));")
            matrix = np.column_stack((np.array([-1.,2.,-1.])/math.sqrt(6),
                                      np.array([-1.,0.,1.])/math.sqrt(2),np.ones(3)/math.sqrt(3)))
            self.assert_frame(text,matrix)
        with self.subTest(case='small component chooses default projection'):
            text = block_text((1.,1.,0.),None).replace(
                "#2=DIRECTION('',(1.0,1.0,0.0));","#2=DIRECTION('',(1.,1.E-400,0.));")
            self.assert_frame(text,[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])

    def test_missing_malformed_or_parallel_declared_placement_is_not_a_default(self):
        valid = block_text()
        cases = [valid.replace("BLOCK('',#4,", "BLOCK('',#999,"),
                 valid.replace("AXIS2_PLACEMENT_3D", "NOT_A_PLACEMENT"),
                 valid.replace("('',#1,#2,#3)", "('',#999,#2,#3)"),
                 valid.replace("CARTESIAN_POINT", "DIRECTION",1),
                 valid.replace("(0.0,0.0,0.0)", "(0.0,0.0)",1),
                 valid.replace("(0.0,0.0,0.0)", "(0.0,0.0,0.0,0.0)",1),
                 valid.replace("#2=DIRECTION", "#222=DIRECTION"),
                 valid.replace("#2=DIRECTION", "#2=CARTESIAN_POINT"),
                 block_text((0.,0.,0.)), block_text(reference=(0.,0.,0.)),
                 block_text(reference=(0.,0.,2.)),block_text(None,(0.,0.,1.)),
                 block_text((math.nan,0.,1.)),block_text((math.inf,0.,1.))]
        for index,text in enumerate(cases):
            with self.subTest(case=index),self.assertRaises(ValueError):
                self.verify(text)

    def test_cylinder_mass_properties_follow_the_same_frame(self):
        def cylinder(text):
            return (text.replace("BLOCK('',#4,2.,3.,4.)", "RIGHT_CIRCULAR_CYLINDER('',#4,4.,2.)")
                    .replace("AXIS2_PLACEMENT_3D('',#1,#2,#3)", "AXIS1_PLACEMENT('',#1,#2)"))
        base = self.verify(cylinder(block_text()))
        moved = self.verify(cylinder(block_text((1e-200,0.,0.),(0.,1e200,0.))))
        matrix = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        self.assertTrue(moved['conservation_verified'],moved)
        for key in ('volume_m3','surface_area_m2','mass_kg','principal_moments_kg_m2'):
            np.testing.assert_allclose(moved[key],base[key],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(moved['centroid_m'],matrix@base['centroid_m'],atol=1e-13)
        np.testing.assert_allclose(moved['inertia_tensor_centroid_kg_m2'],
                                   matrix@base['inertia_tensor_centroid_kg_m2']@matrix.T,atol=1e-12)

    def test_unowned_ifc_sweep_keeps_its_optional_position(self):
        text = "DATA;#1=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,3.);" \
               "#2=IFCDIRECTION((0.,0.,1.));#3=IFCEXTRUDEDAREASOLID(#1,$,#2,4.);ENDSEC;"
        r = self.verify(text)
        self.assertTrue(r['conservation_verified'],r)
        self.assertEqual(r['volume_m3'],24.)
        np.testing.assert_allclose(r['centroid_m'],[0.,0.,2.],atol=1e-14)
        rotation = "#4=IFCCARTESIANPOINT((0.,0.,0.));#5=IFCDIRECTION((1.E-200,0.,0.));" \
                   "#6=IFCDIRECTION((0.,1.E200,0.));#7=IFCAXIS2PLACEMENT3D(#4,#5,#6);"
        r = self.verify(text.replace("(#1,$,#2,4.)","(#1,#7,#2,4.)").replace("ENDSEC;",rotation+"ENDSEC;"))
        self.assertTrue(r['conservation_verified'],r)
        np.testing.assert_allclose(r['centroid_m'],[2.,0.,0.],atol=1e-14)
        np.testing.assert_allclose(r['inertia_tensor_centroid_kg_m2'],np.diag([26.,50.,40.]),atol=1e-12)


if __name__ == '__main__':
    unittest.main()
