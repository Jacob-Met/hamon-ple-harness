"""Rooted STEP CSG must evaluate material sets before integrating properties."""
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry

def two_blocks(operator='UNION', origin_b=(1,1,1), size_a=(2,2,2), size_b=(2,2,2)):
    vec=lambda xyz: ','.join(map(str,xyz))
    return ("DATA;#1=CARTESIAN_POINT('',(0,0,0));#2=AXIS2_PLACEMENT_3D('',#1,$,$);"
            f"#3=BLOCK('',#2,{vec(size_a)});#4=CARTESIAN_POINT('',({vec(origin_b)}));"
            f"#5=AXIS2_PLACEMENT_3D('',#4,$,$);#6=BLOCK('',#5,{vec(size_b)});"
            f"#7=BOOLEAN_RESULT('',.{operator}.,#3,#6);#8=CSG_SOLID('',#7);ENDSEC;")

class StepCsgMaterialTests(unittest.TestCase):
    def verify(self,text,**kw):
        return verify_step_cad_geometry(dict(step_text=text,density_kg_m3=1,**kw))

    def test_regularized_block_operators_and_single_primitive_root(self):
        for op,volume in [('UNION',15),('INTERSECTION',1),('DIFFERENCE',7)]:
            with self.subTest(operator=op):
                result=self.verify(two_blocks(op))
                self.assertTrue(result['conservation_verified'],result)
                self.assertEqual(result['volume_m3'],volume)
        with self.subTest(operator='one primitive'):
            result=self.verify(two_blocks().replace("CSG_SOLID('',#7)","CSG_SOLID('',#3)"))
            self.assertTrue(result['conservation_verified'],result)
            self.assertEqual(result['volume_m3'],8)

    def test_shared_expression_nested_root_and_explicit_selection(self):
        text=two_blocks().replace(".UNION.,#3,#6", ".UNION.,#3,#3")
        self.assertEqual(self.verify(text)['volume_m3'],8)
        text=two_blocks().replace("#8=CSG_SOLID('',#7);",
            "#9=CSG_SOLID('',#7);#10=BOOLEAN_RESULT('',.INTERSECTION.,#9,#3);#8=CSG_SOLID('',#10);")
        r=self.verify(text)
        self.assertTrue(r['conservation_verified'],r)
        self.assertEqual((r['volume_m3'],r['selected_solid_entity_id']),(8,8))
        text=two_blocks().replace('ENDSEC;',"#20=CSG_SOLID('',#3);ENDSEC;")
        with self.assertRaises(ValueError):self.verify(text)
        self.assertEqual(self.verify(text,solid_entity_id=8)['volume_m3'],15)
        self.assertEqual(self.verify(text,solid_entity_id=20)['volume_m3'],8)

    def test_rectilinear_placement_and_selected_representation_units(self):
        text=two_blocks(size_a=(2,3,4)).replace("CSG_SOLID('',#7)","CSG_SOLID('',#3)")
        text=text.replace("#2=AXIS2_PLACEMENT_3D('',#1,$,$)",
                          "#2=AXIS2_PLACEMENT_3D('',#1,#10,#11)")
        text=text.replace("ENDSEC;","#10=DIRECTION('',(1.E-300,0,0));"
            "#11=DIRECTION('',(7.E300,-3.E300,0));"
            "#90=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));"
            "#91=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#90))"
            " REPRESENTATION_CONTEXT('',''));#92=CSG_SHAPE_REPRESENTATION('',(#8),#91);ENDSEC;")
        r=self.verify(text)
        self.assertTrue(r['conservation_verified'],r)
        np.testing.assert_allclose(r['volume_m3'],24e-9,rtol=2e-15)
        np.testing.assert_allclose(r['centroid_m'],[.002,-.001,-.0015],rtol=2e-15)
        np.testing.assert_allclose(r['inertia_tensor_centroid_kg_m2'],
                                   np.diag([26e-15,50e-15,40e-15]),rtol=2e-15,atol=0)

    def test_incomplete_cyclic_or_nonrectilinear_construction_refuses(self):
        base=two_blocks()
        cases=[base.replace(".UNION.",".XOR."),
               base.replace(".UNION.,#3,#6",".UNION.,#7,#6"),
               base.replace(".UNION.,#3,#6",".UNION.,#3,#999"),
               base.replace("#6=BLOCK","#6=RIGHT_CIRCULAR_CYLINDER"),
               base.replace("#2=AXIS2_PLACEMENT_3D","#2=NOT_A_PLACEMENT"),
               base.replace("BLOCK('',#2,2,2,2)","BLOCK('',#2,0,2,2)"),
               base.replace("BLOCK('',#2,2,2,2)","BLOCK('',#2,-2,2,2)"),
               base.replace("BLOCK('',#2,2,2,2)","BLOCK('',#2,NaN,2,2)")]
        tilted=base.replace("#2=AXIS2_PLACEMENT_3D('',#1,$,$)",
                            "#2=AXIS2_PLACEMENT_3D('',#1,#10,$)")
        cases.append(tilted.replace("ENDSEC;","#10=DIRECTION('',(1.E-500,0,1));ENDSEC;"))
        cases.append(tilted.replace("ENDSEC;","#10=DIRECTION('',(0,0,0));ENDSEC;"))
        for index,text in enumerate(cases):
            with self.subTest(case=index),self.assertRaises(ValueError):self.verify(text)
        for selector in (True,'8',999):
            with self.subTest(selector=selector),self.assertRaises(ValueError):
                self.verify(base,solid_entity_id=selector)

    def test_mixed_ifc_material_requires_an_explicit_step_definition(self):
        marker="#100=IFCCARTESIANPOINT((0.,0.,0.));"
        body=("#101=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,3.);"
              "#102=IFCDIRECTION((0.,0.,1.));#103=IFCEXTRUDEDAREASOLID(#101,$,#102,4.);")
        text=two_blocks().replace("ENDSEC;",marker+"ENDSEC;")
        self.assertTrue(self.verify(text)['conservation_verified'])
        text=text.replace("ENDSEC;",body+"ENDSEC;")
        with self.assertRaises(ValueError):self.verify(text)
        result=self.verify(text,solid_entity_id=8)
        self.assertTrue(result['conservation_verified'],result)
        self.assertEqual(result['volume_m3'],15)

if __name__=='__main__':unittest.main()
