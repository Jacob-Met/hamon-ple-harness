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

if __name__=='__main__':unittest.main()
