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


if __name__ == '__main__':
    unittest.main()
