"""Round exact curved-coordinate ties after combining independently non-dyadic terms."""
import math
import unittest
from decimal import Decimal, localcontext
from hamon_conservation_cad_verifier import verify_step_cad_geometry


class CircularCoordinateTies(unittest.TestCase):
    def test_rational_rotated_extents_and_translated_centroids_at_exact_midpoints(self):
        with localcontext() as ctx:
            ctx.prec = 120
            for k in (1,3):
                d = Decimal(k)/(Decimal(2)**53)
                expected = float(1+d)
                specs = [
                    (("0","0","1"), ("0","0","0.1"), Decimal("1.8")+2*d, 1,
                     "centroid_m",2),
                    (("3","4","0"), ("0","0","0"), Decimal(".5"), Decimal("0.4375")+5*d/8,
                     "aabb_extents_m",0),
                ]
                for axis,origin,h,r,field,i in specs:
                    with self.subTest(k=k, field=field):
                        text=("DATA;#1=CARTESIAN_POINT('',("+",".join(origin)+"));"
                              "#2=DIRECTION('',("+",".join(axis)+"));"
                              "#3=AXIS1_PLACEMENT('',#1,#2);"
                              f"#4=RIGHT_CIRCULAR_CYLINDER('',#3,{h},{r});ENDSEC;")
                        result=verify_step_cad_geometry({"step_text":text,
                            "unit_scale_m":1.,"density_kg_m3":1.})
                        self.assertTrue(result["conservation_verified"],result)
                        self.assertEqual(result[field][i],expected)
