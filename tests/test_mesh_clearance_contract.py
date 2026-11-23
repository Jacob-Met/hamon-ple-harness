"""Public precision, threshold and diagnostic contracts for represented meshes."""
import math
import unittest
from fractions import Fraction as F

from hamon_ifc_spatial_clash import run_router_ifc_clash
from hamon_mesh_distance import lower_distance_bound, wire_point


def tetra_pair(gap):
    rows = []
    for j, points in enumerate((
            ((0,0,0),(-2,0,0),(0,2,0),(0,0,2)),
            ((gap,0,0),(gap+2,0,0),(gap,2,0),(gap,0,2)))):
        n = 10*j+1
        coordinates = ",".join("("+",".join(str(float(v)) for v in p)+")" for p in points)
        rows.extend((f"#{n}=IFCCARTESIANPOINTLIST3D(({coordinates}));",
                     f"#{n+1}=IFCTRIANGULATEDFACESET(#{n},$,.T.,((1,3,2),(1,2,4),(1,4,3),(2,3,4)),$);",
                     f"#{n+2}=IFCSHAPEREPRESENTATION($,'Body','Tessellation',(#{n+1}));",
                     f"#{n+3}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{n+2}));",
                     f"#{n+4}=IFCBUILDINGELEMENTPROXY('solid{j}',$,'solid{j}',$,$,$,#{n+3},$,$);"))
    return "DATA;"+"".join(rows)+"ENDSEC;"


class MeshClearanceContract(unittest.TestCase):
    def test_native_integer_threshold_keeps_its_own_boundary(self):
        for requested, status in ((2**53,"clear"),(2**53+1,"clearance_violation")):
            result = run_router_ifc_clash(dict(ifc_text=tetra_pair(2**53),
                required_clearance_m=requested,cross_discipline_only=False))
            self.assertTrue(result["ok"],result)
            pair, = result["all_pairs"]
            self.assertEqual(pair["status"],status)
            self.assertEqual(pair["min_distance_m"],2**53)
            self.assertEqual(pair["required_clearance_m"],requested)

    def test_invalid_clearance_is_a_structured_refusal(self):
        for requested in (True,-1.,float("nan"),float("inf")):
            with self.subTest(requested=requested):
                result = run_router_ifc_clash(dict(ifc_text=tetra_pair(1.),
                    required_clearance_m=requested,cross_discipline_only=False))
                self.assertFalse(result["ok"])
                self.assertIn("clearance",result["reason"])

    def test_witness_bound_contains_exact_rounding_error(self):
        for point in ((F(1,3),F(2,7),F(2)**-1100),
                      (F(2)**1023+F(2)**900,F(2)**-1074,F(0))):
            rounded,error = wire_point(point)
            exact_squared = sum((F(x)-y)**2 for x,y in zip(rounded,point))
            self.assertLessEqual(exact_squared,F(error)**2)
            self.assertTrue(math.isfinite(error))

    def test_pruning_bound_remains_conservative_outside_distance_output_range(self):
        for squared in (F(2)**-2400,F(2),F(2)**2050):
            lower = lower_distance_bound(squared)
            self.assertTrue(math.isfinite(lower))
            self.assertLessEqual(F(lower)**2,squared)
