"""Closed mesh clearance follows attained feature distance, independently of scale."""
import math
import unittest
from hamon_ifc_spatial_clash import run_router_ifc_clash

def prism_pair(scale, gap, crossed=True):
    records=[]
    faces=[(1,3,2),(4,5,6),(1,2,5),(1,5,4),
           (2,3,6),(2,6,5),(3,1,4),(3,4,6)]
    for k in range(2):
        base=10*k+1
        xy=([(0,2),(-2,-1),(2,-1)] if k==0 or not crossed else
            [(0,-2),(2,1),(-2,1)])
        lo,hi=(-scale/4,0.) if k==0 else (gap,gap+scale/4)
        vertices=[(x*scale,y*scale,z) for z in (lo,hi) for x,y in xy]
        points="("+",".join("("+",".join(map(repr,p))+")" for p in vertices)+")"
        triangles="("+",".join("("+",".join(map(str,f))+")" for f in faces)+")"
        records.extend([
            f"#{base}=IFCCARTESIANPOINTLIST3D({points});",
            f"#{base+1}=IFCTRIANGULATEDFACESET(#{base},$,.T.,{triangles},$);",
            f"#{base+2}=IFCSHAPEREPRESENTATION($,'Body','Tessellation',(#{base+1}));",
            f"#{base+3}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{base+2}));",
            f"#{base+4}=IFCBUILDINGELEMENTPROXY('prism{k}',$,'prism{k}',$,$,$,#{base+3},$,$);"])
    return "DATA;"+"".join(records)+"ENDSEC;"

class TriangleClearanceScaleTests(unittest.TestCase):
    def report(self,scale,gap,required,crossed=True):
        result=run_router_ifc_clash({"ifc_text":prism_pair(scale,gap,crossed),
            "required_clearance_m":required,"cross_discipline_only":False})
        self.assertTrue(result["ok"],result)
        self.assertEqual(result["solid_count"],2,result)
        self.assertEqual(result["pair_count"],1,result)
        return result["all_pairs"][0]

    def test_opposed_prisms_have_the_same_clearance_under_scaling(self):
        # Both projected triangular interiors contain (0,0). The opposed
        # z half-spaces prove gap is a lower bound, attained by those two
        # central cap points. No vertex projects into the other triangle.
        for exponent in (-5,-15):
            scale=math.ldexp(1.,exponent)
            gap=scale/8
            with self.subTest(exponent=exponent):
                result=self.report(scale,gap,scale/4)
                self.assertEqual(result["status"],"clearance_violation",result)
                self.assertAlmostEqual(result["min_distance_m"]/gap,1.,places=13)

    def test_clearance_contact_uses_the_requested_side_of_the_boundary(self):
        scale=math.ldexp(1.,-5)
        gap=scale/8
        for required,status in ((math.nextafter(gap,0.),"clear"),
                                (gap,"clear"),
                                (math.nextafter(gap,math.inf),"clearance_violation")):
            with self.subTest(required=required):
                result=self.report(scale,gap,required)
                self.assertEqual(result["status"],status,result)
                self.assertEqual(result["min_distance_m"],gap)

    def test_positive_separation_is_not_a_hard_material_clash(self):
        gap=math.ldexp(1.,-40)
        result=self.report(math.ldexp(1.,-5),gap,0.,crossed=False)
        self.assertEqual(result["status"],"clear",result)
        self.assertEqual(result["min_distance_m"],gap)

if __name__=="__main__":unittest.main()
