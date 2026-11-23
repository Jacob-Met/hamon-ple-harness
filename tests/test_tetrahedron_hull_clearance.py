"""Exact convex-hull distance between two closed tetrahedra at registered IFC terminals."""
import itertools
import math
import unittest
from fractions import Fraction as F
from decimal import Decimal, localcontext

from hamon_ifc_spatial_clash import run_router_ifc_clash


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def linear(a,b):
    """Exact solve; dependent vertex subsets are covered by lower-dimensional ones."""
    rows = [list(row)+[v] for row,v in zip(a,b)]
    for k in range(len(b)):
        p = next((i for i in range(k,len(b)) if rows[i][k]),None)
        if p is None:
            return None
        rows[k],rows[p] = rows[p],rows[k]
        scale = rows[k][k]
        rows[k] = [v/scale for v in rows[k]]
        for i in range(len(b)):
            if i != k:
                scale = rows[i][k]
                rows[i] = [v-scale*w for v,w in zip(rows[i],rows[k])]
    return [row[-1] for row in rows]


def hull_projection(a,b):
    # Every displacement between the two convex solids belongs to the convex
    # hull of pairwise vertex differences. Project the origin onto affine
    # vertex subsets. Nonnegative weights and a supporting half-space for
    # every vertex prove that a candidate is a GLOBAL nearest displacement.
    cloud = list(dict.fromkeys(sub(p,q) for p in a for q in b))
    for n in (1,2,3,4):
        for vertices in itertools.combinations(cloud,n):
            p = vertices[0]
            e = [sub(v,p) for v in vertices[1:]]
            weights = linear([[dot(s,t) for t in e] for s in e],[-dot(s,p) for s in e])
            if weights is None or min([1-sum(weights),*weights])<0:
                continue
            w = tuple(x+sum(t*d[j] for t,d in zip(weights,e)) for j,x in enumerate(p))
            if all(dot(w,sub(x,w))>=0 for x in cloud):
                return w
    raise AssertionError("no convex projection")


def inside(point, tetra):
    edges = [sub(q,tetra[0]) for q in tetra[1:]]
    gram = [[dot(s,t) for t in edges] for s in edges]
    w = linear(gram,[dot(s,sub(point,tetra[0])) for s in edges])
    return [1-sum(w),*w]


def ifc(a,b,incoherent_outer=False):
    rows = []
    for index,tet in enumerate((a,b)):
        n = 10*index+1
        points = "("+",".join("("+",".join(str(float(x)) for x in p)+")" for p in tet)+")"
        ab,ac,ad = [sub(x,tet[0]) for x in tet[1:]]
        triple = ab[0]*(ac[1]*ad[2]-ac[2]*ad[1])+ab[1]*(ac[2]*ad[0]-ac[0]*ad[2])+ab[2]*(ac[0]*ad[1]-ac[1]*ad[0])
        faces = [(1,3,2),(1,2,4),(1,4,3),(2,3,4)]
        if triple<0:
            faces = [tuple(reversed(f)) for f in faces]
        if incoherent_outer and index == 0:
            faces[0] = tuple(reversed(faces[0]))
        triangles = "("+",".join("("+",".join(map(str,f))+")" for f in faces)+")"
        rows += [f"#{n}=IFCCARTESIANPOINTLIST3D({points});",
                 f"#{n+1}=IFCTRIANGULATEDFACESET(#{n},$,.T.,{triangles},$);",
                 f"#{n+2}=IFCSHAPEREPRESENTATION($,'Body','Tessellation',(#{n+1}));",
                 f"#{n+3}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{n+2}));",
                 f"#{n+4}=IFCBUILDINGELEMENTPROXY('solid{index}',$,'solid{index}',$,$,$,#{n+3},$,$);"]
    return "DATA;"+"".join(rows)+"ENDSEC;"


class TetrahedronHullClearance(unittest.TestCase):
    def test_exact_displacement_regimes_and_attained_world_witness(self):
        # A's high ridge and B's low ridge are transverse. A nonaxial
        # orthogonal transform makes their world AABBs overlap.
        a = [(-2,0,0),(2,0,0),(0,1,-1),(0,-1,-1)]
        ridge = [(0,-2,0),(0,2,0),(-1,0,1),(1,0,1)]
        skew = [(-1,-2,0),(1,2,.125),(-1,0,1),(1,0,1.25)]
        face = [(-2,-1,0),(2,-1,0),(0,2,0),(0,0,-1)]
        vertex = [(.25,.125,0),(-1,-1,1),(1,-1,1),(0,1,1)]
        for shape,other,gap,exp,translation in (
                (a,ridge,.125,-4,0), (a,skew,.125,-12,0),
                (face,vertex,.25,-5,8), (a,ridge,2.**-35,-5,0),
                (a,ridge,0.,-5,0), (a,ridge,-.125,-3,0), (a,ridge,.125,-600,0)):
            def world(p,shift):
                x,y,z = p
                z += shift
                t = translation
                return tuple(F(float(v)) for v in
                             (t+2.**exp*(x+2*y+2*z),-t+2.**exp*(2*x+y-2*z),2*t+2.**exp*(-2*x+2*y-z)))
            s,t = [world(p,0) for p in shape],[world(p,gap) for p in other]
            delta = hull_projection(s,t)
            d2 = dot(delta,delta)
            with localcontext() as context:
                context.prec = 90
                d = float((Decimal(d2.numerator)/Decimal(d2.denominator)).sqrt())
            requireds = [0.] if not d2 else [math.nextafter(d,0.),d,math.nextafter(d,math.inf)]
            for required in requireds:
                with self.subTest(gap=gap,exp=exp,threshold=required):
                    result = run_router_ifc_clash(dict(ifc_text=ifc(s,t),required_clearance_m=required,
                                                       cross_discipline_only=False))
                    self.assertTrue(result["ok"],result)
                    pair, = result["all_pairs"]
                    expected = "hard_clash" if not d2 else "clearance_violation" if F(required)**2>d2 else "clear"
                    self.assertEqual(pair["status"],expected,pair)
                    self.assertLessEqual(abs(pair["min_distance_m"]-d),2*math.ulp(d))
                    wa,wb = [tuple(map(F,p)) for p in pair["closest_points_m"]]
                    # Rounded endpoints must actually lie on/in the submitted
                    # materials. Local scale here is at least 2^-12.
                    self.assertGreaterEqual(min(inside(wa,s)+inside(wb,t)),-F(1,10**11))
                    max_coord = max(abs(float(v)) for p in s+t for v in p)
                    for q,ref in zip(sub(wa,wb),delta):
                        self.assertLessEqual(abs(float(q-ref)),8*math.ulp(max_coord))

    def test_nested_solids_do_not_require_representable_aabb_volume(self):
        # The smaller tetrahedron is strictly inside all four half-spaces of
        # the larger. Its world AABB volume rounds to zero, independently
        # of the validity and representability of its vertex coordinates.
        for exponent, incoherent in ((-375,False),(-600,False),(-30,True)):
            k = F(2)**exponent
            outer = [tuple(k*F(x) for x in p) for p in
                     ((0,0,0),(4,0,0),(0,4,0),(0,0,4))]
            inner = [tuple(k*F(x,2) for x in p) for p in
                     ((1,1,1),(2,1,1),(1,2,1),(1,1,2))]
            for p in inner:
                self.assertGreater(min(inside(p,outer)),0)
            result = run_router_ifc_clash(dict(ifc_text=ifc(outer,inner,incoherent),required_clearance_m=0.,
                                               cross_discipline_only=False))
            pair, = result["all_pairs"]
            self.assertEqual(pair["status"],"hard_clash",pair)
            self.assertEqual(pair["min_distance_m"],0.)
            witness = tuple(map(F,pair["witness_point_m"]))
            self.assertGreaterEqual(min(inside(witness,outer)+inside(witness,inner)),0)
