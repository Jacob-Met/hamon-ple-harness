"""Independent oriented cuboid flux, support and three-dimensional volume quadrature."""
import itertools
import math
import unittest
from mpmath import mp
from hamon_conservation_cad_verifier import verify_step_cad_geometry


def block(lengths, axis, ref, origin):
    def coords(v):
        return "("+",".join(v)+")"
    return ("DATA;#1=CARTESIAN_POINT('',"+coords(origin)+");"
            "#2=DIRECTION('',"+coords(axis)+");#3=DIRECTION('',"+coords(ref)+");"
            "#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);#5=BLOCK('',#4,"+
            ",".join(lengths)+");ENDSEC;")


class OrientedBlockVolumeCubature(unittest.TestCase):
    def close(self,actual,ref):
        target=float(ref)
        self.assertLessEqual(abs(actual-target),3*math.ulp(target),(actual,target))

    def test_corner_support_and_volume_quadrature_from_declared_material_frame(self):
        ctx=mp.clone()
        ctx.dps=195
        def cross(a,b):
            return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def dot(a,b):
            return ctx.fsum(x*y for x,y in zip(a,b))
        def norm(a):
            return ctx.sqrt(dot(a,a))
        def unit(a):
            mag=norm(a)
            return [x/mag for x in a]
        for dims,axis,ref,origin,scale,rho in (
            (("1.5","3.125","4.75"),("3","-4","12"),("1","3",".25"),("1e16","-7","3.5"),.125,1.75),
            (("2e38","1e-11","3e-12"),("2","-3","6"),("5","-2","1"),("0","0","0"),.25,1.25),
            ((".75","2.25","9.5"),("3e-150","4e-150","12e-150"),
             ("6","8","24.0000000000000000000000000000001"),("-100","20","-.5"),1.,.5),
            (("1e-8","3e-8","2e-8"),("-7","-1","2"),("2","3","4"),("3e10","4e10","1e10"),.5,0.)):
            with self.subTest(dims=dims,axis=axis):
                out=verify_step_cad_geometry({"step_text":block(dims,axis,ref,origin),
                                              "unit_scale_m":scale,"density_kg_m3":rho})
                self.assertTrue(out["conservation_verified"],out)
                z=unit(list(map(ctx.mpf,axis)))
                y=unit(cross(z,list(map(ctx.mpf,ref))))
                x=unit(cross(y,z))
                axes=(x,y,z)
                edges=[[ctx.mpf(d)*scale*k for k in vector] for d,vector in zip(dims,axes)]
                volume=abs(dot(edges[0],cross(edges[1],edges[2])))
                mass=volume*rho
                self.close(out["volume_m3"],volume)
                self.close(out["mass_kg"],mass)
                self.close(out["surface_area_m2"],2*sum(norm(cross(edges[i],edges[j]))
                                                       for i,j in ((0,1),(0,2),(1,2))))
                # Direct eight-corner support of the translated material.
                corners=[[ctx.mpf(origin[i])*scale+ctx.fsum(s[j]*edges[j][i] for j in range(3))
                          for i in range(3)] for s in itertools.product((0,1),repeat=3)]
                for i in range(3):
                    self.close(out["centroid_m"][i],ctx.fsum(p[i] for p in corners)/8)
                    self.close(out["aabb_extents_m"][i],max(p[i] for p in corners)-min(p[i] for p in corners))
                # Product two-node Gauss cubature exactly integrates every mass
                # tensor component, without using a diagonal box inertia formula.
                nodes=[[ctx.fsum(s[j]*edges[j][i] for j in range(3))/ctx.sqrt(12)
                        for i in range(3)] for s in itertools.product((-1,1),repeat=3)]
                for i in range(3):
                    for j in range(3):
                        exact=mass/8*ctx.fsum(sum(p[k]**2 for k in range(3) if k!=i)
                                             if i==j else -p[i]*p[j] for p in nodes)
                        self.close(out["inertia_tensor_centroid_kg_m2"][i][j],exact)
                principal=sorted(mass/8*ctx.fsum(dot(cross(p,v),cross(p,v)) for p in nodes)
                                 for v in axes)
                for value,reference in zip(out["principal_moments_kg_m2"],principal):
                    self.close(value,reference)

    def test_equal_material_edges_cancel_irrational_world_centroid_terms(self):
        text=block(("1.","2.","1."),("3","3","0"),("4","-4","0"),("0","0","0"))
        out=verify_step_cad_geometry({"step_text":text,"unit_scale_m":1.,"density_kg_m3":1.})
        self.assertTrue(out["conservation_verified"],out)
        self.assertEqual(out["centroid_m"][1],0.)
        self.assertEqual(out["inertia_tensor_centroid_kg_m2"][0][1],0.)
