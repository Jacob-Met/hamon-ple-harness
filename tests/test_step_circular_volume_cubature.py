"""Independent volume cubature and support checks for canonical curved STEP primitives."""
import math
import unittest
from mpmath import mp
from hamon_conservation_cad_verifier import verify_step_cad_geometry


def cylinder(h,r,axis,origin,rooted):
    ref = ("DATA;#1=CARTESIAN_POINT('',("+",".join(origin)+"));"
           "#2=DIRECTION('',("+",".join(axis)+"));#3=AXIS1_PLACEMENT('',#1,#2);"
           f"#4=RIGHT_CIRCULAR_CYLINDER('',#3,{h},{r});")
    return ref + ("#5=CSG_SOLID('',#4);" if rooted else "") + "ENDSEC;"


def sphere(r,origin,rooted):
    return ("DATA;#1=CARTESIAN_POINT('',("+",".join(origin)+"));"
            f"#2=SPHERE('',{r},#1);" + ("#3=CSG_SOLID('',#2);" if rooted else "")+"ENDSEC;")


class CircularVolumeCubature(unittest.TestCase):
    def close(self, reported, exact):
        ref = float(exact)
        self.assertLessEqual(abs(reported-ref), 3*math.ulp(ref), (reported,ref))

    def test_cylinders_from_axial_gauss_and_rotated_disk_cubature(self):
        ctx = mp.clone()
        ctx.dps = 190
        def dot(a,b):
            return ctx.fsum(x*y for x,y in zip(a,b))
        def cross(a,b):
            return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def normal(a):
            n = ctx.sqrt(dot(a,a))
            return [x/n for x in a]
        for height,radius,axis,origin,scale,rho,rooted in (
            ("7.5",".75",("-2","3","-6"),("8","-3","1"),1.,6.25,True),
            ("2.5","1.25",("1","-4","2"),("-4","7","-1"),.03125,2.,False),
            ("1e36","1e-12",("3","-6","2"),("5","-6","0"),1.,.125,False),
            ("2","3",("-4","2","-4"),("1","2","3"),.125,0.,True)):
            with self.subTest(axis=axis,r=radius,scale=scale,rho=rho):
                out=verify_step_cad_geometry({"step_text":cylinder(height,radius,axis,origin,rooted),
                                              "unit_scale_m":scale,"density_kg_m3":rho})
                self.assertTrue(out["conservation_verified"],out)
                h,r = ctx.mpf(height)*scale, ctx.mpf(radius)*scale
                n=normal(list(map(ctx.mpf,axis)))
                u=normal(cross(n,[0,0,1]))
                v=cross(n,u)
                volume=ctx.pi*r*r*h
                mass=volume*rho
                self.close(out["volume_m3"],volume)
                self.close(out["mass_kg"],mass)
                self.close(out["surface_area_m2"],2*ctx.pi*r*(r+h))
                for i in range(3):
                    self.close(out["centroid_m"][i],ctx.mpf(origin[i])*scale+n[i]*h/2)
                    # Circle support from the two *transverse* basis coordinates.
                    self.close(out["aabb_extents_m"][i],abs(n[i])*h+2*r*ctx.sqrt(u[i]**2+v[i]**2))
                # Equal masses on the stated disk ring integrate quadratic
                # x,y monomials exactly. Two Gauss planes integrate z².
                nodes = [[ax*h*n[i]/ctx.sqrt(12)+sg*r*t[i]/ctx.sqrt(2) for i in range(3)]
                         for ax in (-1,1) for t in (u,v) for sg in (-1,1)]
                for i in range(3):
                    for j in range(3):
                        inertia = mass/8*ctx.fsum(sum(p[k]**2 for k in range(3) if k!=i)
                                       if i==j else -p[i]*p[j] for p in nodes)
                        self.close(out["inertia_tensor_centroid_kg_m2"][i][j],inertia)
                # Project each actual mass node onto the plane perpendicular
                # to the requested rotation axis. This retains a slender
                # cylinder's weak moment independent of world matrix rounding.
                principal = []
                for axis_vector in (n,u,v):
                    principal.append(mass/8*ctx.fsum(dot(cross(p,axis_vector),cross(p,axis_vector))
                                                        for p in nodes))
                for reported,exact in zip(out["principal_moments_kg_m2"],sorted(principal)):
                    self.close(reported,exact)

    def test_spherical_surface_flux_and_radial_volume_quadrature(self):
        ctx=mp.clone()
        ctx.dps=150
        for radius,origin,scale,rho,rooted in (
            ("2.75",("-8","4","-5"),1.,2.25,True),
            (".0625",("1e14","-1e14","3"),.25,0.,False),
            ("1e-50",("0","0","0"),1.,3.,False)):
            with self.subTest(r=radius,scale=scale,rho=rho):
                out=verify_step_cad_geometry({"step_text":sphere(radius,origin,rooted),
                                               "unit_scale_m":scale,"density_kg_m3":rho})
                self.assertTrue(out["conservation_verified"],out)
                r=ctx.mpf(radius)*scale
                surface=4*ctx.pi*r*r
                volume=surface*r/3
                mass=rho*volume
                self.close(out["volume_m3"],volume)
                self.close(out["surface_area_m2"],surface)
                self.close(out["mass_kg"],mass)
                # Six axis nodes at the volume-mean square radius. Four
                # have nonzero distance to each Cartesian rotation axis.
                radius_squared_mean=ctx.mpf(3)/5*r*r
                expected=4*(mass/6)*radius_squared_mean
                for i in range(3):
                    self.close(out["principal_moments_kg_m2"][i],expected)
                    self.close(out["centroid_m"][i],ctx.mpf(origin[i])*scale)
                    self.close(out["aabb_extents_m"][i],2*r)
                    for j in range(3):
                        self.close(out["inertia_tensor_centroid_kg_m2"][i][j],expected if i==j else 0)
