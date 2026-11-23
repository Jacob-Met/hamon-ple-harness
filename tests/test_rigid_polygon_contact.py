"""Independent boundary-integral forward loads. The production inverse sweeps material slabs."""
from fractions import Fraction as F
from math import comb, ulp
import unittest
import json
from pathlib import Path
import subprocess
import sys

from hamon_polygon_contact import verify_rigid_polygon_contact as verify


BOX=[(-2,-2),(2,-2),(2,2),(-2,2)]
HOLE=[(-1,-1),(1,-1),(1,1),(-1,1)]


def polygon_moment(poly,i,j):
    # Green: integral x^i y^j dA = line integral x^(i+1)y^j/(i+1) dy.
    total=F(0)
    for (x,y),(xx,yy) in zip(poly,poly[1:]+poly[:1]):
        dx,dy=xx-x,yy-y
        for r in range(i+2):
            for s in range(j+1):
                total+=F(comb(i+1,r)*comb(j,s),(i+1)*(r+s+1))*x**(i+1-r)*dx**r*y**(j-s)*dy**(s+1)
    return total


def oracle(rings,plane):
    a,b,c=map(F,plane)
    moments={(i,j):F(0) for i,j in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]}
    peak=0
    for ri,ring in enumerate(rings):
        ring=[tuple(map(F,p)) for p in ring]
        sign=(1 if polygon_moment(ring,0,0)>0 else -1)*(-1 if ri else 1)
        cut=[]
        for p,q in zip(ring,ring[1:]+ring[:1]):
            hp=a+b*p[0]+c*p[1]
            hq=a+b*q[0]+c*q[1]
            peak=max(peak,hp)
            if hp>=0:
                cut.append(p)
            if (hp>0 and hq<0) or (hp<0 and hq>0):
                cut.append(tuple((hp*y-hq*x)/(hp-hq) for x,y in zip(p,q)))
        for ij in moments:
            moments[ij]+=sign*polygon_moment(cut,*ij)
    def load(i,j):
        return a*moments[i,j]+b*moments[i+1,j]+c*moments[i,j+1]
    return tuple(map(float,(load(0,0),load(1,0),load(0,1),moments[0,0],peak)))


def spec(rings,n,mx,my):
    return dict(calculation='rigid_polygon_contact',outer_vertices_m=rings[0],
                hole_vertices_m=rings[1:],vertical_resultant_N=n,moment_x_Nm=mx,moment_y_Nm=my)


class PolygonContact(unittest.TestCase):
    def assert_plane(self,rings,plane):
        n,mx,my,area,peak=oracle(rings,plane)
        out=verify(spec(rings,n,mx,my))
        self.assertTrue(out['conservation_verified'],(plane,out))
        self.assertAlmostEqual(out['contact_area_m2'],area,delta=2e-8*area)
        self.assertAlmostEqual(out['q_max_Pa'],peak,delta=2e-8*peak)
        actual=out['pressure_plane']
        g=actual['gradient_Pa_per_m']
        ref=actual['reference_point_m']
        for ring in rings:
            for x,y in ring:
                got=actual['pressure_at_reference_Pa']+g[0]*(x-ref[0])+g[1]*(y-ref[1])
                self.assertAlmostEqual(got,float(plane[0]+plane[1]*x+plane[2]*y),delta=1e-7*peak)
        return out

    def test_square_hole_exact_controls(self):
        rings=[BOX,HOLE]
        base=spec(rings,12,0,0)
        full=verify(base)
        self.assertEqual((full['material_area_m2'],full['contact_area_m2'],full['q_max_Pa']),(12,12,1))
        cut=self.assert_plane(rings,(0,1,0))
        self.assertAlmostEqual(cut['contact_area_m2'],6)
        self.assertAlmostEqual(cut['integrated_moment_x_Nm'],10)
        self.assertFalse(verify({**base,'proposed_q_max_Pa':.75})['conservation_verified'])
        self.assertFalse(verify({**base,'proposed_contact_area_m2':16})['conservation_verified'])
        self.assertTrue(verify({**base,'proposed_material_area_m2':12})['conservation_verified'])

    def test_full_contact_preserves_material_sections_exactly(self):
        def intersections(ring, x):
            ring=[tuple(map(F,p)) for p in ring]
            return sorted(y+(v-y)*(x-a)/(b-a)
                          for (a,y),(b,v) in zip(ring,ring[1:]+ring[:1])
                          if min(a,b)<x<max(a,b))

        def check(rings):
            n,mx,my,_,_=oracle(rings,(6,0,0))
            out=verify(spec(rings,n,mx,my))
            self.assertTrue(out['conservation_verified'],out)
            self.assertTrue(out['output_geometry_verified'])
            shapes=[patch['vertices_m'] for patch in out['contact_patches']]
            for patch in out['contact_patches']:
                self.assertTrue(all(F(q)>0 for q in patch['vertex_pressure_Pa']))
            # Between consecutive x events each boundary ordinate is affine.
            # Equality at two interior sections establishes equality throughout.
            xs=sorted({F(p[0]) for r in rings+shapes for p in r})
            for a,b in zip(xs,xs[1:]):
                for x in ((2*a+b)/3,(a+2*b)/3):
                    edges=sorted(y for r in rings for y in intersections(r,x))
                    desired=list(zip(edges[::2],edges[1::2]))
                    sections=sorted(interval for r in shapes if (interval:=intersections(r,x)))
                    actual=[]
                    for lo,hi in sections:
                        if actual and actual[-1][1]==lo:
                            actual[-1]=(actual[-1][0],hi)
                        else:
                            actual.append((lo,hi))
                    self.assertEqual(actual,desired)

        outer=[(-4,-4),(1,-5),(4,-4),(4,4),(-4,4)]
        hole=[(0,0),(3,-1),(3,1)]
        for sx,dy,winding in [(1,0,1),(1,0,-1),(-1,0,1),(1,4096,1)]:
            rings=[[(sx*x,dy+y) for x,y in r][::winding] for r in [outer,hole]]
            with self.subTest(sx=sx,dy=dy,winding=winding):
                check(rings)
        for origin in (1,2**32,2**48):
            gap=ulp(origin)
            span=max(8,2**28*gap)
            rings=[[(-4,origin-span),(1,origin-2*span),(4,origin-span),
                    (4,origin+span),(-4,origin+span)],
                   [(0,origin),(3,origin),(3,origin+gap)]]
            with self.subTest(origin=origin):
                check(rings)
        control=self.assert_plane([BOX,HOLE],(6,0,0))
        self.assertTrue(all(p['encoding']=='binary64' for p in control['contact_patches']))

    def test_full_contact_heights_follow_the_reported_plane(self):
        # A rounded gradient and independently rounded endpoint heights can
        # pass an aggregate load check while describing two different planes.
        square=[[(-1,-1),(1,-1),(1,1),(-1,1)]]
        out=verify(spec(square,12,0.1,0))
        self.assertTrue(out['output_geometry_verified'])
        plane=out['pressure_plane']
        ox,oy=map(F,plane['reference_point_m'])
        gx,gy=map(F,plane['gradient_Pa_per_m'])
        for patch in out['contact_patches']:
            for (x,y),q in zip(patch['vertices_m'],patch['vertex_pressure_Pa']):
                expected=F(plane['pressure_at_reference_Pa'])+gx*(F(x)-ox)+gy*(F(y)-oy)
                self.assertEqual(F(q),expected)

    def test_common_field_on_disconnected_positive_islands(self):
        shape=[(-3,-2),(3,-2),(3,2),(1,2),(1,0),(-1,0),(-1,2),(-3,2)]
        out=self.assert_plane([shape],(-4,0,8))
        self.assertEqual(len(out['contact_patches']),2)
        intervals=[]
        plane=out['pressure_plane']
        ox,oy=map(F,plane['reference_point_m'])
        gx,gy=map(F,plane['gradient_Pa_per_m'])
        for patch in out['contact_patches']:
            points=[tuple(map(F,p)) for p in patch['vertices_m']]
            xs=[p[0] for p in points]
            intervals.append((min(xs),max(xs)))
            for (x,y),q in zip(points,patch['vertex_pressure_Pa']):
                expected=F(plane['pressure_at_reference_Pa'])+gx*(x-ox)+gy*(y-oy)
                self.assertEqual(F(q),expected)
        self.assertEqual(sorted(intervals),[(-3,-1),(1,3)])

    def test_exposed_planes_and_winding(self):
        shapes=[
            [BOX,HOLE],
            [[(-4,-3),(4,-3),(4,4),(-4,4)],[(-2,-1),(0,-2),(1,0),(-1,1)]],
            [[(-3,-3),(3,-3),(3,3),(1,3),(1,0),(-1,0),(-1,3),(-3,3)]],
            [[(-6,-4),(6,-4),(6,4),(-6,4)],[(-4,-2),(-2,-2),(-2,1),(-4,1)],[(1,-1),(3,-2),(4,1),(2,2)]]]
        for rings in shapes:
            for coeff in [(5,0,0),(4,F(1,2),F(-1,4)),(0,1,1),(-1,1,F(1,2)),(0,F(-3,2),1),(F(-7,4),0,1)]:
                for oriented in (rings,[r[1:]+r[:1] for r in rings],[list(reversed(r)) for r in rings]):
                    with self.subTest(rings=rings,coeff=coeff):
                        self.assert_plane(oriented,coeff)

    def test_origin_and_signed_scaling(self):
        rings=[[(1000+2*x,-2000+y) for x,y in r] for r in [BOX,HOLE]]
        self.assert_plane(rings,(-2500,F(1,2),-1))
        self.assert_plane(rings,(2500,F(-1,2),1))

    def test_feasibility_and_degeneracy(self):
        base=spec([BOX,HOLE],12,0,0)
        for n,mx,my in [(1,2,0),(1,2,2),(1,3,0),(0,1,0),(-1,0,0)]:
            self.assertEqual(verify(spec([BOX,HOLE],n,mx,my))['equilibrium_status'],'no_finite_equilibrium')
        self.assertIsNone(verify(spec([BOX,HOLE],0,0,0))['contact_area_m2'])
        for rings in [[[(0,0),(2,2),(0,2),(2,0)]],[BOX,BOX],[BOX,HOLE,HOLE],
                      [BOX,[(0,0),(1,0),(2,2)]],[BOX,[(5,5),(5,6),(6,5)]]]:
            self.assertFalse(verify(spec(rings,1,0,0))['ok'])
        self.assertFalse(verify({**base,'pressure_model':'one_plane_per_triangle'})['ok'])

    def test_normal_json_and_dual_entry(self):
        from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route
        from hamon_dual_verifier import run_dual_verify_spec
        base=spec([BOX,HOLE],7,10,0)
        result=route(base)
        self.assertTrue(result['conservation_verified'])
        self.assertTrue(run_dual_verify_spec(dict(mode='domain',kind='rigid_polygon_contact',spec=base))['dual_verified'])
        file=Path(__file__).resolve().parents[1]/'hamon_conservation_cad_verifier.py'
        cli=subprocess.run([sys.executable,str(file),'--json-stdin'],input=json.dumps(base),
                           capture_output=True,text=True,check=True)
        self.assertEqual(result,json.loads(cli.stdout))

    def test_serialized_narrow_sliver_retains_its_plane_and_vertices(self):
        rings=[[(1000+2*x,-2000+y) for x,y in r] for r in [BOX,HOLE]]
        out=self.assert_plane(rings,(-2500,F(1,2),-1))
        exact=[p for p in out['contact_patches'] if p['encoding']=='rational']
        self.assertTrue(exact)
        for patch in exact:
            points=[tuple(map(F,p)) for p in patch['vertices_m']]
            self.assertEqual(len(set(points)),len(points))
            self.assertGreater(polygon_moment(points,0,0),0)
            plane=out['pressure_plane']
            origin=list(map(F,plane['reference_point_m']))
            gx,gy=map(F,plane['gradient_Pa_per_m'])
            for (x,y),pressure in zip(points,patch['vertex_pressure_Pa']):
                self.assertEqual(F(pressure),F(plane['pressure_at_reference_Pa'])+gx*(x-origin[0])+gy*(y-origin[1]))

    def test_thin_edge_keeps_the_exact_input_centroid_slack(self):
        # These supplied binary64 moments have exact rational edge distances.
        # An absolute global-moment tolerance used to certify a 2.9e-7 peak
        # error at power 14, then refuse every narrower finite contact here.
        square=[(-1,-1),(1,-1),(1,1),(-1,1)]
        for power in (10,14,16,24,32,44):
            moment=1-2.**(-power)/3
            width=3*(1-F(moment))
            base=spec([square],1,moment,0)
            out=verify({**base,'proposed_q_max_Pa':float(1/width),
                        'proposed_contact_area_m2':float(2*width)})
            with self.subTest(power=power):
                self.assertTrue(out['conservation_verified'],out)
                self.assertAlmostEqual(out['contact_area_m2']/float(2*width),1,delta=2e-10)
                self.assertAlmostEqual(out['q_max_Pa']*float(width),1,delta=2e-10)
                wrong=verify({**base,'proposed_q_max_Pa':float(1/width)*(1+3e-8)})
                self.assertFalse(wrong['conservation_verified'],wrong)

    def test_thin_edge_is_affine_covariant_and_ignores_an_unloaded_hole(self):
        square=[(-1,-1),(1,-1),(1,1),(-1,1)]
        hole=[(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]
        transforms=[(1,0,0,1),(0,-1,1,0),(3,1,2,1),(-2,0,0,.5),
                    (3,2,1,3),(2,7,5,1)]
        for power,origin in ((14,(0,0)),(32,(1024,-2048)),(44,(0,0))):
            centroid=1-2.**(-power)
            width=3*(1-F(centroid))
            for a,b,c,d in transforms:
                det=abs(a*d-b*c)
                def transform(p):
                    x,y=p
                    return (origin[0]+a*x+b*y,origin[1]+c*x+d*y)
                for rings in ([square],[square,hole]):
                    moved=[[transform(p) for p in ring] for ring in rings]
                    mx,my=transform((centroid,0))
                    out=verify(spec(moved,1,mx,my))
                    with self.subTest(power=power,transform=(a,b,c,d),holes=len(rings)-1):
                        self.assertTrue(out['conservation_verified'],out)
                        self.assertAlmostEqual(out['contact_area_m2']/float(2*width*F(det)),1,delta=2e-9)
                        self.assertAlmostEqual(out['q_max_Pa']*float(width*F(det)),1,delta=2e-9)

    def test_thin_corner_uses_both_support_distances(self):
        square=[(-1,-1),(1,-1),(1,1),(-1,1)]
        for power in (4,14,24,36,44):
            centroid=1-2.**(-power)
            leg=4*(1-F(centroid))
            out=verify(spec([square],1,centroid,centroid))
            with self.subTest(power=power):
                self.assertTrue(out['conservation_verified'],out)
                self.assertAlmostEqual(out['contact_area_m2']/float(leg*leg/2),1,delta=2e-9)
                self.assertAlmostEqual(out['q_max_Pa']*float(leg*leg/6),1,delta=2e-9)

    def test_rational_oblique_plane_roundtrips_with_its_contact_geometry(self):
        centroid=1-2.**(-44)
        transform=lambda p:(2*p[0]+7*p[1],5*p[0]+p[1])
        square=[transform(p) for p in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        mx,my=transform((centroid,0))
        out=json.loads(json.dumps(verify(spec([square],1,mx,my))))
        self.assertTrue(out['conservation_verified'],out)
        plane=out['pressure_plane']
        self.assertEqual(plane.get('encoding'),'rational')
        gx,gy=map(F,plane['gradient_Pa_per_m'])
        rx,ry=map(F,plane['reference_point_m'])
        a=F(plane['pressure_at_reference_Pa'])-gx*rx-gy*ry
        total=[F(0),F(0),F(0)]
        for patch in out['contact_patches']:
            poly=[tuple(map(F,p)) for p in patch['vertices_m']]
            moments={(i,j):polygon_moment(poly,i,j)
                     for i,j in [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]}
            total[0]+=a*moments[0,0]+gx*moments[1,0]+gy*moments[0,1]
            total[1]+=a*moments[1,0]+gx*moments[2,0]+gy*moments[1,1]
            total[2]+=a*moments[0,1]+gx*moments[1,1]+gy*moments[0,2]
            for p,h in zip(poly,patch['vertex_pressure_Pa']):
                self.assertLessEqual(abs(a+gx*p[0]+gy*p[1]-F(h)),
                                     F(out['q_max_Pa'])*F('1e-8'))
        self.assertLess(abs(total[0]-1),F('1e-8'))
        # Inverse affine coordinates: u=(7*y-x)/33. Its edge slack is tiny.
        actual_u=(7*total[2]-total[1])/33
        residual=(total[0]-actual_u)-(1-F(centroid))
        self.assertLess(abs(residual),(1-F(centroid))*F('1e-8'))
