"""Independent physical-coordinate vertical integration; no solver clipping/inverse helpers."""
from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
import unittest

from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier as route
from hamon_dual_verifier import run_dual_verify_spec


def forward(vertices, heights):
    """Integrate positive affine pressure over vertical sections with an exact open cubic quadrature."""
    vertices = [tuple(map(F, xy)) for xy in vertices]
    heights = list(map(F, heights))
    (x0,y0),(x1,y1),(x2,y2) = vertices
    det = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
    h0,h1,h2 = heights
    gx = ((h1-h0)*(y2-y0)-(h2-h0)*(y1-y0))/det
    gy = ((x1-x0)*(h2-h0)-(x2-x0)*(h1-h0))/det
    def pressure(x,y):
        return h0+gx*(x-x0)+gy*(y-y0)
    edges = list(zip(vertices,vertices[1:]+vertices[:1]))
    breaks = {x for x,y in vertices}
    for (a,b), (c,d) in edges:
        p,q = pressure(a,b),pressure(c,d)
        if p*q < 0:
            breaks.add(a+(c-a)*p/(p-q))
    def section(x):
        values = []
        for (a,b),(c,d) in edges:
            if a == c and x == a:
                values.extend((b,d))
            elif min(a,c) <= x <= max(a,c) and a != c:
                values.append(b+(x-a)*(d-b)/(c-a))
        lo,hi = min(values),max(values)
        p,q = pressure(x,lo),pressure(x,hi)
        if max(p,q) <= 0:
            return (F(0),)*4
        if p*q < 0:
            zero = lo+(hi-lo)*p/(p-q)
            if p < 0: lo,p=zero,F(0)
            else: hi,q=zero,F(0)
        length=hi-lo
        force=length*(p+q)/2
        return length,force,x*force,length*((2*lo+hi)*p+(lo+2*hi)*q)/6
    out=[F(0)]*4
    cuts=sorted(breaks)
    for left,right in zip(cuts,cuts[1:]):
        # At an entire zero-pressure vertical edge, the positive area at that
        # exact abscissa is measure zero. Use four interior nodes so the area
        # quadrature uses this slab's trace on both sides of a vertical break.
        length=right-left
        nodes=(F(1,8),F(3,8),F(5,8),F(7,8))
        weights=(F(13,48),F(11,48),F(11,48),F(13,48))
        for t,w in zip(nodes,weights):
            vals=section(left+length*t)
            out=[total+length*w*v for total,v in zip(out,vals)]
    return out  # area, force, two global first moments


def payload(v, n, mx, my):
    return dict(calculation='rigid_triangle_contact',base_vertices_m=[[float(x),float(y)] for x,y in v],
                vertical_resultant_N=float(n),moment_x_Nm=float(mx),moment_y_Nm=float(my))


class RigidTriangleContact(unittest.TestCase):
    vertices=((0,0),(3,0),(0,3))

    def test_bbox_false_equilibrium_and_wrong_uniform_pressure(self):
        uniform=route(payload(self.vertices,9,9,9))
        self.assertTrue(uniform['conservation_verified'],uniform)
        self.assertEqual((uniform['contact_area_m2'],uniform['q_max_Pa']), (4.5,2.))
        self.assertEqual(uniform['corner_pressure_Pa'],[2.,2.,2.])
        for mx,my in ((18,18),(18,9),(27,0),(0,9)):
            self.assertEqual(route(payload(self.vertices,9,mx,my))['equilibrium_status'],'no_finite_equilibrium')
        rectangle=route(dict(calculation='rigid_rectangle_contact',base_width_m=3,base_length_m=3,
                             base_center_m=[1.5,1.5],vertical_resultant_N=9,moment_x_Nm=18,moment_y_Nm=18))
        self.assertTrue(rectangle['conservation_verified'])

    def test_contact_regimes_permutations_and_coordinates(self):
        shapes=(((0,0),(3,0),(0,3)), ((-4,1),(1,5),(6,-2)),
                ((3,-2),(3,F(-15,8)),(11,-2)))
        fields=((3,7,9),(0,3,7),(-2,4,7),(-3,-5,8),(-2,0,6),(0,0,5),(-32,1,3))
        for v in shapes:
            for h in fields:
                # Exercise six nodal orderings, two-sided transitions and all
                # physical corners via three independent cyclic field rotations.
                for shift in range(3):
                    hs=h[shift:]+h[:shift]
                    area,n,mx,my=forward(v,hs)
                    for order in permutations(range(3)):
                        spec=payload([v[i] for i in order],n,mx,my)
                        out=route(spec)
                        self.assertTrue(out['conservation_verified'],(v,hs,order,out))
                        self.assertAlmostEqual(out['contact_area_m2']/float(area),1.,delta=2e-12)
                        self.assertAlmostEqual(out['q_max_Pa']/max(hs),1.,delta=2e-12)
                        for j,i in enumerate(order):
                            self.assertAlmostEqual(out['corner_pressure_Pa'][j],max(0,hs[i]),delta=8e-12)
                        for i,k in enumerate(('integrated_vertical_resultant_N','integrated_moment_x_Nm','integrated_moment_y_Nm')):
                            self.assertAlmostEqual(out[k],float((n,mx,my)[i]),delta=2e-11*float(n))

    def test_transition_topology_both_sides(self):
        v=self.vertices
        for h, expected in [((0,1,2),'full_triangle'),((F(1,128),1,2),'full_triangle'),
                            ((F(-1,128),1,2),'two_positive_corners'),
                            ((-2,F(1,128),3),'two_positive_corners'),
                            ((-2,0,3),'one_positive_corner'),((-2,F(-1,128),3),'one_positive_corner')]:
            for shift in range(3):
                hs=h[shift:]+h[:shift]
                area,n,mx,my=forward(v,hs)
                out=route(payload(v,n,mx,my))
                self.assertTrue(out['conservation_verified'],out)
                self.assertAlmostEqual(out['contact_area_m2']/float(area),1.,delta=1e-12)
                # At an exact zero, the fractions supplied as binary64 may cross
                # the boundary. Use exact binary force/moment at contact below.
                if 0 not in hs:
                    self.assertEqual(out['footprint_shape'],expected)
        self.assertEqual(route(payload(v,24,18,18))['footprint_shape'],'full_triangle')
        self.assertEqual(route(payload(v,24,9,18))['footprint_shape'],'one_positive_corner')

    def test_near_release_retains_both_loaded_corners_and_distinct_cuts(self):
        vertices = ((0,0),(1,0),(0,1))
        for n, mx, my in ((9007199254740991, 5629499534213119, 2251799813685248),
                          (7.999999999999999, 4.999999999999999, 2.0)):
            with self.subTest(n=n):
                result = route(payload(vertices, n, mx, my))
                self.assertTrue(result['conservation_verified'], result)
                self.assertTrue(result['topology_verified'])
                self.assertEqual(result['footprint_shape'], 'two_positive_corners')
                self.assertEqual(len({tuple(xy) for xy in result['contact_polygon_m']}), 4)
                corners = result['corner_pressure_Pa']
                self.assertEqual(corners[0], 0)
                self.assertGreater(corners[1], corners[2])
                self.assertGreater(corners[2], 0)
                plane = result['pressure_plane']
                peak = F(plane['pressure_at_reference_Pa'])
                ref = tuple(map(F, plane['reference_point_m']))
                gradient = tuple(map(F, plane['gradient_Pa_per_m']))
                signed = [peak + sum(g*(F(x)-r) for g,x,r in zip(gradient,v,ref))
                          for v in vertices]
                self.assertLess(signed[0], 0)
                self.assertGreater(min(signed[1:]), 0)
        boundary = route(payload(vertices, 8, 5, 2))
        self.assertTrue(boundary['conservation_verified'], boundary)
        self.assertEqual(boundary['footprint_shape'], 'one_positive_corner')
        self.assertEqual(len(boundary['contact_polygon_m']), 3)

    def test_unresolved_release_keeps_balance_and_refuses_regime_claim(self):
        # The rounded cuts of this exact input cannot describe a four-sided
        # contact patch. The earlier 378-case quantitative course includes it.
        spec = payload(((-4,1),(1,5),(6,-2)), 45.833333333333336, 103.125, 22.916666666666668)
        out = route(spec)
        self.assertTrue(out['conservation_verified'])
        self.assertFalse(out['topology_verified'])
        self.assertEqual(out['footprint_shape'], 'release_boundary_unresolved')
        for proposed in ('full_contact', 'partial_contact', 'boundary_unresolved'):
            result = route(dict(spec, proposed_contact_regime=proposed))
            self.assertFalse(result['claim_checks']['proposed_contact_regime'])
            self.assertFalse(result['conservation_verified'])

    def test_skinny_and_unresolved_geometry(self):
        v=((0,0),(4,4),(8,8+F(1,2**20)))
        # Use N=3 so both physical first moments of the uniform field are the
        # exact binary sums of vertex positions even for an oblique skinny base.
        for ordered in permutations(v):
            out=route(payload(ordered,3,12,12+F(1,2**20)))
            self.assertTrue(out['conservation_verified'],out)
            self.assertEqual(out['contact_area_m2'],2**-19)
            self.assertEqual(out['q_max_Pa'],3*2**19)
        self.assertFalse(route(payload(((0,0),(4,4),(8,8)),3,12,12))['ok'])
        # The three base vertices are distinguishable, but the contact
        # intersection at offset 4/9 m cannot be rendered faithfully here.
        offset=2**48
        translated=route(payload(((offset,0),(offset+3,0),(offset,3)),9,9*offset+1,2))
        self.assertEqual(translated['equilibrium_status'],'finite')
        self.assertFalse(translated['output_geometry_verified'])
        self.assertFalse(translated['conservation_verified'])
        too_large=2**54
        rounded=payload(((too_large,0),(too_large+1,0),(too_large,3)),9,9*too_large,9)
        self.assertFalse(route(rounded)['conservation_verified'])

    def test_check_out_of_process_json_cli(self):
        script=Path(__file__).resolve().parents[1]/'hamon_conservation_cad_verifier.py'
        for values in ((9,9,9),(9,18,18)):
            spec=payload(self.vertices,*values)
            call=subprocess.run([sys.executable,str(script),'--json-stdin'],input=json.dumps(spec),
                                text=True,capture_output=True,check=True)
            self.assertEqual(json.loads(call.stdout),route(spec))

    def test_unloaded_and_proposed_claims(self):
        v=self.vertices
        self.assertEqual(route(payload(v,0,0,0))['contact_regime'],'unloaded')
        for n,mx,my in ((-1,0,0),(0,1,0)):
            self.assertEqual(route(payload(v,n,mx,my))['equilibrium_status'],'no_finite_equilibrium')
        spec=payload(v,9,9,9)
        out=route(spec)
        self.assertTrue(route({**spec,'proposed_contact_polygon_m':out['contact_polygon_m'][::-1],
                               'proposed_contact_vertex_pressure_Pa':[2.]*3})['conservation_verified'])
        for claim in ({'proposed_q_max_Pa':3.304421935079097},
                      {'proposed_contact_area_m2':7.246876144702549},
                      {'proposed_contact_polygon_m':[[0,0],[3,0],[3,3],[0,3]]},
                      {'proposed_contact_vertex_pressure_Pa':[2.,2.,2.]}):
            self.assertFalse(route({**spec,**claim})['conservation_verified'])
        converted={**spec,'base_vertices_m':[['0*m','0*m'],['3000*mm','0*m'],['0*m','300*cm']]}
        self.assertEqual(route(converted),out)
        d=run_dual_verify_spec(dict(mode='domain',kind='rigid_triangle_contact',spec=spec))
        self.assertTrue(d['dual_verified'],d)


if __name__ == '__main__':
    unittest.main()
