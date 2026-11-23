"""Weak-axis magnitude on partial contact, using independent exact sections."""
from fractions import Fraction as F
from itertools import product
import unittest

from hamon_rectangle_contact import verify_rigid_rectangle_contact as solve


def section_integrals(boundary, pressure_tilt):
    """Integrate the displayed trapezoid, split on its top-right fan diagonal."""
    totals = [F(0)] * 3
    for y, weight in zip(map(F, [-1, '-1/3', '1/3', 1]), [1, 3, 3, 1]):
        t = (1+y)/2
        left, middle, right = -boundary*y, boundary+(1-boundary)*t, F(1)
        peak = F(3,2)*(1+pressure_tilt)
        field = [(left, F(0)), (middle, t*peak), (right, F(3,2)*(1+pressure_tilt*y))]
        for (a,p), (b,q) in zip(field, field[1:]):
            force = (b-a)*(p+q)/2
            x_moment = (b-a)*((2*a+b)*p+(a+2*b)*q)/6
            for j, value in enumerate((force, x_moment, y*force)):
                totals[j] += weight*value/4  # cubic-exact 3/8 rule over [-1, 1]
    return totals


class ClippedRectangleMoments(unittest.TestCase):
    def test_correct_support_and_sign_do_not_establish_weak_magnitude(self):
        e = F(1,2**42)
        self.assertEqual(section_integrals(e,e), [F(3,2)+e*e/2, 1, e])
        panels = [(1,1,1), (1,1024,F(1025,2)), (1024,1,F(1025,2)), (1024,1024,1024)]
        for boundary, pressures, ratio in panels:
            self.assertEqual(section_integrals(boundary*e, pressures*e)[2], ratio*e)

        for swap,sx,sy,center in product((False,True), (-1,1), (-1,1), ((0,0),(4,-2))):
            def orient(x,y):
                return (sy*y,sx*x) if swap else (sx*x,sy*y)

            def point(x,y):
                return [float(c+d) for c,d in zip(center,orient(x,y))]

            dx,dy = orient(F(1), e)
            spec = dict(base_width_m=2,base_length_m=2,vertical_resultant_N=1.5,
                        base_center_m=center,moment_x_Nm=float(F(3,2)*center[0]+dx),
                        moment_y_Nm=float(F(3,2)*center[1]+dy))
            result = solve(spec)
            self.assertEqual(result['footprint_shape'],'edge_trapezoid')
            self.assertTrue(result['conservation_verified'])
            for boundary,pressures,ratio in panels:
                a,b = boundary*e,pressures*e
                polygon = [point(1,1),point(-a,1),point(a,-1),point(1,-1)]
                loads = [float(F(3,2)*(1+b)),0.,0.,float(F(3,2)*(1-b))]
                claim = dict(proposed_contact_polygon_m=polygon,
                             proposed_contact_vertex_pressure_Pa=loads)
                with self.subTest(swap=swap,sx=sx,sy=sy,center=center,ratio=ratio):
                    certified = solve({**spec,**claim})
                    self.assertEqual(certified['conservation_verified'],ratio==1,certified)
            for factor in (1,1024):
                corners = {orient(x,y):float(F(3,2)*max(0,x+factor*e*y))
                           for x,y in product((-1,1),repeat=2)}
                claim = [corners[p] for p in [(-1,-1),(1,-1),(1,1),(-1,1)]]
                self.assertEqual(solve({**spec,'proposed_corner_pressure_Pa':claim})
                                 ['conservation_verified'],factor==1)

    def test_all_clipped_shapes_keep_their_genuine_fields(self):
        for shape,moments in [('corner_triangle',(2,2)),('edge_trapezoid',(2,0.5)),
                              ('clipped_pentagon',(1.6,0.8))]:
            for sx,sy in product((-1,1),repeat=2):
                spec = dict(base_width_m=2,base_length_m=2,vertical_resultant_N=4,
                            moment_x_Nm=sx*moments[0], moment_y_Nm=sy*moments[1])
                ref = solve(spec)
                self.assertEqual(ref['footprint_shape'],shape)
                for reverse in (False,True):
                    order = list(range(len(ref['contact_polygon_m'])))
                    if reverse:
                        order.reverse()
                    claim = dict(proposed_contact_polygon_m=[ref['contact_polygon_m'][i] for i in order],
                                 proposed_contact_vertex_pressure_Pa=[ref['contact_vertex_pressure_Pa'][i] for i in order],
                                 proposed_corner_pressure_Pa=ref['corner_pressure_Pa'])
                    self.assertTrue(solve({**spec,**claim})['conservation_verified'])
