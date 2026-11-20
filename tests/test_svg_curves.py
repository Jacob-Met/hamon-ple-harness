"""Exact transform-aware SVG centerline extrema and falsification controls."""
import math
import unittest
import hamon_drawing as d

I=(1.,0.,0.,1.,0.,0.)

class SvgCurveBounds(unittest.TestCase):
    def assert_bbox(self,received,expected,delta=1e-8):
        self.assertIsNotNone(received)
        for a,b in zip(received,expected):
            self.assertAlmostEqual(a,b,delta=delta)
    def test_cubic_has_interior_extremum_not_control_hull(self):
        self.assert_bbox(d._svg_curve_bbox(
            'M 0 0 C 0 10 10 10 10 0',I),[0,0,10,7.5])
        self.assertIsNone(d._path_points('M 0 0 C 0 10 10 10 10 0'))
    def test_affine_rotated_cubic_extrema_recomputed(self):
        rotated=(0,1,-1,0,5,10)
        self.assert_bbox(d._svg_curve_bbox(
            'M 0 0 C 0 10 10 10 10 0',rotated),[-2.5,10,5,20])
    def test_quadratic_and_smooth_curve(self):
        self.assert_bbox(d._svg_curve_bbox(
            'M 0 0 Q 10 20 20 0',I),[0,0,20,10])
        self.assert_bbox(d._svg_curve_bbox(
            'M 0 0 Q 10 20 20 0 T 40 0',I),[0,-10,40,10])
        got=d._svg_curve_bbox('M 0 0 C 0 10 10 10 10 0 S 20 -10 20 0',I)
        self.assertEqual(got[0],0);self.assertEqual(got[2],20)
        self.assertLess(got[1],0)
    def test_relative_and_close(self):
        self.assert_bbox(d._svg_curve_bbox(
            'm 5 5 c 0 10 10 10 10 0 z',I),[5,5,15,12.5])
    def test_exact_affine_ellipse(self):
        self.assert_bbox(d._svg_ellipse_bbox('ellipse',
            {'cx':'10','cy':'20','rx':'4','ry':'2'},(0,1,-1,0,5,10)),
            [-17,16,-13,24])
    def test_changed_control_changes_bbox_and_region_filter(self):
        a=d._svg_curve_bbox('M 0 0 C 0 10 10 10 10 0',I)
        b=d._svg_curve_bbox('M 0 0 C 0 20 10 20 10 0',I)
        self.assertNotEqual(a,b)
        svg=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M 0 0 C 0 10 10 10 10 0"/></svg>'
        r=d._svg_inventory(svg)
        self.assertEqual(r['unlocalized_vectors'],0)
        self.assertEqual(r['reported_vectors'][0]['svg_bbox'],a)
        self.assertIsNone(r['reported_vectors'][0]['svg_points'])
        self.assertEqual(d._svg_inventory(svg,region=[4,8,6,9])['reported_count'],0)
        self.assertEqual(d._svg_inventory(svg,region=[4,7,6,8])['reported_count'],1)
    def test_unsupported_arc_and_malformed_are_not_guessed(self):
        for p in ['M 0 0 A 10 10 0 0 1 20 0','M 0 0 C 0 1 2',
                  'M 0 0 ??? L 2 2','M 0 0 C NaN 2 2 2 2 2']:
            self.assertIsNone(d._svg_curve_bbox(p,I),p)

if __name__=='__main__':unittest.main()
