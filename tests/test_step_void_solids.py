"""Contained disjoint STEP void shells subtract material and retain their walls."""
import math
import re
import unittest
from fractions import Fraction as F
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry
from test_step_face_regions import prism_step


def box(origin, dimensions, matrix=None, translation=(0., 0., 0.)):
    x, y, z = dimensions
    transform = np.eye(3) if matrix is None else np.asarray(matrix)
    return prism_step([(0.,0.),(x,0.),(x,y),(0.,y)], height=z,
                      rotation=transform, translation=transform@np.asarray(origin)+translation)


def with_voids(outer, *voids):
    rows, shells = [], []
    offset = 0
    for text in (outer, *voids):
        data = text.split("DATA;\n", 1)[1].split("ENDSEC;", 1)[0]
        shell = int(re.search(r"#(\d+)=CLOSED_SHELL", data)[1]) + offset
        data = re.sub(r"^#\d+=MANIFOLD_SOLID_BREP[^\n]*\n", "", data, flags=re.M)
        data = re.sub(r"#(\d+)", lambda m: "#" + str(int(m[1])+offset), data)
        rows.append(data)
        shells.append(shell)
        offset = 1 + max(map(int, re.findall(r"#(\d+)", data)))
    wrappers = []
    for shell in shells[1:]:
        wrappers.append(offset)
        rows.append(f"#{offset}=ORIENTED_CLOSED_SHELL('void',*,#{shell},.F.);")
        offset += 1
    refs = ",".join(f"#{i}" for i in wrappers)
    rows.append(f"#{offset}=BREP_WITH_VOIDS('material',#{shells[0]},({refs}));")
    return "ISO-10303-21;\nHEADER;ENDSEC;\nDATA;\n" + "\n".join(rows) + "\nENDSEC;\nEND-ISO-10303-21;"


def box_difference_properties(parts, matrix, translation):
    a = np.asarray(matrix)
    volumes, centers, covariances = [], [], []
    area = 0.
    for origin, sides, sign in parts:
        sides = np.asarray(sides)
        volumes.append(sign*abs(np.linalg.det(a))*np.prod(sides))
        centers.append(a@(np.asarray(origin)+sides/2))
        covariances.append(a@np.diag(sides*sides/12)@a.T)
        edges = a*sides
        area += 2*sum(math.hypot(*np.cross(edges[:,i],edges[:,j])) for i,j in ((0,1),(0,2),(1,2)))
    volume = sum(volumes)
    center = sum(v*c for v,c in zip(volumes,centers))/volume
    covariance = sum(v*(c+np.outer(p-center,p-center)) for v,p,c in zip(volumes,centers,covariances))
    inertia = np.trace(covariance)*np.eye(3)-covariance
    return volume, area, center+translation, inertia


class StepVoidSolidTests(unittest.TestCase):
    def verify(self, text):
        return verify_step_cad_geometry({"step_text": text, "density_kg_m3": 1.})

    def assert_withheld(self, text):
        try:
            got = self.verify(text)
        except ValueError:
            return
        self.assertFalse(got["conservation_verified"], got)
        for key in ("volume_m3","mass_kg","centroid_m","inertia_tensor_centroid_kg_m2"):
            self.assertNotIn(key, got)

    def test_offcenter_disjoint_cavities_have_material_moments(self):
        parts = [((0.,0.,0.),(4.,5.,6.),1), ((.5,1.,1.5),(1.,2.,1.),-1),
                 ((2.5,3.5,4.),(.5,.75,1.),-1)]
        for matrix, translation in ((np.eye(3),np.zeros(3)),
                (np.array([[1.,.25,-.5],[0.,1.,.5],[0.,0.,1.]]),np.array([500000.,4500000.,80.])),
                (np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]]),np.array([-16.,32.,-8.]))):
            for count in (2,3):
                with self.subTest(count=count, matrix=matrix.tolist()):
                    chosen = parts[:count]
                    texts = [box(o,d,matrix,translation) for o,d,_ in chosen]
                    got = self.verify(with_voids(*texts))
                    expected = box_difference_properties(chosen,matrix,translation)
                    self.assertTrue(got["conservation_verified"], got)
                    self.assertTrue(got["void_containment_verified"], got)
                    self.assertEqual(got["void_shell_count"], count-1)
                    self.assertAlmostEqual(got["volume_m3"], expected[0], places=11)
                    self.assertAlmostEqual(got["surface_area_m2"], expected[1], places=10)
                    np.testing.assert_allclose(got["centroid_m"], expected[2], rtol=0,atol=1e-10)
                    np.testing.assert_allclose(got["inertia_tensor_centroid_kg_m2"], expected[3], rtol=3e-13,atol=1e-12)

    def test_one_ulp_wall_retains_exact_positive_material(self):
        h = math.nextafter(1.,0.)
        text = with_voids(box((-1.,-1.,-1.),(2.,2.,2.)),
                          box((-h,-h,-h),(2*h,2*h,2*h)))
        got = self.verify(text)
        self.assertTrue(got["conservation_verified"], got)
        self.assertEqual(got["volume_m3"], float(8*(1-F(h)**3)))
        moment = float(F(16,3)*(1-F(h)**5))
        self.assertEqual(got["centroid_m"], [0.,0.,0.])
        self.assertEqual(got["principal_moments_kg_m2"], [moment]*3)

    def test_outside_crossing_and_touching_voids_are_refused(self):
        outer = box((0.,0.,0.),(4.,4.,4.))
        for origin, sides in (((5.,1.,1.),(1.,1.,1.)),
                              ((3.5,1.,1.),(1.,1.,1.)),
                              ((0.,1.,1.),(1.,1.,1.)),
                              ((-1.,-1.,-1.),(6.,6.,6.))):
            with self.subTest(origin=origin):
                self.assert_withheld(with_voids(outer,box(origin,sides)))

    def test_voids_must_be_mutually_disjoint_and_not_nested(self):
        outer = box((0.,0.,0.),(6.,6.,6.))
        first = box((1.,1.,1.),(2.,2.,2.))
        for origin, sides in (((2.,2.,2.),(2.,2.,2.)),
                              ((3.,1.,1.),(1.,1.,1.)),
                              ((1.5,1.5,1.5),(.5,.5,.5))):
            with self.subTest(origin=origin):
                self.assert_withheld(with_voids(outer,first,box(origin,sides)))

    def test_concave_outer_and_through_hole_use_solid_membership(self):
        lshape = prism_step([(0.,0.),(4.,0.),(4.,1.),(1.,1.),(1.,4.),(0.,4.)],height=4.)
        ring = prism_step([(0.,0.),(4.,0.),(4.,4.),(0.,4.)],
                          [[(1.,1.),(3.,1.),(3.,3.),(1.,3.)]],height=4.)
        for outer, origin, expected in ((lshape,(.25,2.,1.),27.875),
                                        (ring,(.25,.25,1.),47.875)):
            got = self.verify(with_voids(outer,box(origin,(.5,.5,.5))))
            self.assertTrue(got["conservation_verified"], got)
            self.assertAlmostEqual(got["volume_m3"],expected,places=12)
        for outer in (lshape,ring):
            self.assert_withheld(with_voids(outer,box((2.,2.,1.),(.5,.5,.5))))

    def test_void_reference_and_sense_cannot_be_dropped(self):
        text = with_voids(box((0.,0.,0.),(4.,4.,4.)),box((1.,1.,1.),(1.,1.,1.)))
        wrapper = re.search(r"#(\d+)=ORIENTED_CLOSED_SHELL\('void',\*,#(\d+),\.F\.\);",text)
        cases = [text.replace(wrapper[0],wrapper[0].replace(".F.",".T.")),
                 text.replace(wrapper[0],wrapper[0].replace("#"+wrapper[2],"#999999")),
                 text.replace(f"(#{wrapper[1]})",f"(#{wrapper[1]},#{wrapper[1]})"),
                 text.replace(f"(#{wrapper[1]})",f"(#{wrapper[2]})")]
        for value in cases:
            self.assert_withheld(value)


if __name__ == "__main__":
    unittest.main()
