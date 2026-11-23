"""Independent rigid-patch displacement, signed brace, and strain-energy checks."""
from itertools import product
import math
import unittest

from numpy.testing import assert_allclose

from hamon_dual_verifier import run_dual_verify_spec


class PinnedBraceTests(unittest.TestCase):
    @staticmethod
    def model(hinge, corner, ea, precision, reverse=False):
        x, y = hinge
        a, b = corner
        xy = [(0, 0), (x, 0), hinge, (0, y), (a, y), corner, (x, b)]
        panels = [dict(nodes=ids, Dx_N_m=1000., Dy_N_m=1000., G_N_m=400.)
                  for ids in ([0, 1, 2, 3], [2, 4, 5, 6])]
        return dict(nodes_m=xy, panels=panels,
                    springs=[dict(node=i, axis=axis, k_N_m=1.)
                             for i, axis in ((0, 'x'), (0, 'y'), (1, 'y'))],
                    bars=[dict(nodes=[5, 0] if reverse else [0, 5], EA_N=ea)],
                    loads=[], displacement_precision=precision)

    @staticmethod
    def solve(spec):
        return run_dual_verify_spec(dict(kind='diaphragm_membrane', spec=spec))

    def test_unique_uniform_translation_and_exact_mechanism(self):
        # The three springs restrain the first patch. The remaining rotation
        # changes brace length by -d / hypot(2,2+d) per radian.
        for mode in ('binary64', 'compensated'):
            for d, ea, reverse in ((2**-50, 1., False), (-2**-50, 1e30, True),
                                   (2**-47, 1e3, False), (.125, 1e3, False)):
                with self.subTest(mode=mode, d=d, ea=ea):
                    spec = self.model((1, 1), (2, 2+d), ea, mode, reverse)
                    spec['loads'] = [dict(node=0, force_N=[.004, 0])]
                    result = self.solve(spec)
                    self.assertTrue(result['dual_verified'])
                    assert_allclose(result['displacements_m'], [[.004, 0]]*7, atol=1e-15, rtol=0)
                    assert_allclose(result['spring_transmitted_N'], [.004, 0, 0], atol=1e-15)
                    self.assertEqual(result['bar_tension_N'], [0.])
                    self.assertAlmostEqual(result['strain_energy_J'], .004**2/2)
            with self.assertRaisesRegex(ValueError, 'unrestrained'):
                self.solve(self.model((1, 1), (2, 2), 1e30, mode))

    def test_two_pins_require_two_independent_braces(self):
        # The first relative rotation is restrained by lever -d; the second
        # brace has lever d-e about the next pin. Duplicating one brace cannot
        # replace the missing second restraint.
        d = 2**-49
        for mode, e in product(('binary64', 'compensated'), (d, -d)):
            with self.subTest(mode=mode, e=e):
                spec = self.model((1, 1), (2, 2+d), 1000., mode)
                spec['nodes_m'].extend([(3, 2+d), (3, 3+d+e), (2, 3+d+e)])
                spec['panels'].append({**spec['panels'][0], 'nodes': [5, 7, 8, 9]})
                spec['bars'].extend([dict(nodes=[5, 0], EA_N=100.), dict(nodes=[2, 8], EA_N=500.)])
                spec['loads'] = [dict(node=0, force_N=[.004, 0])]
                if e == d:
                    with self.assertRaisesRegex(ValueError, 'unrestrained'):
                        self.solve(spec)
                else:
                    result = self.solve(spec)
                    self.assertTrue(result['dual_verified'])
                    assert_allclose(result['displacements_m'], [[.004, 0]]*10, atol=1e-15, rtol=0)

    def test_nonzero_hinge_rotation_from_signed_brace_end_loads(self):
        # A 3-4-5 brace lets all physical loads and all nodal displacements
        # be represented exactly. About (1.5,2+d), its extension per radian
        # is 3d/5, so EA=125 gives T=15*d*theta and U=4.5*(d*theta)**2.
        theta = 1/256
        for mode, d in product(('binary64', 'compensated'), (2**-49, -2**-49, .125)):
            with self.subTest(mode=mode, d=d):
                hinge = (1.5, 2+d)
                spec = self.model(hinge, (3, 4), 125., mode, reverse=d<0)
                force = [9*theta*d, 12*theta*d]
                spec['loads'] = [dict(node=0, force_N=[-f for f in force]), dict(node=5, force_N=force)]
                expected = [(0, 0) if i < 4 else (-theta*(y-hinge[1]), theta*(x-hinge[0]))
                            for i, (x, y) in enumerate(spec['nodes_m'])]
                result = self.solve(spec)
                self.assertTrue(result['dual_verified'])
                assert_allclose(result['displacements_m'], expected, rtol=1e-13, atol=1e-16)
                self.assertTrue(math.isclose(result['bar_tension_N'][0], 15*theta*d, rel_tol=1e-12))
                for field in ('external_work_J', 'strain_energy_J'):
                    self.assertTrue(math.isclose(result[field], 4.5*(theta*d)**2, rel_tol=1e-12))
