import unittest

import numpy as np

from hamon_ifc_spatial_clash import SolidMesh3D


class SurveyMassProperties(unittest.TestCase):
    def test_asymmetric_tetrahedron_in_survey_coordinates(self):
        offset = np.array([729011.2258823584, 9063960.607644705, 120.0])
        local = np.array([[0., 0., 0.], [0.25, 0., 0.], [0., 0.5, 0.], [0., 0., 2.]])
        faces = np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
        for winding in (faces, faces[:, ::-1]):
            mesh = SolidMesh3D("t", "IFCBUILDINGELEMENTPROXY", "tetra", "other",
                               local + offset, winding)
            self.assertAlmostEqual(mesh.volume_m3(), 0.25 * 0.5 * 2 / 6, places=12)
            np.testing.assert_allclose(mesh.centroid_m(), offset + local.mean(axis=0),
                                       rtol=0, atol=2e-9)
