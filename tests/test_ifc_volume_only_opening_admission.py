"""A volume-only material certificate must never become a Boolean operand surface."""
from unittest import TestCase
from unittest.mock import patch

import hamon_ifc_spatial_clash as reader
from test_ifc_finite_halfspace_openings import Source


def triangle(scene, points, bottom=0., depth=2.):
    ring = scene.ring(points)
    profile = scene.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{ring})")
    frame = scene.frame_at(bottom)
    return scene.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{frame},#{scene.up},{depth})")


def contact_items(scene):
    # The two outer prisms touch along the vertical through (1,1), in the
    # middle of the first prism's diagonal edge. Its source ring does not
    # have a vertex there. The union kernel splits that edge at the contact.
    # The third prism is entirely inside the first and adds no material.
    return [triangle(scene, [(0., 0.), (2., 0.), (0., 2.)]),
            triangle(scene, [(1., 1.), (2., 1.), (1., 2.)]),
            triangle(scene, [(.25, .25), (.75, .25), (.25, .75)], .5, 1.)]


def scene_with_opening(compound_role):
    scene = Source()
    compound = contact_items(scene)
    distant = [triangle(scene, [(3., 0.), (4., 0.), (3., 1.)])]
    wall = scene.product("IFCWALL", compound if compound_role == "host" else distant)
    cutter = scene.product("IFCOPENINGELEMENT", compound if compound_role == "cutter" else distant)
    scene.void(wall, cutter)
    return scene


class VolumeOnlyOpeningAdmissionTests(TestCase):
    def test_true_union_volume_with_unqualified_component_surfaces(self):
        scene = Source()
        scene.product("IFCWALL", contact_items(scene))
        material, = scene.solids()
        # The source sum includes 0.125 m³ of embedded material twice.
        self.assertAlmostEqual(material.certified_union_volume_m3, 5., places=12)
        self.assertAlmostEqual(material.volume_m3(), 5., places=12)
        self.assertEqual(len(material.material_face_counts), 3)
        with self.assertRaises(reader.UnresolvedMeshMass):
            material.surface_area_m2()
        with self.assertRaises(reader.UnresolvedMeshMass):
            material.centroid_m()

    def test_volume_only_host_and_cutter_are_rejected_before_surface_use(self):
        for role in ("host", "cutter"):
            with self.subTest(role=role):
                scene = scene_with_opening(role)
                # Both the fast box subtraction and the general Boolean require
                # a qualified replacement boundary. Even a disjoint opening
                # must not send unqualified component triangles to either path.
                with patch.object(reader, "_box_bounds", side_effect=AssertionError(
                        f"Volume-only {role} entered opening boundary reconstruction")):
                    with self.assertRaises(reader.UnresolvedOpening):
                        scene.solids()
