"""IFC4 whole-file geometry integration with real IfcOpenShell 0.9.0post1.

Build project solids from source-controlled coordinates; write IFC4, reopen through
the actual optional geometry kernel. No competition files or mock geometry.
"""
from pathlib import Path
import math
import tempfile
import unittest

import ifcopenshell
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.root
import ifcopenshell.api.unit

import hamon_geometry as h


def vertices(origin,edge):
    x,y,z=origin
    return [(x,y,z),(x+edge,y,z),(x,y+edge,z),(x,y,z+edge)]


class RealIFCMesh(unittest.TestCase):
    def test_broadphase_false_positive_vs_surface_intersection_in_ifc4(self):
        model=ifcopenshell.api.project.create_file()
        project=ifcopenshell.api.root.create_entity(model,ifc_class="IfcProject")
        ifcopenshell.api.unit.assign_unit(model)
        context=ifcopenshell.api.context.add_context(model,context_type="Model")
        body=ifcopenshell.api.context.add_context(
            model,context_type="Model",context_identifier="Body",
            target_view="MODEL_VIEW",parent=context)
        def create(kind,origin,edge):
            product=ifcopenshell.api.root.create_entity(model,ifc_class=kind)
            representation=ifcopenshell.api.geometry.add_mesh_representation(
                model,context=body,
                vertices=[vertices(origin,edge)],
                faces=[[(0,2,1),(0,1,3),(0,3,2),(1,2,3)]])
            ifcopenshell.api.geometry.assign_representation(
                model,product=product,representation=representation)
            ifcopenshell.api.geometry.edit_object_placement(model,product=product)
            return product
        duct=create("IfcDuctSegment",(0.0,0.0,0.0),2.0)
        disjoint=create("IfcBeam",(1.6,1.6,1.6),2.0)
        crossing=create("IfcBeam",(0.3,0.3,0.3),2.0)
        near=create("IfcBeam",(2.1,0.0,0.0),2.0)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"fixture.ifc"
            model.write(str(file))
            reopened=ifcopenshell.open(str(file))
            self.assertEqual(len(reopened.by_type("IfcBeam")),3)
            screen=h.ifc_narrowphase(str(file),"IfcDuctSegment","IfcBeam",limit=10)
            threshold=h.ifc_narrowphase(str(file),"IfcDuctSegment","IfcBeam",
                                        limit=10,min_clearance_mm=150)
            stringent=h.ifc_narrowphase(str(file),"IfcDuctSegment","IfcBeam",
                                        limit=10,min_clearance_mm=2000)
        self.assertEqual(screen["element_a_count"],1,screen)
        self.assertEqual(screen["element_b_count"],3,screen)
        self.assertEqual(screen["compared_pairs"],3,screen)
        self.assertEqual(screen["broadphase_overlaps"],2,screen)
        self.assertEqual(screen["surface_contact_pairs"],1,screen)
        self.assertEqual(screen["by_relation"].get("no_surface_contact_found"),1,screen)
        self.assertEqual(screen["elements_skipped"],0,screen)
        self.assertTrue(screen["all_pairs_qualified"],screen)
        pair_rel={entry["b"]:entry["relation"] for entry in screen["witnesses"]}
        self.assertEqual(pair_rel[disjoint.GlobalId],"no_surface_contact_found")
        self.assertEqual(pair_rel[crossing.GlobalId],"surface_contact")
        self.assertEqual(pair_rel[near.GlobalId],"aabb_disjoint")
        self.assertAlmostEqual(
            next(x for x in screen["witnesses"] if x["b"]==disjoint.GlobalId)
            ["surface_distance_mm"],2800/math.sqrt(3),places=5)
        classified={x["b"]:x for x in threshold["witnesses"]}
        self.assertEqual(classified[near.GlobalId]["clearance_status"],"below_specified_gap")
        self.assertAlmostEqual(classified[near.GlobalId]["surface_distance_mm"],100,places=5)
        self.assertEqual(classified[disjoint.GlobalId]["clearance_status"],"meets_triangulated_gap")
        self.assertEqual(classified[crossing.GlobalId]["clearance_status"],"contact_or_solid_overlap")
        self.assertEqual(threshold["clearance_failures"],2)
        self.assertTrue(threshold["all_pairs_qualified"],threshold)
        self.assertEqual(stringent["clearance_failures"],3)


if __name__=="__main__":unittest.main()
