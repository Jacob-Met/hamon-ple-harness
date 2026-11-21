"""Account for unique requested IFC pairs, including missing geometry."""
from pathlib import Path
import tempfile
import unittest

import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.root
import ifcopenshell.api.unit

from hamon_geometry import ifc_narrowphase


class IFCPairCoverage(unittest.TestCase):
    def write_model(self, model):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "coverage.ifc"
        model.write(str(path))
        return str(path)

    def test_partially_overlapping_selections_with_missing_representations(self):
        model = ifcopenshell.api.project.create_file()
        for kind in ("IfcBeam", "IfcBeam", "IfcColumn", "IfcSlab"):
            ifcopenshell.api.root.create_entity(model, ifc_class=kind)
        path = self.write_model(model)
        report = ifc_narrowphase(path, "IfcBeam,IfcColumn", "IfcBeam,IfcSlab")
        self.assertEqual(report["inconclusive_pairs"], 6)
        self.assertEqual(report["requested_pairs"], 6)
        self.assertEqual(report["compared_pairs"], 0)
        self.assertEqual(report["unmeasured_pairs"], 6)
        self.assertEqual(report["elements_skipped"], 4)
        singleton = ifc_narrowphase(path, "IfcColumn", "IfcColumn")
        self.assertEqual(singleton["inconclusive_pairs"], 0)
        self.assertEqual(singleton["requested_pairs"], 0)
        self.assertEqual(singleton["elements_skipped"], 1)

    def test_unresolved_contact_and_clearance_count_as_one_pair(self):
        model = ifcopenshell.api.project.create_file()
        ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject")
        ifcopenshell.api.unit.assign_unit(model)
        context = ifcopenshell.api.context.add_context(model, context_type="Model")
        body = ifcopenshell.api.context.add_context(
            model, context_type="Model", context_identifier="Body",
            target_view="MODEL_VIEW", parent=context)
        # Open parallel patches have overlapping world AABBs but no surface contact.
        for kind, offset in [("IfcBeam", 0.0), ("IfcColumn", 0.1)]:
            product = ifcopenshell.api.root.create_entity(model, ifc_class=kind)
            vertices = [(x + offset, y + offset, z + offset) for x, y, z in
                        [(0, 0, 0), (1, 0, -1), (1, 1, -2), (0, 1, -1)]]
            mesh = ifcopenshell.api.geometry.add_mesh_representation(
                model, context=body, vertices=[vertices], faces=[[(0, 1, 2), (0, 2, 3)]])
            ifcopenshell.api.geometry.assign_representation(model, product=product,
                                                           representation=mesh)
            ifcopenshell.api.geometry.edit_object_placement(model, product=product)
        report = ifc_narrowphase(self.write_model(model), "IfcBeam", "IfcColumn",
                                 min_clearance_mm=1)
        self.assertEqual(report["compared_pairs"], 1)
        self.assertEqual(report["clearance_unresolved_pairs"], 1)
        self.assertEqual(report["by_relation"], {"unresolved_nonwatertight": 1})
        self.assertEqual(report["inconclusive_pairs"], 1)
        self.assertEqual(report["requested_pairs"], 1)
        self.assertEqual(report["unmeasured_pairs"], 0)
