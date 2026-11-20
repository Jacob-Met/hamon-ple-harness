"""Whole-file checks for overlapping IFC class selectors and per-side limits."""
from pathlib import Path
import tempfile
import unittest

import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.root
import ifcopenshell.api.unit

from engineering_toolkit import ifc_broadphase


class PhysicalPairIdentity(unittest.TestCase):
    def test_overlapping_classes_share_one_physical_pair(self):
        model = ifcopenshell.api.project.create_file()
        ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject")
        ifcopenshell.api.unit.assign_unit(model)
        context = ifcopenshell.api.context.add_context(model, context_type="Model")
        body = ifcopenshell.api.context.add_context(
            model, context_type="Model", context_identifier="Body",
            target_view="MODEL_VIEW", parent=context)
        elements = []
        for kind in ("IfcDuctSegment", "IfcBeam", "IfcBeam", "IfcBeam"):
            element = ifcopenshell.api.root.create_entity(model, ifc_class=kind)
            representation = ifcopenshell.api.geometry.add_wall_representation(
                model, context=body, length=1., height=1., thickness=.1)
            ifcopenshell.api.geometry.assign_representation(
                model, product=element, representation=representation)
            elements.append(element.GlobalId)

        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "pairs.ifc"
            model.write(str(file))
            for selectors, counts, expected in (
                (("IfcDuctSegment","IfcBeam"), (1,3), 3),
                (("IfcBeam,IfcBeam","IfcBeam"), (3,3), 3),
                (("IfcElement,IfcBeam","IfcBeam"), (4,3), 6),
                (("IfcBeam","IfcElement,IfcBeam"), (3,4), 6),
            ):
                with self.subTest(selectors=selectors):
                    result = ifc_broadphase(str(file), *selectors, limit=10)
                    self.assertEqual((result["a_count"],result["b_count"]), counts)
                    self.assertEqual(result["comparisons"], expected)
                    self.assertEqual(result["candidate_pairs"], expected)
                    pairs = {frozenset((hit["a"],hit["b"])) for hit in result["witnesses"]}
                    self.assertEqual(len(pairs), expected)
                    self.assertTrue(all(len(pair)==2 and pair<=set(elements) for pair in pairs))
            capped = ifc_broadphase(str(file), "IfcBeam,IfcBeam", "IfcBeam", limit=2)
            self.assertEqual((capped["skipped_a"],capped["skipped_b"]), (1,1))
            self.assertEqual(capped["comparisons"], 1)
