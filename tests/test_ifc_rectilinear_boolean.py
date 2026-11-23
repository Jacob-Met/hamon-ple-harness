"""Nominal material controls using independent exact box moments."""
from fractions import Fraction
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from hamon_ifc_spatial_clash import (
    SolidMesh3D, UnresolvedSolidGeometry, _boolean_meshes, load_ifc_solids,
    run_router_ifc_clash,
)
from hamon_rectilinear_boolean import rectilinear_boolean
from test_ifc_mapped_mass import Scene

FACES = np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),
                  (0,1,5),(0,5,4),(1,2,6),(1,6,5),
                  (2,3,7),(2,7,6),(3,0,4),(3,4,7)], dtype=np.int32)


def box(low, high, reverse=False):
    x0,y0,z0 = low
    x1,y1,z1 = high
    vertices = np.array([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)], dtype=float)
    return vertices, FACES[:, ::-1].copy() if reverse else FACES.copy()


def mesh(parts, operation):
    vertices, faces = _boolean_meshes(parts, operation)
    return SolidMesh3D("control", "IFCBEAM", "control", "structural", vertices, faces)


def box_measure(bounds):
    low, high = bounds
    widths = [max(Fraction(0), b-a) for a,b in zip(low,high)]
    volume = math.prod(widths)
    moment = [volume*(a+b)/2 for a,b in zip(low,high)]
    return volume, moment


def union_measure(boxes):
    """Independent inclusion-exclusion, never the production occupancy grid."""
    volume, moment = Fraction(0), [Fraction(0)]*3
    for count in range(1, len(boxes)+1):
        for selection in itertools.combinations(boxes,count):
            low = [max(Fraction(float(b[0][a])) for b in selection) for a in range(3)]
            high = [min(Fraction(float(b[1][a])) for b in selection) for a in range(3)]
            weight, centre = box_measure((low,high))
            sign = 1 if count % 2 else -1
            volume += sign*weight
            moment = [a+sign*b for a,b in zip(moment,centre)]
    return float(volume), np.array([float(v/volume) for v in moment])


def ifc_intersection(width, unit=1.):
    scene = Scene()
    roots = []
    for low in (0., 1.-width):
        frame = scene.frame(np.eye(3), [low*unit,0,0])
        roots.append(scene.add(f"IFCBLOCK(#{frame},{unit},{unit},{unit})"))
    root = scene.add(f"IFCBOOLEANRESULT(.INTERSECTION.,#{roots[0]},#{roots[1]})")
    scene.product(root, np.eye(3), [0,0,0])
    prefix = ".MILLI." if unit == 1000. else "$"
    length = scene.add(f"IFCSIUNIT(*,.LENGTHUNIT.,{prefix},.METRE.)")
    scene.add(f"IFCUNITASSIGNMENT((#{length}))")
    return scene.text()


class ExactRectilinearMaterial(unittest.TestCase):
    def test_real_ifc4_export_and_reopen_preserve_source_planes(self):
        try:
            import ifcopenshell
            import ifcopenshell.api.context
            import ifcopenshell.api.geometry
            import ifcopenshell.api.project
            import ifcopenshell.api.root
            import ifcopenshell.api.unit
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional IFC authoring dependency unavailable: {exc}")
        for delta in (.125,2.**-40,2.**-50):
            model=ifcopenshell.api.project.create_file()
            ifcopenshell.api.root.create_entity(model,ifc_class="IfcProject")
            unit=ifcopenshell.api.unit.add_si_unit(model,unit_type="LENGTHUNIT")
            ifcopenshell.api.unit.assign_unit(model,units=[unit])
            context=ifcopenshell.api.context.add_context(model,context_type="Model")
            context.Precision=1e-16
            body=ifcopenshell.api.context.add_context(model,context_type="Model",
                 context_identifier="Body",target_view="MODEL_VIEW",parent=context)
            blocks=[]
            for x in (0.,1.-delta):
                point=model.create_entity("IfcCartesianPoint",Coordinates=(x,0.,0.))
                placement=model.create_entity("IfcAxis2Placement3D",Location=point)
                blocks.append(model.create_entity("IfcBlock",Position=placement,
                              XLength=1.,YLength=1.,ZLength=1.))
            boolean=model.create_entity("IfcBooleanResult",Operator="INTERSECTION",
                                        FirstOperand=blocks[0],SecondOperand=blocks[1])
            rep=model.create_entity("IfcShapeRepresentation",ContextOfItems=body,
                    RepresentationIdentifier="Body",RepresentationType="CSG",Items=[boolean])
            product=ifcopenshell.api.root.create_entity(model,ifc_class="IfcBeam")
            ifcopenshell.api.geometry.assign_representation(model,product=product,representation=rep)
            ifcopenshell.api.geometry.edit_object_placement(model,product=product)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"material.ifc"
                model.write(str(path))
                reopened=ifcopenshell.open(str(path))
                self.assertEqual(reopened.schema,"IFC4")
                self.assertEqual(len(reopened.by_type("IfcBooleanResult")),1)
                result=run_router_ifc_clash({"ifc_text":reopened.to_string()})
            self.assertTrue(result["ok"],result)
            self.assertEqual(result["total_solid_volume_m3"],delta)

    def test_adjacent_planes_keep_the_full_rectangular_intersection(self):
        # The former normalized mesh kernel returned a tetrahedron and one-sixth
        # volume from k=40 onward, despite reporting a qualified closed result.
        for k, axis, reverse in itertools.product((20,30,40,45,50,52,53),range(3),(False,True)):
            delta = 2.**(-k)
            a = box((0,0,0),(1,1,1),reverse)
            low, high = np.zeros(3), np.ones(3)
            low[axis], high[axis] = 1.-delta, 2.
            b = box(low,high,not reverse)
            with self.subTest(k=k,axis=axis,reverse=reverse):
                result = mesh([a,b],"INTERSECTION")
                self.assertEqual(len(result.vertices),8)
                self.assertEqual(result.volume_m3(),delta)
                self.assertEqual(result.aabb_min[axis],1.-delta)
                self.assertEqual(result.aabb_max[axis],1.)
                self.assertAlmostEqual(result.surface_area_m2(),2.+4.*delta,places=14)

    def test_registered_ifc_blocks_use_exact_material_without_the_optional_kernel(self):
        with mock.patch.dict(sys.modules,{"manifold3d":None}):
            for k, unit in itertools.product((20,40,50),(1.,1000.)):
                text = ifc_intersection(2.**(-k),unit)
                # Use the actual decoded binary64 planes after declared-unit conversion.
                expected = 1. - (((1.-2.**(-k))*unit)*(1./unit))
                result = run_router_ifc_clash({"ifc_text":text})
                self.assertTrue(result["ok"],result)
                self.assertEqual(result["total_solid_volume_m3"],expected)

    def test_three_box_union_matches_exact_inclusion_exclusion(self):
        rng = np.random.default_rng(6054)
        for case in range(32):
            bounds = []
            parts = []
            for index in range(3):
                low = rng.integers(-3,2,size=3)/4
                high = low+rng.integers(4,9,size=3)/4
                bounds.append((low,high))
                parts.append(box(low,high,reverse=bool((case+index)%2)))
            expected, centroid = union_measure(bounds)
            with self.subTest(case=case):
                result = mesh(parts,"UNION")
                self.assertAlmostEqual(result.volume_m3(),expected,places=13)
                np.testing.assert_allclose(result.centroid_m(),centroid,rtol=0,atol=2e-14)

    def test_void_refill_and_repeated_source_triangulation(self):
        shell = mesh([box((0,0,0),(4,4,4)),box((1,1,1),(3,3,3))],"DIFFERENCE")
        result = mesh([(shell.vertices,shell.faces),box((1,1,1),(2,3,3))],"UNION")
        self.assertEqual(result.volume_m3(),60.)
        self.assertEqual(result.surface_area_m2(),112.)
        np.testing.assert_allclose(result.centroid_m(),[118/60,2.,2.],rtol=0,atol=1e-15)
        # Every face gets independent indices. Only exact source positions agree.
        duplicate = result.vertices[result.faces].reshape(-1,3)
        triangles = np.arange(len(duplicate)).reshape(-1,3)
        again = mesh([(duplicate,triangles),box((0,0,0),(1,1,1))],"UNION")
        self.assertEqual(again.volume_m3(),60.)

    def test_far_component_does_not_set_the_precision_of_a_near_gap(self):
        for gap in (2.**-30,2.**-40,2.**-50):
            parts=[box((0,0,0),(1,1,1)),box((1.+gap,0,0),(2.+gap,1,1)),
                   box((1e6,0,0),(1e6+1,1,1))]
            result=mesh(parts,"UNION")
            self.assertEqual(result.volume_m3(),3.)
            self.assertEqual(result.surface_area_m2(),18.)

    def test_oblique_material_is_left_to_the_general_kernel(self):
        angle=.3
        rotate=np.array([[math.cos(angle),-math.sin(angle),0],
                         [math.sin(angle),math.cos(angle),0],[0,0,1]])
        a=box((0,0,0),(1,1,1))
        oblique=(a[0]@rotate.T,a[1])
        self.assertIsNone(rectilinear_boolean([a,oblique],"UNION"))
        with mock.patch.dict(sys.modules,{"manifold3d":None}):
            with self.assertRaisesRegex(UnresolvedSolidGeometry,"requires manifold3d"):
                _boolean_meshes([a,oblique],"UNION")

    def test_corner_and_edge_only_contacts_stay_unresolved(self):
        for low in ((1,1,1),(1,1,0)):
            with self.subTest(low=low):
                with self.assertRaisesRegex(UnresolvedSolidGeometry,"nonmanifold material contact"):
                    _boolean_meshes([box((0,0,0),(1,1,1)),
                                     box(low,np.asarray(low)+1)],"UNION")

    def test_budget_and_invalid_material_do_not_fall_back_to_tolerance_merging(self):
        a=box((0,0,0),(2,2,2))
        b=box((1,1,1),(3,3,3))
        with mock.patch("hamon_rectilinear_boolean.MAX_CELLS",1):
            with self.assertRaisesRegex(UnresolvedSolidGeometry,"cell budget"):
                _boolean_meshes([a,b],"UNION")
        duplicate=(np.vstack([a[0],a[0]]),np.vstack([a[1],a[1]+8]))
        with self.assertRaises(UnresolvedSolidGeometry):
            _boolean_meshes([duplicate,b],"UNION")
        with self.assertRaises(UnresolvedSolidGeometry):
            _boolean_meshes([a,a],"DIFFERENCE")


if __name__ == "__main__":
    unittest.main()
