"""Frozen IFC mesh SAT, closed-body containment, and conservative uncertainty."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hamon_geometry as g


def tetra(origin=(0.0,0.0,0.0),edge=2.0):
    x,y,z=origin
    verts=[x,y,z,x+edge,y,z,x,y+edge,z,x,y,z+edge]
    faces=[0,2,1,0,1,3,0,3,2,1,2,3]
    return g.make_mesh(verts,faces)


class NarrowPhaseContracts(unittest.TestCase):
    def test_aabb_overlaps_but_meshes_are_disjoint(self):
        a=tetra()
        b=tetra((1.6,1.6,1.6))
        self.assertTrue(g.overlaps(a.bbox,b.bbox))
        self.assertTrue(a.closed_oriented and b.closed_oriented)
        r=g.mesh_relation(a,b)
        self.assertEqual(r["relation"],"no_surface_contact_found")
        self.assertGreater(r["triangle_tests"],0)

    def test_genuine_intersection_is_not_just_envelope(self):
        a=tetra()
        b=tetra((0.3,0.3,0.3),2.0)
        r=g.mesh_relation(a,b)
        self.assertEqual(r["relation"],"surface_contact")
        self.assertGreater(r["triangle_tests"],0)

    def test_fully_enclosed_mesh_detected_without_crossing_faces(self):
        a=tetra()
        b=tetra((0.15,0.15,0.15),0.10)
        self.assertTrue(g.inside_closed((0.15,0.15,0.15),a))
        self.assertEqual(g.mesh_relation(a,b)["relation"],"solid_containment")
        self.assertEqual(g.mesh_relation(b,a)["relation"],"solid_containment")

    def test_triangle_coplanar_2d_sat_axes(self):
        a=((0,0,0),(2,0,0),(0,2,0))
        b=((1.6,1.6,0),(3,1.6,0),(1.6,3,0))
        c=((0.5,0.5,0),(1.5,0.5,0),(0.5,1.5,0))
        self.assertTrue(g.overlaps(g.bounds(a),g.bounds(b)))
        self.assertFalse(g.tri_overlap(a,b))
        self.assertTrue(g.tri_overlap(a,c))

    def test_disjoint_aabb_short_circuit(self):
        a=tetra();b=tetra((10,0,0))
        self.assertEqual(g.mesh_relation(a,b),
                         {"relation":"aabb_disjoint","triangle_tests":0})

    def test_nonmanifold_must_not_report_clear(self):
        a=tetra()
        vertices=[0,0,0,2,0,0,0,2,0]
        open_sheet=g.make_mesh(vertices,[0,1,2])
        self.assertFalse(open_sheet.closed_oriented)
        shifted=g.make_mesh([x+0.2 if i%3!=2 else x for i,x in enumerate(vertices)],
                            [0,1,2])
        self.assertEqual(g.mesh_relation(open_sheet,shifted)["relation"],
                         "surface_contact")
        other=g.make_mesh([0.01,0.01,0.4,2.01,0.01,0.4,0.01,2.01,0.4],[0,1,2])
        self.assertEqual(g.mesh_relation(open_sheet,other)["relation"],
                         "aabb_disjoint")

    def test_hard_triangle_budget_explicit(self):
        a=tetra();b=tetra((0.3,0.3,0.3))
        with self.assertRaisesRegex(ValueError,"budget exhausted"):
            g.surface_contact(a,b,budget=0)

    def test_mesh_validation_rejects_invalid_topology(self):
        for vertices,faces in [
            ([0,0,0,1,0,0,0,1,0],[0,0,2]),
            ([0,0,0,1,0,0,0,1,0],[0,1,7]),
            ([0,0,0,1,0,0,2,0,0],[0,1,2]),
            ([0,0,float("nan"),1,0,0,0,1,0],[0,1,2])]:
            with self.subTest(faces=faces):
                with self.assertRaises(ValueError):g.make_mesh(vertices,faces)

    def test_missing_ifcopenshell_is_actionable_error(self):
        with patch.dict(sys.modules,{"ifcopenshell":None}):
            with self.assertRaisesRegex(RuntimeError,"optional IFC kernel"):
                g.ifc_narrowphase("missing.ifc","IfcDuctSegment","IfcBeam")

    def test_cli_provenance_no_kernel_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as d:
            cli=Path(__file__).resolve().parents[1]/"engineering_toolkit.py"
            p=subprocess.run([sys.executable,str(cli),"ifc-narrowphase","missing.ifc"],
                             text=True,capture_output=True,check=False)
            self.assertEqual(p.returncode,2)
            payload=json.loads(p.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["command"],"ifc-narrowphase")


if __name__=="__main__":unittest.main()
