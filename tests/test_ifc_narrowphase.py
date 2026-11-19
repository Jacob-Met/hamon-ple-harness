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


def face_soup(mesh):
    """A real IFC-style triangle soup: independent indices per face corner."""
    vertices = [coordinate for triangle in mesh.triangles
                for point in triangle for coordinate in point]
    return vertices, list(range(len(vertices)//3))


def gridded_cube(n=12):
    """Closed tessellation with no shared vertex indices across any triangles."""
    vertices=[]
    orient=[
        ((0,0,1),(1,0,0),(0,1,0)),
        ((0,0,0),(0,1,0),(1,0,0)),
        ((1,0,0),(0,1,0),(0,0,1)),
        ((0,0,0),(0,0,1),(0,1,0)),
        ((0,1,0),(0,0,1),(1,0,0)),
        ((0,0,0),(1,0,0),(0,0,1)),
    ]
    for origin,u,v in orient:
        def at(i,j):
            return tuple(origin[k]+(i/n)*u[k]+(j/n)*v[k] for k in range(3))
        for i in range(n):
            for j in range(n):
                a,b,c,d=at(i,j),at(i+1,j),at(i+1,j+1),at(i,j+1)
                for p in (a,b,c,a,c,d):
                    vertices.extend(p)
    return g.make_mesh(vertices,list(range(len(vertices)//3)))


class RealisticTopologyAndRayScaling(unittest.TestCase):
    def test_duplicate_per_triangle_vertices_still_form_a_certified_shell(self):
        original=tetra()
        verts,faces=face_soup(original)
        duplicated=g.make_mesh(verts,faces)
        self.assertTrue(duplicated.closed_oriented)
        self.assertEqual(len(duplicated.representatives),1)
        small=tetra((0.15,0.15,0.15),0.1)
        self.assertEqual(g.mesh_relation(duplicated,small)["relation"],"solid_containment")

    def test_real_seam_gap_is_not_falsely_welded(self):
        verts,faces=face_soup(tetra())
        verts[0]+=1e-6
        imperfect=g.make_mesh(verts,faces)
        self.assertFalse(imperfect.closed_oriented)
        self.assertIsNone(g.inside_closed((0.2,0.2,0.2),imperfect))

    def test_large_tessellation_uses_bvh_without_changing_ray_parity(self):
        mesh=gridded_cube(12)
        self.assertEqual(len(mesh.triangles),1728)
        self.assertTrue(mesh.closed_oriented)
        for point,expected in [
            ((0.5,0.5,0.5),True),
            ((1.5,0.5,0.5),False),
            ((-1.0,0.5,0.5),False),
            ((0.15,0.3,0.81),True),
        ]:
            with self.subTest(point=point):
                stats={}
                self.assertIs(g.inside_closed(point,mesh,stats),expected)
                self.assertEqual(stats["rays"],4)
                self.assertGreater(stats["bvh_nodes"],0)
                self.assertLess(stats.get("triangle_tests",0),len(mesh.triangles)//5)

    def test_bvh_parity_matches_full_scan_on_off_surface_ray_fixtures(self):
        mesh=gridded_cube(8)
        rays=((1.0,0.416234,0.271828),(0.357,1.0,0.618034),
              (0.12345,0.517,1.0),(1.0,0.927,0.211))
        points=((.23,.37,.61),(-.35,.23,.44),(1.7,.7,.2),(.11,.4,1.3))
        for point in points:
            for direction in rays:
                with self.subTest(point=point,direction=direction):
                    reference=[g._ray_triangle(point,direction,t)
                               for t in mesh.triangles]
                    self.assertNotIn(-1,reference)
                    expected=sum(t!=0 for t in reference)
                    self.assertEqual(g._ray_crossings_bvh(point,direction,mesh),
                                     expected)


if __name__=="__main__":unittest.main()
