"""Planar STEP faces are regions; their inner loops are void boundaries."""
import math
import re
import unittest

import numpy as np

from hamon_conservation_cad_verifier import verify_step_cad_geometry


def prism_step(outer, holes=(), height=6., rotation=None, translation=(0., 0., 0.),
               edge_loops=False, reverse_bounds=False, opposite_surface=False,
               explicit_outer=True):
    """Build straight-edge planar B-rep topology, without tessellating its caps."""
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    translation = np.asarray(translation)
    rows = []

    def add(kind, arguments):
        entity = len(rows) + 1
        rows.append(f"#{entity}={kind}({arguments});")
        return entity

    def numbers(values):
        return "(" + ",".join(repr(float(v)) for v in values) + ")"

    def refs(values):
        return "(" + ",".join(f"#{v}" for v in values) + ")"

    points, coordinates, vertices, rings = {}, {}, {}, []
    for ring in (outer, *holes):
        layers = []
        for z in (0., height):
            layer = []
            for x, y in ring:
                p = rotation @ np.array([x, y, z]) + translation
                pid = add("CARTESIAN_POINT", "''," + numbers(p))
                points[pid] = pid
                coordinates[pid] = p
                vertices[pid] = add("VERTEX_POINT", f"'',#{pid}")
                layer.append(pid)
            layers.append(layer)
        rings.append(layers)

    edges = {}

    def loop(pids):
        if not edge_loops:
            return add("POLY_LOOP", "''," + refs(pids))
        oriented = []
        for a, b in zip(pids, pids[1:] + pids[:1]):
            key = tuple(sorted((a, b)))
            if key not in edges:
                start, end = key
                delta = coordinates[end] - coordinates[start]
                length = float(np.linalg.norm(delta))
                direction = add("DIRECTION", "''," + numbers(delta / length))
                vector = add("VECTOR", f"'',#{direction},{length!r}")
                line = add("LINE", f"'',#{start},#{vector}")
                edges[key] = add("EDGE_CURVE", f"'',#{vertices[start]},#{vertices[end]},#{line},.T.")
            sense = ".T." if a == key[0] else ".F."
            oriented.append(add("ORIENTED_EDGE", f"'',*,*,#{edges[key]},{sense}"))
        return add("EDGE_LOOP", "''," + refs(oriented))

    faces = []

    def face(boundaries):
        bounds = []
        for index, pids in enumerate(boundaries):
            stored = list(reversed(pids)) if reverse_bounds else list(pids)
            lid = loop(stored)
            sense = ".F." if reverse_bounds else ".T."
            kind = "FACE_OUTER_BOUND" if index == 0 and explicit_outer else "FACE_BOUND"
            bounds.append(add(kind, f"'',#{lid},{sense}"))
        p = np.array([coordinates[pid] for pid in boundaries[0]])
        local = p - p[0]
        normal = np.cross(local, np.roll(local, -1, axis=0)).sum(axis=0)
        normal /= np.linalg.norm(normal)
        if opposite_surface:
            normal = -normal
        x = p[1] - p[0]
        x /= np.linalg.norm(x)
        zref = add("DIRECTION", "''," + numbers(normal))
        xref = add("DIRECTION", "''," + numbers(x))
        placement = add("AXIS2_PLACEMENT_3D", f"'',#{boundaries[0][0]},#{zref},#{xref}")
        plane = add("PLANE", f"'',#{placement}")
        same = ".F." if opposite_surface else ".T."
        kind = "ADVANCED_FACE" if edge_loops else "FACE_SURFACE"
        faces.append(add(kind, f"'',{refs(bounds)},#{plane},{same}"))

    face([rings[0][0][::-1], *(ring[0] for ring in rings[1:])])
    face([rings[0][1], *(ring[1][::-1] for ring in rings[1:])])
    for index, (bottom, top) in enumerate(rings):
        for i in range(len(bottom)):
            j = (i + 1) % len(bottom)
            wall = [bottom[i], bottom[j], top[j], top[i]]
            face([wall if index == 0 else wall[::-1]])
    shell = add("CLOSED_SHELL", "''," + refs(faces))
    add("MANIFOLD_SOLID_BREP", f"'perforated prism',#{shell}")
    return "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n" + "\n".join(rows) + "\nENDSEC;\nEND-ISO-10303-21;\n"


def prism_properties(outer, holes, height, density, rotation, translation):
    """Independent polygon area integrals and prism product moments."""
    area, first, second, perimeter = 0., np.zeros(2), np.zeros((2, 2)), 0.
    for index, ring in enumerate((outer, *holes)):
        sign = 1 if index == 0 else -1
        for a, b in zip(ring, ring[1:] + ring[:1]):
            x, y = a
            u, v = b
            cross = sign * (x*v - u*y)
            area += cross / 2
            first += cross * np.array([x+u, y+v]) / 6
            second[0, 0] += cross * (x*x + x*u + u*u) / 12
            second[1, 1] += cross * (y*y + y*v + v*v) / 12
            second[0, 1] += cross * (2*x*y + x*v + u*y + 2*u*v) / 24
            perimeter += math.hypot(x-u, y-v)
    second[1, 0] = second[0, 1]
    center = first / area
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = height * (second - area*np.outer(center, center))
    covariance[2, 2] = area*height**3 / 12
    tensor = density * (np.trace(covariance)*np.eye(3) - covariance)
    rotation = np.asarray(rotation)
    return {"volume_m3": area*height, "mass_kg": density*area*height,
            "surface_area_m2": 2*area + height*perimeter,
            "centroid_m": rotation @ np.r_[center, height/2] + translation,
            "inertia_tensor_centroid_kg_m2": rotation @ tensor @ rotation.T}


class StepFaceRegionTests(unittest.TestCase):
    def check_prism(self, outer, holes=(), height=6., density=1., **options):
        rotation = options.get("rotation", np.eye(3))
        translation = np.asarray(options.get("translation", (0., 0., 0.)))
        got = verify_step_cad_geometry({"step_text": prism_step(outer, holes, height, **options),
                                       "density_kg_m3": density})
        expected = prism_properties(outer, holes, height, density, rotation, translation)
        self.assertTrue(got["conservation_verified"], got)
        self.assertTrue(got["watertight_manifold"], got)
        for key, value in expected.items():
            np.testing.assert_allclose(got[key], value, rtol=2e-10, atol=1e-9,
                                       err_msg=key)
        return got

    def test_annular_caps_do_not_count_the_void_twice(self):
        outer = [(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)]
        holes = [[(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]]
        result = self.check_prism(outer, holes)
        self.assertEqual(result["volume_m3"], 72.)
        self.assertEqual(result["surface_area_m2"], 168.)

    def test_multiple_holes_and_implicit_outer_bound(self):
        outer = [(0., 0.), (9., 0.), (9., 6.), (0., 6.)]
        holes = [[(1., 1.), (3., 1.), (3., 3.), (1., 3.)],
                 [(6., 2.), (8., 2.), (7., 5.)]]
        for explicit in (True, False):
            with self.subTest(explicit_outer=explicit):
                self.check_prism(outer, holes, height=2.5, density=7.8,
                                 explicit_outer=explicit)

    def test_concave_cap_and_concave_cap_with_hole(self):
        outer = [(4., 1.), (1., 1.), (1., 4.), (0., 4.), (0., 0.), (4., 0.)]
        hole = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.7), (0.2, 0.7)]
        for holes in ([], [hole]):
            with self.subTest(holes=len(holes)):
                self.check_prism(outer, holes, height=3., density=2.)

    def test_edge_loops_bound_orientation_and_surface_sense(self):
        outer = [(0., 0.), (8., 0.), (8., 4.), (0., 4.)]
        holes = [[(1., 1.), (2., 1.), (2., 3.), (1., 3.)]]
        rotation = np.array([[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]]) @ np.array(
            [[1., 0., 0.], [0., .6, -.8], [0., .8, .6]])
        for reverse in (False, True):
            for opposite in (False, True):
                with self.subTest(reverse=reverse, opposite=opposite):
                    self.check_prism(outer, holes, height=2., edge_loops=True,
                                     reverse_bounds=reverse, opposite_surface=opposite,
                                     rotation=rotation, translation=(10000., -20000., 30000.))


    def test_invalid_face_regions_are_refused(self):
        outer = [(0., 0.), (5., 0.), (5., 5.), (0., 5.)]
        inner = [(1., 1.), (3., 1.), (3., 3.), (1., 3.)]
        cases = [
            (outer, [[(6., 1.), (7., 1.), (7., 2.), (6., 2.)]]),
            (outer, [[(0., 1.), (2., 1.), (2., 2.), (0., 2.)]]),
            (outer, [inner, [(2., 2.), (4., 2.), (4., 4.), (2., 4.)]]),
            (outer, [inner, [(1.5, 1.5), (2., 1.5), (2., 2.), (1.5, 2.)]]),
            (outer, [inner[::-1]]),
            ([(0., 0.), (5., 4.), (0., 5.), (4., 0.)], []),
        ]
        for index, (outline, holes) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, "STEP"):
                verify_step_cad_geometry({"step_text": prism_step(outline, holes)})


    def test_incomplete_or_inconsistent_declared_geometry_is_refused(self):
        outer = [(0., 0.), (5., 0.), (5., 5.), (0., 5.)]
        holes = [[(1., 1.), (2., 1.), (2., 2.), (1., 2.)]]
        poly = prism_step(outer, holes)
        edge = prism_step(outer, holes, edge_loops=True)
        cases = {
            "duplicate_outer": poly.replace("FACE_BOUND(", "FACE_OUTER_BOUND(", 1),
            "missing_point": re.sub(r"#\d+=CARTESIAN_POINT[^\n]*\n", "", poly, count=1),
            "missing_loop": poly.replace("POLY_LOOP(", "UNSUPPORTED_LOOP(", 1),
            "nonplanar": poly.replace("CARTESIAN_POINT('',(0.0,0.0,0.0))",
                                       "CARTESIAN_POINT('',(0.0,0.0,0.025))", 1),
            "curved_surface": poly.replace("PLANE(", "CYLINDRICAL_SURFACE(", 1),
            "curved_edge": edge.replace("LINE(", "CIRCLE(", 1),
            "plane_normal": re.sub(r"DIRECTION\([^;]+", "DIRECTION('',(1.,1.,1.))", poly, count=1),
            "line_direction": re.sub(r"DIRECTION\([^;]+", "DIRECTION('',(1.,1.,1.))", edge, count=1),
            "surface_sense": re.sub(r"(FACE_SURFACE\([^;]+),\.T\.\)", r"\1,.F.)", poly, count=1),
            "curve_sense": re.sub(r"(EDGE_CURVE\([^;]+),\.T\.\)", r"\1,.F.)", edge, count=1),
            "disconnected_edge": re.sub(r"(ORIENTED_EDGE\([^;]+),\.([TF])\.\)",
                                         lambda m: m[1]+(",.F.)" if m[2] == "T" else ",.T.)"), edge, count=1),
        }
        for name, text in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, "STEP"):
                verify_step_cad_geometry({"step_text": text})


    def test_distant_support_anchors_do_not_hide_offsets_or_change_winding(self):
        outer = [(0., 0.), (5., 0.), (5., 5.), (0., 5.)]
        hole = [(1., 1.), (2., 1.), (2., 2.), (1., 2.)]
        for line in (False, True):
            original = prism_step(outer, [hole], edge_loops=line)
            if line:
                match = re.search(r"(#\d+=LINE\(''),#(\d+),#(\d+)\)", original)
                pid, vector = int(match[2]), int(match[3])
                base = [float(value) for value in re.search(
                    rf"#{pid}=CARTESIAN_POINT\('',\(([^()]*)", original)[1].split(',')]
                changed = original.replace(match[0], f"{match[1]},#990001,#{vector})")
            else:
                placement = re.search(r"PLANE\('',#(\d+)\)", original)[1]
                changed = re.sub(rf"(#{placement}=AXIS2_PLACEMENT_3D\('',)#\d+",
                                 r"\g<1>#990001", original)
                base = [0., 0., 0.]
            for shift, normal_offset in ((1e12, 0.), (1e12, .001), (1e15, 1.), (1e16, 0.)):
                anchor = f"#990001=CARTESIAN_POINT('',({shift},{base[1]},{normal_offset}));\n"
                text = changed.replace("ENDSEC;\nEND-ISO", anchor + "ENDSEC;\nEND-ISO")
                with self.subTest(line=line, shift=shift, offset=normal_offset):
                    if normal_offset:
                        with self.assertRaisesRegex(ValueError, "declared line|declared plane"):
                            verify_step_cad_geometry({"step_text": text})
                    else:
                        got = verify_step_cad_geometry({"step_text": text})
                        self.assertTrue(got["conservation_verified"], got)
                        self.assertEqual(got["volume_m3"], 144.)
                        self.assertEqual(got["surface_area_m2"], 192.)


if __name__ == "__main__":
    unittest.main()
