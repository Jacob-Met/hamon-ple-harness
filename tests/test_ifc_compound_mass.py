"""Mass and first moments of independent IFC Body items through the registered router."""
import numpy as np
import itertools
import unittest

from hamon_ifc_spatial_clash import load_ifc_solids, run_router_ifc_clash

FACES = [(1, 3, 2), (1, 2, 4), (1, 4, 3), (2, 3, 4)]


def course(*, reverse=True, reflect=False, millimetres=False, cavity=False,
           open_surface=False, declared_open=False, swap=False, shift=4.0, mirror_left=False, probe=False, right_scale=2., misorient=False):
    unit = 1000 if millimetres else 1
    entities = []

    def add(kind):
        entities.append(f"#{len(entities) + 1}={kind};")
        return f"#{len(entities)}"

    def coords(values):
        return "(" + ",".join(str(float(v) * unit) for v in values) + ")"

    zero = add("IFCCARTESIANPOINT((0.,0.,0.))")
    placement = add(f"IFCAXIS2PLACEMENT3D({zero},$,$)")
    context = add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-6,{placement},$)")
    length = add("IFCSIUNIT(*,.LENGTHUNIT.," + (".MILLI." if millimetres else "$") + ",.METRE.)")
    units = add(f"IFCUNITASSIGNMENT(({length}))")
    add(f"IFCPROJECT('0AzR0sseD4bBOcPJ50wSE1',$,'course',$,$,$,$,({context}),{units})")

    vertices = np.array([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])

    def tessellation(points, faces, closed=True):
        pts = add("IFCCARTESIANPOINTLIST3D((" + ",".join(coords(p) for p in points) + "))")
        triangles = ",".join("(" + ",".join(map(str, f)) + ")" for f in faces)
        flag = ".T." if closed else ".F."
        return add(f"IFCTRIANGULATEDFACESET({pts},$,{flag},({triangles}),$)")

    left_points = vertices
    left_faces = FACES.copy()
    if cavity:
        left_points = np.vstack([vertices, vertices / 4 + 1/8])
        left_faces += [tuple(i + 4 for i in reversed(f)) for f in FACES]
    left = tessellation(left_points, left_faces)
    right_points = right_scale * vertices
    right_faces = [f[::-1] if reverse else f for f in FACES]
    if misorient:
        right_faces[0] = right_faces[0][::-1]
    if open_surface:
        right_faces.pop(0)
    if reflect or mirror_left:
        def mapped(item, translation, mirror):
            source = add(f"IFCSHAPEREPRESENTATION({context},'Body','Tessellation',({item}))")
            rep_map = add(f"IFCREPRESENTATIONMAP({placement},{source})")
            origin = add(f"IFCCARTESIANPOINT({coords(translation)})")
            y = add("IFCDIRECTION((0.,-1.,0.))") if mirror else "$"
            transform = add(f"IFCCARTESIANTRANSFORMATIONOPERATOR3D($,{y},{origin},1.,$)")
            return add(f"IFCMAPPEDITEM({rep_map},{transform})")

        left = mapped(left, (0., 0., 0.), mirror_left)
        right = mapped(tessellation(right_points, right_faces, not declared_open), (shift, 0., 0.), reflect)
        rep_type = "MappedRepresentation"
    else:
        right = tessellation(right_points + [shift, 0., 0.], right_faces, not declared_open)
        rep_type = "Tessellation"
    items = [right, left] if swap else [left, right]
    shape = add(f"IFCSHAPEREPRESENTATION({context},'Body','{rep_type}',({','.join(items)}))")
    body = add(f"IFCPRODUCTDEFINITIONSHAPE($,$,({shape}))")
    local = add(f"IFCLOCALPLACEMENT($,{placement})")
    add(f"IFCBEAM('0AzR0sseD4bBOcPJ50wSE2',$,'compound',$,$,{local},{body},$,.BEAM.)")
    if probe:
        witness = tessellation(vertices + [12., 2., 1.], FACES)
        witness_shape = add(f"IFCSHAPEREPRESENTATION({context},'Body','Tessellation',({witness}))")
        witness_body = add(f"IFCPRODUCTDEFINITIONSHAPE($,$,({witness_shape}))")
        add(f"IFCDUCTSEGMENT('0AzR0sseD4bBOcPJ50wSE3',$,'probe',$,$,{local},{witness_body},$,.RIGIDSEGMENT.)")
    return ("ISO-10303-21;\nHEADER;FILE_DESCRIPTION(('Compound mass course'),'2;1');\n"
            "FILE_NAME('compound.ifc','2026-01-01T00:00:00',('test'),('test'),'course','test','');\n"
            "FILE_SCHEMA(('IFC4'));ENDSEC;\nDATA;\n"
            + "\n".join(entities) + "\nENDSEC;END-ISO-10303-21;")


class CompoundMassTest(unittest.TestCase):
    def test_moments(self):
        for settings in itertools.product([False, True], repeat=6):
            reverse, reflect, millimetres, cavity, swap, mirror_left = settings
            with self.subTest(settings=settings):
                text = course(reverse=reverse, reflect=reflect, millimetres=millimetres,
                              cavity=cavity, swap=swap, mirror_left=mirror_left)
                result = run_router_ifc_clash({"ifc_text": text})
                volumes = np.array([1., 8., -1/64 if cavity else 0.]) / 6
                left_y = -1 if mirror_left else 1
                centres = np.array([[.25, left_y*.25, .25], [4.5, -.5 if reflect else .5, .5],
                                    [3/16, left_y*3/16, 3/16]])
                self.assertAlmostEqual(result["total_solid_volume_m3"], volumes.sum(), places=12)
                mesh, = load_ifc_solids(text)
                np.testing.assert_allclose(mesh.centroid_m(), np.average(centres, weights=volumes, axis=0),
                                           rtol=0, atol=1e-12)

    def test_equal_opposing_items(self):
        # Outward source triangles mapped through a reflection in only one item.
        text = course(reverse=False, reflect=True, right_scale=1.)
        self.assertAlmostEqual(run_router_ifc_clash({"ifc_text": text})["total_solid_volume_m3"],
                               1/3, places=12)
        mesh, = load_ifc_solids(text)
        np.testing.assert_allclose(mesh.centroid_m(), [2.25, 0., .25], rtol=0, atol=1e-12)

    def test_unresolved_quantity_preserves_clash_report(self):
        for options in [{"open_surface": True}, {"misorient": True}, {"declared_open": True},
                        {"declared_open": True, "reflect": True}]:
            with self.subTest(options=options):
                result = run_router_ifc_clash({"ifc_text": course(probe=True, **options)})
                self.assertTrue(result["ok"])
                self.assertEqual(result["solid_count"], 2)
                pair, = result["all_pairs"]
                self.assertEqual(pair["status"], "clear")
                self.assertGreater(pair["min_distance_m"], 0.)
                self.assertIsNotNone(pair["witness_point_m"])
                self.assertIsNone(result["total_solid_volume_m3"])
                self.assertEqual(result["volume_status"], "unresolved")
