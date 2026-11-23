"""Mass belongs to a declared STEP solid, not every geometric definition."""
import re
import unittest
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry
from test_step_face_regions import prism_step

PROPERTIES = ("volume_m3", "surface_area_m2", "mass_kg", "centroid_m",
              "inertia_tensor_centroid_kg_m2", "principal_moments_kg_m2")

def model(size=2., offset=(0., 0., 0.)):
    return prism_step([(0., 0.), (size, 0.), (size, size), (0., size)],
                      height=size, translation=offset)

def add_records(original, records):
    return original.replace("ENDSEC;\nEND-ISO", "\n".join(records) + "\nENDSEC;\nEND-ISO")

def append_entities(original, other, keep_solid=False):
    data = other.split("DATA;\n", 1)[1].split("ENDSEC;", 1)[0]
    if not keep_solid:
        data = re.sub(r"^#\d+=MANIFOLD_SOLID_BREP[^\n]*\n", "", data, flags=re.M)
    data = re.sub(r"#(\d+)", lambda m: "#" + str(10000 + int(m[1])), data)
    return add_records(original, [data])


def oriented_faces(original, cycle=False):
    text, wrappers = original, []
    pattern = r"#(\d+)=FACE_SURFACE\('',\(([^()]*)\),#\d+,\.T\.\);"
    for index, match in enumerate(re.finditer(pattern, original)):
        fid = int(match[1])
        if index % 2:
            for bid in re.findall(r"#(\d+)", match[2]):
                text = re.sub(rf"(#{bid}=FACE_OUTER_BOUND\('',#\d+,)\.T\.", r"\1.F.", text)
            text = text.replace(match[0], match[0].replace(",.T.);", ",.F.);"))
        wrapper = 900000 + index
        sense = ".F." if index % 2 else ".T."
        target = wrapper if cycle and index == 0 else fid
        wrappers.append(f"#{wrapper}=ORIENTED_FACE('',*,#{target},{sense});")
    shell_refs = ",".join(f"#{900000+i}" for i in range(len(wrappers)))
    text = re.sub(r"(CLOSED_SHELL\('',)\([^()]*\)", rf"\1({shell_refs})", text, count=1)
    return add_records(text, wrappers)

class StepSolidRootTests(unittest.TestCase):
    def verify(self, text, **options):
        return verify_step_cad_geometry({"step_text": text, "density_kg_m3": 1., **options})

    def assert_properties(self, actual, expected):
        self.assertTrue(actual["conservation_verified"], actual)
        for key in PROPERTIES:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-12,
                                       err_msg=key)

    def assert_withheld(self, text, **options):
        try:
            result = self.verify(text, **options)
        except ValueError:
            return
        self.assertFalse(result.get("conservation_verified", False), result)
        self.assertTrue(all(key not in result for key in PROPERTIES), result)

    def test_unreferenced_closed_and_curved_geometry_is_not_added(self):
        original = model()
        expected = self.verify(original)
        orphan = model(1., (5., 0., 0.))
        for extra in (orphan, orphan.replace("PLANE(", "CYLINDRICAL_SURFACE(", 1)):
            with self.subTest(curved="CYLINDRICAL_SURFACE" in extra):
                self.assert_properties(self.verify(append_entities(original, extra)), expected)

    def test_unreferenced_primitive_is_not_added_to_the_solid(self):
        original = model()
        extra = ["#900001=CARTESIAN_POINT('',(20.,0.,0.));",
                 "#900002=AXIS2_PLACEMENT_3D('',#900001,$,$);",
                 "#900003=BLOCK('',#900002,1.,1.,1.);"]
        self.assert_properties(self.verify(add_records(original, extra)), self.verify(original))

    def test_missing_root_shell_or_declared_face_withholds_properties(self):
        original = model()
        cases = [re.sub(r"^#\d+=MANIFOLD_SOLID_BREP[^\n]*\n", "", original, flags=re.M),
                 re.sub(r"^#\d+=CLOSED_SHELL[^\n]*\n", "", original, flags=re.M),
                 re.sub(r"(CLOSED_SHELL\('',\()#\d+,", r"\1", original, count=1),
                 re.sub(r"(CLOSED_SHELL\('',\()#\d+", r"\g<1>#999999", original, count=1)]
        for index, text in enumerate(cases):
            with self.subTest(case=index):
                self.assert_withheld(text)

    def test_multiple_definitions_require_selection(self):
        first, second = model(), model(1., (5., 0., 0.))
        text = append_entities(first, second, keep_solid=True)
        ids = [int(value) for value in re.findall(r"#(\d+)=MANIFOLD_SOLID_BREP", text)]
        self.assert_withheld(text)
        self.assert_properties(self.verify(text, solid_entity_id=ids[0]), self.verify(first))
        self.assert_properties(self.verify(text, solid_entity_id=ids[1]), self.verify(second))
        self.assert_withheld(text, solid_entity_id=999999)


    def test_oriented_face_occurrences_follow_the_shell(self):
        self.assert_properties(self.verify(oriented_faces(model())), self.verify(model()))
        self.assert_withheld(oriented_faces(model(), cycle=True))

    def test_faceted_faces_and_reversed_shell_are_complete_solids(self):
        original = model()
        faceted = re.sub(r"FACE_SURFACE\('',\(([^()]*)\),#\d+,\.T\.\)",
                         r"FACE('',(\1))", original).replace("MANIFOLD_SOLID_BREP(", "FACETED_BREP(")
        self.assert_properties(self.verify(faceted), self.verify(original))
        shell = re.search(r"#(\d+)=CLOSED_SHELL", original)[1]
        reverse = re.sub(r"(FACE_OUTER_BOUND\('',#\d+,)\.T\.", r"\1.F.", original)
        reverse = re.sub(r"(FACE_SURFACE\([^;\n]+),\.T\.\);", r"\1,.F.);", reverse)
        reverse = re.sub(r"(MANIFOLD_SOLID_BREP\('[^']*',)#\d+", r"\g<1>#900500", reverse)
        reverse = add_records(reverse, [f"#900500=ORIENTED_CLOSED_SHELL('',*,#{shell},.F.);"])
        self.assert_properties(self.verify(reverse), self.verify(original))

    def test_duplicate_or_disconnected_face_sets_withhold_properties(self):
        original = model()
        duplicate = re.sub(r"(CLOSED_SHELL\('',\()(#\d+),", r"\1\2,\2,", original, count=1)
        self.assert_withheld(duplicate)
        disconnected = append_entities(original, model(1., (5., 0., 0.)))
        faces = ",".join("#" + value for value in re.findall(r"#(\d+)=FACE_SURFACE", disconnected))
        disconnected = re.sub(r"(CLOSED_SHELL\('',)\([^()]*\)", rf"\1({faces})", disconnected, count=1)
        self.assert_withheld(disconnected)

    def test_enclosed_voids_are_not_silently_added_as_material(self):
        combined = append_entities(model(), model(.5, (.75, .75, .75)))
        shells = re.findall(r"#(\d+)=CLOSED_SHELL", combined)
        combined = re.sub(r"MANIFOLD_SOLID_BREP\('[^']*',#\d+\)",
                          f"BREP_WITH_VOIDS('',#{shells[0]},(#900666))", combined)
        combined = add_records(combined, [f"#900666=ORIENTED_CLOSED_SHELL('',*,#{shells[1]},.F.);"])
        with self.assertRaisesRegex(ValueError, "void shells"):
            self.verify(combined)


if __name__ == "__main__":
    unittest.main()
