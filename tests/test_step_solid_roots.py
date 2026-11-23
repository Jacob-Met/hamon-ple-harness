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

if __name__ == "__main__":
    unittest.main()
