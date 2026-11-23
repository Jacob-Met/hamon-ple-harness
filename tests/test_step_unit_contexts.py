"""Length is attached to the represented geometry, not words in a STEP file."""
import re
import unittest

import numpy as np

from hamon_conservation_cad_verifier import verify_step_cad_geometry
from test_step_solid_roots import add_records, model


def with_context(text, *, length_ref=900001, prefix="$", angle_prefix="$", extra=(), base=900000):
    solid = int(re.search(r"#(\d+)=MANIFOLD_SOLID_BREP", text)[1])
    records = [
        f"#{base+1}=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT({prefix},.METRE.));",
        f"#{base+2}=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT({angle_prefix},.RADIAN.));",
        f"#{base+3}=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());",
        f"#{base+4}=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT"
        f"((#{length_ref},#{base+2},#{base+3}))REPRESENTATION_CONTEXT('','3D'));",
        f"#{base+5}=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{solid}),#{base+4});",
        *extra,
    ]
    return add_records(text, records)


class StepUnitContextTests(unittest.TestCase):
    def verify(self, text, **options):
        return verify_step_cad_geometry({"step_text": text, "density_kg_m3": 1., **options})

    def assert_cube(self, result, scale):
        self.assertTrue(result["conservation_verified"], result)
        self.assertEqual(result["unit_scale_m"], scale)
        for key, expected in {
            "volume_m3": 8 * scale**3,
            "mass_kg": 8 * scale**3,
            "surface_area_m2": 24 * scale**2,
            "centroid_m": np.full(3, scale),
            "inertia_tensor_centroid_kg_m2": np.eye(3) * (16/3) * scale**5,
            "principal_moments_kg_m2": np.full(3, (16/3) * scale**5),
        }.items():
            np.testing.assert_allclose(result[key], expected, rtol=5e-14, atol=0, err_msg=key)

    def test_angle_prefix_labels_and_unused_units_do_not_rescale_metres(self):
        original = model()
        cases = {
            "metre": with_context(original),
            "milliradian": with_context(original, angle_prefix=".MILLI."),
            "point label": with_context(original.replace("CARTESIAN_POINT('',", "CARTESIAN_POINT('INCH',", 1)),
            "unused millimetres": with_context(original, extra=[
                "#900006=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"]),
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                self.assert_cube(self.verify(text), 1.)

    def test_conversion_graph_determines_length_independently_of_unit_label(self):
        original = model()
        records = [
            "#900010=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);",
            "#900011=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#900001);",
            "#900012=(CONVERSION_BASED_UNIT('arbitrary label',#900011)LENGTH_UNIT()NAMED_UNIT(#900010));",
            "#900013=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(12.),#900012);",
            "#900014=(CONVERSION_BASED_UNIT('another label',#900013)LENGTH_UNIT()NAMED_UNIT(#900010));",
        ]
        for reference, scale in ((900012, .0254), (900014, .3048)):
            with self.subTest(reference=reference):
                self.assert_cube(self.verify(with_context(original, length_ref=reference,
                                                          prefix=".MILLI.", extra=records)), scale)
        renamed = [
            "#900010=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);",
            "#900011=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.),#900001);",
            "#900012=(CONVERSION_BASED_UNIT('INCH',#900011)LENGTH_UNIT()NAMED_UNIT(#900010));",
        ]
        self.assert_cube(self.verify(with_context(original, length_ref=900012, extra=renamed)), 2.)


if __name__ == "__main__":
    unittest.main()
