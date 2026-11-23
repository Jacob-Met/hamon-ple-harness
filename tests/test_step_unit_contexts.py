"""Length is attached to the represented geometry, not words in a STEP file."""
from fractions import Fraction
import math
import re
import unittest

import numpy as np

from hamon_conservation_cad_verifier import verify_step_cad_geometry
from test_step_solid_roots import add_records, append_entities, model


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

    def assert_withheld(self, text, **options):
        try:
            result = self.verify(text, **options)
        except ValueError:
            return
        self.assertFalse(result.get("conservation_verified", False), result)
        self.assertNotIn("volume_m3", result)

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

    def test_each_selected_definition_uses_its_own_context(self):
        first = with_context(model())
        second = with_context(model(), prefix=".MILLI.", base=910000, length_ref=910001)
        combined = append_entities(first, second, keep_solid=True)
        ids = [int(value) for value in re.findall(r"#(\d+)=MANIFOLD_SOLID_BREP", combined)]
        self.assert_withheld(combined)
        for eid, expected in zip(ids, (1., .001)):
            with self.subTest(solid=eid):
                result = self.verify(combined, solid_entity_id=eid)
                self.assert_cube(result, expected)
                self.assertEqual(result["unit_scale_source"], "representation_context")
        repeated = with_context(first, base=910000, length_ref=910001)
        self.assert_cube(self.verify(repeated), 1.)
        conflict = with_context(first, prefix=".MILLI.", base=910000, length_ref=910001)
        self.assert_withheld(conflict)
        self.assert_cube(self.verify(conflict, unit_scale_m=.01), .01)

    def test_incomplete_or_inconsistent_unit_graphs_withhold_properties(self):
        good = with_context(model())
        cases = {
            "missing context": good.replace(",#900004);", ",#999999);"),
            "missing unit": good.replace("((#900001,#900002,#900003))", "((#999999,#900002,#900003))"),
            "no length": good.replace("((#900001,#900002,#900003))", "((#900002,#900003))"),
            "repeated unit": good.replace("((#900001,#900002,#900003))", "((#900001,#900001,#900003))"),
            "two length units": add_records(good.replace(",#900003))", ",#900006))"), [
                "#900006=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.CENTI.,.METRE.));"]),
            "wrong SI dimension": good.replace("SI_UNIT($,.METRE.)", "SI_UNIT($,.SECOND.)"),
            "unknown SI prefix": good.replace("SI_UNIT($,.METRE.)", "SI_UNIT(.UNKNOWN.,.METRE.)"),
            "extra length attributes": good.replace("LENGTH_UNIT()", "LENGTH_UNIT(1.)"),
            "unknown contextual length": good.replace("SI_UNIT($,.METRE.)", "CONTEXT_DEPENDENT_UNIT('custom')"),
            "unattached units": re.sub(r"#900005=[^;]+;", "", good),
        }
        records = [
            "#900010=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);",
            "#900011=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.),#900001);",
            "#900012=(CONVERSION_BASED_UNIT('custom',#900011)LENGTH_UNIT()NAMED_UNIT(#900010));",
        ]
        converted = with_context(model(), length_ref=900012, extra=records)
        for factor in ("0.", "-2.", "NaN", "1.E400", "1.E-400"):
            cases[f"invalid final factor {factor}"] = converted.replace("LENGTH_MEASURE(2.)", f"LENGTH_MEASURE({factor})")
        cases["cycle"] = converted.replace("LENGTH_MEASURE(2.),#900001", "LENGTH_MEASURE(2.),#900012")
        cases["factor dimension"] = converted.replace("LENGTH_MEASURE(2.),#900001", "LENGTH_MEASURE(2.),#900002")
        cases["declared dimension"] = converted.replace("DIMENSIONAL_EXPONENTS(1.,0.,", "DIMENSIONAL_EXPONENTS(2.,0.,")
        cases["measure type"] = converted.replace("LENGTH_MEASURE(2.)", "AREA_MEASURE(2.)")
        for name, text in cases.items():
            with self.subTest(case=name):
                self.assert_withheld(text)
                # A declared coordinate scale remains a deliberate override.
                self.assert_cube(self.verify(text, unit_scale_m=1.), 1.)

    def test_all_standard_length_prefixes_and_exact_conversion_chains(self):
        prefixes = {
            "YOCTO": -24, "ZEPTO": -21, "ATTO": -18, "FEMTO": -15, "PICO": -12,
            "NANO": -9, "MICRO": -6, "MILLI": -3, "CENTI": -2, "DECI": -1,
            "DECA": 1, "HECTO": 2, "KILO": 3, "MEGA": 6, "GIGA": 9, "TERA": 12,
            "PETA": 15, "EXA": 18, "ZETTA": 21, "YOTTA": 24,
        }
        for prefix, exponent in prefixes.items():
            with self.subTest(prefix=prefix):
                self.assert_cube(self.verify(with_context(model(), prefix=f".{prefix}.")),
                                 float(Fraction(10)**exponent))
        records = [
            "#900010=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);",
            "#900011=MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E300),#900001);",
            "#900012=(CONVERSION_BASED_UNIT('large',#900011)LENGTH_UNIT()NAMED_UNIT(#900010));",
            "#900013=(LENGTH_MEASURE_WITH_UNIT()MEASURE_WITH_UNIT(LENGTH_MEASURE(2.E-300),#900012));",
            "#900014=CONVERSION_BASED_UNIT(#900010,'balanced chain',#900013);",
        ]
        self.assert_cube(self.verify(with_context(model(), length_ref=900014, extra=records)), 2.)

    def test_labels_comments_and_record_order_do_not_change_geometry(self):
        text = with_context(model())
        text = text.replace("CARTESIAN_POINT('',", "CARTESIAN_POINT('INCH; ) /* .MILLI. */ O''Brien',", 1)
        text = text.replace("LENGTH_UNIT()NAMED_UNIT(*)", "LENGTH_UNIT() /* separator */ NAMED_UNIT(*)")
        self.assert_cube(self.verify(text), 1.)
        lines = text.splitlines()
        records = [line for line in lines if line.startswith("#")]
        shuffled = "DATA;\n" + "\n".join(reversed(records)) + "\nENDSEC;"
        self.assert_cube(self.verify(shuffled), 1.)
        complex_rep = re.sub(r"(#900005=)ADVANCED_BREP_SHAPE_REPRESENTATION\(([^;]+)\);",
                             r"\1(ADVANCED_BREP_SHAPE_REPRESENTATION()REPRESENTATION(\2)SHAPE_REPRESENTATION());", text)
        self.assert_cube(self.verify(complex_rep), 1.)

    def test_explicit_scale_and_unambiguous_legacy_inputs_remain_supported(self):
        unitless = model().replace("CARTESIAN_POINT('',", "CARTESIAN_POINT('FOOT',", 1)
        result = self.verify(unitless)
        self.assert_cube(result, 1.)
        self.assertEqual(result["unit_scale_source"], "implicit_metre")
        for record in ("#900001=SI_UNIT(.MILLI.,.METRE.);",
                       "#900001=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"):
            with self.subTest(record=record):
                legacy = self.verify(add_records(unitless, [record]))
                self.assert_cube(legacy, .001)
                self.assertEqual(legacy["unit_scale_source"], "standalone_length_unit")
        ambiguous = add_records(unitless, ["#900001=SI_UNIT($,.METRE.);",
                                           "#900002=SI_UNIT(.MILLI.,.METRE.);"])
        self.assert_withheld(ambiguous)
        override = self.verify(ambiguous, unit_scale_m=.3048)
        self.assert_cube(override, .3048)
        self.assertEqual(override["unit_scale_source"], "explicit_override")
        for scale in (0., -1., math.nan, math.inf, -math.inf, True, False):
            with self.subTest(scale=scale):
                self.assert_withheld(unitless, unit_scale_m=scale)

    def test_unrelated_ifc_records_cannot_choose_units_for_a_step_definition(self):
        text = with_context(model(), prefix=".MILLI.")
        solid = int(re.search(r"#(\d+)=MANIFOLD_SOLID_BREP", text)[1])
        for records in (["#999001=IFCCARTESIANPOINT((6.,8.,3.));"],
                        ["#999001=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);",
                         "#999002=IFCUNITASSIGNMENT((#999001));"]):
            for selection in ({}, {"solid_entity_id": solid}):
                with self.subTest(records=records, selection=selection):
                    self.assert_cube(self.verify(add_records(text, records), **selection), .001)
        ifc = [
            "#999001=IFCCARTESIANPOINT((0.,0.,0.));",
            "#999002=IFCAXIS2PLACEMENT3D(#999001,$,$);",
            "#999003=IFCLOCALPLACEMENT($,#999002);",
            "#999004=IFCCARTESIANPOINT((0.,0.));",
            "#999005=IFCAXIS2PLACEMENT2D(#999004,$);",
            "#999006=IFCRECTANGLEPROFILEDEF(.AREA.,$,#999005,1.,1.);",
            "#999007=IFCDIRECTION((0.,0.,1.));",
            "#999008=IFCEXTRUDEDAREASOLID(#999006,#999002,#999007,1.);",
            "#999009=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#999008));",
            "#999010=IFCPRODUCTDEFINITIONSHAPE($,$,(#999009));",
            "#999011=IFCBUILDINGELEMENTPROXY('foreign',$,$,$,$,#999003,#999010,$,$);",
        ]
        self.assert_cube(self.verify(add_records(text, ifc), solid_entity_id=solid), .001)
        primitive = "DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));#2=AXIS2_PLACEMENT_3D('',#1,$,$);" \
                    "#3=BLOCK('',#2,2.,2.,2.);#4=SI_UNIT(.MILLI.,.METRE.);" \
                    "#5=IFCCARTESIANPOINT((6.,8.,3.));ENDSEC;"
        self.assert_withheld(primitive)


if __name__ == "__main__":
    unittest.main()
