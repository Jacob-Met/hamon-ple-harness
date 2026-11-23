"""World-space clash and clearance witnesses for positioned IFC extrusions."""
import math
import unittest

from hamon_ifc_spatial_clash import run_router_ifc_clash


def scene(probe, *, sign=1, hollow=False, rotated=True, scale=1.0):
    # Parent Rz(90) @ child Rx(90) @ solid Ry(90) gives
    # world(x,y,z)=(8.5-x, 22.5+z, 30.25+y); rotations do not commute.
    x, y, z = probe
    loc = (8.5 - x, 22.5 + z, 30.25 + y - 0.05)
    profile = ("IFCRECTANGLEHOLLOWPROFILEDEF(.AREA.,$,#18,1.6,2.4,0.25,$,$)" if hollow
               else "IFCRECTANGLEPROFILEDEF(.AREA.,$,#18,0.4,0.8)")
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('sweep.ifc','2026-01-01T00:00:00',('test'),('test'),'geometry test','test','');
FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCAXIS2PLACEMENT3D(#1,$,$);
#3=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-7,#2,$);
#4=IFCCARTESIANPOINT((10.,20.,30.));
#5=IFCDIRECTION((0.,0.,1.));
#6=IFCDIRECTION((0.,1.,0.));
#7=IFCAXIS2PLACEMENT3D(#4,#5,#6);
#8=IFCLOCALPLACEMENT($,#7);
#9=IFCCARTESIANPOINT((2.,1.,0.));
#10=IFCDIRECTION((0.,-1.,0.));
#11=IFCDIRECTION((1.,0.,0.));
#12=IFCAXIS2PLACEMENT3D(#9,#10,#11);
#13=IFCLOCALPLACEMENT(#8,#12);
#14=IFCCARTESIANPOINT((0.5,0.25,-0.5));
#15=IFCDIRECTION((0.,0.,-1.));
#16=IFCAXIS2PLACEMENT3D(#14,#11,#15);
#17=IFCCARTESIANPOINT((0.25,-0.125));
#18=IFCAXIS2PLACEMENT2D(#17,{"#19" if rotated else "$"});
#19=IFCDIRECTION((0.,1.));
#20={profile};
#21=IFCDIRECTION(({3 * scale:.16E},0.,{4 * sign * scale:.16E}));
#22=IFCEXTRUDEDAREASOLID(#20,#16,#21,5.);
#23=IFCSHAPEREPRESENTATION(#3,'Body','SweptSolid',(#22));
#24=IFCPRODUCTDEFINITIONSHAPE($,$,(#23));
#25=IFCBEAM('0AzR0sseD4bBOcPJ50wSE2',$,'sweep',$,$,#13,#24,$,.BEAM.);
#26=IFCCARTESIANPOINT(({loc[0]},{loc[1]},{loc[2]}));
#27=IFCAXIS2PLACEMENT3D(#26,$,$);
#28=IFCLOCALPLACEMENT($,#27);
#29=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,0.1,0.1);
#30=IFCEXTRUDEDAREASOLID(#29,#2,#5,0.1);
#31=IFCSHAPEREPRESENTATION(#3,'Body','SweptSolid',(#30));
#32=IFCPRODUCTDEFINITIONSHAPE($,$,(#31));
#33=IFCDUCTSEGMENT('0AzR0sseD4bBOcPJ50wSE3',$,'probe',$,$,#28,#32,$,.RIGIDSEGMENT.);
#34=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#35=IFCUNITASSIGNMENT((#34));
#36=IFCPROJECT('0AzR0sseD4bBOcPJ50wSE4',$,'Project',$,$,$,$,(#3),#35);
ENDSEC;END-ISO-10303-21;"""


def solid_coordinates(point):
    return 8.5 - point[0], point[2] - 30.25, point[1] - 22.5


class ExtrusionFrames(unittest.TestCase):
    def physical_pair(self, probe, sign, *, hollow=False, rotated=True, scale=1.0, axes=()):
        model = scene(probe, sign=sign, hollow=hollow, rotated=rotated, scale=scale)
        for old, new in axes:
            self.assertIn(old, model)
            model = model.replace(old, new)
        out = run_router_ifc_clash({"ifc_text": model, "required_clearance_m": 0.1})
        self.assertTrue(out["ok"])
        self.assertEqual((out["solid_count"], out["pair_count"]), (2, 1))
        (pair,) = out["all_pairs"]
        # Verify each reported point against the independently defined solids.
        # At height z the section center has moved by (3/4)*|z| in solid X.
        w, h = ((2.4, 1.6) if hollow else ((0.8, 0.4) if rotated else (0.4, 0.8)))

        def in_sweep(world):
            x, y, z = solid_coordinates(world)
            along = sign * z
            dx, dy = abs(x - (0.25 + 0.75 * along)), abs(y + 0.125)
            self.assertTrue(-1e-10 <= along <= 4 + 1e-10)
            self.assertLessEqual(dx, w / 2 + 1e-10)
            self.assertLessEqual(dy, h / 2 + 1e-10)
            if hollow:
                self.assertTrue(dx >= w / 2 - 0.25 - 1e-10 or dy >= h / 2 - 0.25 - 1e-10)

        def in_probe(world):
            self.assertLessEqual(max(abs(a - b) for a, b in zip(solid_coordinates(world), probe)), 0.05 + 1e-10)

        a, b = pair["closest_points_m"]
        in_sweep(a)
        in_probe(b)
        self.assertAlmostEqual(math.dist(a, b), pair["min_distance_m"], places=10)
        if pair["status"] == "hard_clash":
            in_sweep(pair["witness_point_m"])
            in_probe(pair["witness_point_m"])
        return pair

    def test_signed_relocated_rectangle_and_clearance(self):
        for sign in (1, -1):
            for scale in (1., 1.e-200, 1.e200):
                with self.subTest(sign=sign, scale=scale):
                    inside = self.physical_pair((2.5, -0.125, 3 * sign), sign, scale=scale)
                    self.assertEqual(inside["status"], "hard_clash")
            outside = self.physical_pair((0.25, 2.125, 3 * sign), sign)
            self.assertEqual(outside["status"], "clear")
            # Gap normal to the slanted x-z face is 1.7625 / 1.25.
            self.assertAlmostEqual(outside["min_distance_m"], math.hypot(2., 1.41), places=10)
            control = self.physical_pair((2.5, -0.125, 3 * sign), sign, rotated=False)
            self.assertEqual(control["status"], "hard_clash")

    def test_hollow_wall_and_void_follow_the_same_sweep(self):
        for sign in (1, -1):
            wall = self.physical_pair((3.575, -0.125, 3 * sign), sign, hollow=True)
            self.assertEqual(wall["status"], "hard_clash")
            void = self.physical_pair((2.5, -0.125, 3 * sign), sign, hollow=True)
            self.assertEqual(void["status"], "clear")
            self.assertAlmostEqual(void["min_distance_m"], 0.5, places=10)

    def test_invalid_or_unrepresentable_sweep_has_no_clearance_certificate(self):
        base = scene((2.5, -0.125, 3))
        for ratios in ("(0.,0.,0.)", "(1.,2.,0.)", "(1.e250,0.,1.e-250)", "(0.,1.)"):
            with self.subTest(ratios=ratios):
                result = run_router_ifc_clash({
                    "ifc_text": base.replace("#21=IFCDIRECTION((3.0000000000000000E+00,0.,4.0000000000000000E+00))",
                                             f"#21=IFCDIRECTION({ratios})")
                })
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "unresolved_solid_geometry")

    def test_equivalent_placement_directions_preserve_contact_and_clearance(self):
        changes = [
            ("#19=IFCDIRECTION((0.,1.))", "#19=IFCDIRECTION((0.,1.E-250))"),  # profile
            ("#19=IFCDIRECTION((0.,1.))", "#19=IFCDIRECTION((0.,1.E250))"),
            ("#6=IFCDIRECTION((0.,1.,0.))", "#6=IFCDIRECTION((0.,1.E-250,0.))"),  # parent X
            ("#10=IFCDIRECTION((0.,-1.,0.))", "#10=IFCDIRECTION((0.,-1.E250,0.))"),  # child Z
            ("#15=IFCDIRECTION((0.,0.,-1.))", "#15=IFCDIRECTION((0.,0.,-1.E-250))"),  # solid X
            # Axis2Placement3D projects RefDirection onto the plane normal to Axis.
            # The same positive Y reference survives even when its projection
            # would underflow after normalizing the two raw components together.
            ("#6=IFCDIRECTION((0.,1.,0.))", "#6=IFCDIRECTION((0.,1.E-250,1.E250))"),
        ]
        for change in changes:
            with self.subTest(change=change):
                embedded = self.physical_pair((2.8, -0.125, 3), 1, axes=[change])
                self.assertEqual(embedded["status"], "hard_clash")
                separated = self.physical_pair((0.25, 2.125, -3), -1, axes=[change])
                self.assertAlmostEqual(separated["min_distance_m"], math.hypot(2., 1.41), places=10)

    def test_invalid_explicit_placement_axes_withhold_geometry(self):
        base = scene((2.8, -0.125, 3))
        for replacement in ("(0.,0.,1.)", "(0.,0.,0.)", "(0.,1.)"):
            with self.subTest(replacement=replacement):
                model = base.replace("#6=IFCDIRECTION((0.,1.,0.))", f"#6=IFCDIRECTION({replacement})")
                result = run_router_ifc_clash({"ifc_text": model})
                self.assertEqual(result["status"], "unresolved_solid_geometry")
                self.assertFalse(result["ok"])
