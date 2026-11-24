"""Signed half-space and complete polygon opening controls through the ordinary IFC reader."""
import unittest

from hamon_ifc_spatial_clash import load_ifc_solids


class Source:
    def __init__(self):
        self.rows = []
        si = self.add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)")
        self.add(f"IFCUNITASSIGNMENT((#{si}))")
        self.frame = self.frame_at(0.)
        self.place = self.add(f"IFCLOCALPLACEMENT($,#{self.frame})")
        self.context = self.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-6,#{self.frame},$)")
        self.up = self.add("IFCDIRECTION((0.,0.,1.))")

    def add(self, s):
        self.rows.append(f"#{len(self.rows) + 1}={s};")
        return len(self.rows)

    def frame_at(self, z, normal=1):
        p = self.add(f"IFCCARTESIANPOINT((0.,0.,{z}))")
        axis = self.add(f"IFCDIRECTION((0.,0.,{normal}))")
        x = self.add("IFCDIRECTION((1.,0.,0.))")
        return self.add(f"IFCAXIS2PLACEMENT3D(#{p},#{axis},#{x})")

    def ring(self, pts, closed=True):
        ids = [self.add(f"IFCCARTESIANPOINT(({x},{y}))") for x, y in pts]
        if closed:
            ids.append(ids[0])
        return self.add("IFCPOLYLINE((" + ",".join(f"#{i}" for i in ids) + "))")

    def prism(self, profile, depth=2.):
        return self.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{self.frame},#{self.up},{depth})")

    def box(self):
        return self.prism(self.add("IFCRECTANGLEPROFILEDEF(.AREA.,$,$,2.,2.)"))

    def product(self, kind, items):
        ids = ",".join(f"#{i}" for i in items)
        shape = self.add(f"IFCSHAPEREPRESENTATION(#{self.context},'Body','SolidModel',({ids}))")
        body = self.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        return self.add(f"{kind}('0{str(body).zfill(21)}',$,'solid',$,$,#{self.place},#{body},$,$)")

    def void(self, host, opening):
        self.add(f"IFCRELVOIDSELEMENT('000000000000000000000v',$,$,$,#{host},#{opening})")

    def clip(self, z, normal, flag, bounded=False):
        host = self.box()
        plane = self.add(f"IFCPLANE(#{self.frame_at(z, normal)})")
        if bounded:
            ring = self.ring([(-2., -2.), (0., -2.), (0., 2.), (-2., 2.)])
            cutter = self.add(f"IFCPOLYGONALBOUNDEDHALFSPACE(#{plane},{flag},#{self.frame},#{ring})")
        else:
            cutter = self.add(f"IFCHALFSPACESOLID(#{plane},{flag})")
        return self.add(f"IFCBOOLEANCLIPPINGRESULT(.DIFFERENCE.,#{host},#{cutter})")

    def solids(self):
        text = "ISO-10303-21;HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;DATA;\n"
        return load_ifc_solids(text + "\n".join(self.rows) + "\nENDSEC;END-ISO-10303-21;")


class PolygonClipTests(unittest.TestCase):
    def test_plane_agreement_normals_and_bounded_footprint(self):
        for bounded in (False, True):
            for normal in (1, -1):
                for flag in (".T.", ".F."):
                    with self.subTest(bounded=bounded, normal=normal, flag=flag):
                        scene = Source()
                        scene.product("IFCWALL", [scene.clip(.5, normal, flag, bounded)])
                        solid, = scene.solids()
                        below = (normal == 1) == (flag == ".T.")
                        removed = (2. if bounded else 4.) * (.5 if below else 1.5)
                        self.assertAlmostEqual(solid.volume_m3(), 8. - removed, places=10)

    def test_full_removal_and_disjoint_plane(self):
        for height, expected in ((-1., 8.), (3., 0.)):
            scene = Source()
            scene.product("IFCWALL", [scene.clip(height, 1, ".T.")])
            self.assertAlmostEqual(sum(s.volume_m3() for s in scene.solids()), expected)

    def test_overlapping_authored_polygon_cutters_are_subtracted_once(self):
        scene = Source()
        wall = scene.product("IFCWALL", [scene.box()])
        ring = scene.ring([(-.8, -.8), (.8, -.8), (-.8, .8)])
        profile = scene.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{ring})")
        opening = scene.product("IFCOPENINGELEMENT", [scene.prism(profile, 1.), scene.prism(profile)])
        scene.void(wall, opening)
        solid, = scene.solids()
        self.assertAlmostEqual(solid.volume_m3(), 8. - .5 * 1.6**2 * 2., places=10)

    def test_open_and_nonnumeric_opening_inputs_refuse_the_host(self):
        for damage in ("open-ring", "ring-coordinate", "depth", "direction", "dimension"):
            for invalid in (".T.", "'1.'"):
                with self.subTest(damage=damage, invalid=invalid):
                    scene = Source()
                    wall = scene.product("IFCWALL", [scene.box()])
                    if damage == "dimension":
                        profile = scene.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{invalid},1.)")
                    else:
                        x = invalid if damage == "ring-coordinate" else 1.
                        ring = scene.ring([(0., 0.), (x, 0.), (0., 1.)], damage != "open-ring")
                        profile = scene.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{ring})")
                    if damage == "direction":
                        scene.up = scene.add(f"IFCDIRECTION((0.,0.,{invalid}))")
                    item = scene.prism(profile, invalid if damage == "depth" else 1.)
                    scene.void(wall, scene.product("IFCOPENINGELEMENT", [item]))
                    with self.assertRaises(ValueError):
                        scene.solids()
