"""Analytic occupied-volume controls for compound IFC Body representations."""
import pytest

from hamon_ifc_spatial_clash import run_router_ifc_clash


class Scene:
    def __init__(self):
        self.records = []
        unit = self.add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)")
        self.add(f"IFCUNITASSIGNMENT((#{unit}))")
        self.up = self.add("IFCDIRECTION((0.,0.,1.))")

    def add(self, text):
        self.records.append(f"#{len(self.records) + 1}={text};")
        return len(self.records)

    def sweep(self, profile, xyz, depth):
        point = self.add(f"IFCCARTESIANPOINT(({xyz[0]},{xyz[1]},{xyz[2]}))")
        placement = self.add(f"IFCAXIS2PLACEMENT3D(#{point},$,$)")
        return self.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{placement},#{self.up},{depth})")

    def box(self, side, x, z=None, y=0.0):
        profile = self.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{side},{side})")
        return self.sweep(profile, (x, y, -side / 2 if z is None else z), side)

    def element(self, kind, name, items, rep_type="SweptSolid"):
        refs = ",".join(f"#{i}" for i in items)
        shape = self.add(f"IFCSHAPEREPRESENTATION($,'Body','{rep_type}',({refs}))")
        product = self.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        return self.add(f"{kind}('{name}',$,'{name}',$,$,$,#{product},$,$)")

    def report(self, tmp_path):
        text = ("ISO-10303-21;\nHEADER;FILE_SCHEMA(('IFC4'));ENDSEC;\nDATA;\n"
                + "\n".join(self.records) + "\nENDSEC;END-ISO-10303-21;")
        source = tmp_path / "clearance.ifc"
        source.write_text(text)
        report = run_router_ifc_clash({"path": str(source), "required_clearance_m": 0.01,
                                      "cross_discipline_only": True})
        assert report["ok"]
        return report


@pytest.mark.parametrize("reverse_parts", [False, True])
@pytest.mark.parametrize("reverse_elements", [False, True])
@pytest.mark.parametrize("count", [2, 26])
def test_nested_component_order(tmp_path, reverse_parts, reverse_elements, count):
    scene = Scene()
    parts = [scene.box(1., 10.), scene.box(1., 0.)]
    if reverse_parts:
        parts.reverse()
    elements = [("IFCBEAM", "compound", parts), ("IFCDUCTSEGMENT", "container", [scene.box(4., 0.)])]
    for element in (reversed(elements) if reverse_elements else elements):
        scene.element(*element)
    for i in range(count - 2):
        scene.element("IFCBEAM", f"far{i}", [scene.box(1., 100. + 10 * i)])
    report = scene.report(tmp_path)
    assert report["hard_clash_count"] == 1 and report["min_clearance_m"] == 0
    a, b = report["clashes"][0]["closest_points_m"]
    assert a == b and all(abs(v) <= 0.5 for v in a)


@pytest.mark.parametrize("wall,probe_x,expected", [(1., 0., 0.9), (1., 1.5, 0.),
                                                   (0.01, 1.995, 0.), (0.01, 1.98, 0.008)])
def test_enclosed_ifc_opening_and_thin_wall(tmp_path, wall, probe_x, expected):
    scene = Scene()
    host = scene.element("IFCDUCTSEGMENT", "host", [scene.box(4., 0.)])
    void = scene.element("IFCOPENINGELEMENT", "closed-cavity", [scene.box(4. - 2 * wall, 0.)])
    scene.add(f"IFCRELVOIDSELEMENT('void',$,$,$,#{host},#{void})")
    side = 0.2 if wall == 1 else 0.004
    scene.element("IFCBEAM", "compound", [scene.box(1., 10.), scene.box(side, probe_x)])
    report = scene.report(tmp_path)
    assert report["min_clearance_m"] == pytest.approx(expected, abs=1e-8)
    assert report["hard_clash_count"] == int(expected == 0.)


@pytest.mark.parametrize("xy,expected", [((0., 0.), 0.9), ((-1.5, 1.), 0.)])
def test_nonconvex_notch_and_material(tmp_path, xy, expected):
    scene = Scene()
    polygon = [(-2., -2.), (2., -2.), (2., -1.), (-1., -1.), (-1., 2.), (-2., 2.), (-2., -2.)]
    points = [scene.add(f"IFCCARTESIANPOINT(({x},{y}))") for x, y in polygon]
    line = scene.add("IFCPOLYLINE((" + ",".join(f"#{i}" for i in points) + "))")
    profile = scene.add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#{line})")
    scene.element("IFCDUCTSEGMENT", "L", [scene.sweep(profile, (0., 0., -1.), 2.)])
    scene.element("IFCBEAM", "compound", [scene.box(1., 10.), scene.box(0.2, xy[0], y=xy[1])])
    report = scene.report(tmp_path)
    assert report["min_clearance_m"] == pytest.approx(expected)
    assert report["hard_clash_count"] == int(expected == 0.)


@pytest.mark.parametrize("centres,sides", [((-0.9, 0.9), (4., 4.)), ((0., 0.), (4., 2.)),
                                          ((0., 0.), (4., 4.))])
@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_and_nested_positive_body_items(tmp_path, centres, sides, reverse):
    scene = Scene()
    positive = [scene.box(side, centre) for side, centre in zip(sides, centres)]
    scene.element("IFCDUCTSEGMENT", "additive body", positive[::-1] if reverse else positive)
    scene.element("IFCBEAM", "inside both", [scene.box(0.2, 0.)])
    report = scene.report(tmp_path)
    assert report["min_clearance_m"] == 0
    assert report["hard_clash_count"] == 1


@pytest.mark.parametrize("inner_void,expected", [(False, 0.), (True, 0.9)])
def test_nested_shell_orientation_within_one_tessellated_item(tmp_path, inner_void, expected):
    # The same vertices describe either an inward cavity or a second outward
    # shell. This single tessellated item has no separate Body metadata.
    scene = Scene()
    points = []
    triangles = []
    faces = [(1, 3, 2), (1, 4, 3), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
    for i, radius in enumerate([2, 1]):
        points.extend((x, y, z) for z in (-radius, radius)
                      for x, y in [(-radius, -radius), (radius, -radius),
                                   (radius, radius), (-radius, radius)])
        for face in faces:
            order = reversed(face) if i == 1 and inner_void else face
            triangles.append(tuple(index + 8*i for index in order))
    vertices = scene.add("IFCCARTESIANPOINTLIST3D((" + ",".join(map(str, points)) + "))")
    shell = scene.add(f"IFCTRIANGULATEDFACESET(#{vertices},$,.T.,("
                      + ",".join(map(str, triangles)) + "),$)")
    scene.element("IFCDUCTSEGMENT", "shells", [shell], rep_type="Tessellation")
    scene.element("IFCBEAM", "probe", [scene.box(0.2, 0.)])
    report = scene.report(tmp_path)
    assert report["min_clearance_m"] == pytest.approx(expected)
    assert report["hard_clash_count"] == int(expected == 0.)
