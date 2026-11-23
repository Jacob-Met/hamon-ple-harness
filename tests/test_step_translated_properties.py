"""Centroidal mass properties of one explicitly bounded rectangular prism."""
import numpy as np
import pytest

from hamon_conservation_cad_verifier import verify_step_cad_geometry


SIDES_M = np.array([.1, .2, .3])
CORNERS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
OUTWARD = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
ROT = np.array([[.6,-.8,0],[.8,.6,0],[0,0,1.]])


def prism(offset, scale, rotation, issue=None):
    vertices=(CORNERS*SIDES_M)@rotation.T+offset
    lines=["ISO-10303-21;","HEADER;","ENDSEC;","DATA;"]
    def add_point(pid,xyz):
        values=','.join(format(v/scale,'.16g') for v in xyz)
        lines.append(f"#{pid}=CARTESIAN_POINT('',({values}));")
    for j,xyz in enumerate(vertices):add_point(10+j,xyz)
    faces=[]
    for j,boundary in enumerate(OUTWARD):
        if issue=='missing' and j==5:continue
        ids=[10+k for k in boundary]
        if issue=='hairline' and j==0:
            add_point(99,vertices[0]+[0,0,2e-10]);ids[0]=99
        # Half the face loops are supplied with the opposite topological orientation.
        reversed_loop=(j%2==0)
        if reversed_loop:ids.reverse()
        bound_flag=reversed_loop ^ (issue=='flipped' and j==1)
        bound='.F.' if bound_flag else '.T.'
        same='.F.' if j%2==0 else '.T.'
        refs=','.join(f'#{k}' for k in ids)
        lines.extend([f"#{20+j}=POLY_LOOP('',({refs}));",
                      f"#{30+j}=FACE_OUTER_BOUND('',#{20+j},{bound});",
                      f"#{40+j}=ADVANCED_FACE('',(#{30+j}),$,{same});"])
        faces.append(f'#{40+j}')
    lines.extend([f"#50=CLOSED_SHELL('',({','.join(faces)}));",
                  "#51=MANIFOLD_SOLID_BREP('',#50);","ENDSEC;","END-ISO-10303-21;"])
    return {'step_text':'\n'.join(lines),'unit_scale_m':scale,'density_kg_m3':1000}


@pytest.mark.parametrize('scale',[1.,.001])
@pytest.mark.parametrize('rot',[np.eye(3),ROT])
@pytest.mark.parametrize('offset',[(0.,0.,0.),(500000.,4500000.,80.),(-500000.,-4500000.,80.)])
def test_closed_prism_translation_and_full_tensor(scale,rot,offset):
    got=verify_step_cad_geometry(prism(offset,scale,rot))
    assert got['conservation_verified'] and got['watertight_manifold']
    assert got['net_oriented_area_residual_m2']<=got['net_oriented_area_tolerance_m2']
    assert abs(got['volume_m3']-.006)<1e-10
    assert abs(got['surface_area_m2']-.22)<1e-9
    np.testing.assert_allclose(got['centroid_m'],offset+rot@(SIDES_M/2),atol=2e-8,rtol=0)
    principal=6*(sum(SIDES_M**2)-SIDES_M**2)/12
    np.testing.assert_allclose(got['inertia_tensor_centroid_kg_m2'],rot@np.diag(principal)@rot.T,atol=1e-8,rtol=0)
    np.testing.assert_allclose(got['principal_moments_kg_m2'],sorted(principal),atol=1e-8,rtol=0)


@pytest.mark.parametrize('issue',['missing','flipped','hairline'])
def test_open_inconsistent_or_area_unbalanced_prism_withholds_mass_properties(issue):
    got=verify_step_cad_geometry(prism((0,0,0),1.,np.eye(3),issue))
    assert got['conservation_verified'] is False
    if issue=='hairline':
        # Vertices match under the existing nanometre weld but oriented area exposes the gap.
        assert got['watertight_manifold']
        assert got['net_oriented_area_residual_m2']>got['net_oriented_area_tolerance_m2']
    else:assert not got['watertight_manifold']
    for prop in ['volume_m3','mass_kg','centroid_m','inertia_tensor_centroid_kg_m2','principal_moments_kg_m2']:
        assert prop not in got


def rectangular_ifc_sweep(origin, scale, direction, dims=(0.002, 0.006, 0.010)):
    dx, dy, dz = np.asarray(dims) / scale
    point = ",".join(str(v / scale) for v in origin)
    ratios = ",".join(map(str, direction))
    data = f"""ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1=IFCCARTESIANPOINT(({point}));
#2=IFCDIRECTION((3.,0.,4.));
#3=IFCDIRECTION((0.,1.,0.));
#4=IFCAXIS2PLACEMENT3D(#1,#2,#3);
#5=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{dx},{dy});
#6=IFCDIRECTION(({ratios}));
#7=IFCEXTRUDEDAREASOLID(#5,#4,#6,{dz});
ENDSEC;END-ISO-10303-21;"""
    return {"step_text": data, "unit_scale_m": scale, "density_kg_m3": 7800}


@pytest.mark.parametrize("origin", [(0., 0., 0.), (500000., 4500000., 80.)])
@pytest.mark.parametrize("scale", [1., 0.001])
@pytest.mark.parametrize("direction", [(0., 0., 1.), (0., 0., -1.), (3., 0., 4.), (2., -3., -6.)])
def test_rectangular_ifc_sweep_uses_its_body_placement(origin, scale, direction):
    # Integrate the uniform parallelepiped in its three independent edge coordinates.
    result = verify_step_cad_geometry(rectangular_ifc_sweep(origin, scale, direction))
    assert result["conservation_verified"], result
    rotation = np.array([[0., -0.8, 0.6], [1., 0., 0.], [0., 0.6, 0.8]])
    sweep = rotation @ np.asarray(direction) * (0.010 / np.linalg.norm(direction))
    edges = np.array([0.002 * rotation[:, 0], 0.006 * rotation[:, 1], sweep])
    volume = abs(np.linalg.det(edges))
    covariance = edges.T @ edges / 12
    inertia = (7800 * volume) * (np.trace(covariance) * np.eye(3) - covariance)
    assert result["volume_m3"] == pytest.approx(volume, rel=2e-6, abs=0)
    np.testing.assert_allclose(result["centroid_m"], origin + sweep / 2, atol=2e-9, rtol=0)
    np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"], inertia, atol=1e-16, rtol=2e-6)


@pytest.mark.parametrize("direction,dims", [
    ((1., 2., 0.), (0.002, 0.006, 0.010)), ((0., 0., 0.), (0.002, 0.006, 0.010)),
    ((0., 0., 1.), (0.002, 0., 0.010)), ((0., 0., 1.), (0.002, 0.006, 0.)),
])
def test_rectangular_ifc_sweep_rejects_zero_volume(direction, dims):
    with pytest.raises(ValueError, match="IFC extrusion"):
        verify_step_cad_geometry(rectangular_ifc_sweep((0., 0., 0.), 0.001, direction, dims))


@pytest.mark.parametrize("profile_axis", ["$", "#12"])
@pytest.mark.parametrize("scale", [1., 0.001])
def test_ifc_profile_placement_is_independent_of_sweep_direction(profile_axis, scale):
    origin = np.array([10., -20., 30.])
    spec = rectangular_ifc_sweep(origin, scale, (3., 0., 4.))
    profile_location = np.array([0.020, -0.030])
    xy = ",".join(str(v / scale) for v in profile_location)
    frame = (f"#11=IFCCARTESIANPOINT(({xy}));\n"
             "#12=IFCDIRECTION((0.,2.));\n"
             f"#13=IFCAXIS2PLACEMENT2D(#11,{profile_axis});\n")
    spec["step_text"] = spec["step_text"].replace("#5=", frame + "#5=").replace(
        "IFCRECTANGLEPROFILEDEF(.AREA.,$,$,", "IFCRECTANGLEPROFILEDEF(.AREA.,$,#13,")
    from hamon_conservation_cad_verifier import run_router_conservation_cad_verifier
    result = run_router_conservation_cad_verifier({"mode": "step_cad", **spec})
    assert result["conservation_verified"]
    # In the body frame the sweep translates by (6, 0, 8) mm. Only the
    # cross-section follows the profile axis; the sweep stays in body coordinates.
    body = np.array([[0., -0.8, 0.6], [1., 0., 0.], [0., 0.6, 0.8]])
    profile = np.eye(3) if profile_axis == "$" else np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    sweep = np.array([0.006, 0., 0.008])
    centroid = origin + body @ (np.r_[profile_location, 0.] + sweep / 2)
    edges = body @ np.column_stack((0.002 * profile[:, 0], 0.006 * profile[:, 1], sweep))
    volume = 96e-9
    covariance = edges @ edges.T / 12
    inertia = 7800 * volume * (np.trace(covariance) * np.eye(3) - covariance)
    assert result["volume_m3"] == pytest.approx(volume, rel=1e-10, abs=0)
    np.testing.assert_allclose(result["centroid_m"], centroid, rtol=0, atol=2e-13)
    np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"], inertia, rtol=1e-10, atol=1e-18)


def product_scene(mode):
    # A reusable rectangle, its unowned sibling, and an auxiliary Axis.
    rows = [
        "#1=IFCCARTESIANPOINT((0.,0.,0.));",
        "#2=IFCAXIS2PLACEMENT3D(#1,$,$);",
        "#3=IFCDIRECTION((0.,0.,1.));",
        "#4=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,.020,.040);",
        "#5=IFCEXTRUDEDAREASOLID(#4,#2,#3,.060);",
        "#6=IFCEXTRUDEDAREASOLID(#4,#2,#3,1.);",
        "#7=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-8,#2,$);",
        "#8=IFCSHAPEREPRESENTATION(#7,'Body','SweptSolid',(#5));",
        "#9=IFCSHAPEREPRESENTATION(#7,'Axis','Curve3D',(#6));",
        "#10=IFCPRODUCTDEFINITIONSHAPE($,$,(#9,#8));",
        "#11=IFCCARTESIANPOINT((2.,0.,0.));",
        "#12=IFCAXIS2PLACEMENT3D(#11,$,$);",
        "#13=IFCLOCALPLACEMENT($,#2);",
        "#14=IFCLOCALPLACEMENT(#13,#12);",
        "#15=IFCBUILDINGELEMENTPROXY('first',$,$,$,$,#14,#10,$,$);",
    ]
    if mode == "parent":
        rows += ["#16=IFCCARTESIANPOINT((4.,5.,0.));", "#17=IFCDIRECTION((0.,1.,0.));",
                 "#18=IFCAXIS2PLACEMENT3D(#16,#3,#17);"]
        rows[12] = "#13=IFCLOCALPLACEMENT($,#18);"
    if mode == "shared":
        rows.append("#19=IFCBEAM('second',$,$,$,$,#13,#10,$,$);")
    return {"step_text": "DATA;\n" + "\n".join(rows) + "\nENDSEC;", "density_kg_m3": 1000}


@pytest.mark.parametrize("mode,center,inertia,extents,count", [
    ("one", [2.,0.,.03], [2.08e-5,1.60e-5,8.e-6], [.02,.04,.06], 1),
    ("parent", [4.,7.,.03], [1.60e-5,2.08e-5,8.e-6], [.04,.02,.06], 1),
    ("shared", [1.,0.,.03], [4.16e-5,.096032,.096016], [2.02,.04,.06], 2),
])
def test_ifc_mass_counts_each_placed_product(mode, center, inertia, extents, count):
    got = verify_step_cad_geometry(product_scene(mode))
    assert got["conservation_verified"] and got["watertight_manifold"]
    assert got["geometry_basis"] == "additive_ifc_product_occurrences"
    assert got["ifc_product_count"] == count and got["triangle_count"] == 12 * count
    assert got["volume_m3"] == pytest.approx(count * .000048)
    assert got["mass_kg"] == pytest.approx(count * .048)
    assert got["surface_area_m2"] == pytest.approx(count * .0088)
    np.testing.assert_allclose(got["centroid_m"], center, atol=1e-15)
    np.testing.assert_allclose(got["inertia_tensor_centroid_kg_m2"], np.diag(inertia), atol=1e-15)
    np.testing.assert_allclose(got["aabb_extents_m"], extents, atol=1e-15)


@pytest.mark.parametrize("before,after", [
    ("#14,#10", "#99,#10"), ("#14,#10", "#14,#99"), ("#14,#10", "#98,#99"),
    ("#14,#10", "#14,$"), ("IFCBUILDINGELEMENTPROXY", "IFCSPACE"),
    ("#13=IFCLOCALPLACEMENT($,#2)", "#13=IFCLOCALPLACEMENT(#14,#2)"),
    ("#13,#12", "#13,#11"), ("#11,$,$", "#11,#3,#3"),
    ("'SweptSolid',(#5)", "'SweptSolid',(#5,#6)"),
    ("'SweptSolid',(#5)", "'MappedRepresentation',(#99)"),
    ("#5=IFCEXTRUDEDAREASOLID(#4,#2,#3,.060)", "#5=IFCMAPPEDITEM(#4,#2)"),
    ("#4,#2,#3,.060", "#4,#99,#3,.060"),
    ("#2,$);\n#8=", "#12,$);\n#8="),
    ("ENDSEC;", "#22=IFCRELVOIDSELEMENT('void',$,$,$,#15,#99);\nENDSEC;"),
])
def test_ifc_mass_rejects_partial_or_unsupported_product(before, after):
    spec = product_scene("shared")
    assert before in spec["step_text"]
    spec["step_text"] = spec["step_text"].replace(before, after)
    with pytest.raises(ValueError):
        verify_step_cad_geometry(spec)


@pytest.mark.parametrize("spec", [product_scene("one"), rectangular_ifc_sweep((0.,0.,0.), 1., (0.,0.,1.))])
def test_ifc_mass_refuses_short_physical_owner_even_without_product_shapes(spec):
    from hamon_ifc_spatial_clash import _product_shapes, parse_ifc_entities

    body, trailer = spec["step_text"].rsplit("ENDSEC;", 1)
    text = body + "#99=IFCCOLUMN('short');\nENDSEC;" + trailer
    with pytest.raises(ValueError, match="Incomplete IFC product"):
        verify_step_cad_geometry(dict(spec, step_text=text))
    # The spatial consumer retains its historical permissive product enumeration.
    assert all(eid != 99 for eid, *_ in _product_shapes(parse_ifc_entities(text)))
