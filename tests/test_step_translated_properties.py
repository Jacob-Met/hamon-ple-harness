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


@pytest.mark.parametrize("origin", [(0., 0., 0.), (500000., 4500000., 80.)])
@pytest.mark.parametrize("scale", [1., 0.001])
def test_rectangular_ifc_sweep_uses_its_body_placement(origin, scale):
    # The 2 × 6 mm centered profile sweeps 10 mm along local Z. Local X
    # points north and local Z has a 3:4 east/up tilt in the survey frame.
    dims = np.array([0.002, 0.006, 0.010])
    dx, dy, dz = dims / scale
    point = ",".join(str(v / scale) for v in origin)
    data = f"""ISO-10303-21;
HEADER;FILE_SCHEMA(('IFC4'));ENDSEC;
DATA;
#1=IFCCARTESIANPOINT(({point}));
#2=IFCDIRECTION((3.,0.,4.));
#3=IFCDIRECTION((0.,1.,0.));
#4=IFCAXIS2PLACEMENT3D(#1,#2,#3);
#5=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{dx},{dy});
#6=IFCDIRECTION((0.,0.,1.));
#7=IFCEXTRUDEDAREASOLID(#5,#4,#6,{dz});
ENDSEC;END-ISO-10303-21;"""
    result = verify_step_cad_geometry({"step_text": data, "unit_scale_m": scale, "density_kg_m3": 7800})
    assert result["conservation_verified"]
    rotation = np.array([[0., -0.8, 0.6], [1., 0., 0.], [0., 0.6, 0.8]])
    volume = np.prod(dims)
    mass = 7800 * volume
    moments = mass * (sum(dims**2) - dims**2) / 12
    assert result["volume_m3"] == pytest.approx(volume, rel=2e-6, abs=0)
    np.testing.assert_allclose(result["centroid_m"], origin + rotation[:, 2] * dims[2] / 2,
                               atol=2e-9, rtol=0)
    np.testing.assert_allclose(result["inertia_tensor_centroid_kg_m2"],
                               rotation @ np.diag(moments) @ rotation.T, atol=1e-16, rtol=2e-6)
