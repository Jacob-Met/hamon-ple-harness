"""Independent analytic controls for indexed, volume-only material unions."""
import numpy as np
import pytest
from hamon_ifc_indexed_union import certified_indexed_union_volume

V = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),
              (0,0,1),(1,0,1),(1,1,1),(0,1,1)], dtype=float)
F = np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
              (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)])

def certify(vertices, faces, volume, operands, *, world=None, scale=1.):
    return certified_indexed_union_volume(vertices, faces, kernel_volume=volume,
        operand_volumes=operands, world_vertices=vertices if world is None else world, scale=scale)

def test_separately_indexed_point_contact_retains_two_material_components():
    vertices, faces = np.vstack([V, V+1]), np.vstack([F, F+8])
    assert certify(vertices, faces, 2., [1.,1.]) == 2.
    welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
    with pytest.raises(ValueError, match="separate surface fans"):
        certify(welded, inverse[faces], 2., [1.,1.])

def test_cavity_winding_and_scaled_world_volume():
    vertices, faces = np.vstack([2*V, V+.5]), np.vstack([F, F[:,::-1]+8])
    world = vertices*.01 + np.array([34.,-12.,60.])
    assert certify(vertices, faces, 7., [7.,7.], world=world, scale=.01) == pytest.approx(7e-6, rel=1e-10)

@pytest.mark.parametrize("faces", [F[:-1], np.vstack([F,F[0]]),
                                   np.vstack([F[0,::-1],F[1:]])])
def test_open_duplicate_or_misoriented_output_is_not_admitted(faces):
    with pytest.raises(ValueError):
        certify(V, faces, 1., [1.,1.])

def test_volume_certificate_rejects_wrong_kernel_and_operand_bounds():
    with pytest.raises(ValueError, match="kernel"):
        certify(V,F,1.1,[1.,1.])
    with pytest.raises(ValueError, match="bounds"):
        certify(V,F,1.,[.4,.4])

def test_world_roundoff_and_changed_geometry_withhold_quantity():
    shifted = V.copy()
    shifted[6,2] += .1
    with pytest.raises(ValueError, match="World"):
        certify(V,F,1.,[1.,1.],world=shifted)
    with pytest.raises(ValueError, match="collapses"):
        certify(V,F,1.,[1.,1.],world=V+1e16)

def test_nonfinite_or_nonintegral_indices_refuse():
    with pytest.raises(ValueError):
        certify(V,F.astype(float),1.,[1.,1.])
    damaged = V.copy()
    damaged[0,0] = np.nan
    with pytest.raises(ValueError):
        certify(damaged,F,1.,[1.,1.])
