# Stress on a specified plane

The registered `hamon_conservation_cad_verifier` accepts a symmetric Cartesian stress tensor in `solid_mohr_vessel_buckling` mode with `subkind: "mohr_stress"`. Specify the tensor or the existing normal and shear component fields. When both forms are present their values must agree. Tensor entries use the same stress unit throughout. The optional `plane_normal` is a nonzero three-component direction in the tensor’s Cartesian frame. The solver normalizes it.

```json
{"mode":"solid_mohr_vessel_buckling","spec":{"subkind":"mohr_stress","stress_tensor":[[120,0,0],[0,0,0],[0,0,0]],"plane_normal":[1,1,1]}}
```

For a unit normal $n$, the traction vector is $t=\sigma n$. Its signed normal component is $n^Tt$, and its tangential component is $t-(n^Tt)n$. The `plane_traction` result includes the normalized direction, traction vector, signed normal stress, tangential vector, and tangential magnitude. Reversing the direction reverses both traction vectors and preserves the two scalar stresses.

Every orthant diagonal of the **principal** frame defines an octahedral plane. Their normal stress `sigma_oct` is the mean principal stress. Their shear magnitude `tau_oct` is $\sqrt{2J_2/3}$, where $J_2$ is half the squared Frobenius norm of the deviatoric tensor. The solver checks the tractions of the four unoriented diagonals after transforming those normals to Cartesian coordinates. Under a hydrostatic load every plane carries the same signed normal stress and zero shear.

In the example, the octahedral plane carries normal stress 40 and shear $40\sqrt{2}$. The maximum shear `tau_max_abs` is 60 on a different plane. Repeated principal stresses need no preferred basis because the octahedral stress magnitudes remain invariant.
