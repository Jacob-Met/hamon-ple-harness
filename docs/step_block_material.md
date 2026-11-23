# Positioned STEP BLOCK material at source precision

This route integrates one standalone BLOCK from its source dimensions and
AXIS2_PLACEMENT_3D. It does not construct globally rounded vertices or weld a
mesh. The existing rooted rectilinear Boolean route and IFC swept-profile route
retain their own geometry contracts.

Primary schema definitions:
- https://www.steptools.com/stds/stp_aim/html/t_block.html
- https://www.steptools.com/stds/stp_aim/html/t_axis2_placement_3d.html

Finite decimal dimensions and placement ratios remain exact fractions. Exact
cross products construct the orthogonal material axes. The determinant and
Gram projectors prove the normalized frame is positively oriented and
orthonormal. Local volume, area and principal inertia are rational; the complete
world tensor uses exact axis projectors before publication and must satisfy
all three characteristic identities.

Centroid and support extents are sums b/sqrt(q). Terms are grouped by exact
rational square class, recognized with integer square roots without factoring.
This preserves cancellation between independently scaled irrational axes and
combines rational sums before nearest-even midpoint rounding. Remaining
irrational sums use the existing interval publication contract: one determined
binary64 rounding and a proved error bound of1e-12. Primary material properties,
centroids, extents, diagonal inertia and principal moments retain their own
relative scale and nonzero values. Off-diagonal tensor entry error is normalized
by sqrt(I_ii*I_jj), permitting a negligible cross entry to round to zero.
The output explicitly states this separate tensor error model; principal
moments are qualified from the source material frame rather than a rounded
world-tensor eigensolve. No fixed physical volume threshold, welding grid or global-origin
subtraction controls material existence.

One complete standalone material definition is required. Extra blocks, other
primitives and unresolved representation/assembly items are refused rather
than added as overlapping meshes. The unused standalone box-mesh producer is
removed. Density may be zero; positive density requires positive principal
moments in the public representation.

## Preserved cases and independent qualification

Tests-only89e8a8bcca1e0f5d974ed144e5d1caff0c5e3e81 on released65bec536
ran in CI45159016743/job137639259194:1079 methods, exactly one error and two
failures. A 1e-7 m cube hit the fixed volume floor, a 2^-40 m slab lost topology,
and translating a 1 m cube to x=1e16 m erased its global vertex separation.

The additional tests-only1434ce2bbe6b45d72f3a2fd000523ec5a095efbd preserves
a long block's lost weak inertia and a 1e-60 m source centroid produced by
opposing coordinates of order1e16 m. Full run receipts are kept in PR #354.

The independent peer oracle uses oriented edge-volume flux, eight actual
corners for bounds and centroid, and three-dimensional eight-node Gauss
cubature for all world inertia components and principal-axis moments. It also
checks exact irrational centroid cancellation. Original circular primitive
tests qualify the extracted shared interval publication helper.

Production execution and independent course outcomes are reported only from
official CI. No local Python execution is claimed during the executor outage.

First production70b5b3a9 ran1106 methods in CI45160030630/job137642750654.
All new material and independent cubature cases passed. One inherited frame
control with z=(1,1e-400,0) and omitted reference failed because an overly strict
per-entry nonzero requirement refused its negligible cross inertia. The exact
direction must still choose the correct material frame. The revised publication
contract permits nearest-zero cross entries only with an interval proof against
their diagonal energy scale; all primary and principal material values retain
the strict contract. The original failed candidate is preserved.

The tensor bound is entrywise. With D=diag(I_ii), the off-diagonal publication
error satisfies |v^T E_off v| <= 2e-12 v^T D v in three dimensions.
For a nearly rank-deficient rotated tensor this does not supply the same
relative bound on v^T I v in its weakest direction. The separately reported
principal moments and exact characteristic identities refer to the original
material tensor. Symmetric published entries are identical.
