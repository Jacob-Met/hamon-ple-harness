# Canonical circular STEP solids

The bounded analytic route accepts one complete RIGHT_CIRCULAR_CYLINDER or
SPHERE, either directly or as the direct root expression of CSG_SOLID. Cylinder
position is AXIS1_PLACEMENT; sphere centre is a three-dimensional CARTESIAN_POINT.
The omitted cylinder axis is global +Z. Negative or zero physical dimensions,
wrong coordinate types, ambiguous material, curved Boolean expressions and
assembly occurrences are refused.

Primary EXPRESS entity definitions:
- https://www.steptools.com/stds/stp_aim/html/t_right_circular_cylinder.html
- https://www.steptools.com/stds/stp_aim/html/t_axis1_placement.html
- https://www.steptools.com/stds/stp_aim/html/t_sphere.html

Decimal source dimensions, coordinates and direction ratios remain rational.
The resolved length scale follows the existing unit-context resolver. With
q(z) equal to the squared circular radius, the kernel integrates q, z*q,
z*z*q and q*q exactly. It verifies the volume against the separate primitive
formula and verifies the rotated inertia tensor's three characteristic
invariants against its axial and transverse principal moments. Tensor
coefficients are assembled before multiplication by pi, retaining tiny axial
moments for large aspect ratios without a rounded eigensolve.

Centroids and exact curved bounding extents use interval square roots.
Opposite-sign translation is rationalized before evaluation. Each output must
have an unambiguous binary64 rounding and a proved relative error no greater
than 1e-12; nonzero outputs cannot disappear. Interval precision doubles from
80 through 5120 decimal digits, with a clean refusal if qualification remains
unresolved. Zero density is allowed and produces exactly zero mass/inertia.

The report identifies analytic geometry and exact circular section moments.
There is no polygonal approximation or fictitious triangle count. Watertightness
and connectedness follow from the validated complete positive primitive.
The dual volume comparison is an exact section-integral/formula identity;
independent source-equation tests provide separate validation.

Standalone BLOCK keeps its existing mesh route, but now requires exactly three
finite positive dimensions. The historical cylinder frame fixture is corrected
from AXIS2_PLACEMENT_3D to the schema's AXIS1_PLACEMENT.

## Preserved regression

Tests-only commit 0b91b1be45484a6b20eedc4e42516c6b4638d3b9 ran in official
CI 45154458943 / job 137623737105: 1037 methods, exactly four errors and four
failures. The four schema-correct standalone/rooted cylinder and sphere cases
were unsupported. A wrongly typed cylinder placement and three individually
negative BLOCK dimensions were incorrectly certified. The production candidate
is executed by official CI; no local execution is claimed during the executor
outage.
