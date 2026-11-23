# Rooted rectilinear STEP CSG

The STEP geometry route evaluates a selected `CSG_SOLID` built from `BLOCK` leaves and `BOOLEAN_RESULT` union, intersection and ordered difference. The result must have positive volume, connected interior and a closed embedded manifold boundary. Shared expression nodes are allowed; cycles, missing references and unsupported selected operands are refused.

The [STEP geometric model schema](https://www.steptools.com/stds/smrl/data/resource_docs/geometric_and_topological_representation/sys/6_schema.htm) defines the root expression and regularized Boolean operations. Regularization removes lower-dimensional remnants. Here, a face-to-face union joins material interiors; an edge-only or point-only contact does not. An intersection that leaves only a face, edge or point has no supported volume.

## Geometry and arithmetic

Every selected block placement is validated from typed coordinates and directions. Exact cross products project the reference direction onto the axis-normal plane. The resulting frame must be an exact signed permutation of the global coordinate axes. Independent direction scaling and a component parallel to the axis do not change the frame. A genuinely oblique frame is outside this path, including one whose decisive nonzero source component would disappear in binary64.

Decimal source positions, lengths and direction ratios remain rational through the construction. The selected representation's length scale is resolved by the existing STEP unit-context resolver; its finite public scale is applied without intermediate coordinate rounding.

All block planes partition space into open rectangular cells. Boolean operations select material cells, so overlaps are counted once and subtraction removes material. A face-neighbor traversal verifies the selected solid's connected interior. Exposed cell faces form the oriented boundary. Exact paired-edge checks and connected vertex links exclude nonmanifold contacts; the material partition prevents surface crossings.

Two integrations qualify the same geometry. Rectangular-cell formulas compute volume, first moments and the full second-moment tensor. Separately, the existing signed tetrahedral boundary integrator computes the same mass properties from exposed faces. Their rounded volume, centroid and full centroidal inertia tensor must agree. Oriented boundary area must balance exactly. Principal moments use the existing conditioned high-precision symmetric eigensolve.

## Selection and reporting

A single unconsumed CSG root may be selected automatically. Multiple roots, other STEP solid definitions or assembly occurrences require an explicit `solid_entity_id`. Another IFC solid or shape representation also requires an explicit selection. Unrelated IFC point/direction metadata does not replace the selected STEP geometry or its units.

An explicit selector qualifies that solid definition. It does not expand assembly placements or combine unselected material. The output identifies `selected_solid_entity_id` and `solid_geometry_kind: rectilinear_csg`.

Volume, positive-density mass, three positive extents, positive principal moments and surface area must remain finite and positive when reported. If a positive extent underflows to zero, the path withholds a geometry certificate even if volume remains representable. Zero density retains the supported zero-mass convention.

## Bounded scope and tests

The path admits at most 32 distinct block leaves, 128 expression nodes and 32,768 arrangement cells. Curved primitives, oblique blocks, half-spaces and transformed solid replicas require a different geometry kernel.

Registered tests cover basic Boolean operations, shared and nested expressions, selected roots, unit contexts, signed rectilinear placements and malformed constructions. An independent reference expands signed box indicators, integrates their monomials and computes exposed area from squared planar indicator jumps. Its cases include cavities, through-cuts, material restoration, touching solids, a wall below binary64 coordinate resolution, empty or disconnected interiors and finite-reporting limits.
