# Rectilinear IFC material

Axis-aligned closed triangle surfaces use a shared source-coordinate cell
arrangement for union, intersection and difference. Every finite binary64
coordinate is converted to an integer with its exact power-of-two scale. Face
coverage and half-open triangle edge ties are evaluated with integer predicates;
the midpoint of two adjacent representable planes is never rounded to either
plane. Independent winding sweeps on all three axes must agree on the material.

The output boundary uses the original coordinate planes. It must have closed,
consistently oriented edges and one connected link at every vertex. Empty
results, corner/edge-only material contacts and unqualified boundaries remain
unresolved. Distinct source planes are never welded by a tolerance. Whole-shell
winding reversal and signed cavity boundaries are supported.

This path applies to rectilinear source meshes, including nested Boolean results,
at the point where the Boolean is evaluated. Later product placements can rotate
the result. Oblique source faces continue through the existing general polyhedral
kernel. Curved-profile tessellation remains a polygonal approximation. These are
nominal source-coordinate quantities, not manufactured-tolerance measurements.

The bounded path permits at most 250,000 cells, 3,000,000 projected face tests and
100,000 output triangles. Exceeding a bound withholds the result; it does not retry
the same rectilinear material through tolerance-based merging.

## Verification

The former normalized Manifold 3.5.5 path changed an intersection of two unit
boxes, with overlap `2**-40` m, into a four-vertex solid. It reported
`1.5158245029535018e-13` m³ instead of `9.094947017729282e-13` m³. The exact-cell
path retains eight vertices and the full rectangular volume through `2**-53` m.

The maintained tests compare three-box unions against independent rational
inclusion-exclusion moments, exercise all three operations, cavity refill,
reversed winding, repeated source indices, declared metre/millimetre units, a
remote component, resource bounds and nonmanifold contacts. Real IfcOpenShell
IFC4 export/reopen checks preserve the supplied source planes. No competition
score or foundation-design accuracy improvement is inferred from these controls.
