# Concave enclosed IFC cavities

`hamon_ifc_spatial_clash` accepts an `IfcFacetedBrepWithVoids` only after checking
the actual polygonal boundaries. The exterior and cavity shells may be concave;
two disjoint cavities may have overlapping bounding boxes or convex hulls.

Each shell must be a connected, consistently oriented closed two-manifold,
including a single cycle in every vertex link. Triangle intersections must be
limited to their shared vertex or edge within a shell. Different shells must
have no boundary contact. Exact ray intersections then establish that each
cavity lies inside the exterior and that no cavity contains another. The
exterior has outward winding, and every cavity has inward winding.

Finite binary64 coordinates are converted to a common integer lattice before
these checks. Integer and rational predicates retain a gap of one representable
coordinate step; they never merge nearby source planes. The existing face
triangulation and unit/placement readers still define the polygonal input.
Accepted boundaries are passed through unchanged for volume, centroid,
surface-area and clash calculations. This is a nominal polygonal model, not
an analytic reconstruction of curved surfaces or a physical tolerance claim.

The shared limit is two million triangle-pair and ray-triangle predicates.
The integer ray direction is selected by excluding the exact polynomial roots
that would make it parallel to a face or incident on an edge. This needs one
ray count, without a fixed retry limit or movement of the test point. Exceeding
the predicate budget leaves the geometry unresolved and withholds its material
volume. Self-intersection, touching shells, nested
cavities, missing faces and incorrect winding also remain unresolved.

## Checks

The previous convex-shell certificate refused a 6 by 6 by 4 metre box with an
enclosed L-shaped cavity. Its cavity area is 7 square metres and its height is
2 metres, leaving 130 cubic metres of material. The replacement checks that
volume, its 214 square metre boundary, the independent first moment, a probe in
the cavity and a probe in its solid notch. A concave exterior with a concave
cavity retains 62 cubic metres. Additional cases cover disjoint cavities whose
convex hulls overlap, survey coordinates, declared metre/millimetre units,
shear, exact contact and a one-step gap. A cavity that crosses a concave exterior
is rejected even when all of its vertices are individually inside.

The maintained tests also export/reopen the two concave models through
IfcOpenShell and compare its independent geometry volume. Those cases require
the repository's pinned geometry dependencies; they are not silently reported
as executed when the optional local installation is absent.

The source semantics follow buildingSMART's
[IFC4 IfcFacetedBrepWithVoids definition](https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD1/HTML/schema/ifcgeometricmodelresource/lexical/ifcfacetedbrepwithvoids.htm):
void shells are disjoint, enclosed, non-nested boundaries with normals pointing
into the cavity. These tests do not establish foundation-design accuracy or a
hosted competition score.
