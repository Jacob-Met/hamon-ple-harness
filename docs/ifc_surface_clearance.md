# Source-qualified surface clearance, not a code signoff

Use the IFC4 geometric narrow-phase verifier to obtain nearest world-space
triangle surfaces for true AABB overlaps and, with the clearance flag, near
misses even when object envelopes do not intersect:

    python3 /tmp/hamon-engineering-toolkit.py ifc-narrowphase site.ifc \
       --a IfcDuctSegment,IfcPipeSegment --b IfcBeam,IfcColumn \
       --min-clearance-mm 50

The optional minimum is in **millimetres**. Geometry from IfcOpenShell
explicitly uses world placement and its default **metre** output
(\`convert-back-units = false\`); the tool emits world coordinates, exact
minimum triangulated-surface distance, closest-point witnesses, triangle
test count, and a threshold status per bounded pair.

Its best-first BVH uses lower bounds between AABBs and evaluates exact
feature pairs (6 vertex-to-face + 9 edge-to-edge comparisons), plus SAT
surface intersections. This avoids reporting misleading AABB intersection
as material collision, and prevents an AABB-disjoint pair with a too-small
gap from escaping a user-specified surface clearance threshold. Closed
shell containment always violates the material separation check even if
its *surface-to-surface* distance is positive. Unsupported/open/self-
intersecting meshes cannot establish a volume clearance.

Independent controls freeze analytic disjoint tetrahedra with overlapping
AABBs, real crossing solids, nested solids, sub-threshold near misses
whose AABBs are disjoint, and coplanar-parallel surface gaps. Genuine
generated IFC4 is opened through IfcOpenShell in CI before any claim.

This is a tessellated geometry-only screen. Real design tolerances,
architectural opening semantics, survey datum, penetrations, partial/
self-intersecting BReps, and governing structural/MEP clearance standards
must be evaluated separately. Do not interpret this output as a safety
certification or hosted PLE performance score.

References: Ericson, *Real-Time Collision Detection*, 2005, chapters
5.1.5/5.1.9/5.1.11. IfcOpenShell
https://docs.ifcopenshell.org/ifcopenshell/geometry_settings.html
