# IFC geometry-aware narrow-phase screening

The Pi task container receives both \`/tmp/hamon-engineering-toolkit.py\`
and \`/tmp/hamon_geometry.py\` via the taskbridge relay. When the evaluator
task container has the optional IfcOpenShell geometry kernel, invoke:

\`\`\`sh
python3 /tmp/hamon-engineering-toolkit.py ifc-narrowphase site.ifc --a IfcDuctSegment,IfcPipeSegment --b IfcBeam,IfcColumn
\`\`\`

This uses IfcOpenShell USE_WORLD_COORDS tessellation, per-element triangle
AABBs, a median-split BVH, surface triangle separating-axis tests (with
coplanar in-plane axes), and manifold-closed triangulated-shell point containment
checks using multiple independent ray/triangle probes.

It separates:
- \`aabb_disjoint\`: provable envelope separation in emitted coordinates;
- \`no_surface_contact_found\`: overlapping envelopes, tested mesh triangles
  do not meet, no detected closed-solid containment;
- \`surface_contact\`: triangles touch/cross, possible but not proven clash;
- \`solid_containment\`: disjoint triangle surfaces, closed-shell point inclusion;
- \`unresolved_nonwatertight\`/\`unresolved_containment_ray\`/
  \`unresolved_budget_or_geometry\`: uncertainty cannot be silently treated
  as clearance.

All witnesses include IFC GlobalIds, class, explicit relation and triangle
comparison counts. An absent IfcOpenShell kernel is a loud machine-readable
error; parse/geometry failures are reported and \`all_pairs_qualified=false\`.
Per-element hard cap: 20,000 faces, 50,000 vertices; per-pair SAT budget is
bounded. More than 100,000 IFC product pairs is rejected, not reported as clear.

**Not a design signoff:** Tessellation can hide exact BRep tolerances, open or
nonmanifold meshes can obscure interior relations, surface contact is not the
same as material penetration, and some intentional duct/beam openings are
permitted. Confirm project units, true BRep solids, object semantics, clearance
distance, physical tolerances and governing code independently. This tool
reports no compliance or hosted PLE score.

References: IfcOpenShell geometry docs
https://docs.ifcopenshell.org/ifcopenshell-python/geometry_processing.html;
Tomas Möller, *A Fast Triangle-Triangle Intersection Test* (1997),
https://doi.org/10.1080/10867651.1997.10487472;
Möller & Trumbore, *Fast, Minimum Storage Ray-Triangle Intersection* (1997),
https://doi.org/10.1080/10867651.1997.10487468.
