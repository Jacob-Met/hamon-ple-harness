# Exact clearance of represented triangle meshes

The clearance decision uses the finite binary64 coordinates stored in each loaded mesh. It does not recover decimal geometry lost during import or promise exact curved-surface tessellation.

The three public baseline cases in test_triangle_clearance_scale.py failed in CI45162563776, job137651373248: an adjacent greater threshold was accepted, a uniformly smaller pair lost its edge-interior distance, and a positive 2^-40 separation became a hard clash. Commit d61203a819c9b7bd06e2e902b75021ee035d01a2 retains that executed baseline.

hamon_mesh_distance uses rational arithmetic on represented coordinates. Segment/face intersections, vertex/face projections and interior edge/edge stationary points give the complete closest-feature set, including degenerate triangles. Exact squared distances decide hard contact and the strict required-clearance comparison; the native integer threshold is preserved. Axiswise rational AABB gaps supply conservative pruning bounds, never an attained pair distance.

Closed material containment uses exact nondegenerate ray crossings. Coherently oriented components retain signed winding, including cavities and positive Body unions; inconsistent closed winding retains parity. AABB-volume or vector-norm underflow does not disable containment. The existing material/component topology still determines which surfaces represent solids.

Positive attained distances must have a uniquely rounded binary64 value with at most 1e-12 relative error and retain nonzero magnitude. An unqualified output returns an unresolved geometry result. The two closest_points_m entries are nearest binary64 world points. closest_points_rounding_error_m gives a conservative absolute Euclidean error bound for each; rounded points need not lie exactly on the mathematical face. The witness midpoint is a display point between the exact endpoints.

A positive overlap_aabb_volume_m3 diagnostic becomes null when its magnitude underflows or overflows binary64. It is not a solid intersection volume. Existing volume takeoff and upstream tessellation are separate from this distance contract.

The independent tetrahedron suite projects the origin onto the convex hull of all pairwise vertex differences, using exact linear algebra and a global supporting half-space certificate. It exercises transformed ridge, skew, vertex/face, contact, penetration, tiny separation and nested-shell cases through the registered IFC route. Additional contract checks cover an integer threshold above 2^53, invalid requests, certified witness rounding bounds and conservative AABB bounds outside the attained-distance output range.

Validation requires the actual official workflow and independent source review. No local execution is claimed while the executor is unavailable.
