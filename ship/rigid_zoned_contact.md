# Rigid contact on regions with declared vertical stiffness

The registered conservation verifier accepts calculation=rigid_zoned_contact.
It solves one rigid displacement field over every declared support region:

    w(x,y) = a + b*x + c*y
    p_i(x,y) = k_i * max(w(x,y), 0)

Here k_i is the positive normal stiffness per area in N/m³. Regions may have
holes and disconnected material, and may meet at shared edges or vertices.
Their material interiors must not overlap. A region may occupy another
region's hole. Each region has one constant stiffness; no value is inferred.

This adds an input model that the homogeneous polygon calculation could not
express. For the two adjacent 2-by-2 regions below, a common displacement of
one metre gives force 20 N, first moment 12 N m, and peak pressure 4 Pa.
Treating the same base as homogeneous gives peak 4.75 Pa. Both are different
models of the same resultant, so the homogeneous route remains unchanged.

## Registered input

Call hamon_conservation_cad_verifier with mode=auto and this spec, or pass it
directly to the maintained Python JSON CLI:

    {
      "calculation": "rigid_zoned_contact",
      "support_regions": [
        {
          "id": "left",
          "outer_vertices_m": [[-2,-1],[0,-1],[0,1],[-2,1]],
          "vertical_stiffness_N_m3": 1
        },
        {
          "id": "right",
          "outer_vertices_m": [[0,-1],[2,-1],[2,1],[0,1]],
          "vertical_stiffness_N_m3": 4
        }
      ],
      "vertical_resultant_N": 20,
      "moment_x_Nm": 12,
      "moment_y_Nm": 0
    }

Each support accepts optional hole_vertices_m. IDs must be distinct nonempty
strings. Numeric quantities are SI; dimensional expressions use explicit
multiplication and Python powers, for example "4*Pa/m" or "1*N/m**3".
Unknown fields, unsupported pressure models, nonpositive/nonfinite stiffness,
invalid boundaries, and positive-area region overlaps are rejected.

The moment convention is Mx=integral(x*p dA), My=integral(y*p dA), about the
global origin. These are pressure first moments. A coordinate translation
must translate both moment inputs. The optional pressure_model is
positive_part_affine_displacement; independent regional displacement planes
are deliberately not part of this model.

Optional proposed_contact_area_m2, proposed_material_area_m2 and
proposed_q_max_Pa are checked against the reconstructed result. A false
claim sets conservation_verified=false while retaining the physical result.

## Shared inverse and independent output checks

Every convex material patch contributes its positive stiffness weight to
one three-variable inverse problem. The objective is the integral of
k_i*max(w,0)^2/2 minus the prescribed work. Its gradient contains the three
equilibrium residuals; its active Hessian is the weighted second-moment
matrix. The full-contact solve is exact rational arithmetic. Partial contact
uses the existing support-distance coordinate conditioning and safeguarded
Newton solve, then integrates the actual reported geometry exactly.

The optional weights on the shared polygon inverse do not change the
homogeneous path. A frozen 62-case comparison checks complete JSON equality.

The result includes a common displacement_plane and a regions list. Each
region reports material/contact area, integrated force and both global first
moments, peak pressure, constitutive residual, and clipped contact_patches.
Each patch carries vertices, vertex_displacement_m and vertex_pressure_Pa.
Constitutive closure checks the reported displacement against the common
plane and the reported pressure divided by the declared stiffness. Total
force and moments are checked after serialization; moment tolerances scale
with actual distances to support edges.

Ordinary geometry uses binary64 JSON numbers. When that cannot preserve
the field, an encoding="rational" plane or patch uses fraction strings.
Consumers must preserve those strings when independently integrating.
output_geometry_verified and common_displacement_verified describe these
reconstructions; conservation_verified also includes any proposed claims.

Zero force and moments produce equilibrium_status="unloaded", zero pressure,
and null contact areas, because the touching set is not determined.
A positive resultant on or outside the support hull cannot have a finite
compression-only solution. Numerical failures return an unverified result.

## Frozen independent checks

The six reference cases use two adjacent regions, k_left=1 and k_right=1 or 4,
with prescribed displacement 1, x+1/4, or x+1/4+y/2. Reference loads and
regional results are rational integrals computed independently of the solver.
Four quarter-turn orientations and origins (0,0) and (1000,-2000) give 48
ordinary JSON CLI requests:

    python ship/verify_zoned_contact.py --repo . --out /path/to/report

The receiver independently integrates the reported common field over every
serialized contact polygon by Green's theorem. It checks regional force,
both centred moments, peak, contact area and every pressure/displacement
vertex. The reference integrals and receiver share neither the production
clipping nor its triangle integration.

Maintained tests also cover common stiffness scaling, equal-stiffness region
subdivision and ordering, disconnected supports with a fully unloaded region,
a region filling another's hole, one-ULP material overlap, invalid stiffness
and model inputs, false peak claims, dimensional expressions, and the actual
registered TypeScript-to-Python route.

This is a linear, compression-only normal support model. It does not infer
stiffness, model friction or nonlinear ground response, establish bearing
capacity, or constitute a hosted PLE score.
