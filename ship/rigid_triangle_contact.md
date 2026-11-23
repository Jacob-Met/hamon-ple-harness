# Normal pressure on a rigid triangular support

Use `civil_geotech` with `calculation: rigid_triangle_contact` for a finite, compression-only pressure field equal to the positive part of one affine function over three planar vertices.

```json
{"calculation":"rigid_triangle_contact","base_vertices_m":[[0,0],[3,0],[0,3]],"vertical_resultant_N":9,"moment_x_Nm":9,"moment_y_Nm":9}
```

Both moments are global **first moments of pressure**. Their definitions are ∫x p dA and ∫y p dA. The force is ∫p dA. Vertices may be supplied in any order. Positive load requires the resultant point to lie strictly inside the actual triangle. An exact determinant on the parsed coordinates rejects collinear footprints and boundary or exterior resultants before inversion. Units follow the rectangle verifier. This is a mathematical equilibrium model. Site-specific stiffness, settlement, load combinations, lateral sliding and bearing capacity require their own models.

The example has 4.5 m² of contact at uniform 2 Pa. Using the same bounding square gives 7.246876 m² and a 3.304422 Pa peak. At moments (18,18) N·m the square still admits a solution, while the resultant point (2,2) lies outside the triangle.

## Inverse problem in barycentric coordinates

Let z be the barycentric coordinates of the resultant. Their sum is one. Let h be the affine pressures at the three vertices, up to a common positive scale. For full contact, h_i = 4 z_i − 1 suffices. If exactly one vertex carries positive pressure, give it unit pressure. Each zero lies a fraction 4 z_i of the way to the other vertex i. These formulas follow by integrating a linear pressure over a triangle. They also handle zero-pressure boundary vertices.

For the remaining regime name the unloaded corner A, the highest pressure corner B, and the other loaded corner C. Set h_B=1 and r=h_C. The zero line enters edges BA and CA at contact depths w and u=w r/(1−w+r w), measured from their loaded ends as fractions of the corresponding edges. Splitting the loaded quadrilateral along the diagonal from B to the intersection on CA gives D=u(1+r)+w(1−u). Its centroid satisfies

```text
4 z_A D = u²(1+r) + w(1−u)(u+w)
4 z_B D = u(2+r) + w(1−u)(3−w).
```

The solver brackets w at fixed r using the measured depth z_A, then brackets r between zero and one. At these r endpoints z_B equals 3/4−z_A and (1−z_A)/2 respectively. The implementation factors the common w from D and works with w/(4 z_A) so it can resolve thin loaded edges. It recovers the negative vertex pressure from w, clips the actual input triangle, and scales the field to its specified force.

At a two-corner transition, binary64 can round a small positive corner pressure to zero. When the first inverse reaches such a boundary, the solver retries using decimal arithmetic and checks whether the four contact vertices stay distinct and convex after conversion to the output coordinates. A verified two-corner label requires two positive emitted vertex pressures, two positive base corners and the matching signs of the published affine plane. The other shapes are also compared with the emitted material vertices. If the numerical field balances the prescribed force and moments but the output cannot distinguish the exact topology, `topology_verified` is false, `footprint_shape` is `release_boundary_unresolved` and `contact_regime` is `boundary_unresolved`. Any proposed regime then fails its claim check.

A field which passes all three moment checks is unique within this model. If two affine planes f,g gave equal force and first moments, ∫(f−g)(max(f,0)−max(g,0)) would vanish. The integrand is everywhere nonnegative. Positive total load provides a positive-area region on which the planes agree, fixing both planes everywhere.

## Verification and limits

The returned polygon, vertex pressures, pressure plane and global moments use binary64 numbers. The verifier independently reintegrates the displayed values using exact fractions, including normalized first moments along all three barycentric coordinates. This rejects errors across a skinny oblique triangle that global x/y moment tolerances alone can hide. Output geometry and supplied claims must both qualify before `conservation_verified` is true. The numerical scope is representable geometry and pressure. Materially inaccurate displayed moments cause refusal.

The independent test oracle slices the physical input triangle vertically, splitting at vertex abscissae and crossings of the zero-pressure line. An open, four-node cubic quadrature integrates width, force and both global moments using exact rational arithmetic. It exercises full fields, one or two loaded corners, two-sided releases, vertex reordering, thin supports, collapsed geometry, the bounding-square adverse control and the ordinary JSON entry point. Run `python3 -m unittest discover -s tests -p test_rigid_triangle_contact.py`.
