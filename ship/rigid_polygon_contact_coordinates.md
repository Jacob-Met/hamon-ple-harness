# Thin polygon contact in support coordinates

The polygon verifier fits one compression-only affine pressure field over the declared material, including holes and disconnected engaged patches. This change preserves that model and the exact full-contact solve.

## Why global coordinates fail

For a square from -1 to 1 in both directions, a triangular pressure strip of width d along the right edge has unit force, x moment 1-d/3, contact area 2d, and peak pressure 1/d. The supplied floating-point moment defines an exact rational d through 3*(1-Fraction(moment)).

The former partial-contact Gram matrix used (1,x,y) in full-footprint coordinates. Its constant and x columns become almost dependent as the strip narrows. An absolute moment residual on the footprint scale can conceal a large relative error in d and peak pressure. Rounding a large intercept and gradient about the footprint center introduces further cancellation at the loaded edge.

## A basis determined by the support and supplied load

Let c=M/N be the load centroid. For each convex-hull edge, let n_i be its inward normal and s_i=n_i dot (c-a_i)>0. Form the exact row r_i=n_i/s_i. Select two rows u,v with maximal absolute determinant and order them with positive determinant.

In coordinates q=(u dot (x-c), v dot (x-c)), the target centroid is exactly zero and the two chosen supporting lines lie at -1. The small support distances are retained as rational values until this transformation is complete.

Every other row r_i=alpha*u+beta*v has |alpha|<=1 and |beta|<=1 by maximality of the chosen determinant. Small force and moment residuals in this basis control every centroid-to-edge moment residual, including extremely small original edge distances. The choice is affine invariant apart from ties.

If J=det(u,v), the transformed pressure density is p/J because dA_q=J*dA_x. The inverse solves for unit force in q coordinates, then maps its affine field back with the exact coordinate transform and J. One positive-part variational objective continues to couple all material patches.

## The serialized witness

Intersections start at the retained compressive endpoint. Partial-contact pressure is referenced at a support vertex with maximal pressure. If separate binary64 gradient rounding cannot preserve the field on the support, pressure_plane has encoding='rational' and its pressure_at_reference_Pa and gradient_Pa_per_m values are exact fraction strings. Decode them as rational values when reproducing the witness. Ordinary representable planes retain their existing numeric fields.

Contact patches independently declare their existing binary64 or rational encoding. The emitted polygons and pressures are integrated again with exact rational arithmetic. Force and moments must agree on every centroid-to-support-edge scale; full-footprint checks alone are insufficient.

## Independent controls

Closed-form edge and corner contacts cover exact interpretations of supplied binary64 moments, signs, shears, translations, and unloaded holes. For a triangular corner contact with legs d and unit force, the centroid lies d/4 from each loaded edge, area is d*d/2, and peak pressure is 6/(d*d).

The maintained boundary-integral tests are independent of production slab integration and cover nonconvex material, holes, winding changes, and ordinary JSON and dual-verifier routes.

This qualifies the stated mathematical pressure model. Hosted PLE score, material capacity, and settlement remain separate, unmeasured conditions.
