# Volume-only material union at indexed contacts

A successful polygonal UNION can have distinct indexed vertices at the same
physical point. Position welding then creates a nonmanifold surface even when
the Boolean kernel has a closed indexed boundary. Summing the original Body
volumes is also wrong when those components overlap.

The fallback is deliberately limited to aggregate material volume after a
successful kernel UNION. It requires two opposite uses of every indexed edge,
one connected cycle in every vertex link, finite coordinates, nondegenerate
triangles, positive volume, and the union bounds set by its operand volumes.
It independently integrates all three Cartesian divergence fields using exact
rational values of the binary64 output coordinates. Their enclosed volumes
must agree exactly. The output must agree with the kernel volume and survive
reconstruction in world coordinates within the stated numerical tolerance.

This does not relax input-geometry admission. It is never accepted as an
opening cutter or a replacement clipping surface. Original component geometry
is retained for geometric queries. A dedicated field supplies only the checked
material volume; area and centroid are withheld because overlapping component
surfaces and moments cannot be added.

Analytic controls include separately indexed point contacts, a welded vertex
with two disjoint surface fans, inward cavity winding, open and reversed faces,
incorrect kernel and operand measures, and world-coordinate loss. Public-source
qualification additionally compares both eleven-part FZK window unions with an
independent closed-solid OpenCASCADE union.
