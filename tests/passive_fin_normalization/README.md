# Passive fin normalization

The old passive fin reporter returns efficiency 1 and effectiveness 0 whenever the base-to-ambient temperature difference is at most 1e-12 in magnitude. This changes dimensionless ratios at an arbitrary absolute temperature threshold. It also hides positive geometry ratios when the physical heat rate rounds to zero.

The candidate forms passive unit-drive conductance before multiplying by the signed drive. It retains the historical lateral reference hPL for fin_efficiency and bare-footprint reference hA_c for fin_effectiveness. Both reference products are formed without intermediate binary64 range loss. The existing q_max_ref_w still includes the Robin tip, and the new evidence labels make the difference explicit. Infinite mode continues to use the provided length as the reporting reference.

At zero base excess, both literal heat-rate ratios are undefined. The ordinary zero-heat physical result retains its pass decision, while the ratios are null and numeric claims on them explicitly refuse. A prescribed-temperature tip has an independent drive and retains its original output, including its existing zero-base behavior.

## Frozen comparison

Base edecde96ecd6b155825e97014b9b69da146f3085 has tree d3c4982c85505eafcb4508313fc7122c3480f235, exactly the qualified scalar-tolerance CI tree. The stock and candidate contain the same test files and this protocol; only hamon_pde_control_psychro_fem.py differs.

The independent receiver uses a 100-digit direct hyperbolic ODE solution and separately specified exact dyadic matched-tip identities. It does not import the producer's helpers. It executes 102 router calls across 32 systems before assertion: 32 ordinary calls and 70 claims, expected 54 approvals and 16 refusals. General comparisons use a magnitude-scaled absolute tolerance with relative tolerance zero; named exact dyadic and refusal controls use zero absolute tolerance.

The cases cover three passive modes with ordinary/signed tiny/zero drive, adjacent binary64 values around the old threshold, both signs of the smallest subnormal drive, physical q rounding to zero, different lateral/full-tip references, a ratio retained despite an overflowed intermediate reference product, the smallest subnormal ratio, a large finite ratio, ratios below/above float range, and two independently heated prescribed-tip controls. Matched Robin eta 1/2 and epsilon 2 are distinct from the full-surface quotient 1/3.

All 32 full physical projections must match across the original pair, excluding only the two intentionally changed ratios and the new fin_normalization dictionary. The two prescribed-tip entire ordinary outputs must match without exclusions. These pairwise equalities are receipt-level gates. Each branch separately reports all raw decisions, wrong approvals/refusals, missing metric/metadata checks and strict-JSON or solver errors; these categories must not be conflated.

The actual registered tool fixture adds 13 requests: 4 unclaimed and 9 numeric claims, expected 6 approvals and 3 refusals. It covers signed matched Robin, zero-drive refusal, a non-scalar evidence refusal, and positive ratios when physical q rounds to zero. The unchanged batch executes every request before assertions. A unittest wrapper ensures these requests are exercised even if the stock unit stage fails before pytest.

## Interpretation and inherited gates

The seven-field passive evidence dictionary identifies the reference areas, provided length, unit-drive method, zero policy and separate finite/below-range/above-range/undefined statuses. Nonzero subnormals are retained; overflow and nonzero values that round to zero are withheld. Exact rational composition protects the numerical range of already rounded analytic factors. It does not certify transcendental functions or an exact physical solution.

Lienhard and Lienhard define whole-surface fin efficiency, specialize hPL to an insulated tip, and define effectiveness against the bare wall. The existing Robin fin_efficiency uses a different, lateral-only reference; retaining and labeling that convention prevents a silent API change. A large lateral-only ratio is not clamped. See A Heat Transfer Textbook, 6th edition, section 4.5, equations 4.31 and4.52–4.54:
https://bpb-us-e1.wpmucdn.com/sites.mit.edu/dist/6/1810/files/2024/04/AHTTv600.pdf

Keep every inherited scalar-tolerance, DC, Bode, transfer, rank, Lyapunov and diagnostic course, the other-owner passive-fin flux and pipe changes, actual Pi checks, public contract, pinned static kit, CAD and provenance gates. Preserve the first original negative and successful logs, verify their actual checkout trees, and compare the landed tree directly with the successful CI tree. No workflow or oracle changes after execution. This is maintained source qualification, not a new model score, contest result or timing result.
