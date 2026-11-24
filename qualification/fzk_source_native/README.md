# FZK-Haus public-source native qualification

The workflow downloads the unchanged AC20-FZK-Haus IFC from IfcOpenShell commit
90c840c68fb35a54e1a61505c6417ab854f826a5 and checks its exact digest before export
and measurement. The native importer (IfcOpenShell 0.8.3.post2) and independent
OpenCASCADE bindings (cadquery-ocp 7.8.1.1.post1) run in separate interpreters.
The candidate uses the repository's pinned runtime. Neither native stage
imports the candidate reader or its Boolean kernel.

Native parts must form closed, valid solids. Body parts are combined by a
native material union before per-occurrence volume is reported. Summing
overlapping window components is not accepted. The oracle also measures gross
walls, opening-subtracted walls, and all 22 window components. It checks
material intersections both at the original source context precision and in a
separate native conversion whose in-memory context precision is 10 nm; every
authored coordinate remains unchanged. This second conversion is a sensitivity
variant, not a change to the downloaded public file.

The historical source investigation found 81 independently measurable physical
occurrences and one unresolved stair, STEP #14502. The intact model has 82
physical occurrences. This expected coverage is an explicit gate; failures
remain visible. No repair of the stair is applied. A sum of per-occurrence
volumes is not a union of the whole building.

The bounded walls have independent source integrals of 5.779930866498654 m³
gross and 5.544406029727043 m³ net. The authored 128-edge opening polygons remove
0.235524836771612 m³. The unbounded-plane wall volumes are 2.056376840517137 m³.
These public-source expectations are qualification data, never production
reader constants.

A workflow run qualifies only its exact source commit. Reconstructed code
receives no credit from measurements of an earlier lost local candidate.
This is engineering source qualification, not an organizer score, hidden-task
accuracy measurement, or portal submission.

The registered candidate step reads the complete original 2.5 MB file once via
its ordinary TypeScript task receiver. It compares every reported physical
quantity with the independent native table, accounts for all requested pairs,
and requires the one unresolved stair and whole-model uncertainty to stay
explicit. The full pair matrix is not independently verified by the native
oracle; only the two wall/window relations have native intersection evidence.
