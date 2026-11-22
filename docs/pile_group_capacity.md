# Axial compression of a rectangular pile group

Run `python engineering_toolkit.py dual-verify spec.json` with `mode: "domain"`, `kind: "pile_group_capacity"`, and a `spec` containing the fields below. The calculation rates an idealized rectangular array of identical circular piles in uniform undrained clay. A positive `applied_load_N` is a compressive demand. Zero is permitted. Uplift needs a separate resistance calculation, so a negative demand returns a domain error.

| Input | Meaning |
|---|---|
| `n_rows`, `n_cols` | Positive integral pile counts. |
| `pile_D_m`, `length_m` | Positive shaft diameter and embedded length in metres. |
| `spacing_m` | Positive center-to-center pitch in metres. With more than one shaft, pitch must equal or exceed diameter to avoid a geometric overlap. The pitch has no effect on a single shaft. |
| `cu_Pa` | Nonnegative uniform undrained shear strength in pascals. |
| `adhesion_alpha`, `Nc_tip` | Nonnegative shaft adhesion and toe bearing factors, defaulting to 0.75 and 9. A zero value omits the corresponding single-pile contribution. |
| `FS` | Positive supplied divisor for nominal resistance, default 2.5. |
| `applied_load_N` | Nonnegative axial compressive demand in newtons. |

The model computes the individual pile resistance from circular toe area and exterior shaft area, multiplies by pile count and the Converse–Labarre group efficiency, and compares that result with compression of a rectangular soil block. The block spans the outside shaft surfaces. Its toe term uses the block footprint and its shaft term uses undrained shear strength along its four outside faces. The smaller group resistance divided by `FS` defines `Q_allowable_N`. An excessive positive demand returns a valid calculation with `passes_pile_group: false`. The independent numerical check reconstructs the single-pile resistance, group efficiency, both competing group capacities and the allowable capacity from the inputs.

This is a calculation under the stated empirical assumptions. A touching layout establishes the geometric boundary of distinct shafts. The suitability of a spacing, clay model or design factor for a specific project requires separate evidence. FHWA distinguishes nominal axial compression from group uplift in cohesive soils and describes additional conditions for selecting group efficiency. [FHWA GEC 12, Volume I, sections 7.2.2.2 and 7.2.3.2](https://www.fhwa.dot.gov/engineering/geotech/pubs/gec12/nhi16009_v1.pdf)
