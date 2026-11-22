# Explicit wet-bulb feasibility

The former explicit wet-bulb branch clipped every negative humidity balance to dry air. A sufficiently low wet bulb can therefore be accepted even when it describes no nonnegative humidity state. This comparison freezes a narrow repair: validate the signed balance, refuse infeasible explicit inputs, and preserve the established RH, humidity-ratio and dew-point routes.

A negative result is admitted as dry air only when the very next representable wet-bulb input gives a nonnegative result in the same vapor and wick phases, with both endpoints checked. That exceptional input carries `wet_bulb_dry_bracket_c:[lower,upper]`. Positive results stay positive. Exact calculated zero has no bracket field. This is a declared input-resolution convention; it does not certify an exact thermodynamic root or guarantee every public dry-air round trip.

## Frozen independent comparison

The independent receiver issues 20 fixed requests and 12 relational requests before its aggregate assertion. The fixed set expects 12 approvals and eight refusals: six physical states checked with table-based energy bounds, five alternate-input controls, four mixing/cooling cases and five scalar claims. Three impossible states with a claimed humidity of zero must refuse before scalar comparison. Genuine dry and saturated claims must remain accepted.

The physical oracle uses saturation-pressure intervals around ASHRAE Table3 values at -20, -10 and +5 degrees Celsius, inserted into a separately expressed adiabatic energy balance. It imports no producer saturation or wet-bulb helper. This checks the sign and a broad feasible humidity interval in the maintained approximate model.

For each dry-bulb anchor, 40 and -10 degrees Celsius at101325Pa, the public RH0 inverse supplies one wet-bulb input. Five fixed probes use a far lower offset, its predecessor, itself, its successor and a far upper offset. The receiver checks the resolved adjacent dry bracket and the distant signs, while leaving the unrelated near-neighbor verdict unfixed. These are self-consistency relations, distinct from the independent physical oracle. The protocol deliberately specifies no exact combined approval count.

The actual registered Pi tool receives eight additional frozen requests: liquid and ice infeasibility, a false zero-humidity claim, invalid mixing and cooling inlets, and chart/dry/saturated controls. The chart check uses ASHRAE Example1's approximate40/20-degree values with explicit chart-read tolerances. All calls enter the existing shared tool batch before assertions.

## Preservation and first execution

Both comparison branches carry identical receivers and this protocol. They differ only in `hamon_pde_control_psychro_fem.py`. The first numerical runs are the maintained pull-request CI runs. Source and all expectations are frozen before opening those PRs.

Retain whole-output hashes for every request, including every unaffected fixed approval. Only the named near-boundary comparisons may remove the single new bracket field; the original hashes are always retained. Keep all204 ordered inherited records across the fin, scalar-tolerance, DC, Bode, transfer-function, rank, Lyapunov and diagnostics courses, plus every maintained delivery gate.

Both complete original logs are retained once. Qualification checks the actual CI checkout tree and the actual landed tree. Any concurrent main change requires separate qualification of the combined tree while preserving the original pair.

This is source qualification of the maintained approximate psychrometric verifier. It makes no new model/contest score, timing, exact real-fluid or exact transcendental claim.

Primary reference: [ASHRAE Fundamentals2025 SI, Chapter1](https://handbook.ashrae.org/handbooks/F25/SI/F25_Ch01/F25_Ch01_si.aspx), Table3, thermodynamic wet-bulb section, and Example1.
