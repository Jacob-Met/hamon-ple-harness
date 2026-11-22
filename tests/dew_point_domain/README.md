# Bounded dew/frost-point diagnostic

The former computed dew/frost-point helper always bisected between-80 and150 degrees Celsius. A supported-90-degree frost point therefore returned-80, and high supported liquid points returned150. Separately, dry air was assigned a finite-100-degree sentinel even though-100 is itself a real positive-pressure endpoint of the maintained correlation.

This change classifies vapor pressure against the supported ice and liquid branches before inversion. Supported values receive a numerical temperature; zero vapor, out-of-range pressure and the gap between the stitched phase formulas receive null and an optional `t_dp_status`. The existing scalar checker refuses numeric claims on unavailable output. A valid physical state need not fail because one diagnostic is unavailable.

The computed diagnostic retains its default0-degree phase policy. Its coefficient support comes from the actual ASHRAE2017 equations in the source: ice down to-100 and water up to200 degrees Celsius. The newer2025 equations are a different model. The ordinary70-step calculation is preserved when its result remains on the selected branch; supported extensions use a branch bracket. Exact pressure equality at the closed-100/0/200 endpoints follows an explicitly frozen endpoint convention. Finite ordinary results gain no status field.

This does not change numeric input types or precedence, saturation coefficients, moisture/energy/volume fields, wet-bulb behavior, physical admission, process equations or the shared scalar comparator. Null output is not a new accepted temperature input. The inherited alternate process forward phase switch at0.01 degrees is also unchanged.

## Frozen first comparison

The independent receiver makes17 ordinary requests and14 numeric claims before its aggregate assertion. Expected claim results are six approvals and eight refusals. Every ordinary request remains admitted under the existing physical predicate.

Two pressure-input cases use an independent80-digit Decimal evaluation of the published equations at-90 and170 degrees Celsius. The explicit-90 case is separately a public round trip. Three closed endpoints check the declared exact output convention. Five unavailable cases distinguish dry RH/W inputs, positive vapor below range, a phase-gap pressure, and pressure above range within the existing RH admission tolerance. Three ordinary finite states, a strictly negative ice-side limit, mixing and cooling complete the set.

The above-range row deliberately exercises an existing slightly-supersaturated admission tolerance. It does not assert strict physical undersaturation or establish real-fluid accuracy. The phase-gap false-zero claim allows1e-8 degrees so it includes the legacy bisection limit; other unavailable-value challenges use zero tolerance. The protocol spells out all tolerances and evidence classes before execution.

The actual registered Pi tool receives eight frozen requests covering supported-90 inversion, truthful/false finite claims, dry null/status, an ordinary finite state and below-range null/status. All eight enter the existing batch before assertions. Four ordinary requests pass; one of four claims passes and three refuse.

## Preservation and qualification

Every one of the17 new ordinary outputs is hashed in full and with exactly `t_dp_c` and `t_dp_status` removed. The latter hashes must match stock for all17. Five named unaffected finite/process controls must also match in full.

The inherited wet-bulb receiver receives one reporting-only field: a hash excluding exactly those same two diagnostic keys. It retains all32 requests, ordering, expectations, bracket fields and original hashes. The new stock must match the previous landed wet-bulb output hashes. All32 non-dew projections and all other wet-bulb evidence must match across this pair.

All204 other inherited case records and their totals, plus every maintained CI gate, remain required. Both first complete original logs are retained once. Qualification binds actual CI checkout trees and verifies the actual landed tree. The branches differ only in the solver file; test and protocol changes are identical.

This is qualification of the maintained approximate verifier. There is no new competition score, timing claim, exact real-fluid certificate or uniquely identified floating-temperature claim.

Primary reference: [ASHRAE Fundamentals2017 SI Chapter1](https://handbook.ashrae.org/Handbooks/F17/SI/f17_ch01/f17_ch01_si.aspx), equations5 and6.
