# Exact scalar tolerance comparisons

This preregistered regression course starts from exact DC merge 145418856e4f14fcc003da2ce9e28524c0fd3b56, tree abf94bb76de552b4815a723492a014997cbe177c. That landed tree exactly matches the successful DC CI tree and includes the concurrent passive-fin and pipe-network repairs. The new tolerance course has not been numerically executed at preregistration.

## Decision contract

The Wave 5 scalar checker serves PDE, control, psychrometrics and FEM. A claim concerns the scalar actually returned by the verifier. Preserve the existing float conversion of numeric claims and tolerances, including numeric text and Boolean 0/1 values.

Let a be the accepted finite scalar, c the accepted finite claim, A the nonnegative absolute tolerance, R the nonnegative relative tolerance, and F the exact binary64 value of the existing literal 1e-12. The decision is:

E = |a-c|
L = max(A, R max(|a|, F))
within_tolerance = (E <= L)

Evaluate this inequality using the exact dyadic values of the accepted numbers. The formula keeps the existing actual-value floor, maximum and inclusive boundary. The ordinary solver result remains authoritative: a truthful claim cannot turn an existing passed=False into success.

Require a complete claim pair with a nonempty string field selecting an existing numeric scalar. Unknown fields, rational or structured evidence, unavailable values, invalid numeric text, nonfinite claims and negative or nonfinite tolerances explicitly refuse. Keep the existing specific reasons for undefined results and DC rational evidence.

## Why intermediate rounding matters

The source has three ways for an outside-tolerance value to collapse onto a rounded boundary:

- With actual 1, claim -2^-54, absolute tolerance 1 and relative tolerance zero, the exact error exceeds 1, while the floating subtraction rounds to 1.
- With d=2^-27, actual 1+3d, relative tolerance 1+d and absolute tolerance zero, the exact limit is 1+4d+3d^2. A claim -d-4d^2 has the larger exact error 1+4d+4d^2, equal to the upward-rounded limit.
- With actual zero, claim u=2^-1074 and relative tolerance 2^-1035, the exact floor-based limit is about 0.55u. Its floating product rounds to u, although the exact error is larger.

Overflow gives another boundary collapse. For the accepted actual 1e308 and its opposite claim, an exact comparison with relative tolerance 1.9 refuses; tolerance 2 is exact equality. The old floating error and both limits overflow, obscuring that distinction. A nonfinite error diagnostic also fails strict JSON serialization.

These are analytic source witnesses. Original CI results must establish the observed failures and counts.

## Evidence

Valid comparisons retain field, actual, claimed, abs_error and within_tolerance. They add tolerance_limit, canonical abs_error_exact and tolerance_limit_exact strings, and a range status for each diagnostic. Each display is finite or null. Preserve nonzero subnormals; label a nonzero value that rounds to zero below_float_range and an overflow above_float_range.

The exact evidence explains a refusal even when the two displayed values coincide. Invalid requests return a reason without claim_check. The checker validates its inputs; the numerical correctness of each underlying physical calculation remains a separate question.

## Independent frozen course

The independent receiver defines 29 paired cases, giving 58 route calls before the final assertion. Its oracle uses integer multiples of powers of two, exponent alignment, integer multiplication and gcd normalization. It does not use the producer's Fraction comparator.

Twelve core cases cover the absolute, product, overflow and magnitude-floor families, with seven approvals and five refusals. Six valid controls cover sign reflection, a rounded third, a Boolean claim, a truthful claim on a failed feedback calculation, a positive limit that rounds to zero, and the distinction between maximum and sum when both tolerances are positive. Eleven cases cover explicit malformed or unsupported requests.

Expected totals are 10 claimed approvals and 19 refusals, plus 28 unclaimed approvals and one unclaimed refusal. Every full unclaimed output must match across stock and candidate, with no excluded fields. Keep raw old decisions available for analysis if their diagnostics fail strict JSON.

The separate actual registered Pi course has 14 request specifications: six valid comparisons and eight malformed or unsupported requests. It requires three approvals and 11 refusals. It additionally covers numeric-text acceptance and an unhashable list selector. Both maintained unittest and pytest routes exercise the tool.

## Source identities

- Prerequisite DC solver SHA256: 064606a5df98b2c85521b45647968fe4bea7d58d48ac500dd7b7dee3ad0f44d8
- Comparator candidate SHA256: 1a49d00803e9e6464448fd43ecbab5292ed58366c8f9cfe766c7b36522a4b460
- Independent receiver v2 SHA256: 4f537ccc6ed79a8a680aa6c8e865722e78d5826404ae3ef496fe40a37231511c
- Registered Pi receiver SHA256: 195fdce3917c759667cb68daf5d1dfdea8f525d663a77063f877fb0f73e6be71
- Protocol SHA256: 67284ea890a5667b4c9d3609816a40b1b1137fba97136c062eff13a6250098e8

The original 28-case receiver remains retained. The only pre-execution addition was the maximum-versus-sum discriminator identified by independent review. The implementation patch was ported onto the exact landed DC source with zero fuzz; only its later hunk coordinate shifted by 32 lines for the existing fin change. Its semantics and all test expectations remained unchanged.

## Qualification boundary

Preregister identical tests and protocol on matched stock/candidate branches, with only the solver source differing. The maintained PR workflow must be their first numerical execution. Retain all original logs, failures and partial observations.

Require all 29 cases and registered Pi requests, zero solver/JSON errors, exact full unclaimed projection equality, unchanged inherited DC/Bode/transfer projections and every existing maintained gate. Record actual CI checkout trees and compare the eventual landed tree directly against the successful CI tree.

This course makes no model-score, organizer-score, universal numerical-accuracy or performance claim.

## Primary references

Python 3.12 documents [exact conversion of a supplied float to Fraction](https://docs.python.org/3.12/library/fractions.html), the [exact integer ratio of a finite float](https://docs.python.org/3.12/library/stdtypes.html#float.as_integer_ratio), and [float spacing and subnormal values](https://docs.python.org/3.12/library/math.html). The particular tolerance formula is the repository's existing contract; the analytic examples above follow from that formula.
