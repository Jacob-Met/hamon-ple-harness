# State-space characteristic coefficients and diagnostic range

After the exact Lyapunov repair, floating characteristic calculations could
still prevent the verifier from returning a valid stability certificate.
For `M = 2**512`, `A = diag(-M, -1)` has the finite positive solution
`P = diag(1/(2M), 1/2)`. Its floating characteristic recurrence first loses
the `+1` in the trace and then returns `M/2` for the constant coefficient,
whose correct value is `M`. The quadratic discriminant overflows, and the
explicit floating scale power raises before the result can be returned.

Other cases distinguish information lost during arithmetic from information
that cannot be rendered in binary64. With `H = 2**520`, the polynomial of
`diag(-H, -H)` has an unrepresentable constant `H**2`, although both roots
and its positive Lyapunov matrix are representable. With `h = 2**-550`,
`diag(-h, -2h)` has a nonzero constant coefficient that rounds to zero.
Neither limitation invalidates the exact stability certificate.

## Exact characteristic certificate

Accepted finite binary64 coefficients are interpreted as exact dyadic
rationals. Clear their common denominator `D`, compute the characteristic
polynomial of the integer matrix `D*A` by Faddeev-LeVerrier recurrence, and
divide its degree-k coefficient by `D**k`. Every trace division by `k` is
exact for an integer matrix. Evaluate the complete recurrence residual in
the original units by dividing the terminal matrix by `D**n`.

The result preserves all exact coefficients as rational strings in
`char_poly_exact`. A float `char_poly` is included only when no coefficient
overflows and no nonzero coefficient rounds to zero. Such a float list may
still round its coefficients. The Cayley-Hamilton certificate explicitly
identifies `char_poly_exact` as its basis; it does not assert that the rounded
float polynomial has zero exact residual.

The new gate checks the exact characteristic residual and the existing
identity/requested Lyapunov certificates. Numeric diagnostic availability
does not decide that verification result. Rank and transfer-function
implementations remain outside this repair.

## Numeric pole diagnostics

Algebraic zero roots are factored from the exact coefficient list. Linear
roots are converted directly. Quadratics use the exact discriminant and
a scaled square root. One real root uses the sum of like-signed magnitudes;
the other is recovered from the exact root product before conversion. This
avoids subtracting two large, nearly equal approximations to recover a slow
pole. Real and imaginary parts of complex conjugate roots are rendered
separately.

Higher-degree diagnostics scale the polynomial variable by a power of two,
call the existing numerical root iteration, and restore physical units.
`finite_approximation` means that the result is a finite numerical estimate.
It is not an exact symbolic root certificate or a condition-independent
accuracy guarantee; the iterative solver has no new backward-error check.
The frozen cubic controls below test specific accuracy and scaling behavior.

| Field | Meaning |
| --- | --- |
| `char_poly_exact` | Descending exact monic coefficients, as rational strings |
| `char_poly_float_status` | `finite` or `outside_float_range` |
| `cayley_hamilton_residual_basis` | Always `char_poly_exact` |
| `cayley_hamilton_residual_exact` | Exact original-unit residual as a rational string |
| `zero_eigenvalue_multiplicity` | Exact number of trailing zero polynomial coefficients |
| `eigenvalues_status` | `finite_approximation`, `outside_float_range`, or `unavailable_numerical_diagnostic` |

An unavailable numeric coefficient or root list is null. The exact
characteristic and stability certificates remain available. Arithmetic and
storage costs still depend on matrix dimension and integer coefficient growth;
this change makes no general performance claim.

## Frozen independent comparison

The source base is `d71933ab5aae721cb21106ca2144428a1d5bf23c` (PR 162).
The protocol was fixed before any numerical execution, with SHA256
`40a015293e008fa2a73efdb959a7b0bfb431da6bddd8ce3f20514175072826f2`.

The 17 systems include ordinary stable and unstable diagonals, mixed time
scales in both coordinate orders, repeated poles, large spirals and rotations,
tiny coefficients, a genuinely unrepresentable eigenvalue, a cancellation
example, three uniformly scaled stable cubics, a signed cubic, and an exact
zero pole. There are 14 unique identity-forcing solutions and three
inconsistent equations. The 68 Boolean claims require exactly 31 approvals
and 37 refusals; every system is also called without a claim.

The independent receiver compares analytical rational coefficients and
Lyapunov matrices, substitutes both into their full matrix equations, and
matches all roots with multiplicity. Each nonzero root component has its own
relative tolerance; a fast root cannot conceal a missing slow root. True zero
poles must be exactly zero. It checks correct float coefficient rounding,
explicit withholding outside range, and strict JSON serialization. All 85
calls complete before the aggregate assertion. Twenty representative real
registered Pi requests run in both unittest and pytest.

The first numerical execution is a matched stock/candidate pair using the
maintained repository PR CI. Both branches have identical test and protocol
files. Only the solver source differs. The inherited rank and Lyapunov
courses and all other maintained gates remain required. Original failures
are retained without case or tolerance changes. This is a synthetic/static
qualification, with no organizer, dev-set, scored, or paid-model evaluation.

## Primary references

- Christian Baer, *The Faddeev-LeVerrier algorithm and the Pfaffian*, Section 2:
  <https://arxiv.org/pdf/2008.04247>.
- Fredrik Johansson, *On a fast and nearly division-free algorithm for the
  characteristic polynomial*, Algorithm 1 and exact division discussion:
  <https://arxiv.org/pdf/2011.12573>.
- William Kahan, *Baleful Effect*, quadratic root-product formulation and
  numerical accuracy cautions:
  <https://people.eecs.berkeley.edu/~wkahan/ieee754status/baleful.pdf>.
