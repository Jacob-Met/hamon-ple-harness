# Exact transfer-function root locations

A transfer denominator can have a strictly stable slow pole that the previous
fixed real-part band labeled marginal. For D = s + 2^-55 and N = 1, the
open-loop pole is strictly negative, but the previous real < -1e-6 test
returned a false strict-Hurwitz flag. Agreement between the approximate root
count and the numerical Routh count did not detect that incorrect flag.

Feedback introduced a second loss of information. For
D = (s + 1)(s^2 + 1) and N = +/-2^-55, binary64 addition rounds the
closed-loop constant back to one. The exact constants 1 - 2^-55 and
1 + 2^-55 put the conjugate pair on opposite sides of the imaginary axis.
The stability calculation must retain this addition exactly.

## Coefficients and scope

The existing float conversion defines the accepted finite input values.
Each accepted binary64 coefficient is then interpreted as its exact rational
value. This does not reconstruct an intended decimal value before conversion.

Align D and N by degree and add them in their common original units.
Only then normalize D and D + N separately to monic polynomials. Changing
the leading sign of D changes the sign of N as well. Common nonzero gains
therefore leave both normalized characteristics unchanged.

The certificate describes the authored denominator and formal unity
negative-feedback characteristic. It retains numerator/denominator common
factors. It does not silently replace the authored dynamics with a canceled
minimal transfer function. The existing open denominator degree requirement
remains at least one.

A nonzero constant D + N has no finite roots and receives a true
strict-Hurwitz flag under this finite-root convention. This is not a
certificate of physical causality, properness, or descriptor-system
well-posedness. An identically zero characteristic instead has undefined
degree, counts, and closed-loop stability. Its open certificate may still be
displayed, but the overall verification and every Boolean claim refuse.

## Exact two-sided shift

For a monic real polynomial

    p(s) = sum(a_j * s**(n-j), j=0..n), a_0=1,

write its translated coefficients as polynomials in an indeterminate t:

    c_k(t) = sum(a_j * binom(n-j, k-j) * t**(k-j), j=0..k).

Build the Hurwitz matrix with zero-based entry H[r,c] = c_(2*c-r+1),
using zero outside the coefficient range. Its leading principal
determinants are Delta_1(t), ..., Delta_n(t).

For a regular Routh array, its first column consists of one followed by
successive Hurwitz-minor ratios. The number of first-column sign changes
counts roots in the strict right half-plane. Barnett's Section 3 gives
this regular-array relation and sign criterion.

The extension used here follows by translation. For sufficiently large
positive t, every root of p(s+t) is in the strict left half-plane.
Its Hurwitz determinants are positive there, so none of the determinant
polynomials is identically zero. Each therefore has a first nonzero term

    Delta_k(t) = b_k * t**v_k + higher powers.

Near zero its sign at positive t is sign(b_k), and its sign at negative
t is (-1)^v_k * sign(b_k). These signs determine the complete Routh
first column exactly, without substituting a small numerical epsilon.
The finite nonzero roots of the determinant polynomials leave a punctured
interval in which the regular-array criterion applies.

Choose positive t inside that interval and smaller than every nonzero
absolute root real part of the original polynomial. Translation by +t
moves original imaginary-axis roots into the left half-plane, while
translation by -t moves them into the right half-plane. Off-axis roots
remain on their original sides. Translation preserves algebraic
multiplicity, including repeated origins and repeated imaginary pairs.

Thus, if V_plus and V_minus are the two sign-variation counts,

    R = V_plus
    J = V_minus - V_plus
    L = n - V_minus

are the exact right-half-plane, imaginary-axis, and left-half-plane root
multiplicities. Strict Hurwitz stability is exactly R == 0 and J == 0.
This two-sided specialization is our deduction from the regular theorem,
not a claim that the cited references directly present this implementation.

One fraction-free Bareiss elimination over rational polynomials produces
all leading minors. Every division by the previous pivot must have zero
remainder. No pivot tolerance, numerical root band, or row rescaling
chooses the certificate's signs. Bit growth and polynomial degree still
affect runtime.

## Returned evidence

Both open_loop_root_certificate and closed_loop_root_certificate contain:

| Field | Meaning |
| --- | --- |
| method | infinitesimal_shift_hurwitz |
| status | classified, constant_nonzero, or identically_zero |
| degree | Exact polynomial degree, or null if undefined |
| normalized_coefficients | Descending monic rational strings |
| minor_leading_terms | First nonzero coefficient and power of each shifted minor |
| first_column_signs_plus | Exact signs for a positive infinitesimal shift |
| first_column_signs_minus | Exact signs for a negative infinitesimal shift |
| rhp_poles, jw_axis_poles, lhp_poles | Exact multiplicities, or null if undefined |
| hurwitz_stable | Exact strict-Hurwitz flag, or null if undefined |

The existing top-level stability flags and rhp_poles_routh /
jw_axis_poles now come from these exact certificates. The minor
coefficients and normalized polynomial are rational strings, avoiding
nonfinite numeric encodings.

Numerical poles remain separate diagnostics. They reuse the existing
range-aware state-space root renderer after a power-of-two variable
scaling. Their status reports a finite approximation, a root outside
float range, an unavailable numerical diagnostic, or an undefined
characteristic. Approximate roots do not decide exact acceptance.

rhp_poles_eigen counts strictly positive real components in the rendered
approximate open-loop list, or is null if no list is available.
rhp_poles_eigen_is_approximate is true. A small rendered real part near
an exact imaginary-axis root can therefore disagree with the exact count;
that diagnostic disagreement is not a certificate failure.

## Frozen comparison

The source base is beb98262f623430673e3ac56f22bc8918ffb9a63.
It contains both the PR 165 diagnostic repair and the concurrent PR 163
AC power-flow changes. The next substantive candidate's maintained CI
therefore also qualifies their combined base.

The protocol was frozen before numerical execution with SHA256
eb43024c5dfd9824c5cd744e6847400b2b504561428d666dfa95bfe7776ef265.

The 20 systems cover slow poles of both signs, repeated origins and axis
pairs, symmetric real roots, a zero first pivot with a nonzero row, common
positive and negative gains, a retained unstable common factor, three
exact feedback-boundary cases, a constant feedback characteristic, an
identically zero characteristic, and an invalid constant denominator.
There are 18 fully defined systems, one undefined feedback system, and
one invalid input.

The independent receiver obtains root counts from known factors and a
separate quartic homotopy. It expands every shifted leading minor using
the Leibniz permutation formula, independently of the producer's Bareiss
recurrence. It checks each first nonzero term, both sign sequences, the
known multiplicities, strict flags, serialization, and claim evidence.

All 20 unclaimed requests and all 80 Boolean requests complete before
the aggregate assertion. The required outcome is 36 approvals and
44 refusals with no solver or serialization exceptions. Missing new
schema, wrong historical flags, false approvals, false refusals, and
exceptions are reported separately for stock. Twenty further actual
registered Pi calls cover four representative systems in both the
unittest and pytest stages.

The first numerical execution is one matched stock/candidate pair in
the maintained repository PR workflow. Tests and protocol are identical
between branches; only the solver source differs. All inherited rank,
Lyapunov, characteristic-diagnostic, AC, and other maintained gates
remain required. The actual CI merge tree and actual landed merge tree
must each be inspected; concurrent merges cannot be assumed absent.

## Preserved behavior and limits

The Bode and DC implementations and their inputs remain unchanged.
Their twelve output fields receive per-case and aggregate digests for
the paired runs. Exact output equality is required wherever both
branches return those fields. A stock exception before the output is
assembled is preserved and is not described as a measured equality.

The Bode calculation retains its prior decimal-string rational
interpretation, distinct from the exact accepted binary64 interpretation
of the new stability certificate. It also retains a known range limit:
finite coefficients can imply an unrepresentable crossover frequency,
and the inherited frequency conversion can raise before the complete
response is assembled. This repair makes no unconditional delivery
claim for all finite input arrays.

This qualification does not establish exact symbolic numeric poles,
general conditioning-independent root accuracy, uniform performance,
physical feedback well-posedness, or a benchmark score improvement.
There is no organizer, dev-set, scored, or paid-model evaluation.

## Primary references

- S. Barnett, *A New Formulation of the Theorems of Hurwitz, Routh and
  Sturm* (1971), Sections 2 and 3, especially equations 25-28:
  <https://webhomes.maths.ed.ac.uk/~v1ranick/papers/barnett.pdf>.
- Jongeun Choi, *ME451 Lecture 10: Routh-Hurwitz stability criterion*,
  including the distinction between zero-pivot and zero-row cases:
  <https://www.egr.msu.edu/classes/me451/jchoi/2010/notes/ME451_L10_RouthHurwitz.pdf>.
