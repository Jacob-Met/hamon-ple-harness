# Exact continuous-time stability and requested Lyapunov equations

The state-space verifier previously inferred stability from numerical roots and
required that result to agree with a thresholded positive-definiteness flag for
the requested Lyapunov equation. Agreement between those two approximations
could approve an incorrect stability claim. For example, with
`A = diag(-1, -2**-55)` and `Q = I`, the exact solution is
`P = diag(1/2, 2**54)` and the system is strictly Hurwitz. The root cutoff and
relative linear-system pivot cutoff can both report false, allowing an incorrect
`hurwitz_stable = false` claim to pass.

The sign of a requested `Q` is a separate issue. A stable system with a negative
or zero forcing matrix need not have positive-definite `P`. Conversely,
`A = [1]`, `Q = [-1]` has the positive solution `P = [1/2]` although the system
is unstable. Symmetrizing a solution for a nonsymmetric `Q` changes its equation.

## Contract and method

Accepted finite binary64 coefficients are treated as exact rational numbers.
The symmetric operator `P -> A^T P + P A` has `n(n+1)/2` independent coordinates.
Primitive integer elimination is shared by the identity forcing and the
requested symmetric forcing. Each right-hand side is classified as having a
unique solution, being inconsistent, or having nonunique solutions. A unique
solution is checked against every entry of the original matrix equation.
Strict positive definiteness is decided by exact Schur pivots.

For `Q = I`, a positive-definite solution exists exactly when `A` is Hurwitz;
that solution is then unique. This identity-forcing certificate determines
`hurwitz_stable`. The separately requested `Q` may be positive, semidefinite,
zero, or indefinite. Nonsymmetric, nonfinite, and incorrectly shaped `Q` are
explicitly refused.

Every unique requested solution retains its lossless rational-string matrix in
`lyapunov_P_exact`. The float matrix is omitted if any entry would overflow or
if a nonzero entry would round to zero. Singular equations have null `P` and
null `lyapunov_pd`, and either Boolean claim about that undefined PD flag is
explicitly refused. Nonunique does not mean that no positive-definite solution
exists; it means no single requested solution was certified.

The existing characteristic polynomial, Cayley-Hamilton residual gate,
approximate pole diagnostics, algebraic controllability/observability ranks,
and transfer-function implementation remain outside this repair. Approximate
poles no longer determine the state-space stability flag. Large-dimensional
or extreme-magnitude floating diagnostics can still limit acceptance; this
course does not establish unrestricted numerical robustness or runtime bounds.
Exact elimination also has coefficient-dependent integer growth.

## Frozen independent course

`protocol.json` was fixed before any solver execution. Its SHA256 is
`8b6486d66c109710933a9ade73b3c28055c1c1f98e70f325e71fba9f555b40e9`.
The base is `4c8bf73ef4676291b60984f36ec04add92f68bd0`.

The course contains 32 analytically specified systems and 128 Boolean claims:
25 unique solutions, three inconsistent equations, three nonunique equations,
and one invalid nonsymmetric forcing. Cases cover mixed and scalar time
scales, signs and scales of `Q`, semidefinite forcing with and without state
coupling, zero dynamics, a rotation, opposite-sign real eigenvalues,
coordinate/time transformations, and exact solutions outside float range.

The receiver compares requested solutions with analytic rational formulas,
checks full residuals without solving another linear system, and checks the
one- and two-state principal minors independently. Every system is also called
without a claim. Strict JSON serialization and every Boolean claim outcome
must succeed without exceptions. Representative requests go through the real
registered Pi tool, including both claims for undefined positive definiteness.
The stock and candidate branches receive identical test and protocol files;
only the solver source differs. Both use the maintained repository PR CI,
including all inherited rank, physics, registration, and static intake gates.
Original failures remain evidence; no case or tolerance is changed after the
first execution.

## Primary mathematical references

- MIT, *Lectures on Dynamic Systems and Control*, Lecture 14, Theorem 14.1:
  <https://ocw.mit.edu/courses/6-241j-dynamic-systems-and-control-spring-2011/996025f6db0d90b00f11c44fc49b85f9_MIT6_241JS11_textbook.pdf>.
- Stanford, Sanjay Lall, *Controllability and Observability*, slides 9-10 and
  9-19 (continuous-time Lyapunov solvability and equivalence):
  <https://lall.stanford.edu/engr210a/lectures/lecture9_2001_10_30_01.pdf>.
- Rutgers, Gajić, Chapter 5 (state-coordinate invariance):
  <https://eceweb1.rutgers.edu/~gajic/psfiles/chap5.pdf>.
