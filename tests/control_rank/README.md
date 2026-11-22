# Algebraic controllability and observability

The state-space verifier returns the Kalman ranks of the finite binary64 coefficients accepted by its existing parser. These are algebraic properties of that supplied system. They do not measure the distance to an uncontrollable or unobservable system.

## Two distinct losses of information

For `A = diag(-1,-2)`, `B = (1,2^-42)^T`, the controllability determinant is `-2^-42`. A single relative cutoff scaled by the largest entry can discard the independent second direction. Changing the state units by `diag(1,2^k)` also gives `C = (1,2^-k)`; the input-output transfer function is unchanged for every frozen coordinate case.

For `A = [[-1,0],[delta,-1]]`, `B = (1,1)^T`, and `C = (1,1)`, the two Kalman determinants are `delta` and `-delta`. With `delta = 2^-55`, every supplied coefficient is representable, but ordinary floating multiplication rounds `-1+delta` to `-1` while constructing the Krylov blocks. Changing an elimination tolerance cannot recover the lost direction.

## Exact span closure

A common power-of-two denominator converts `A` to an integer matrix `M = D*A`. Each input column is independently converted to a primitive integer vector. These nonzero scalings preserve the reachable subspace.

The solver reduces candidate directions against an ordered exact integer basis, dividing common factors to limit integer growth. Every independent residual is queued once, and its image under `M` is considered. All accepted directions are reachable. At exhaustion the span contains the input columns and is invariant under `M`, hence also under `A`; it is therefore exactly the Kalman reachable subspace. Full rank permits immediate termination. Observability uses the same algorithm on `A^T,C^T`.

There are at most `n` retained basis vectors. Input matrices and integer bit lengths remain part of the cost. No float Krylov products, approximate pivot cutoff, or fitted threshold enter this computation. A distinction already rounded away in the input remains absent; the frozen MIMO control explicitly checks that case.

## Frozen qualification

`protocol.json` defines 49 systems with ranks fixed by diagonal-mode support, nonsingular input/output matrices, a two-by-two determinant, or a Jordan-chain argument. The comparison also checks 196 correct and opposite Boolean claims, omitted and empty ports, and the real registered Pi entry. Shared output digests retain stability, poles, Lyapunov results and characteristic polynomials for comparison between stock and candidate. The test oracle contains no generic rank implementation.

The maintained repository workflow provides the integration gate. These constructed engineering controls use no organizer task content and establish no new competition score.

## Primary references

- Zoran Gajić, [Linear Dynamic Systems and Signals, Chapter 5](https://eceweb1.rutgers.edu/~gajic/psfiles/chap5.pdf): continuous-time Kalman controllability and the invariance of controllability and observability under nonsingular state changes, Theorems 5.4–5.6.
- Shreyas Sundaram, [Fault-Tolerant and Secure Control Systems](https://engineering.purdue.edu/~sundara2/misc/ft_control_lecture_notes.pdf), Sections 2.6.1, 2.6.2 and 2.6.5.
