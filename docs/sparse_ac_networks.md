# Sparse verification of larger AC networks

The existing `ac_power_flow` operation uses sparse numerical methods for
networks with at least 128 buses and no explicit PV reactive-power limits.
The input and output schema is unchanged. A network with specified
`q_min_injection_pu` and `q_max_injection_pu` continues to use the established
dense regulator solver. The sparse path does not qualify generator-limit
switching, security constraints, dynamics, or field operating safety.

The polar solve stores the actual sparse bus admittance and Newton Jacobian.
A second solver independently forms currents from individual branch terminals
and solves the rectangular current-balance equations with an analytic sparse
Jacobian. It starts from the original flat voltages. If that start stalls, it
tries an independently computed DC angle estimate, then solves the full AC
equations again. The DC estimate does not certify a voltage or replace any
physical check; zero reactance or a singular DC graph makes that seed unavailable.

The answer still has to pass the existing polar/rectangular agreement,
terminal-current agreement, Tellegen balance, and emitted-voltage checks.
Proposed voltage maps must satisfy the current inputs. Reusing an operating
point from a different load state is rejected.

## Reproduce the public network qualification

Run `ship/verify_sparse_ac_networks.py` using its documented pinned MATPOWER
and PYPOWER checkouts. No MATLAB input is executed. The receiver parses the
public case matrices, independently runs upstream PYPOWER from a flat start,
then calls the harness's registered conservation router.

The course covers IEEE 14, IEEE 118, IEEE 300, PEGASE 1,354, and Polish 2,383.
For each case it checks the original operating point, a 20% increase in one
positive PQ load, and a 17-degree global phase rotation with reversed input
ordering. It compares every complex bus voltage and all four real/reactive
branch-terminal components. It also gives the changed network the original
voltage map and requires rejection. PV reactive-limit enforcement is disabled
on both engines for this comparison.

PEGASE is a fictitious European-scale engineering benchmark; the Polish case
describes a winter-peak snapshot. Both are evaluated as steady-state
simulations here. Case metadata and attribution remain in the upstream files:

- [MATPOWER case1354pegase](https://github.com/MATPOWER/matpower/blob/b6c85b0323d20fb801cab3377c803a285d730847/data/case1354pegase.m)
- [MATPOWER case2383wp](https://github.com/MATPOWER/matpower/blob/b6c85b0323d20fb801cab3377c803a285d730847/data/case2383wp.m)
- [Independent PYPOWER implementation](https://github.com/rwl/PYPOWER/tree/c7f1cf50bedd34012b92fae4f092ec55617546a8)

The first cold comparison against harness source `aa8c9dcca1d708278168ec0feb4fda31f2e99c6d`
used one BLAS thread and a 35-second process limit. Both large dense baselines
reached that limit. The sparse candidate's separately observed processes
completed in 3.78 seconds / 106,020 KiB peak RSS and 8.08 seconds / 191,260 KiB,
respectively. These are measurements from a shared CPU environment, not fixed
runtime promises. The dedicated CI workflow rechecks numerical correctness
against the pinned external solver and retains its machine-readable result.

This course verifies public engineering calculations and packaging. It does
not measure a PLE model score, paid-model usage, or a portal submission. PLE
artifacts must still be produced by the actual authorized evaluator on the
exact harness source.
