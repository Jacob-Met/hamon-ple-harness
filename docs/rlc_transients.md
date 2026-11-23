# Stable series RLC transients

The `rf_transmission_rlc` solver's `rlc_transient` mode evaluates the capacitor voltage and series current after a constant voltage source is applied at time zero. It uses the supplied initial capacitor voltage and inductor current.

| Input | Meaning |
|---|---|
| `R` | Resistance in ohms, nonnegative |
| `L` | Inductance in henries, positive |
| `C` | Capacitance in farads, positive |
| `V_s` | Constant source voltage in volts; default 1 |
| `v_c0` | Initial capacitor voltage; default 0 |
| `i_l0` | Initial series current in amperes; default 0 |
| `t` | Elapsed time in seconds, nonnegative |

All inputs must be finite. Derived scales and returned values must also fit the supported floating-point range.

## Endpoint propagation and independent verification

The analytical path uses a stable two-state propagator. In the overdamped case it computes the slow root using the product of the two roots, avoiding subtraction of nearly equal rates. For a nonzero initial state with distinct real poles, it projects the voltage and current onto those poles using the same decimal input values as the checker. An exact product of the two projection numerators preserves the smaller signed amplitude when the state lies near a pure mode. It evaluates source response terms with `expm1` and uses an affine increment series near time zero. Near-critical inputs retain their actual damping ratio instead of being snapped to the critical solution.

An independent checker computes a Decimal matrix exponential with a directly propagated forcing column. Its Taylor expansion has an explicit tail bound and is scaled before repeated squaring. It uses neither modal roots nor time stepping across the circuit's fast time scale. Keeping the forcing column explicit preserves very small zero-state responses in the checker as well as the analytical path.

The returned `residual` is the larger voltage/current discrepancy scaled by that component's reference magnitude or nonzero initial magnitude. When an overdamped component decays below its initial magnitude, the checker sizes its precision for the endpoint and repeats the calculation with 16 additional digits. Verification requires both runs to give identical binary64 endpoints and both component errors to pass relative to their final magnitudes, with the smallest positive binary64 value as a floor. `verified` requires a residual below 1e-8. This prevents a tiny zero-state voltage from disappearing inside a tolerance based on the full source voltage. The result names the verification method and reports the number of matrix squarings. `reference_converged` records the agreement when the checker runs at two precisions and is null otherwise.

The original explicit DOP853 check could take impractical numbers of steps on stiff circuits. SciPy's [solver documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) distinguishes explicit methods for nonstiff problems from implicit methods for stiff problems. Here the system is linear with constant coefficients, so endpoint propagation removes the need for an integration trajectory altogether. The same checker works when SciPy is absent.

## Reproduce the public numerical controls

From the repository root, with the normal requirements installed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_rlc_transient_stability.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ship/verify_rlc_transients.py --source . --cases tests/fixtures/rlc_transient_controls.json --output /var/tmp/rlc-results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python ship/verify_rlc_transients.py --source . --cases tests/fixtures/rlc_transient_controls.json --output /var/tmp/rlc-without-scipy.json --without-scipy
```

The frozen 50-case comparison spans undamped, underdamped, near-critical, critical, and stiff overdamped models, with three source/initial-state combinations. Each process has the same four-second deadline and one BLAS thread; two processes run concurrently. An independent 80-digit mpmath matrix exponential supplies the endpoint references. Passing requires a verified output and scaled endpoint error below 1e-10.

| Version | Passed | Process deadlines | Maximum scaled error among returned results |
|---|---:|---:|---:|
| baseline | 31/50 | 14 | 2.11e-10 |
| candidate | 50/50 | 0 | 3.2e-16 |
| candidate-without-scipy | 50/50 | 0 | 3.2e-16 |


Fourteen baseline calls exceeded the process deadline. Five other baseline cases missed the stricter endpoint-accuracy threshold because the original implementation approximated near-critical models as exactly critical; their original 1e-8 verifier still accepted them. These are distinct failure types.

The additional regression tests cover responses as small as 5e-51 V, a stiff circuit whose fast mode changes before its tiny voltage response, exact initial/equilibrium states, invalid passive inputs, and deliberate erasure of a tiny response. The actual registered TypeScript tool executes both stiff and tiny cases, and rejects an invalid inductance.

This comparison uses manufactured public linear-circuit inputs. It does not measure organizer tasks, model accuracy, a four-discipline score, or contest qualification. Very long oscillatory phases can exceed the accuracy available from floating-point input parameters; discrepancies remain unverified. No inherited competition artifacts are regenerated by these checks.
