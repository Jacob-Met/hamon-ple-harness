# Sampled state-space stability

Call `hamon_conservation_cad_verifier` with outer mode `control`, inner mode `state_space`, and `timebase` set to `discrete` when A advances the state by one sample. For example, the rounded transition matrix below describes the CTMS mass–spring–damper sampled every 0.01 seconds. [CTMS digital controller design](https://ctms.engin.umich.edu/CTMS/index.php?example=Introduction&section=ControlDigital)

```json
{
  "mode": "control",
  "spec": {
    "mode": "state_space",
    "timebase": "discrete",
    "sample_time_s": 0.01,
    "A": [[0.9990, 0.0095], [-0.1903, 0.9039]]
  }
}
```

The model is `x[k+1] = A x[k] + B u[k]`. The optional positive finite `sample_time_s` records the period associated with the supplied transition matrix. If the available matrix describes a continuous derivative, discretize the model first or use `timebase="continuous"` to check the continuous dynamics. Omitting the timebase selects continuous time. A sampling period requires an explicit discrete timebase.

The period is supplied metadata. Scalar claims must select a computed quantity such as `schur_stable`.

## Reading the result

`schur_stable` reports whether every unforced state tends to zero. A pole exactly on the unit circle fails this strict criterion. The certificate uses the exact rational values of the supplied binary64 matrix entries, so rounding an input transition matrix can affect systems near the boundary.

- The **Stein equation** gives an energy test. The solver computes a symmetric matrix P satisfying `P − Aᵀ P A = I`, checks the full equation, and tests whether P is positive definite. If it is, the energy `xᵀ P x` decreases by `xᵀ x` at each step. `stability_certificate` gives the identity-forced solution, its definiteness and its exact residual status. [Stanford EE363 Lyapunov theory](https://web.stanford.edu/class/ee363/notes/lq-lyap-notes.pdf)
- The **Schur recursion** checks the exact characteristic polynomial independently. At each degree, the constant coefficient must have magnitude strictly below that of the leading coefficient. The solver then reduces the polynomial by one degree and repeats. `schur_recursion_stable` reports this result. A disagreement between the tests makes the verification fail.

An optional symmetric `Q` changes the right-hand side in the reported `lyapunov_P_exact` solution. The identity-forced stability check still runs. For a singular Stein operator the solution status can be `nonunique` or `inconsistent`. Inspect `schur_stable` for the stability verdict and `passed` for verification or claim acceptance. Use `claimed_field="schur_stable"` with a Boolean or numeric 0/1 for an exact stability claim.

## Published examples

| Transition matrix | Spectral radius | Strictly stable | Identity-forced solution |
| --- | ---: | --- | --- |
| CTMS mass–spring–damper, using the displayed four-decimal coefficients | 0.97273738 | Yes | Positive definite |
| SciPy discrete Lyapunov example, transposed to use Aᵀ P A | 1.15777472 | No | Unique and indefinite |

The regression test compares all four entries of each solution with an independent SciPy solve and verifies the exact matrix identity entry by entry. The SciPy example illustrates why a unique solution alone cannot certify stability. SciPy uses the equation `A X Aᵀ − X + Q = 0` for real inputs. [SciPy solver and worked example](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html)

These comparisons measure results on public matrices. The change has no measured organizer development-set or leaderboard score.
