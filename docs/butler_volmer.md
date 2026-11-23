# Current-driven activation in the electrochemistry verifier

Select the registered conservation tool with outer mode `electrochem_separations` and `spec.subkind="electrochemistry"`. To compute activation for one effective charge-transfer step, provide `exchange_current_A` and the **anodic** `transfer_coefficient` together. Both input currents are total currents in amperes for the same electrode. The exchange current must be positive and the coefficient must lie strictly between zero and one. The effective transfer count `kinetic_n_electrons` defaults to `n_electrons`; set it explicitly if the written overall reaction and the charge-transfer step use different electron counts.

For thermal voltage `vT = RT/(nF)`, the signed current satisfies

```text
I / I0 = exp(alpha * eta / vT) - exp(-(1 - alpha) * eta / vT)
```

The existing `current_A` field is a nonnegative magnitude. In `spec.mode="galvanic"` this loss model applies cathodic polarization to the equilibrium terminal voltage, with negative signed kinetics current and overpotential. In `spec.mode="electrolytic"` it applies anodic polarization with positive signs. The ohmic term uses the same loss direction. Zero current gives exactly zero kinetic overpotential in both modes. Use the existing measured `eta_activation_V` input for a known combined polarization loss. The verifier rejects conflicting measured and kinetic inputs.

For an equilibrium voltage of 1.1 V, transfer count 1, temperature 298.15 K, 1 A current, 0.1 A exchange current, alpha 0.5 and zero internal resistance, the activation magnitude is 0.118825010093 V. The galvanic terminal voltage is 0.981174989907 V and the electrolytic terminal voltage is 1.218825010093 V. `kinetic_current_relative_residual` reports the relative error when the returned overpotential is substituted back into the current law. The result certifies the supplied one-step model and its unit convention.
