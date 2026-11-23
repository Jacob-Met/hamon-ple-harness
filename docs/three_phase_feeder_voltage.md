# Balanced three-phase feeder voltage

Use the registered `three_phase_vdrop` calculation for a balanced, steady-state short feeder with per-phase series impedance. `V_ll_V` is the known terminal voltage, `I_A` is the line current magnitude, and `pf` is its power factor at the receiving terminal. `length_m` and the per-kilometre resistance and reactance give the series impedance of one phase. Resistance is nonnegative. Signed reactance supports inductive or capacitive series elements.

| Setting | Meaning |
| --- | --- |
| `voltage_reference` | `sending` (default) or `receiving`, identifying the terminal at which `V_ll_V` was supplied |
| `power_factor_direction` | `lagging` (default) or `leading`, with `pf` between zero and one |
| `receiving_voltage_branch` | For known sending voltage, `high` (default) selects the branch connected to the no-load state. `low` requests a distinct positive alternate solution. |
| `max_vdrop_pct` | Maximum signed drop as a percentage of the supplied reference magnitude. Defaults to 3. A voltage rise gives a negative drop. |

With receiving voltage on the real axis and balanced line-line voltage expressed in the corresponding rotated reference frame, the terminal equation is `Vs = Vr + sqrt(3) I Z`. A known receiving magnitude gives one sending phasor. For known sending magnitude, the allowable receiving voltages follow from the intersections of the phasor line with the circle of that magnitude. Only strictly positive receiving voltages define the receiving power-factor reference. The reported list is in increasing order. No physical root gives `feasible=false`. A request for an unavailable distinct low branch is refused. The labels identify mathematical branches and convey no dynamic-stability assessment.

`V_sending_phasor_ll_V` holds the rectangular sending phasor in the receiving-voltage basis. `vdrop_ll_V` and `vdrop_pct` use the signed magnitude difference, sending minus receiving. The former approximate projection is also available as `longitudinal_projection_ll_V`. The reported complex terminals are checked against the original series equation. Small physical voltage changes are computed separately so that a limit check can resolve a change smaller than one terminal-voltage rounding increment. Each nonzero reported electrical quantity must also be representable within a relative error of 1e-10. The percentage criterion is a user-specified one-sided limit.

[Eaton, *Consulting Application Guide*, Power Distribution Systems, Voltage Drop Formulas](https://www.eaton.com/content/dam/eaton/products/design-guides---consultant-audience/canada/cag/eaton-power-distribution-systems-consulting-application-guide-tb08104003e-tab-1-ca08104001e-ca.pdf)
