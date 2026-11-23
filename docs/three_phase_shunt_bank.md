# Three-phase power-factor correction

The registered `pf_correction` calculation sizes a balanced ideal shunt at a fixed sinusoidal terminal voltage and frequency. Supply positive total load power `P_kW`, line-line RMS voltage `V_ll_V`, `freq_Hz`, and the initial and target power-factor magnitudes in (0, 1]. A zero power factor cannot carry positive real power at finite current. This sizing model assumes the load's real power stays fixed.

Specify `initial_pf_direction` and `target_pf_direction` as `lagging` or `leading`. Both default to lagging. Unity magnitude has zero reactive demand independently of the direction setting. `bank_connection` is `delta` (default) or `wye`. Each delta leg sees line-line voltage and each wye leg sees line-line voltage divided by the square root of three.

| Result | Interpretation |
| --- | --- |
| `Q_initial_kvar`, `Q_target_kvar` | Signed total reactive demand, positive for a lagging load and negative for a leading load |
| `Q_shunt_kvar` | Target minus initial demand. Positive means a reactor absorbs VARs; negative means a capacitor supplies VARs. |
| `Qc_required_kvar` | The same adjustment in the original capacitive-supply-positive convention |
| `bank_type` | `capacitor`, `reactor` or `none` |
| `bank_capacitance_per_leg_F`, `bank_inductance_per_leg_H` | Positive physical component, with null for an unneeded device |
| `C_equiv_delta_per_phase_F` | Equivalent delta capacitance for a capacitor bank, zero for no bank, null for a reactor |

Sizing follows the signed complex-power balance for the three legs. The solver independently reconstructs the reactive power from the published component and its leg current, and reports that result as `Q_shunt_replay_kvar`. Its relative error is measured against the bank's own reactive power, including small adjustments between nearly equal power factors. The reported component must also attain the target when reconnected to the original load. The tolerance is 1e-10 relative to the magnitude of nonzero target reactive power. At unity power factor the tolerance applies to residual reactive power relative to the positive real load. A component whose binary64 value misses that target is refused. `Q_net_replay_kvar` reports the net reactive power obtained from the actual reported component. The calculation assumes ideal components and one fundamental frequency. Harmonic, protection, switching and discrete commercial component selection require additional system data.

[ABB, *Power factor correction and harmonic filtering in electrical plants*, section 7](https://library.e.abb.com/public/4ffab4cf79db502fc12574ef004deed3/1SDC007107G0201.pdf) · [Hitachi Energy, *Shunt reactors*](https://www.hitachienergy.com/us/en/products-and-solutions/transformers/reactors-and-inductors/shunt-reactors)
