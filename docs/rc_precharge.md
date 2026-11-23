# Passive RC precharge endpoint

The registered `rc_precharge` route models a capacitor connected through a positive series resistance to a constant signed DC bus. Supply `V_bus_V`, `R_precharge_ohm`, and `C_uF`. The optional signed `V_initial_V` defaults to 0 V. Specify exactly one target. `target_fraction` is a fraction in [0,1] of the bus voltage as before. `V_target_V` accepts a signed voltage, including targets on a discharge toward a zero-volt bus.

The capacitor starts at its initial voltage and moves monotonically toward the bus. `target_time_status` reports the physical outcome.

| Status | Time and endpoint |
| --- | --- |
| `already_at_target` | The initial voltage equals the target. Time, source work, energy change and heat are zero. |
| `finite` | The target is strictly between the initial voltage and the bus. The solution reports the first crossing time and its signed endpoint quantities. |
| `asymptotic` | The capacitor approaches a different bus voltage that is also the target. Time is null. The endpoint reports the limiting state and integrated work and heat. |
| `unreachable` | The target is outside the trajectory, including any different target at equilibrium. Time, reached endpoint and integrated work and heat are null. The requested voltage is retained and `dual_verified` is false. |

For example, charging from 9 V toward 12 V cannot reach a 6 V target. A 100 µF capacitor discharged from 12 V toward 0 V through 10 Ω reaches 6 V after approximately 0.693147 ms. The bus contributes zero work and the resistor dissipates 5.4 mJ drawn from the capacitor.

`stored_energy_J` retains the full-bus capacity ½ C Vbus². `initial_capacitor_energy_J` and `target_capacitor_energy_J` describe the initial and reached states. `capacitor_energy_change_J` is their signed difference. Current is positive from the bus into the capacitor. Source work is signed and equals stored-energy change plus nonnegative resistor heat. The verifier replays the transient voltage coordinates and the integrals of source power and resistor heating. A nonzero quantity outside the supported binary64 output range cannot receive a finite endpoint certificate.
