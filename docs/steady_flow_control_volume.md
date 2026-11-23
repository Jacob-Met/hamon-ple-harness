# Steady-flow control-volume verification

Select `calculation: "steady_flow_control_volume"` in the registered thermal/fluid verifier. Supply a steady, open control volume with known SI stream properties, reservoir heat rates and shaft work. The result evaluates steady mass, energy and entropy balances. Its feasibility flag concerns these three balances conditional on the supplied properties. The caller supplies compatible thermodynamic states and a common enthalpy and entropy reference, including any required chemical or phase contributions.

| Input | Physical meaning |
| --- | --- |
| `property_reference` | Nonempty identifier for the common material-property reference. Each stream inherits it. Any explicit stream reference must match. |
| `inlets`, `outlets` | Nonempty lists. Each stream supplies nonnegative `mdot_kg_s`, signed `h_J_kg` and signed `s_J_kgK`. Each side requires positive total throughput. |
| `speed_m_s`, `elevation_m` | Optional stream speed magnitude and elevation above a shared datum, both defaulting to zero. |
| `gravity_m_s2` | Positive uniform gravitational acceleration, default 9.80665. |
| `heat_reservoirs` | Required list. Each entry gives signed `heat_into_W` and positive absolute `temperature_K` at its heat-transfer boundary. Use an empty list for adiabatic operation. |
| `shaft_work_out_W` | Required signed shaft power, positive out of the volume. Flow work is already included in enthalpy. |
| `entropy_generation_W_K` | Optional measured entropy generation rate. It must be nonnegative and agree with the calculated generation. |

For any stream property (a), its flux change is the sum of outgoing mass flow times (a) minus the corresponding incoming sum. The energy residual is net heat in minus shaft work out minus the flux changes of enthalpy, kinetic energy (v²/2) and potential energy (gz). The calculated entropy generation is the outgoing-minus-incoming specific entropy flux minus the sum of each boundary heat rate divided by its own temperature.

Integer inputs are exact. Decimal JSON measurements processed as floats use their shortest round-trip decimal numerals. On that nominal basis mass flow must close exactly, so adding shared constants to enthalpy, entropy or elevation leaves the balances of a mass-conserving volume unchanged. The first law allows an absolute residual up to 1e-10 times the largest magnitude among net heat, shaft work and the three **changes** in energy flux. Computed entropy generation must be nonnegative regardless of its magnitude. A supplied generation rate must agree within 1e-10 of the magnitude of the computed rate, including exact agreement at zero. Returned numeric values are independently rounded to binary64 and must retain relative accuracy of 1e-12.

`mass_balance_ok`, `first_law_ok` and `second_law_ok` expose the individual arithmetic checks. A supplied generation rate additionally has an `entropy_generation_observation_ok` flag. `conservation_verified` and `thermodynamically_feasible` require every applicable check. With a mass imbalance, the raw energy and entropy differences remain reference-dependent; the combined flags stay false. Bare mass/species inventories retain their existing route.

The sign conventions follow the [MIT steady-flow energy equation](https://ocw.mit.edu/ans7870/16/16.unified/thermoF03/chapter_6.htm) and [Cambridge thermofluids control-volume entropy balance](https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/THERMOFLUIDS_DATA_v_20.pdf).
