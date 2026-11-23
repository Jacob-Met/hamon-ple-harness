# Passive RC precharge endpoint

The registered `rc_precharge` route models an initially discharged capacitor, a constant signed DC bus voltage and a positive series resistor. Capacitance is supplied in microfarads. The target fraction is the requested capacitor voltage divided by the nonzero bus voltage, between zero and one inclusive.

For a positive time constant τ=RC and nonzero bus, fractions strictly inside the interval are reached at t=−τ log(1−f). The zero target is initially reached. A target of exactly one is an asymptotic limit with `t_target_s=null`. If the bus voltage itself is zero, the requested physical voltage is already present at time zero.

`stored_energy_J` retains the full-bus capacitor energy. The reached state separately reports `target_capacitor_energy_J`, capacitor charge and voltage, remaining resistor voltage and signed current. Source work and resistor heat are cumulative to the reported endpoint. They satisfy source work = capacitor energy + resistor heat. These quantities, the voltage loop and both the reached and remaining voltage fractions are replayed using the reported finite values. A nonzero quantity outside the supported binary64 output range cannot receive a finite endpoint certificate.
