# Passive heat-exchanger rating and sizing

The registered `heat_exchanger_ntu` route rates a constant-capacity exchanger from a nonnegative conductance `UA_W_K`. The `heat_exchanger_lmtd` route sizes area from the measured hot-stream outlet temperature and positive `U_W_m2K`. Both routes require positive stream mass rates, specific heat capacities and absolute temperatures.

Sizing accepts `arrangement="counterflow"` (default) and `arrangement="parallel"`. For counterflow, each terminal pairs the hot-stream inlet with the cold-stream outlet, or the hot-stream outlet with the cold-stream inlet. A cold outlet above the hot outlet can therefore be feasible. For parallel flow the outlets are at the same terminal and cannot cross. A measured pinch does not identify a finite positive area. Rating also retains the `shell_and_tube_1_2` arrangement.

Heat duty is signed from the stream labeled hot to the stream labeled cold. Zero UA gives zero duty. Equal inlet temperatures give zero duty for every UA: rating returns this state, while sizing reports that area is unidentifiable. With distinct inlets and zero transfer, the inferred area is zero.

The calculations check signed duty against a separate effectiveness–NTU evaluation and require nonnegative stream entropy production. Output `hot_temperature_drop_K` and `cold_temperature_rise_K` preserve small changes that a rounded absolute temperature can lose. The `energy_balance_residual_W` field uses the displayed absolute cold outlet. The `temperature_change_balance_residual_W` field replays both streams from their signed changes and, for sizing, checks the displayed area and logarithmic mean temperature difference.
