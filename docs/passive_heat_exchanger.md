# Passive heat-exchanger rating and sizing

The registered `heat_exchanger_ntu` route rates a constant-capacity exchanger from a nonnegative conductance `UA_W_K`. The `heat_exchanger_lmtd` route sizes area from the measured hot-stream outlet temperature and positive `U_W_m2K`. Both routes require positive stream mass rates, specific heat capacities and absolute temperatures.

Sizing accepts `arrangement="counterflow"` (default) and `arrangement="parallel"`. For counterflow, each terminal pairs the hot-stream inlet with the cold-stream outlet, or the hot-stream outlet with the cold-stream inlet. A cold outlet above the hot outlet can therefore be feasible. For parallel flow the outlets are at the same terminal and cannot cross. A measured pinch does not identify a finite positive area. Rating also retains the `shell_and_tube_1_2` arrangement.

Heat duty is signed from the stream labeled hot to the stream labeled cold. Zero UA gives zero duty. Equal inlet temperatures give zero duty for every UA: rating returns this state, while sizing reports that area is unidentifiable. With distinct inlets and zero transfer, the inferred area is zero.

The calculations check signed duty against a separate effectiveness–NTU evaluation and require nonnegative stream entropy production. Output `hot_temperature_drop_K` and `cold_temperature_rise_K` preserve small changes that a rounded absolute temperature can lose. The `energy_balance_residual_W` field uses the displayed absolute cold outlet. The `temperature_change_balance_residual_W` field replays both streams from their signed changes and, for sizing, checks the displayed area and logarithmic mean temperature difference.


## Inside and outside fouling

A `fouling` object supplies the clean overall coefficient and both surface-specific fouling resistances. Its `area_basis` is `"inside"` or `"outside"`. Supply `U_clean_W_m2K` on that basis, `outside_to_inside_area_ratio` as the positive ratio of outside to inside area, and the nonnegative `R_inside_m2K_W` and `R_outside_m2K_W` on their respective surfaces. A zero resistance specifies a clean surface. In rating, also provide `area_m2` on the named basis. In sizing, the returned `area_m2` uses that basis. The existing effective-coefficient inputs `UA_W_K` and `U_W_m2K` are mutually exclusive with this model.

The complete-device resistance in K/W is
$$
  (UA)^{-1}=(U_{\rm clean}A_{\rm ref})^{-1}
  +R''_{\rm inside}/A_{\rm inside}+R''_{\rm outside}/A_{\rm outside}.
$$
The reported `U_effective_W_m2K` and `fouling_on_basis_m2K_W` use the named area. Changing that reference consistently preserves the device conductance and duty. This constant-capacity model takes fouling factors as inputs. [Morrison, *Heat Transfer*, Michigan Technological University](https://pages.mtu.edu/~fmorriso/cm310/heat_lecture_14.pdf)

For example, take 4.5 m² inside, 9 m² outside, a clean outside coefficient of 256 W/(m²·K), and inside/outside fouling factors of 1/512 and 1/1024 m²·K/W. The three device resistances sum to 1/1024 K/W. The clean inside coefficient for the same device is 512 W/(m²·K), yielding the same 1,024 W/K conductance.
