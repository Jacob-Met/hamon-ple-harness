# Electrochemical potential and Faradaic yield

Use the registered `hamon_conservation_cad_verifier` with mode `electrochem_separations` and subkind `electrochemistry`. The solver calculates a Nernst potential for a specified reaction and a constant-current yield for one selected product. It assumes constant temperature, reaction quotient and current efficiency over the supplied duration.

## Reaction basis

Supply `E0_V`, the positive electron count `n_electrons` per written reaction, and the positive coefficient `product_stoich` of the selected product. The coefficient defaults to one. `Q_reaction` is the dimensionless activity quotient for the same written reaction; `T_K` is the absolute temperature. Multiplying the reaction by a positive factor multiplies both coefficients by that factor and raises the quotient to that power. The potential and species yield stay invariant. Reaction extent scales inversely, and the Gibbs energy per mole of written reaction scales with the factor.

For constant `current_A` and `time_s`, delivered charge is their product. `faradaic_efficiency` is the fraction of that charge assigned to the specified reaction, from zero through one. The reaction extent in moles is assigned charge divided by the electron count and the Faraday constant. Multiplying the extent by the product coefficient gives `product_moles`. `moles_reacted` remains an alias of `reaction_extent_mol`. A positive `molar_mass_g_mol` gives `mass_deposited_g`. With missing molar mass, positive production has unknown mass (JSON null); zero production has zero mass.

For example, 0.5 A over 7,200 s delivers 3,600 C. Normalizing silver deposition to one silver atom and one electron, or to three silver atoms and three electrons, gives 4.0247104 g in either case at unit efficiency and 107.8682 g/mol.

## Optional inverse

Supply `target_mass_g` to calculate the time required at the same constant current and efficiency. The forward `time_s` still controls the reported forward yield. Positive rate and a known molar mass yield a unique time. A positive target at zero rate is unreachable. At zero rate a zero target is satisfied at every nonnegative time and `target_time_s` is null. At positive rate a zero target has unique time zero. If the rate is positive and a positive target is specified without molar mass, the time is undetermined. An unresolved or unreachable positive target returns `verified: false` together with the valid forward quantities and `target_mass_status`.

The `galvanic` and `electrolytic` modes retain the existing voltage convention, subtracting or adding the specified nonnegative activation and ohmic losses to the Nernst potential, respectively. Inputs provide the reaction and its direction. The calculation does not infer product identity, chemical activity, side reactions or reaction selectivity.

IUPAC defines species yield in terms of reaction current efficiency, species stoichiometry and reaction electron number. Its Nernst relation uses the activity quotient on that same reaction basis. [Current efficiency](https://goldbook.iupac.org/terms/view/C01458) · [Nernst equation](https://goldbook.iupac.org/terms/view/09068)

## Electrical operating boundary

`external_circuit` defaults to `prescribed_current`, preserving the fixed-current interpretation. An external source may deliver or absorb the required terminal power. Set `external_circuit="passive_load"` when the external element may absorb electrical energy or be short-circuited but cannot provide power. At a prescribed point requiring external power, the passive request returns the signed electrical diagnostics, `operating_point_feasible=false` and `verified=false`. It omits unattained forward yields. A positive product target has no attainable time at that point; a zero target is already met at time zero.

Voltage retains the reference polarity of the Nernst potential E. Define signed outward terminal current j as +I for the galvanic convention and −I for the electrolytic convention. With the supplied nonnegative activation loss η and internal resistance R, the existing terminal voltages are E−η−IR and E+η+IR respectively. The power check is

    j E = j V_terminal + I² R + I η.

Positive `terminal_power_out_W` is electrical delivery to the external circuit. Negative power requires external supply. Zero current is the open-circuit case; a nonzero current with zero terminal voltage is a short circuit. Classification comes from the stated reference and computed power, even when the terminal voltage difference cannot be recovered by subtracting the separately rounded reported total voltages. At zero current the irreversible powers vanish.

`reversible_port_power_out_W` is the reversible source power of this lumped electrical model. `assigned_reference_extent_rate_mol_s` is the signed rate referred to the Nernst reaction orientation. Multiplying by −ΔG gives `assigned_reaction_reversible_power_out_W` for the Faradaic current allocated to that reaction. Forward production and time-to-target fields keep their existing nonnegative yield convention for the selected product. With current efficiency below one, `unassigned_current_magnitude_A` reports the remainder. Its reactions and chemical energies are unspecified. [IUPAC galvanic cell](https://goldbook.iupac.org/terms/view/09066) · [IUPAC electrolytic cell](https://goldbook.iupac.org/terms/view/09062)
