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
