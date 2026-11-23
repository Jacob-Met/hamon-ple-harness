# Chemical stream balances

Each inlet or outlet record describes one transported formula unit. Use `mdot_kg_s` for a mass-flow record, or `formula` and `ndot_mol_s` for an elemental molar balance. Supplying all three fields also checks the formula mass of each stream. Enter rates as nonnegative magnitudes on the appropriate side.

For joint records the default molar mass uses the sum of the constituent [CIAAW 2024 abridged standard atomic weight intervals](https://ciaaw.org/abridged-atomic-weights.htm), in grams per mole of the whole formula. Grouped subscripts and hydrate coefficients contribute to that same formula unit. These intervals cover normal terrestrial materials. The stream passes when its measured mass rate agrees with at least one value in the resulting interval at the requested relative tolerance (default 0.0001). Zero mass and molar rates must coincide.

An optional positive finite `molar_mass_g_mol` supplies the whole-formula molar mass of a particular sample, including one with specified isotope composition. For example, a water sample with a declared molar mass of 20.0276 g/mol at 2 mol/s carries 0.0400552 kg/s. This check treats the declared value as an input property. It does not determine isotope abundance from the bulk flow. The response identifies records using this property in `declared_molar_mass_streams`. An element without a standard atomic weight requires this property for a joint record.

`mass_balance_ok` checks total inlet and outlet mass and `species_balance_ok` checks molar flow of each element. `stream_mass_mole_ok` checks that each joint record is internally consistent and must also pass for `conservation_verified`. Stream errors identify the inlet or outlet and its zero-based position.
