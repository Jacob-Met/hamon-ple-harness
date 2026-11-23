# Multicomponent flash with prescribed equilibrium ratios

Use `hamon_conservation_cad_verifier` in mode `electrochem_separations` with `subkind="equilibrium_flash"`. Supply two arrays in the same species order. `feed_mol_s` contains nonnegative component mole rates with positive total flow. `K_values` contains a strictly positive, finite vapor-to-liquid mole-fraction ratio for each component at the specified operating condition. The material calculation uses those supplied ratios. Temperature, pressure, energy duties and an equation of state belong to the external model that determines them.

For example, `feed_mol_s=[3,2,3]` and `K_values=[2,1,0.5]` give liquid and vapor rates of 4 mol/s each. The liquid mole fractions are `[0.25,0.25,0.5]` and vapor mole fractions are `[0.5,0.25,0.25]`.

Write the vapor fraction as β and the feed mole fractions as z. The liquid and vapor compositions in a two-phase state satisfy xᵢ = zᵢ/[1+β(Kᵢ−1)] and yᵢ = Kᵢxᵢ. The Rachford–Rice equation sets the sum of zᵢ(Kᵢ−1)/[1+β(Kᵢ−1)] to zero. That sum is strictly decreasing whenever a present component has Kᵢ≠1. Evaluating it at both bounds distinguishes a unique interior split, a bubble or dew point, and a single-phase state. [Rachford and Rice, 1952](https://doi.org/10.2118/952327-G)

| Phase state | Returned composition and split |
|---|---|
| `two_phase` | Both phases have positive flow. Each phase has normalized mole fractions and a separate per-component inventory in mol/s. |
| `liquid` or `vapor` | All feed enters the named phase. The absent phase has zero flow and null mole fractions. |
| `bubble_point` or `dew_point` | The incipient phase has zero flow and a normalized equilibrium composition. |
| `indeterminate` | Every present species has K=1. Both phase compositions equal the feed and any vapor fraction in [0,1] satisfies the balances. Individual phase flows and fractions are null. |

Rates and ratios use their submitted binary64 values. Zero-feed species carry zero inventory and have no effect on classification. Vapor and liquid fractions, totals, compositions and component inventories are rounded independently from the common state. Near an endpoint the larger fraction can round to 1 while its complementary phase still carries measurable flow. The calculation reports an error if a positive output falls outside its finite, nonzero reporting range.
