# Binary distillation at the reflux and boilup limits

Use `hamon_conservation_cad_verifier` with mode `electrochem_separations` and subkind `distillation`. The model uses constant relative volatility, constant molar overflow in each section, a total condenser, and equilibrium stages. The feed quality `q` gives the change in internal liquid flow divided by feed flow and can lie outside [0, 1] for superheated or subcooled feed. The solvent and thermal properties needed to establish that quality must come from the caller. [McCabe and Thiele, *Graphical Design of Fractionating Columns*](https://doi.org/10.1021/ie50186a023)

With feed rate F, overhead rate D, bottoms rate B, and external reflux ratio R = L/D, the section flows are

| Reported field (mol/s) | Flow |
|---|---|
| `L_rect_mol_s` | RD |
| `V_rect_mol_s` | (R+1)D |
| `V_strip_mol_s` | (R+1)D − (1−q)F |
| `L_strip_mol_s` | V_strip + B |

`R_feed_pinch` is the signed reflux from the Underwood feed pinch. A negative value can arise when the working separation is weak. `R_flow_floor` is the minimum imposed by nonnegative circulation, max(0, (1−q)F/D−1). `R_min` gives the lower limiting reflux including any active equilibrium pinch. A genuine pinch requires infinitely many stages in this model. The solver also withholds finite-stage verification inside its relative numerical guard band at an active positive feed-pinch minimum.

The two active section lines satisfy Vy = Lx + D x_D above the feed and V′y = L′x − B x_B below it. At the flow floor the following cases have distinct boundary conditions.

| Section mode | Stage convention |
|---|---|
| `two_section` | The staircase uses both section operating lines and the feed intersection. |
| `stripping_only` | Zero reflux. The top vapour has composition x_D, the active stripping line determines subsequent ordinates, and the feed is at the first contact. |
| `rectifying_only` | Zero stripping vapour. The bottom balance fixes x=x_B. Only the rectifying line is active, and the last contact is the feed stage. The inactive reboiler contributes no equilibrium stage. |
| `single_equilibrium_flash` | Zero reflux and zero stripping vapour. Exactly one contact is supported when the prescribed flow split and both product compositions satisfy equilibrium. |

For F = 100 mol/s, z_F = 0.5, x_D = 0.6, x_B = 0.4 and α = 2, a saturated-liquid feed permits R = 0 and gives 1.216 theoretical stages. A saturated-vapour feed has minimum R = 1 and gives 101/75 theoretical stages at that limit. At R = 0.8 the latter case would require −10 mol/s of stripping vapour. The callback returns `verified: false` and the offending flows. If the two prescribed products lie on exactly the same equilibrium tie line, one ideal contact attains the Fenske bound whenever the flow and pinch checks pass. A fractional count interpolates the last graphical stage and does not specify an integer tray design.

If the user omits `R`, the solver applies `R_mult` to the physical lower limit (default factor 1.25). A zero lower limit therefore tests the zero-reflux boundary itself. Where that boundary is a pinch or an incompatible no-circulation flash, choose an explicit positive R to test a different operating point. The enforced flow signs, endpoint balances and flash identity use the round-trip decimal representation of the parsed finite inputs.
