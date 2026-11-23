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

For F = 100 mol/s, z_F = 0.5, x_D = 0.6, x_B = 0.4 and α = 2, a saturated-liquid feed permits R = 0 and gives 1.216 theoretical stages. A saturated-vapour feed has minimum R = 1 and gives 101/75 theoretical stages at that limit. At R = 0.8 the latter case would require −10 mol/s of stripping vapour. The callback returns `verified: false` and the offending flows. If the two prescribed products lie on exactly the same equilibrium tie line, one ideal contact attains the Fenske bound whenever the flow and pinch checks pass. The whole equilibrium-contact count must satisfy the Fenske product-odds bound. The reported fractional staircase count interpolates the final step linearly in liquid composition, whereas the continuous Fenske expression interpolates in log odds. For x_D=0.9, z_F=0.8, x_B=0.7, α=2, q=1 and R=1000, the requested bottoms falls between contacts one and two. Liquid-composition interpolation gives 1.939763 and Fenske gives 1.947533. Two complete contacts satisfy the bound.

If the user omits `R`, the solver applies `R_mult` to the physical lower limit (default factor 1.25). A zero lower limit therefore tests the zero-reflux boundary itself. Where that boundary is a pinch or an incompatible no-circulation flash, choose an explicit positive R to test a different operating point. The enforced flow signs, endpoint balances and flash identity use the round-trip decimal representation of the parsed finite inputs.

## Using an equilibrium table

Supply `equilibrium_points` as ordered `[liquid mole fraction, vapour mole fraction]` pairs for the same component at the column pressure. The table defines a piecewise-linear equilibrium model and takes precedence over `alpha`. Supply liquid compositions in strictly increasing order. Each fraction must be finite and between zero and one. The table must cover the requested product interval and every horizontal contact needed for the stage calculation. The solver uses the supplied points without extrapolation. Its reflux and stage results apply to that interpolated curve. With `alpha` alone the constant-relative-volatility model above still applies.

The whole-interval certificate requires equilibrium above the diagonal on the closed product interval. It tests the equilibrium-to-operating-line gap on every interval formed by the supplied liquid compositions, the product compositions and the operating-line intersection. With this interpolation, the minimum on each interval lies at an endpoint. The limiting reflux calculation also tests every equilibrium intersection with the feed line, including intersections between supplied points. `limiting_pinch_x` gives one maximizer when a pinch raises the limit above the nonnegative-flow floor. At the requested reflux, `minimum_operating_gap` and `minimum_gap_x` identify the worst actual gap. Touching the operating line fails finite-column qualification.

The stage calculation solves each inverse contact on the supplied curve. A nonmonotone curve can qualify when every visited vapour composition has a unique liquid contact. Multiple contacts or a coincident horizontal segment give `ambiguous_equilibrium_inverse`. A diagonal crossing, a missing contact, or an unresolved staircase returns `verified: false`. The exact staircase budgets 2,048 contacts and 32,768 bits per intermediate composition. Table results omit constant-volatility Fenske and Underwood quantities.

For example, with `x_B=0.1, z_F=0.5, x_D=0.9, q=1`, the points `[[0,0],[0.1,0.2],[0.3,0.4],[0.5,0.75],[0.7,0.78],[0.9,0.95],[1,1]]` give a minimum reflux of 1.5. The rectifying line touches equilibrium at liquid composition 0.7. At reflux 2 the same table gives 16 whole stages with the feed on stage 7 and a fractional count of 15.32477.
