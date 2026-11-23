# Binary distillation from an equilibrium table

Use the registered conservation verifier in `electrochem_separations` mode with `subkind: "distillation"`. Supply liquid–vapor mole-fraction pairs for one pressure, from pure component to pure component. The model uses straight segments between supplied points. Both coordinates must strictly increase so each vapor fraction has a unique liquid equilibrium partner. Use either the table or constant relative volatility `alpha`.

```json
{"mode":"electrochem_separations","spec":{"subkind":"distillation","x_B":0.05,"z_F":0.5,"x_D":0.95,"q":0.5,"R":12,"equilibrium_table":[[0,0],[0.25,0.46],[0.5,0.63],[0.7,0.72],[0.85,0.94],[1,1]]}}
```

This example needs 23 whole equilibrium contacts, including the partial reboiler and excluding the total condenser. The interpolation of the final contact gives 22.6200 theoretical stages. The minimum reflux ratio is 11.5. At exactly that reflux the rectifying line touches the table at liquid composition 0.7, so the verifier returns the pinch location and omits a stage count.

## Minimum reflux and attainment

For constant molar overflow with positive vapor flows, the active operating line at each liquid composition is the lower of its rectifying and stripping lines. Let $y$ denote the tabulated equilibrium ordinate, $F$ the feed, $D$ the distillate and $B$ the bottoms. When $y>x$, the reflux threshold at $x$ is the lesser of

$$\frac{x_D-x}{y-x}-1,\qquad \frac{B}{D}\frac{x-x_B}{y-x}-1+(1-q)\frac FD.$$

Each expression is monotone or constant along a table segment. The two are equal where $z_F=qx+(1-q)y$. The global threshold is therefore the largest of the internal-flow floor and the thresholds at both product endpoints, all interior table knots and **every** intersection with the feed-quality line. A coincident segment needs only its endpoints. The solver separately tests attainment at the requested reflux, including the single-flash and zero-circulation section boundaries. A point with $y\le x$ in the product interval blocks the separation even at total reflux.

All balances, inversions and stage-boundary decisions use rational arithmetic on the submitted numbers’ round-trip decimal representations. The reported residual of zero refers to those model balances. Displayed numbers are rounded to binary64. A finite, near-pinched column can exceed the evaluation limit of 4,096 contacts or 32,768 bits in a stage coordinate. The solver reports the finite-budget condition and withholds the count. The result qualifies the supplied interpolation and constant-flow model. Measurements between the provided knots, tray efficiencies and pressure variation require additional evidence.
