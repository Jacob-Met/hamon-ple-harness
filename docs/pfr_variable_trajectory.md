# Variable-rate plug-flow trajectories

The registered conservation verifier accepts `mode=reactor_kinetics` and `spec.reactor_type=pfr`. Its existing power-law reaction, constant-pressure expansion and prescribed adiabatic temperature line now retain both conversion and remaining reactant for variable-rate trajectories.

## Representation

The former general path integrated directly in conversion and inverted volume with an absolute conversion tolerance. For a first-order gas with unit rate, feed and inlet concentration, 300 K inlet temperature and a 300 K adiabatic rise, the exact model is `V=-X-2*log(1-X)`. At volume 120 the former solver refused the outlet. Its remaining fraction is `5.311092249679095e-27`, although binary64 conversion rounds to one. A separate heated Arrhenius case at volume `1e-26` returned zero conversion with a failed balance.

The general path now integrates in `t=-log(U)`, with `X=-expm1(-t)` and `U=exp(-t)` evaluated separately. Temperature and expansion use the smaller complementary coordinate. Retain both public fields, `conversion_X` and `unreacted_fraction`: a rounded conversion of one does not prove complete depletion.

## Governing balance and algorithm

The model remains `dV/dX=F_A0/(-r_A)` with `-r_A=k(T)*C_A**n`. The University of Michigan's [mole balance](https://websites.umich.edu/~elements/7e/02chap/summary.html), [gas-phase design lecture](https://websites.umich.edu/~elements/7e/powerpoints/2013lectures/Lec6_PDF.pdf) and [adiabatic reactor chapter](https://websites.umich.edu/~elements/7e/11chap/summary.html) describe the underlying relationships. The coordinate transformation is derived from them here.

The transformed volume derivative is positive inside the physical interval. A bracketed Newton solve uses a relative volume balance. Reference-temperature Arrhenius evaluation forms the temperature difference before division. Constant-rate first-order expansion uses analytic moments, with a series near zero conversion. The existing constant-coefficient closed form is retained.

For order below one with a regular endpoint, `z=U**(1-n)` removes the integrable depletion singularity. Positive-order kinetics may stay depleted downstream; zero and negative orders refuse volume beyond depletion. Nonpositive temperature or bulk volume, unrepresentable outputs and unresolved inverses are refused.

The result replays volume from the returned coordinates and checks rounding of temperature, rate and residence time plus quadrature error. This verifies numerical consistency of this idealized steady model. It does not introduce pressure drop, reversible reactions, heat-transfer dynamics or a stability classification.

## Checks and scope

Run `python -m unittest discover -s tests -p 'test_pfr*.py' -v` with the repository dependencies. New checks use an independent Lambert-W solution for heated gas, a forward DOP853 initial-value solution for Arrhenius cases, finite fractional-order depletion, invalid continuations and the registered JSON CLI. Existing power-law and broader wave-6 tests cover the unchanged routes.

These are mathematical engineering controls, not experimental observations or PLE benchmark scores. No organizer dev-set run or portal submission was performed. Existing dev-set artifacts were not edited.
