# Rankine turbine state and cycle balance

The registered `hamon_conservation_cad_verifier` accepts `mode="thermo_cycles"` and a Rankine `spec`. Pressures are in Pa, enthalpies in J/kg, entropies in J/(kg·K), specific volume in m³/kg and reservoir temperatures in K. The solver uses the supplied state properties. It contains no water or refrigerant property lookup.

## First turbine outlet

`h4s` is the first turbine's **isentropic** outlet enthalpy. The actual outlet enthalpy, returned in `first_turbine_outlet_enthalpy_J_kg`, is `h4 = h3 − eta_t × (h3 − h4s)`. With reheat these are intermediate turbine states. The optional `h_f, h_fg` pair is the saturated-liquid enthalpy and positive latent enthalpy **at that outlet pressure**. The analogous entropy pair is `s_f, s_fg`. All supplied saturation properties must describe the same fluid, pressure and reference convention. The callback can check consistency among the supplied scalars, but their provenance and pressure assignment must be established by the caller.

- With direct `h4s`, a valid cycle can be calculated even when the first outlet's quality is unknown. The corresponding phase label is `unknown` and `x4s_quality` is null.
- A complete saturation enthalpy pair identifies isentropic and actual quality for states on or between the two endpoints. A state beyond an endpoint has a directional phase label and null quality. The same-fluid coexisting liquid–vapor mass fraction is defined only from zero through one.
- With inlet entropy `s3` and a complete entropy pair, the isentropic state can be classified from entropy. In the absence of direct `h4s`, both property pairs are required and only a saturated or mixed state supports the enthalpy interpolation. Single-phase extrapolation through the mixture formula returns a structured refusal. If both pairs independently give a quality, they must agree within relative tolerance 1e−8 and absolute tolerance 1e−10.
- The separate fields `x4_quality` and `first_turbine_outlet_phase` refer to the **actual** first outlet. Without a saturation enthalpy pair, they carry no quality unless unit turbine efficiency equates the two states and the isentropic state is known.
- An optional `x4s` declaration is recorded as `declared_x4s_quality`. Its checked flag states whether a co-located bracket supports comparison. A declaration alone cannot supply a calculated phase. Out-of-range or contradictory declarations are refused.

The existing 10 kPa water fixture supplies `h4s = 2100.0` kJ/kg. At this pressure NIST lists saturated enthalpies of 191.81 and 2583.9 kJ/kg. Adding that bracket gives an isentropic mass quality of 0.7977083. With the fixture’s turbine efficiency of 0.88 the actual exhaust is 2249.808 kJ/kg and has quality 0.8603347. The cycle efficiency is 0.3460633 with or without the optional phase evidence. These comparisons use the published rounded table entries. [NISTIR 5078, Table 3](https://www.nist.gov/system/files/documents/srd/NISTIR5078-Tab3.pdf) · [IAPWS region boundaries](https://iapws.org/public/documents/Hb6vJ/Supp-phs3-2014.pdf)

## Reheat and verification

A reheat calculation requires both `h5` and `h6s`. Each turbine efficiency is in (0, 1]. Under this heat-engine model each turbine produces nonnegative work, and the boiler and reheater supply nonnegative heat. Net work, total supplied heat and condenser heat rejection must be positive. A reservoir-temperature Carnot check requires both `T_cond` and `T_boiler_max`; `carnot_checked` records its use.

The implementation accumulates specific work and heat using rational arithmetic from the round-trip decimal representations of finite parsed input values. It checks the first-law residual on the returned floating-point numbers. A verified cycle establishes these balance and domain checks under the supplied properties. Independent fluid state measurements or an equation of state are needed to validate the inputs themselves.
