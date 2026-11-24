# Normal shock set by nozzle back pressure

Call `hamon_conservation_cad_verifier` with the following dimensionless request.

```json
{"mode":"compressible_acoustics","spec":{"subkind":"cd_nozzle_shock","gamma":1.4,"exit_to_throat_area":1.5,"back_to_total_pressure":0.75}}
```

This solves steady one-dimensional flow of a calorically perfect gas in a converging-diverging nozzle with a monotone divergent section. A normal shock separates two isentropic regions. The area ratio at the shock locates the jump on any supplied area profile. All static and total pressure outputs use the inlet reservoir total pressure as their common reference. The model includes neither duct friction nor heat transfer. Boundary and shock-free cases return an unverified result for this internal-shock calculation.

The solver conserves choked mass through the exit and inverts the total-pressure loss for upstream Mach. Qualification replays the published states for mass, shock momentum, both stagnation-pressure relations and back pressure. The downstream state uses its own reduced total pressure. An explicitly covered complete JSON request with an empty inventory can return the independently replayed result via the existing deterministic delivery path.

## External verification

NASA Glenn's CDV (Converging-Diverging Verification) case has a 1.0-square-inch throat and a 1.5-square-inch exit. At a back-to-reservoir pressure of 0.75 the tabulated jump is bracketed by x = 7.562273 and 7.562293 inches. The published pressure and Mach data are analytic verification values. The 0.89 and 0.16 cases respectively exercise fully subsonic and supersonic shock-free nozzle interiors.

- https://www.grc.nasa.gov/www/wind/valid/cdv/cdv.html
- https://www.grc.nasa.gov/www/wind/valid/cdv/axial.Mex.p75.gen
- https://www.grc.nasa.gov/www/wind/valid/cdv/axial.Pex.p75.gen

Run `make verify-cd-nozzle-shock` with repository runtime dependencies installed. It exercises the JSON-stdin entry point, both exterior controls and 36 prescribed-shock forward checks across three gas properties. The test maps the solved area to NASA's published cosine profile independently.
