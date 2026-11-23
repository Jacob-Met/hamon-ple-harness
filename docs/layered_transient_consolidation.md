# Transient settlement of connected saturated layers

Use `layered_consolidation_transient` with an instantaneous vertical stress step, small strain, constant material coefficients and one-dimensional drainage. Each of up to four contiguous layers specifies its top and bottom depths, volume compressibility `mv_per_Pa` and consolidation coefficient `cv_m2_s`. Specify `drained` (zero excess pressure) or `impermeable` (zero Darcy flux) at each external end. The step can be a positive uniform surcharge or stresses beneath up to four surface rectangles at a specified column. Nonlinear compression indices, construction histories and independently draining internal boundaries require other models.

Within each layer, storage times the pressure rate equals the divergence of the pressure gradient multiplied by `mv*cv`. Pressure and vertical Darcy flux are continuous across material interfaces. Settlement integrates the vertical effective-stress increment against the local storage. The fully drained reference is reported even for a closed column, where total settlement stays zero and nonuniform initial pressures can redistribute. See [Huang and Griffiths, one-dimensional consolidation theory for layered soil](https://www.newcastle.edu.au/__data/assets/pdf_file/0018/22293/One-dimensional-consolidation-theories-for-layered-soil-and-coupled-and-uncoupled-solutions-by-the-finite-element-method.pdf).

For ordinary CLI use, save the following as `column.json` and run `python engineering_toolkit.py dual-verify column.json`.

```json
{
  "kind": "layered_consolidation_transient",
  "spec": {
    "delta_sigma_v_Pa": 100000, "time_s": 1000000,
    "top_boundary": "drained", "bottom_boundary": "impermeable",
    "hydraulic_layers": [
      {"top_depth_m": 0, "bottom_depth_m": 1, "mv_per_Pa": 1e-7, "cv_m2_s": 1e-6},
      {"top_depth_m": 1, "bottom_depth_m": 2, "mv_per_Pa": 4e-7, "cv_m2_s": 1e-6}
    ]
  }
}
```

Settlement is 0.012478821 m. Reversing the two material coefficients gives 0.040267417 m. An independent two-layer eigenfunction expansion checks settlement, interface pressure and the signed downward flux. Separate controls cover single, double and sealed drainage, homogeneous subdivisions, a spatially varying footing load and the drained limit.

The solver propagates a conservative cell-centred finite-volume system and a linear, mass-lumped nodal finite-element system. The finite-volume face resistance is the sum of the two material half-cell resistances. For surface rectangles, an independent area-integral quadrature checks the grid's integrated corner-stress initial condition. Both systems refine separately with second-order extrapolation. The fast route requires normalized discrepancies below one part per million for layer storage, total settlement and interface quantities. Flux discrepancies use the applied pressure divided by the two adjacent whole-layer resistances. The mesh must resolve each external drainage front. Unresolved spectra, refinements or inconsistent claims withhold the certificate. At the instant of loading the pressure profile is available and the interface flux is reported as null.

The ordinary loopback provider test passes the certified JSON result to Pi and checks the projected thinking budget. The historical 29 task prompts have separate obligations; a synthetic column does not establish savings or accuracy on those tasks.

The companion `layered_consolidation_1d` route handles spatially uniform load histories with jumps and ramps. Its Laplace transfer solution is independently checked against this route on their shared uniform-step domain. Use the transient-footing route for depth-dependent rectangular loading, sealed or bottom-only drainage, and interface pressure and flux.

At early times a graded grid places faces on the physical diffusion length `sqrt(cv*time_s)` near a drained boundary. It preserves every layer interface and becomes uniform as diffusion reaches the layer scale. The two spatial discretizations use the same nested geometry. When the stiffness spectrum loses slow-mode precision, the solver instead factors the time-scaled mass-normalized Darcy gradient, whose singular values are the square roots of the dimensionless decay rates. A poorly conditioned gradient, insufficient spatial agreement or insufficient floating-point precision in the settlement difference each withholds verification.

For a 2 m homogeneous column with a single drain, `cv=1e-6 m²/s`, `mv=1e-7 Pa⁻¹` and a 100 kPa surcharge, the 0.01 s drainage front is 0.1 mm thick. The half-space settlement is 1.1283791671 µm. Run `make verify-drainage-front` with the locked Node dependencies installed to compare the registered tool against this limit, its mirrored and double-drained variants, an interface with distinct material coefficients but matched `mv*sqrt(cv)`, and the later finite-column modal solution.
