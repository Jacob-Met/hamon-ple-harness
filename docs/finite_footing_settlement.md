# Settlement under rectangular surface loads

Use `layered_footing_settlement` to integrate ultimate primary settlement down a specified column. Give coordinates and depths in metres and positive net service pressures and initial effective stresses in pascals. Layers are ordered and contiguous. Each layer has its own initial void ratio, compression index and initial effective stress. Overconsolidated layers also require a recompression index and preconsolidation stress.

This input runs through the provisioned task CLI with `python hamon-engineering-toolkit.py dual-verify footing.json`.

```json
{
  "mode": "domain",
  "kind": "consolidation_settlement",
  "spec": {
    "settlement_column": {"x_m": 0, "y_m": 0},
    "rectangular_loads": [{"center_x_m": 0, "center_y_m": 0, "width_m": 2, "length_m": 2, "net_pressure_Pa": 200000}],
    "settlement_layers": [{"top_depth_m": 0.1, "bottom_depth_m": 6.1, "e0": 0.8, "Cc": 0.32, "sigma_v0_eff_Pa": 75000}],
    "max_settlement_m": 0.2
  }
}
```

This column settles 0.2406348715 m, exceeding the 0.2 m limit. Applying only the stress at 3.1 m throughout its thickness would give 0.1725689762 m. The solver superposes rectangular loads in a homogeneous elastic half-space, applies each layer’s compression law at depth and integrates the strain. Two independent stress/depth quadratures check the result. An unresolved transition suppresses the certificate. Layer changes affect compressibility and prescribed stress. The stress-transfer solution assumes a single elastic half-space.

For conservation verification, set the mode to `civil_geotech` and the calculation to `layered_footing_settlement`. Add `proposed_settlement_m` to check a claimed value. The existing `stress_point` mode remains available for a specified point increment. Time-dependent pore-pressure and drainage inputs belong to their separately defined consolidation models.
