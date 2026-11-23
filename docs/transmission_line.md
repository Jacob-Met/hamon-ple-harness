# Transmission-line load boundaries

Use `hamon_conservation_cad_verifier` with mode `rf_transmission_rlc` and subkind `transmission_line`. Choose `load_kind: "open"` for zero load current or `load_kind: "short"` for zero load voltage. A finite impedance uses `R_L` and `X_L` in ohms. The finite load defaults to 100 Ω resistance and zero reactance when those fields are omitted.

An ideal open has no finite resistance or reactance fields. An explicit short accepts omitted or zero-valued fields. Supply the phase either as `length_wavelengths` or using `beta_rad_m` and `length_m`. For a lossy line, `alpha_np_m` and `length_m` determine the total attenuation.

```json
{"mode":"rf_transmission_rlc","spec":{"subkind":"transmission_line","Z0":50,"load_kind":"open","length_wavelengths":0.25}}
```

This lossless quarter-wave example gives `load_kind="open"`, `Z_in_kind="short"`, and zero input impedance. At one eighth of a wavelength the same load gives −50j Ω. At zero electrical length it gives an open port, with zero admittance and no finite impedance. Exact node calculations use fractional wavelengths. A rounded radian approximation to a node represents a nearby, generally finite impedance.

The solver propagates voltage and characteristic-impedance-scaled current. Its voltage reflection coefficient at an ideal open is +1 and at an ideal short is −1. Both have 0 dB return loss and unbounded load VSWR. For complex electrical length $u=(\alpha+j\beta)\ell$, the two boundary impedances have independent checks $Z_{\mathrm{open}}=Z_0\coth u$ and $Z_{\mathrm{short}}=Z_0\tanh u$. [Michigan State University transmission-line notes](https://www.egr.msu.edu/emrg/sites/default/files/content/module2_fundamental_behavior.pdf)

## Uniform distributed lines

A uniform passive line can instead be specified by four per-metre constants, frequency in hertz and physical length in metres. All six must be finite nonnegative numbers. Use this description separately from `Z0`, `alpha_np_m`, `beta_rad_m` and `length_wavelengths`. The same load conventions apply.

```json
{"mode":"rf_transmission_rlc","spec":{"subkind":"transmission_line","R_ohm_per_m":2,"L_H_per_m":0.000001,"G_S_per_m":0,"C_F_per_m":1e-10,"frequency_Hz":0,"length_m":8,"R_L":100}}
```

For sinusoidal steady state with the positive-frequency phasor convention, the series impedance per metre is $z=R+j2\pi fL$ and the shunt admittance per metre is $y=G+j2\pi fC$. The passive propagation root is $\gamma=\sqrt{zy}$ and the characteristic impedance is $Z_0=z/\gamma$. The result includes `alpha_np_m`, `beta_rad_m`, `Z0_real` and `Z0_imag`. A distortionless line, including a lossless line at DC by continuity, has $Z_0=\sqrt{L/C}$. The load VSWR and return loss use voltage-wave reflection at the load.

At DC the example reduces to an 8 m series resistor and the input is 116 Ω. If one distributed branch vanishes, voltage and current propagate using the exact linear two-port. The line still has a well-defined terminal impedance or admittance. Modal quantities that require finite, nonzero characteristic impedance are null.
