# Transformer efficiency and voltage regulation

Run `python3 /tmp/hamon-engineering-toolkit.py dual-verify spec.json` in the task container. Set `mode` to `domain` and `kind` to `transformer_efficiency_regulation`. Put the rated apparent power `S_rated_VA`, secondary terminal voltage `V_sec_rated_V`, equivalent series resistance and reactance `R_eq_ohm` and `X_eq_ohm`, fixed loss `P_core_W` and load power-factor magnitude `pf` in `spec`. The calculator accepts nonnegative load fraction and losses, positive ratings, and power-factor magnitudes from zero to one. The optional `load_fraction` defaults to 1 and `pf_mode` defaults to `lagging`. The other supported mode is `leading`.

The calculator holds secondary terminal voltage fixed. It cross-checks the no-load emf using the complex-phasor and component forms of the secondary equivalent circuit. At load fraction λ the modeled useful power is Aλ, with A = rated apparent power × power factor. The modeled losses are C + Bλ², where C is core loss and B is the copper loss at rated current. Operating efficiency is the useful power divided by total real input power. When that denominator is zero, `efficiency` is null and `efficiency_status` is `undefined_zero_input_real_power`. The voltage and copper-loss fields still describe the given operating point.

The load for maximum efficiency is a separate result. Its search domain is λ ≥ 0 with no upper bound. A value above one describes this model and does not grant an overload rating. The derivative on the domain where efficiency is defined has the sign of A(C − Bλ²).

| Condition | `optimal_load_fraction_status` | `optimal_load_fraction` |
|---|---|---|
| A, B and C positive | `unique` | √(C/B) |
| Unique optimum outside the positive binary64 range | `unique_outside_float_range` | null |
| A and C positive, B zero | `supremum_as_load_increases` | null |
| A and B positive, C zero | `supremum_as_load_approaches_zero` | null |
| A zero with some loss, or positive A with zero losses | `all_loads_with_defined_efficiency` | null |
| A, B and C all zero | `undefined_at_all_loads` | null |

For example, with a 1,000 VA, 100 V rating, 5 W core loss, zero winding resistance, 0.4 Ω reactance and a half-rated load at 0.8 lagging, useful power is 400 W and operating efficiency is 400/405. The model's efficiency approaches one as the load increases. The CLI returns the finite operating results and a null optimum with status `supremum_as_load_increases`.

The reported voltage regulation is the signed relative change in voltage magnitude, `100 (|E₂| − V₂) / V₂`. It is evaluated before rounding the absolute no-load voltage. Near a leading-power-factor zero crossing, `E2_no_load_V` can round to the rated terminal voltage while `voltage_regulation_pct` still reports the direction and size of the change.
