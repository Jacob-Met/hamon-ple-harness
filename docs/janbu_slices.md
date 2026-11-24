# Simplified Janbu on a specified slip surface

Use `calculation: "janbu_slices"` to compute the uncorrected simplified Janbu factor of safety for a supplied table of vertical slices. The Python entry point is `hamon_janbu_slices.verify_janbu_slices(spec)`. The method enforces vertical force equilibrium for each slice and horizontal force equilibrium for the whole mass, with zero interslice shear.

## Input

| Field | Units and convention |
| --- | --- |
| `width_m` | Horizontal slice width in metres |
| `weight_N_m` | Total slice weight in newtons per metre of out-of-plane thickness |
| `base_angle_deg` | Degrees, positive where gravity drives the slice in the declared direction of sliding. Preserve the negative sign for resisting slices. |
| `pore_pressure_Pa` | Mean base pore pressure in pascals. Defaults to zero. |
| `c_prime_Pa`, `phi_deg` | Effective cohesion in pascals and effective friction angle in degrees. A slice can override either common value. |
| `id` | Optional unique slice identifier, supplied as a string |

For example, this dry planar slice has a factor of safety of 2.0. Its 1,000 N/m weight has a 707.107 N/m downslope component. Cohesion acts over a base length of 1.414214 m.

```python
from hamon_janbu_slices import verify_janbu_slices

result = verify_janbu_slices({
    "calculation": "janbu_slices",
    "c_prime_Pa": 1000.0,
    "phi_deg": 0.0,
    "slices": [{"width_m": 1.0, "weight_N_m": 1000.0, "base_angle_deg": 45.0}],
})
```

## Reading the result

`factor_of_safety` is a dimensionless ratio for the supplied surface. `equilibrium_verified` checks the equations enforced by the method. `base_contact_verified` checks that each reconstructed effective base normal force is nonnegative within numerical tolerance. A tensile base is listed in `tensile_slice_ids`. In that case a mathematical solution can still exist, and the overall `verified` flag is false.

Use `required_factor_of_safety` to check a design threshold through `meets_required_factor`. An equilibrium solution can satisfy its numerical checks while falling short of that threshold. A supplied `proposed_factor_of_safety` is compared to the computed value through `proposed_factor_consistent`.

The calculation assumes the input geometry and material properties. The unordered calculation does not check interslice thrust. Neither mode searches for a critical surface, checks moment equilibrium, models tension cracks, or applies Janbu’s empirical correction factor. Geometric validation and a slope-wide assessment require independent evidence.

## Reference cases

| Specified surface | Uncorrected Janbu factor | Evidence |
| --- | ---: | --- |
| Seequent, 29 slices with published pore pressures | 1.1247783031 | Independent force reconstruction using the published slice data |
| Ibaraki Kozo Lab, six dry slices | 1.8293 | Published Janbu result |
| Ibaraki Kozo Lab, 50 dry slices | 1.8677 | Published Janbu result. Reconstruction gives tensile contact at slice 50. |
| Ibaraki Kozo Lab, 50 slices with water at elevation 4 m | 1.427 | Published Janbu result. Reconstruction gives tensile contact at slice 50. |

Seequent’s 13-page *Verification – Hand Calculations* gives slice widths, weights, angles and pressures in Table 1 on PDF page 4. Its published factor of 1.249 uses a different equilibrium method. For the Janbu check, the test converts weights and pressures to SI units, negates the table’s geometric angle to match the declared sliding direction, and reconstructs each base force from the slice free body. The summed horizontal force vanishes at the independent reference value. [Seequent source PDF](https://files.seequent.com/GeoStudio/SlopeW/Verification%20-%20Hand%20Calculations.pdf)

The Ibaraki dry and wet cases prescribe the same circular arc. The tests regenerate the slice weights and inclinations from the circle and ground coordinates and compare against the published Janbu results at their printed precision. [LEM Primer, sections 1 and 5](https://ibaraki-kozo-lab.github.io/lem-primer/en/practice-slices-2d.html)

## Checking thrust across ordered slices

Set `slice_order` to `toe_to_crest` or `crest_to_toe` when the supplied table follows that physical sequence. The two end boundaries carry zero horizontal thrust and the simplified method assumes zero interslice shear. For a toe-to-crest sweep, each base contributes `T cos(α) − N sin(α)` to the cumulative horizontal resisting force. Compression has positive sign. Reversing both the table and its declared order describes the same contacts.

Each entry in `interfaces` identifies two neighboring slices and reports total horizontal thrust, a sign tolerance and an independent interval in N/m. The interval calculation starts from the original dimensional loads and angles in degrees. It encloses the unique root above every denominator pole and sweeps both ends. A strictly positive interval and agreement with the ordinary force reconstruction support compression. A tensile or unresolved contact withholds overall verification. The ordinary equilibrium solution remains available. The ordered threshold check uses the lower endpoint of `factor_of_safety_interval`.

At 33 slices and water elevation 4 m, the Ibaraki geometry gives a factor of approximately 1.4256398871, yet the interface between slices 32 and 33 requires about −5131.37 N/m of tension. The 1.4 threshold is therefore withheld for this ordered table. This 33-slice result is an independent calculation on the published geometry. Basal pore pressure does not specify side-water forces. The total-thrust check cannot establish effective side pressure, thrust-line location or moment equilibrium. [Ibaraki geometry](https://ibaraki-kozo-lab.github.io/lem-primer/en/practice-slices-2d.html) · [USACE EM 1110-2-1902, Appendix C](https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1902.pdf)
