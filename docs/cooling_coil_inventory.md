# Cooling-coil water and energy inventories

The `hvac_coil_and_duct` route models steady cooling and dehumidification without a water supply. The humidity ratios are kilograms of vapor per kilogram of dry air. Removed water is the dry-air rate times the inlet-to-outlet humidity-ratio decrease. A positive water residual and a failed certificate identify an outlet that would require an unmodeled water supply.

`Q_total_cooling_W` retains its air-side meaning, the decrease in moist-air enthalpy flux. The reported `Q_sensible_W` uses the mean humidity ratio of the two states. `Q_vapor_transfer_W` is the complementary contribution from water vapor transfer using the mean air temperature. Their sum independently reconstructs the air-side duty. The air-side and water balance flags apply to the submitted states and flows. They do not establish a coil surface temperature or a particular saturation pressure.

For a coil removing condensate, supply `condensate_enthalpy_J_kg` on the same reference as the air enthalpy model (zero for liquid water at 0°C) to obtain `Q_coil_heat_removed_W`. The coil heat removed equals the air-side decrease minus the outgoing condensate enthalpy flux. A dry coil needs no drain property. The separation of these duties follows the [ASHRAE cooling and dehumidifying coil balance](https://xp20.ashrae.org/SupplementalFiles/PHVAC9/Air_Processing_Equipment.pdf).

The combined certificate requires the coil and duct checks to pass. The rectangular duct retains its existing definition of volume flow from the dry-air mass rate and the supplied density.
