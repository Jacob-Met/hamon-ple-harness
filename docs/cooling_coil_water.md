# Water and heat removed by a cooling coil

Supply measured inlet and outlet air states, one dry-air mass flow or inlet volumetric flow, and the phase of water removed from the airstream. The declared product may be deposited ice, drained liquid, or both. Outlet air temperature alone cannot establish the fraction frozen.

```json
{"mode":"psychrometrics","spec":{"mode":"cooling_coil","inlet":{"t_db_c":-2,"rh":0.9},"outlet":{"t_db_c":-12,"rh":1},"m_da_kg_s":1,"condensate_phase":"ice","t_condensate_c":-12}}
```

The example deposits 0.00153642 kg/s of ice and removes 14.47199 kW. Omitting the phase retains the historical liquid-property calculation, which gives 13.99892 kW under those conditions and labels its assumption `assumed_liquid`. The result identifies a supplied phase and whether the water-product temperature was supplied or taken from the outlet dry bulb.

| Product | Accepted conditions | Water accounting |
|---|---|---|
| `ice` | −60 through 0 °C | Deposited ice mass per second |
| `liquid` | 0 °C through boiling at the stated pressure, within the 0–200 °C correlation range | Drained liquid mass per second |
| `mixed` | 0 °C and a supplied `ice_mass_fraction` from 0 to 1 | Separate deposited and drained rates |

The heat balance subtracts both the liquid-drain enthalpy flow and the enthalpy accumulated with deposited ice from the loss in moist-air enthalpy. Both water phases use liquid water at 0 °C as the reference. Ice at 0 °C has enthalpy −333.5 kJ/kg. Each pure phase approaches its own freezing-point value continuously. At 0 °C the supplied ice mass fraction connects those endpoints continuously. For the explicit-phase calculation the moist-air saturation correlations switch at the water triple point, 0.01 °C.

The default sensible/latent split uses the midpoint moist-air heat capacity. The ASHRAE inlet-dew-point split remains available for a liquid drain. This calculation describes product mass and heat rates at the supplied conditions. Frost-layer resistance, defrost, fans and compressor work require separate models.

For a complete explicit-phase inline request, participant intake confirms the dry-air and water rates and independently reconstructs the first-law load before issuing the registered result. The referenced properties and liquid-drain balance come from ASHRAE. [Psychrometrics](https://handbook.ashrae.org/handbooks/F21/IP/F21_Ch01/F21_Ch01_ip.aspx) · [Air-cooling and dehumidifying coils](https://handbook.ashrae.org/handbooks/S24/SI/s24_ch23/s24_ch23_si.aspx)
