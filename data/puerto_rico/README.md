# USGS hillslope monitoring stations, Puerto Rico (Utuado and Toro Negro)

**Source (public domain, U.S. Geological Survey):**

Smith, J.B., Thomas, M.A., Ashland, F., Michel, A., Mirus, B.B., Wayllace, A., 2020. *Hillslope hydrologic monitoring data following Hurricane Maria in 2017, Puerto Rico, July 2018 to June 2020.* U.S. Geological Survey data release. https://doi.org/10.5066/P9548YK2

Process interpretation of these stations follows Thomas et al. (2020), cited in the manuscript.

The files are used exactly as downloaded **[authors: confirm the original file names in the data release and note any renaming]**:

| File | Content |
|---|---|
| `utuado_15min.csv` | Utuado station, 12 Jul 2018 – 1 Jun 2020, 15-min: rain (`precipitation_mm`), volumetric water content at 12, 27, 42, 57 and 72 cm (`vwc_SP1_*cm_ccpercc`), tensiometers (62, 78 cm), vibrating-wire piezometers (40, 55 cm), temperatures, battery |
| `toroNegro_15min.csv` | Toro Negro station, 12 Jul 2018 – 12 Apr 2020, 15-min: rain, water content at 30, 50, 70, 90 and 110 cm (profile 1) and 20, 80 cm (profile 2), tensiometers (72, 102 cm), piezometers (83, 126 cm) |
| `PuertoRico_TestingData.xlsx` | laboratory soil tests: summary of van Genuchten–Mualem parameters (sheet `Summary`, used for R-lab and the P3a test), Atterberg limits, particle-size distributions, saturated density, direct shear, falling-head permeability, soil-water retention (TRIM) |

Timestamps are UTC (`timestamp_UTC`). Water content is in cm³ cm⁻³.

Sensor models: **[authors: copy the water-content sensor, tensiometer and piezometer models from the data-release metadata]**.

## How the study uses the data (puerto_rico/pr_data.py)

- Input (the "low-cost-like" single probe): Utuado 27 cm, Toro Negro 30 cm.
- Targets (virtual deep sensor): primary Utuado 72 cm and Toro Negro 90 cm; secondary Utuado 57 cm and Toro Negro 110 cm.
- Storm rules and all tests were pre-registered (`puerto_rico/PREREGISTRATION.md`) before the storm catalogue was computed.
