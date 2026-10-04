# Thai LoRaWAN soil-moisture nodes (Dev108 and Dev107)

Installed and operated by the authors (School of Engineering and Technology, Walailak University) on a landslide-prone slope in southern Thailand. **[Authors: add site name/district, coordinates (or a coarsened location), slope angle, soil description and installation date.]**

| File | Node | Period | Rows |
|---|---|---|---|
| `pinn_108_new.csv` | Dev108 | 1 Nov 2025 06:35 – 28 Jan 2026 07:42 (rows before 1 Nov 2025 10:40 are pre-installation and are dropped by the pipeline) | 61,954 |
| `107_pinn.csv` | Dev107 | 1 Nov 2025 11:22 – 24 Apr 2026 14:35 | 43,940 |

Timestamps are `m/d/Y H:M` **[confirm: local time (UTC+7) or UTC]**, 1-min rows interpolated from the LoRaWAN uplinks.

## Columns

| Column | Meaning |
|---|---|
| `timestamp` | record time |
| `devID` | node ID (107 or 108) |
| `soil_` | RK520 FDR volumetric water content, % |
| `soil` | the same as a fraction, m³ m⁻³ (used by the models) |
| `rain` | tipping-bucket rain counter within the hour, mm (0.2794 mm per tip); resets hourly and was interpolated to 1-min rows |
| `temp`, `humi` | air temperature (°C) and relative humidity (%) |
| `depth`, `horizon`, `geo`, `geo_`, `devID.1`, first unnamed column | auxiliary fields of the network export, not used in the analysis |

## Processing (thailand/data_pipeline.py)

1. Rain increments = positive counter differences plus the bottom value of each interpolated hourly reset ramp (the true counter restarted at zero, so that value is new-hour rain). This reproduces the 437 mm Dev108 total.
2. Everything is aggregated to a 5-min grid; empty rain bins stay missing (NaN), not dry.
3. Water content is corrected for the sensor's temperature artefact of −0.0009 m³ m⁻³ per °C.
4. Storms are rain clusters separated by at least 6 h without rain, with at least 10 mm in total (7 storms at Dev108, 4 at Dev107).

## Sensor

RK520 frequency-domain reflectometry probe at 30 cm, one per node, with a tipping-bucket rain gauge and an air-temperature/humidity sensor; data transmitted by LoRaWAN. **[Authors: add probe firmware/calibration used (factory or site calibration), gauge model and gateway details.]**

## Licence

See `../../DATA_LICENSE.md`.
