# Pre-registration — Puerto Rico two-depth test of physics-informed inference

Frozen: 2026-09-27 (Asia/Bangkok), before any storm catalogue, lag statistic or model result was computed.
At this point only file headers, column names and the lab workbook (van Genuchten parameters) had been inspected.
The SHA-256 of this file is recorded in PREREG_SHA256.txt immediately after writing. Any analysis not listed here is exploratory.

## Data
- Utuado station (coarse SM soil, 42°): 15-min series 2018-07-12 to 2020-06-01. VWC at 12, 27, 42, 57, 72 cm; tensiometers 62/78 cm; piezometers 40/55 cm; rain.
- Toro Negro station (fine CL/MH soil, 45°; lab samples labelled ELT): 15-min series 2018-07-12 to 2020-04-12. VWC sp1 at 30, 50, 70, 90, 110 cm; piezometers 83/126 cm; rain.
- Lab: drying and wetting van Genuchten parameters and Ksat for UTU (3 depths) and ELT (2 depths).

## Storms and windows
Storms: rain clusters separated by >= 6 h without rain, total >= 10 mm (same rule as the Thai study).
Window: 24 h before storm start to 72 h after storm end. Rows with missing input or target are excluded from scoring, not filled.
Leave-one-storm-out: train on all other storm windows after removing anything within 24 h of the held-out window.

## Task (virtual deep sensor)
Input: the shallow probe (Utuado 27 cm; Toro Negro 30 cm) and rain for the whole window, plus the observed profile at the window start only (initial condition).
Target: VWC at the primary deep sensor for every 15-min step of the window. Primary deep sensor: Utuado 72 cm; Toro Negro 90 cm. Secondary: Utuado 57 cm; Toro Negro 110 cm.
The deep sensors are never inputs after the window start.
Skill = 1 − SSE/SSE_ref, where the reference predicts no change from the window-start value, pooled over held-out storms.

## Models
- R-cal: 1-D Richards (homogeneous van Genuchten–Mualem, implicit solver), top boundary = observed shallow θ (Dirichlet), free drainage at the pit base; Ks, α, n, θs, θr calibrated on the training storms of each fold.
- R-lab: same solver with lab parameters (no calibration).
- PG-MLP: MLP on shallow θ, its rolling means and changes, rain sums, window-start deep θ and elapsed time; storage-bounded output; 3 seeds.
- Exploratory: hybrid R-cal + learned correction.

## Hypotheses and decision rules (per site)
- P1 inference: SUPPORTED if pooled primary-target skill of R-cal exceeds PG-MLP for all 3 seeds, with storm-bootstrap (10 000 resamples) 95% interval of the difference above 0. NOT SUPPORTED otherwise.
- P2 mechanism: among storms where both shallow and primary deep sensors rise by >= 0.005 within 48 h of storm start, median lag between rise onsets (first step exceeding pre-storm mean + 0.005) > 2 h → matrix-dominated; median <= 30 min or >50% of storms <= 30 min → preferential flow reaches depth.
- P3 identifiability: (a) fold-median calibrated α and Ks within a factor of 3 of the lab range for that site and n within ±0.25 of the lab range; (b) between-fold SD of log-parameters no larger than the median within-fold posterior SD (from P4 posteriors).
- P4 calibration: held-out 90% interval coverage of the primary target between 0.85 and 0.95, and interval score no worse than PG-MLP + cross-conformal.

Every outcome is reported, including failures.
