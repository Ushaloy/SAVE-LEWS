# SAVE-LEWS: reproduce all figures and tables (no training)

Code, data and stored model results for the manuscript

> **Matching physics to observations: physics-guided soil-moisture prediction from low-cost IoT sensors for landslide early warning.**

This repository regenerates **Figs. 1-11 and S1-S9** and the **tables (Tables 2, 5-9, S1-S9)** of the manuscript from the raw sensor data and the stored model outputs. **Nothing is trained**; the full run takes about 1-2 minutes on a CPU.

## Run it (Google Colab, no installation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ushaloy/SAVE-LEWS/blob/main/notebooks/00_reproduce_all_figures_and_tables.ipynb)

Open the notebook and choose `Runtime -> Run all`. It clones this repository, installs the packages in `requirements.txt`, writes the figures to `figures/output/` and the tables to `tables/`, and checks the SHA-256 of the data files and of the Puerto Rico pre-registration.

## Run it locally

```bash
git clone https://github.com/Ushaloy/SAVE-LEWS.git && cd SAVE-LEWS
pip install -r requirements.txt
python figures/make_all_figures.py     # Figs 1-11, S1-S9 -> figures/output/
python tables/make_tables.py           # table CSVs -> tables/
python figures/sensor_compare.py       # Table 2 (sensor systems)
```

## What is in the repository

```
notebooks/00_reproduce_all_figures_and_tables.ipynb   entry point (stored with its executed outputs)
data/            raw data + CHECKSUMS.sha256 (Thailand: authors; Puerto Rico: USGS)
thailand/        model code, stored results (res*.json), 3 checkpoints, bts_out/, conf_out/
puerto_rico/     model code, stored results (out/), PREREGISTRATION.md + SHA-256, DEVIATIONS.md
figures/         figure scripts (make_all_figures.py runs them all)
tables/          make_tables.py
```

Figures and tables are drawn from the raw data and the stored result files (`*.json`, `*.npz`, three PyTorch checkpoints). The models were trained beforehand with scripts that are not part of this reduced repository.

## Data

| Folder | Source |
|---|---|
| `data/thailand/` | Dev108 and Dev107 LoRaWAN stations, Walailak University (see `data/thailand/README.md`) |
| `data/puerto_rico/` | Utuado and Toro Negro stations. U.S. Geological Survey, public domain. Smith et al. (2020), https://doi.org/10.5066/P9548YK2 |

Licences: `LICENSE` (code) and `DATA_LICENSE.md` (data).

## Pre-registration

`puerto_rico/PREREGISTRATION.md` was frozen before the Puerto Rico results were computed. Its SHA-256 is in `puerto_rico/PREREG_SHA256.txt` and is verified by the notebook. Later changes are listed in `puerto_rico/DEVIATIONS.md`.

## Software

Tested with Python 3.11-3.12 and the package versions in `requirements.txt` (NumPy, pandas, SciPy, Matplotlib, PyTorch, JAX, NumPyro, Optax, openpyxl).
