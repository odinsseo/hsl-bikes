# Multi-Graph Fusion for HSL City Bike Demand Forecasting

Spatio-temporal demand forecasting for the Helsinki Region Transport (HSL) city bike system using multi-graph Graph Neural Networks (ST-GNN). This repository implements the pipeline described in the [multidisciplinary research plan](multidisciplinary_research_plan.pdf).

## Overview

The project aims to improve short-term bike availability forecasting by:

1. **Multi-graph construction** — Building four adjacency matrices from OD data: Spatial Distance (SD), Demand (DE), Demand Correlation (DC), and Average Trip Duration (ATD)
2. **Graph fusion** — Dynamically weighting multiple graph views for better predictions
3. **Spatial aggregation** — Evaluating station-level vs. cluster-level (Louvain) predictions for sparse suburban stations (e.g. Espoo)

## Repository structure

```
├── notebooks/
│   └── eda_city_bikes.ipynb   # Exploratory data analysis
├── scripts/
│   └── prepare_data.py        # Load, clean, merge OD data; create train/val/test splits
├── config/
│   └── constants.py           # Station renames and cleaning rules
├── data/
│   ├── trips/                 # Raw OD CSVs by year/month (from HSL)
│   ├── stations/              # Station coordinates
│   ├── merged/                # Merged trips (output of prepare_data.py)
│   ├── train/                 # Training split
│   ├── validation/            # Validation split
│   └── test/                  # Test split
├── requirements.txt
└── multidisciplinary_research_plan.pdf
```

## Setup

```bash
# Clone and create environment
git clone https://github.com/YOUR_USERNAME/hsl-bike-demand-forecasting.git
cd hsl-bike-demand-forecasting

python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Data

Trip data is available from [HSL Open Data](https://www.hsl.fi/en/hsl/open-data). Download OD trip CSVs by year/month and place them under `data/trips/od-trips-YYYY/`. Station coordinates go in `data/stations/`.

**Prepare data:**

```bash
python scripts/prepare_data.py
```

This will:

- Load and clean all trip CSVs
- Merge with station coordinates
- Write `data/merged/trips_merged.csv`
- Split by time into train (before 2022), validation (2022), test (2023+)

## Usage

**Exploratory data analysis:**

```bash
jupyter notebook notebooks/eda_city_bikes.ipynb
```

The notebook covers temporal patterns, trip characteristics, station usage, graph-building blocks (SD, DE, ATD, DC), and data quality checks.

## Roadmap (from research plan)

| Phase | Milestone | Status |
|-------|-----------|--------|
| **1. Data preparation** | M1.1 Download/clean OD data; M1.2 Hourly inflow/outflow tensors | ✅ In progress |
| **2. Graph construction** | M2.1 SD, DE, DC, ATD adjacency matrices; M2.2 Louvain clustering | 🔜 Planned |
| **3. Model implementation** | M3.1 A3T-GCN single-graph; M3.2 Multi-graph fusion; M3.3 Station/cluster-level training | 🔜 Planned |
| **4. Evaluation** | M4.1 WMAPE/MASE metrics; M4.2 Final report | 🔜 Planned |

## Research questions

- **RQ1** — How do functional graphs (DE, DC) compare to geometric (SD) for short-term prediction?
- **RQ2** — Does multi-graph fusion improve accuracy in the heterogeneous Helsinki–Espoo network?
- **RQ3** — Does cluster-level aggregation improve robustness for sparse suburban stations?

## References

- Lin et al. (2018) — Graph construction for bike-sharing demand prediction
- Wang et al. (2023) — Heterogeneous spatio-temporal GNNs
- PyTorch Geometric Temporal — [Documentation](https://pytorch-geometric-temporal.readthedocs.io/)

## License

[Specify license, e.g. MIT]
