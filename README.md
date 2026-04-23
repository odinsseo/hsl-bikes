# Multi-Graph Fusion for HSL City Bike Demand Forecasting

Spatio-temporal demand forecasting for the Helsinki Region Transport (HSL) city bike system using multi-graph Graph Neural Networks (ST-GNN). This repository implements the pipeline described in the [multidisciplinary research plan](multidisciplinary_research_plan.pdf).

## Overview

The project aims to improve short-term bike availability forecasting by:

1. **Multi-graph construction** — Building four adjacency matrices from OD data: Spatial Distance (SD), Demand (DE), Demand Correlation (DC), and Average Trip Duration (ATD)
2. **Graph fusion** — Dynamically weighting multiple graph views for better predictions
3. **Spatial aggregation** — Evaluating station-level vs. cluster-level (Louvain) predictions for sparse suburban stations (e.g. Espoo)

## Shared target preprocessing

All ST-GNN and baseline training/evaluation pipelines use a single train-fitted target preprocessing stack:

1. per-node winsorization (`q0.5`/`q99.5` defaults)
2. `log1p`
3. robust scaling (median/IQR)
4. optional residualization with lag tuned on validation from `{24, 168}`

Primary metrics (WMAPE/MAE/RMSE/MASE) are always reported on the original bike-count scale after inverse-transform.

Full methodology and reviewer checklist: `docs/target_preprocessing_methodology.md`.

## Repository structure

```text
├── notebooks/
│   └── eda_city_bikes.ipynb   # Exploratory data analysis
│   └── network_analysis_city_bikes.ipynb  # Network analysis and community exploration
│   └── rq1_functional_vs_geometric.ipynb  # RQ1 publication notebook (functional vs geometric)
│   └── rq2_fusion_heterogeneity.ipynb  # RQ2 publication notebook (fusion gains)
│   └── rq3_cluster_robustness.ipynb  # RQ3 publication notebook (sparse suburban robustness)
│   └── rq_synthesis.ipynb  # Thesis-ready synthesis matrix notebook
├── scripts/
│   └── prepare_data.py        # Load, clean, merge OD data; create train/val/test splits
│   ├── data_audit.py          # Validate schema, split integrity, and leakage guards
│   ├── graph_construction.py  # Build SD/DE/DC/ATD adjacency matrices
│   ├── experiment_runners.py  # Run RQ1/RQ2/RQ3 ablation experiments
│   ├── train_eval_pipeline.py # 1-hour train/eval for station and community levels
│   ├── train_stgnn_pipeline.py # Single-graph ST-GNN (A3T-GCN style) train/eval
│   ├── run_stgnn_milestones.py # M3.1/M3.2 sweep + leakage sensitivity orchestration
│   ├── pre_notebook_quality_gate.py # Canonical artifact gate before notebook reporting
│   ├── validate_notebook_suite.py # Clean-kernel execution and manifest lock for notebooks
│   ├── notebook_reporting.py # Shared artifact-loading utilities for publication notebooks
├── tests/
│   ├── test_data_audit.py
│   └── test_graph_construction.py
│   └── test_experiment_runners.py
│   └── test_train_eval_pipeline.py
│   └── test_stgnn_pipeline.py
│   └── test_validate_notebook_suite.py
│   └── test_notebook_reporting.py
├── docs/
│   └── phase_0_1_implementation.md
│   └── reproducibility_runbook.md
├── config/
│   └── constants.py           # Station renames and cleaning rules
├── data/
│   ├── primary/               # Raw/source data
│   │   ├── trips/             # Raw OD CSVs by year/month (from HSL)
│   │   └── stations/          # Station coordinates
│   ├── prepared/              # Cleaned and model-ready datasets
│   │   ├── merged/            # Merged trips (output of prepare_data.py)
│   │   └── splits/            # Time-based train/validation/test splits
│   └── artifacts/             # Derived analysis and graph outputs
│       ├── audit/             # Data audit reports
│       ├── graphs/            # SD/DE/DC/ATD matrices
│       └── network/           # Notebook analysis exports
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

Trip data is available from [HSL Open Data](https://www.hsl.fi/en/hsl/open-data). Download OD trip CSVs by year/month and place them under `data/primary/trips/od-trips-YYYY/`. Station coordinates go in `data/primary/stations/`.

**Prepare data:**

```bash
python scripts/prepare_data.py
```

This will:

- Load and clean all trip CSVs
- Merge with station coordinates
- Write `data/prepared/merged/trips_merged.csv`
- Split by time into `data/prepared/splits/{train,validation,test}`

## Usage

**Exploratory data analysis:**

```bash
jupyter notebook notebooks/eda_city_bikes.ipynb
```

The notebook covers temporal patterns, trip characteristics, station usage, graph-building blocks (SD, DE, ATD, DC), and data quality checks.

**Data integrity audit:**

```bash
python scripts/data_audit.py --strict
```

This writes `data/artifacts/audit/data_audit_report.json` with split boundary checks, overlap checks, null-rate profiling, duplicate counts, and timestamp diagnostics.

**Graph construction (train-only recommended):**

```bash
python scripts/graph_construction.py --input data/prepared/splits/train/train.csv --output-dir data/artifacts/graphs/train
```

Outputs SD/DE/DC/ATD adjacency matrices and metadata.

**Network analysis notebook (centrality + communities + extended exploration):**

```bash
jupyter notebook notebooks/network_analysis_city_bikes.ipynb
```

The notebook covers centrality, Louvain/Fluid community analysis, temporal partition stability, resilience experiments, and multi-graph comparison, and exports analysis artifacts under `data/artifacts/network/notebook/`.

**Run tests:**

```bash
pytest -q
```

**Run RQ ablation experiments (Phase 2 starter):**

```bash
python scripts/experiment_runners.py --rqs RQ1,RQ2,RQ3
```

To run holiday sensitivity with Finland national-only holidays (no subdivision), add:

```bash
--holiday-national-only
```

This writes experiment outputs to `data/artifacts/experiments/rq_runner/`:

- `experiment_matrix.csv` (planned ablation matrix)
- `alpha_search.csv` (graph-propagation alpha tuning table)
- `baseline_search.csv` (seasonal/linear/tree baseline tuning table)
- `results.csv` (side-by-side validation/test metrics across all models)
- `summary.json` (best overall and best graph-only configuration per RQ)

**Run 1-hour train/eval pipeline (station + community + station cohorts):**

```bash
python scripts/train_eval_pipeline.py --graph-set SD,DE,DC,ATD
```

The same holiday sensitivity toggle is supported:

```bash
--holiday-national-only
```

This writes outputs to `data/artifacts/experiments/train_eval_1h/`:

- `train_eval_results.csv` (overall station/community model metrics)
- `train_eval_alpha_search.csv` (graph alpha tuning by aggregation)
- `train_eval_baseline_search.csv` (baseline tuning by aggregation)
- `station_cohort_results.csv` (station-level cohorts: Helsinki/Espoo/sparse/dense)
- `station_robustness_statistics.csv` (bootstrap CIs and paired significance by cohort/model)
- `sensitivity_summary.csv` (threshold/resolution/hyperparameter sensitivity table for notebook ingestion)
- `summary.json` (best model by aggregation and run metadata)

**Run ST-GNN baseline (single graph, M3.1):**

```bash
python scripts/train_stgnn_pipeline.py --aggregation station --graph DE --history 24
```

National-only holiday sensitivity is available with:

```bash
--holiday-national-only
```

**Run ST-GNN multi-graph fusion (M3.2):**

```bash
python scripts/train_stgnn_pipeline.py --aggregation station --graph-set SD,DE,DC,ATD --fusion-mode learned --history 24
```

This writes outputs to `data/artifacts/experiments/stgnn_single_graph/`:

- `training_curve.csv` (epoch-wise train/validation Huber loss and validation WMAPE)
- `results.csv` (validation/test metrics for selected graph set, aggregation, and fusion mode)
- `summary.json` (hyperparameters, best epoch, and learned/equal fusion weights)

Notes:

- The ST-GNN baseline uses a lightweight A3T-GCN-style temporal attention architecture over graph-propagated inputs.
- `--fusion-mode single` requires one graph. `--fusion-mode equal` and `--fusion-mode learned` support multi-graph fusion.
- PyTorch is required for this pipeline (`pip install torch`).

**Run milestone sweep (M3.1 + M3.2 + optional leakage sensitivity):**

```bash
python scripts/run_stgnn_milestones.py --include-community --include-leakage-sensitivity --build-leaky-graphs
```

This writes outputs to `data/artifacts/experiments/stgnn_milestones/`:

- `milestone_results.csv` (all M3.1/M3.2 runs)
- `milestone_best_by_milestone.csv` (best run per milestone and aggregation)
- `milestone_best_by_graph_set.csv` (best run per aggregation/graph-set/fusion mode)
- `leakage_sensitivity.csv` (train-only graph vs leaky full-period graph deltas)
- `summary.json` (best overall and best by aggregation)

**Run pre-notebook quality gate (canonical artifacts + metadata + schemas):**

```bash
python scripts/pre_notebook_quality_gate.py
```

This validates canonical experiment outputs under `data/artifacts/experiments/` and fails fast when required files, metadata fields, schema columns, preprocessing lineage fields, or minimum row counts are missing.

**Run publication notebook suite validation (clean kernel execution):**

```bash
python scripts/validate_notebook_suite.py --timeout-seconds 1800
```

This writes a lock manifest to `data/artifacts/notebooks/validation_manifest.json` and executed notebook copies under `data/artifacts/notebooks/executed/`.

**Open publication notebooks (artifact-first):**

```bash
jupyter notebook notebooks/rq1_functional_vs_geometric.ipynb
jupyter notebook notebooks/rq2_fusion_heterogeneity.ipynb
jupyter notebook notebooks/rq3_cluster_robustness.ipynb
jupyter notebook notebooks/rq_synthesis.ipynb
```

For full end-to-end command order and expected runtime bands, use `docs/reproducibility_runbook.md`.

## Roadmap (from research plan)

| Phase | Milestone | Status |
| ----- | --------- | ------ |
| **1. Data preparation** | M1.1 Download/clean OD data; M1.2 Hourly inflow/outflow tensors | ✅ In progress |
| **2. Graph construction** | M2.1 SD, DE, DC, ATD adjacency matrices; M2.2 Louvain clustering | ✅ Initial implementation |
| **3. Model implementation** | M3.1 A3T-GCN single-graph; M3.2 Multi-graph fusion; M3.3 Station/cluster-level training | 🟡 M3.1+M3.2 tooling implemented |
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
