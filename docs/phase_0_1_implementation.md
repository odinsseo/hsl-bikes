# Phase 0-1 Implementation Notes

This document describes the first implementation slice for the HSL multi-graph forecasting project.

## Scope completed

1. Data audit utilities for merged/train/validation/test integrity checks.
2. Leakage-safe graph construction for SD, DE, DC, and ATD views.
3. Network analysis notebook with centrality metrics and Louvain community detection.
4. Unit tests for core functions (data audit, graph construction, network analysis).

## New scripts

1. `scripts/data_audit.py`
2. `scripts/graph_construction.py`
3. `notebooks/network_analysis_city_bikes.ipynb`
4. `scripts/experiment_runners.py`
5. `scripts/train_eval_pipeline.py`

## Data audit outputs

Run:

```bash
python scripts/data_audit.py --strict
```

Output artifact:

- `data/artifacts/audit/data_audit_report.json`

Main checks:

1. Split boundary checks against configured train/validation cutoffs.
2. Temporal overlap checks across train/validation/test.
3. Null-rate and duplicate-row profiling per dataset.
4. Timestamp monotonicity indicator per dataset.

## Graph construction outputs

Run:

```bash
python scripts/graph_construction.py --input data/prepared/splits/train/train.csv --output-dir data/artifacts/graphs/train
```

Output artifacts:

1. `SD.npy` (Spatial Distance adjacency)
2. `DE.npy` (Demand Edge adjacency)
3. `DC.npy` (Demand Correlation adjacency)
4. `ATD.npy` (Average Trip Duration adjacency)
5. `station_index.txt`
6. `metadata.json`

Path:

- `data/artifacts/graphs/train/`

Leakage policy:

1. Graph statistics should be constructed from train-period data only.
2. Optional `--cutoff-end` allows strict end-date filtering.

## Network analysis notebook outputs

Run:

```bash
jupyter notebook notebooks/network_analysis_city_bikes.ipynb
```

Output artifacts:

1. `data/artifacts/network/notebook/node_metrics.csv`
2. `data/artifacts/network/notebook/communities.csv`
3. `data/artifacts/network/notebook/community_summary.csv`
4. `data/artifacts/network/notebook/partition_stability.csv` (if temporal section executed)
5. `data/artifacts/network/notebook/figures/*.png` (publication-focused figures)

Path:

- `data/artifacts/network/notebook/`

Metrics included:

1. Degree and flow strength (in/out)
2. Betweenness and closeness
3. PageRank and eigenvector centrality
4. Louvain modularity and community count
5. NMI stability between adjacent temporal partitions

## Testing

Run:

```bash
pytest -q
```

Current tests validate:

1. Split boundary and overlap logic.
2. SD/DE/DC adjacency properties.
3. Network metrics and community assignment outputs.

## Next implementation slice

1. Add experiment runners for RQ1/RQ2/RQ3 ablation matrices. (implemented)
2. Add baseline forecasting models (seasonal naive, lagged linear, tree-based baseline). (implemented in runner)
3. Add model training/evaluation pipeline for 1-hour ahead station and cluster levels. (implemented)
4. Expand documentation with formulas, assumptions, and reproducibility checklist.

## RQ experiment runner outputs

Run:

```bash
python scripts/experiment_runners.py --rqs RQ1,RQ2,RQ3
```

Output artifacts:

1. `data/artifacts/experiments/rq_runner/experiment_matrix.csv`
2. `data/artifacts/experiments/rq_runner/alpha_search.csv`
3. `data/artifacts/experiments/rq_runner/baseline_search.csv`
4. `data/artifacts/experiments/rq_runner/results.csv`
5. `data/artifacts/experiments/rq_runner/summary.json`

What it does:

1. Builds ablation matrices for RQ1/RQ2/RQ3 from SD/DE/DC/ATD graph sets.
2. Runs 1-hour ahead graph-propagation model plus seasonal naive, lagged linear, and tree-based baselines at station and community aggregation levels.
3. Tunes alpha for graph propagation and baseline hyperparameters on validation.
4. Reports side-by-side held-out test metrics across all models.

## 1-hour training/evaluation pipeline outputs

Run:

```bash
python scripts/train_eval_pipeline.py --graph-set SD,DE,DC,ATD
```

Output artifacts:

1. `data/artifacts/experiments/train_eval_1h/train_eval_results.csv`
2. `data/artifacts/experiments/train_eval_1h/train_eval_alpha_search.csv`
3. `data/artifacts/experiments/train_eval_1h/train_eval_baseline_search.csv`
4. `data/artifacts/experiments/train_eval_1h/station_cohort_results.csv`
5. `data/artifacts/experiments/train_eval_1h/summary.json`

What it does:

1. Tunes graph-propagation alpha and baseline hyperparameters independently for station and community aggregation.
2. Evaluates 1-hour ahead metrics on validation and held-out test splits for graph, seasonal-naive, linear-lagged, and tree-lagged models.
3. Computes station-level cohort metrics for Helsinki/Espoo and sparse/dense station groups.

## Phase 3 Starter (M3.1) - single-graph ST-GNN baseline

Run:

```bash
python scripts/train_stgnn_pipeline.py --aggregation station --graph DE --history 24
```

Output artifacts:

1. `data/artifacts/experiments/stgnn_single_graph/training_curve.csv`
2. `data/artifacts/experiments/stgnn_single_graph/results.csv`
3. `data/artifacts/experiments/stgnn_single_graph/summary.json`

What it does:

1. Builds leakage-safe supervised windows from split-local hourly demand tensors.
2. Trains a lightweight A3T-GCN-style single-graph model with temporal attention and early stopping.
3. Reports validation/test WMAPE, MAE, RMSE, and MASE using the same evaluation conventions as prior experiment runners.
