# Phase 0-1 Implementation Notes

This document describes the first implementation slice for the HSL multi-graph forecasting project.

## Scope completed

1. Publication quality and artifact contract with lean/efficient implementation rules.
2. Data audit utilities for merged/train/validation/test integrity checks.
3. Leakage-safe graph construction for SD, DE, DC, and ATD views.
4. Network analysis notebook with centrality metrics and Louvain community detection.
5. Unit tests for core functions (data audit, graph construction, network analysis).

## New scripts

1. `scripts/data_audit.py`
2. `scripts/graph_construction.py`
3. `notebooks/network_analysis_city_bikes.ipynb`
4. `scripts/experiment_runners.py`
5. `scripts/train_eval_pipeline.py`
6. `scripts/experiments/provenance.py`
7. `scripts/experiments/safeguards.py`
8. `scripts/experiments/contracts.py`
9. `scripts/validate_experiment_artifacts.py`
10. `scripts/pre_notebook_quality_gate.py`

## Phase 0 quality contract

Reference:

1. `docs/publication_quality_contract.md`

Additions implemented:

1. Metadata sidecar contract (`metadata.json`) for experiment output directories.
2. Graph source leakage safeguards (strict mode by default in experiment pipelines).
3. Deterministic seed initialization in experiment runner and train/eval pipeline entrypoints.
4. Canonical artifact schema validation utility for publication notebook inputs.

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

1. `data/artifacts/experiments/train_eval_3h/train_eval_results.csv`
2. `data/artifacts/experiments/train_eval_3h/train_eval_alpha_search.csv`
3. `data/artifacts/experiments/train_eval_3h/train_eval_baseline_search.csv`
4. `data/artifacts/experiments/train_eval_3h/station_cohort_results.csv`
5. `data/artifacts/experiments/train_eval_3h/station_robustness_statistics.csv`
6. `data/artifacts/experiments/train_eval_3h/sensitivity_summary.csv`
7. `data/artifacts/experiments/train_eval_3h/summary.json`

What it does:

1. Tunes graph-propagation alpha and baseline hyperparameters independently for station and community aggregation.
2. Evaluates 1-hour ahead metrics on validation and held-out test splits for graph, seasonal-naive, linear-lagged, and tree-lagged models.
3. Computes station-level cohort metrics for Helsinki/Espoo and sparse/dense station groups.
4. Computes cohort-stratified bootstrap confidence intervals and paired significance deltas against a reference model.
5. Exports a compact sensitivity table over threshold, aggregation resolution, and hyperparameter search axes.

## Phase 3 Starter (M3.1) - single-graph ST-GNN baseline

Run:

```bash
python scripts/train_stgnn_pipeline.py --aggregation station --graph DE --history 8
```

Output artifacts:

1. `data/artifacts/experiments/stgnn_single_graph/training_curve.csv`
2. `data/artifacts/experiments/stgnn_single_graph/results.csv`
3. `data/artifacts/experiments/stgnn_single_graph/summary.json`

What it does:

1. Builds leakage-safe supervised windows from split-local hourly demand tensors.
2. Trains a lightweight A3T-GCN-style single-graph model with temporal attention and early stopping.
3. Reports validation/test WMAPE, MAE, RMSE, and MASE using the same evaluation conventions as prior experiment runners.

## Phase 3 Expansion (M3.2) - multi-graph ST-GNN fusion

Run:

```bash
python scripts/train_stgnn_pipeline.py --aggregation station --graph-set SD,DE,DC,ATD --fusion-mode learned --history 8
```

Output artifacts:

1. `data/artifacts/experiments/stgnn_single_graph/training_curve.csv`
2. `data/artifacts/experiments/stgnn_single_graph/results.csv`
3. `data/artifacts/experiments/stgnn_single_graph/summary.json`

What it does:

1. Supports equal-weight and learned-weight fusion over multiple graph views.
2. Keeps single-graph mode for direct M3.1 comparisons under one shared pipeline.
3. Exports learned fusion weights to support interpretation in RQ2 analysis.

## Milestone orchestration (M3.1 + M3.2 + leakage sensitivity)

Run:

```bash
python scripts/run_stgnn_milestones.py --include-community --include-leakage-sensitivity --build-leaky-graphs
```

Output artifacts:

1. `data/artifacts/experiments/stgnn_milestones/milestone_results.csv`
2. `data/artifacts/experiments/stgnn_milestones/milestone_best_by_milestone.csv`
3. `data/artifacts/experiments/stgnn_milestones/milestone_best_by_graph_set.csv`
4. `data/artifacts/experiments/stgnn_milestones/leakage_sensitivity.csv`
5. `data/artifacts/experiments/stgnn_milestones/summary.json`

What it does:

1. Runs M3.1 single-graph sweeps over SD/DE/DC/ATD.
2. Runs M3.2 equal-fusion and learned-fusion experiments on multi-graph sets.
3. Generates a compact leakage-sensitivity report by comparing train-only graph artifacts against intentionally leaky full-period graph artifacts.

## Phase 3 notebook suite (strict RQ scope)

Notebooks:

1. `notebooks/rq1_functional_vs_geometric.ipynb`
2. `notebooks/rq2_fusion_heterogeneity.ipynb`
3. `notebooks/rq3_cluster_robustness.ipynb`
4. `notebooks/rq_synthesis.ipynb`

What it does:

1. Keeps one notebook per RQ plus one synthesis notebook.
2. Restricts each notebook to artifact consumption (no heavy model retraining in notebook cells).
3. Uses shared `scripts/notebook_reporting.py` utilities to avoid duplicate loading/parsing logic.

## Phase 4 notebook reliability and lock

Run:

```bash
python scripts/validate_notebook_suite.py --timeout-seconds 1800
```

Output artifacts:

1. `data/artifacts/notebooks/validation_manifest.json`
2. `data/artifacts/notebooks/executed/executed_rq1_functional_vs_geometric.ipynb`
3. `data/artifacts/notebooks/executed/executed_rq2_fusion_heterogeneity.ipynb`
4. `data/artifacts/notebooks/executed/executed_rq3_cluster_robustness.ipynb`
5. `data/artifacts/notebooks/executed/executed_rq_synthesis.ipynb`

What it does:

1. Executes each publication notebook in a clean kernel.
2. Captures pass/fail status with stderr/stdout tails in a single lock manifest.
3. Provides a final evidence-lock gate before thesis submission packaging.

## Reproducibility runbook

Use `docs/reproducibility_runbook.md` for exact command order, runtime bands, and claim-to-artifact mapping.
