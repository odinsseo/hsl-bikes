# Reproducibility Runbook (Publication Notebook Suite)

This runbook defines the exact command order used to regenerate artifacts, execute notebook evidence, and lock final outputs.

Target transform methodology reference: `docs/target_preprocessing_methodology.md`.

## Preconditions

1. Activate project environment.
2. Install dependencies from requirements.
3. Ensure source data is available under data/primary.

## Command Order

1. Data preparation and audit.

```bash
python scripts/prepare_data.py
python scripts/data_audit.py
```

Expected runtime band: 5-20 minutes depending on raw file count and storage speed.

1. Graph construction.

```bash
python scripts/graph_construction.py
```

Expected runtime band: 3-15 minutes depending on trip volume.

1. RQ ablation pipeline artifacts.

```bash
python scripts/experiment_runners.py --rqs RQ1,RQ2,RQ3
```

Expected runtime band: 5-25 minutes depending on baseline search space.

1. Train/eval robustness + sensitivity artifacts.

```bash
python scripts/train_eval_pipeline.py --graph-set SD,DE,DC,ATD
```

Expected runtime band: 5-30 minutes depending on split length and model search options.

1. ST-GNN milestones and leakage sensitivity.

```bash
python scripts/run_stgnn_milestones.py --include-community --include-leakage-sensitivity --build-leaky-graphs
```

Expected runtime band: 20-120 minutes depending on hardware and ST-GNN hyperparameters.

1. Canonical experiment contract gate.

```bash
python scripts/pre_notebook_quality_gate.py
```

This must pass before notebook rendering.

The gate enforces strict preprocessing lineage by default. Use `--no-require-preprocessing-lineage` only for draft/debug runs.

1. Notebook suite validation in clean kernels.

```bash
python scripts/validate_notebook_suite.py --timeout-seconds 1800
```

This writes:

- data/artifacts/notebooks/validation_manifest.json
- data/artifacts/notebooks/executed/executed_rq1_functional_vs_geometric.ipynb
- data/artifacts/notebooks/executed/executed_rq2_fusion_heterogeneity.ipynb
- data/artifacts/notebooks/executed/executed_rq3_cluster_robustness.ipynb
- data/artifacts/notebooks/executed/executed_rq_synthesis.ipynb

## Claim-to-Artifact Mapping

- RQ1 notebook: functional vs geometric evidence from data/artifacts/experiments/rq_runner/results.csv, uncertainty support from data/artifacts/experiments/train_eval_1h/station_robustness_statistics.csv.
- RQ2 notebook: fusion ablations from data/artifacts/experiments/rq_runner/results.csv, learned fusion interpretation from data/artifacts/experiments/stgnn_milestones/milestone_results.csv.
- RQ3 notebook: station vs community trade-offs from data/artifacts/experiments/rq_runner/results.csv, sparse/suburban failure slices from data/artifacts/experiments/train_eval_1h/station_cohort_results.csv.
- Synthesis notebook: conclusion matrix aggregated from the three RQ evidence streams above, plus milestone and robustness optional context layers.

## Final Lock Criteria

1. pre_notebook_quality_gate.py exits with code 0.
2. validate_notebook_suite.py exits with code 0.
3. validation_manifest.json overall_status is passed.
4. Executed notebooks are archived with the same git commit as experiment artifacts.

## Sidecar Inspection Checklist

Before freezing notebook claims, inspect `metadata.json` in each canonical experiment directory and confirm:

1. `preprocessing.fitted_split == "train"`
2. `preprocessing.train_only_fit == true`
3. `preprocessing.train_time_bounds` is present
4. `preprocessing.quantile_bounds` and `preprocessing.scaler_type` are present
5. `preprocessing.residual_lag_policy` and `preprocessing.selected_residual_lag` are present
6. `preprocessing.calendar_source` documents country/subdivision
