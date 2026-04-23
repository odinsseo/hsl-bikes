# Publication Quality And Artifact Contract

This document defines minimum quality gates for publication-grade notebook evidence.

## Goals

1. Make every major claim traceable to a stable artifact row/figure.
2. Ensure deterministic reruns with explicit provenance metadata.
3. Enforce leakage-safe graph sources for all non-leakage experiments.
4. Keep implementation lean and efficient by preferring reusable, optimized libraries.
5. Enforce train-fitted preprocessing lineage and inverse-scale reporting discipline.

## Required Metadata Sidecar

Every experiment output directory consumed by notebooks must include `metadata.json`
with at least:

1. `generated_at_utc`
2. `git_commit`
3. `command`
4. `script`
5. `args`
6. `stage`

Optional but recommended:

1. `python_version`
2. `platform`
3. `strict_graph_source`

Required for publication-quality transform lineage:

1. `preprocessing.preprocessing_version`
2. `preprocessing.fitted_split`
3. `preprocessing.train_time_bounds`
4. `preprocessing.quantile_bounds`
5. `preprocessing.scaler_type`
6. `preprocessing.residual_lag_policy`
7. `preprocessing.selected_residual_lag`
8. `preprocessing.calendar_source`
9. `preprocessing.dynamic_feature_definitions`
10. `preprocessing.static_feature_definitions`
11. `preprocessing.sparse_feature_definitions`
12. `preprocessing.train_only_fit`

## Leakage Guard Policy

1. Default mode requires graph bundle metadata source to match the train split path.
2. Leaky graph sources are only allowed for explicitly marked leakage sensitivity runs.
3. Missing graph metadata in strict mode is a hard error.
4. Missing preprocessing lineage in strict contract mode is a hard error.

## Reporting Scale Policy

1. Model fitting/tuning may occur in transformed target space.
2. Final WMAPE/MAE/RMSE/MASE reported in canonical result files must be computed on original bike-count units after inverse-transform.
3. Transformed-space metrics are secondary diagnostics and must be clearly labeled when present.

## Artifact Naming Contract

The following outputs are considered canonical for notebook consumption:

1. `data/artifacts/experiments/rq_runner/results.csv`
2. `data/artifacts/experiments/train_eval_1h/train_eval_results.csv`
3. `data/artifacts/experiments/stgnn_single_graph/results.csv`
4. `data/artifacts/experiments/stgnn_milestones/milestone_results.csv`

Each corresponding output directory must also contain:

1. `summary.json`
2. `metadata.json`

And each canonical results CSV must include at least one data row by default (empty tables are only allowed for explicitly marked draft/incomplete runs).

## Lean Implementation Rules

1. Reuse shared pipeline modules rather than duplicating notebook logic.
2. Prefer vectorized and library-native operations over manual loops where practical.
3. Only add custom algorithms when no reliable library primitive exists.
4. Keep public APIs small and composable.
