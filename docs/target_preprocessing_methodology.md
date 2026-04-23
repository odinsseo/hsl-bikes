# Target Preprocessing Methodology

This document defines the shared target preprocessing stack used by ST-GNN and baseline models.

## Transform Sequence

The fitted transform is applied per node (station or community) in this fixed order:

1. Winsorization with train-fitted quantiles (default `q0.5` and `q99.5`).
2. `log1p` to reduce heavy-tail skew while preserving zero counts.
3. Robust scaling with train-fitted median and IQR.
4. Optional seasonal residualization (`y - seasonal_baseline(y, lag)`) with lag selected from `{24, 168}` by validation MAE in transformed space.

All transform parameters are fit on the train split only and reused for validation/test/inference.

## Leakage Policy

Leakage is prevented with two controls:

1. Graph source guard: train split path must match graph metadata source in strict mode.
2. Transform fit guard: preprocessing parameters are fit from train only, then applied to validation/test without refit.

The metadata sidecar stores `train_only_fit: true` and split/time lineage fields for review.

## Holiday And Calendar Features

Dynamic covariates include:

1. `hour_sin`, `hour_cos`
2. `dow_sin`, `dow_cos`
3. `is_weekend`
4. `is_holiday`

Default holiday source is Finland + Uusimaa (`holidays.country_holidays("FI", subdiv="18")`).
For sensitivity checks, national-only holidays are available via `--holiday-national-only`.

## Static And Sparse Features

Static node context features are computed from train split only:

1. `train_mean`
2. `train_variance`
3. `train_zero_rate`

Sparse activity channels include:

1. `recent_activity_mask` (default enabled)
2. `long_zero_run_indicator` (optional)

## Reporting Policy

Model fitting and tuning may happen in transformed space, but final scientific/reporting metrics are always computed on the original bike-count scale.

This is done by inverse-transforming predictions before WMAPE/MAE/RMSE/MASE computation.

## Reviewer Sidecar Checklist

Check `metadata.json` for:

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
