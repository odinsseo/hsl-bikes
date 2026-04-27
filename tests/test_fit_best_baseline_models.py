from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from scripts.experiments.models import (
    compute_metrics,
    fit_best_baseline_models,
    predict_baseline,
)
from scripts.experiments.preprocessing import (
    TargetPreprocessingConfig,
    apply_target_preprocessing,
    fit_target_preprocessing,
    inverse_target_predictions,
)


def test_fit_best_baseline_inverse_without_raw_raises() -> None:
    train_z = np.ones((20, 2))
    val_z = np.ones((15, 2))
    state, _ = fit_target_preprocessing(
        np.ones((20, 2)) * 5.0,
        validation_series=np.ones((15, 2)) * 5.0,
        config=TargetPreprocessingConfig(
            winsor_lower_quantile=0.0,
            winsor_upper_quantile=1.0,
            enable_residualization=False,
        ),
    )
    args = Namespace(
        seasonal_lags="1",
        linear_lag_candidates="1|1,2",
        tree_lag_candidates="1,2|1,2,3",
        tree_max_depths="4",
        tree_estimators=10,
        linear_max_samples=5000,
        tree_max_samples=5000,
        random_state=0,
    )
    with pytest.raises(ValueError, match="val_series_raw"):
        fit_best_baseline_models(
            train_z,
            val_z,
            args,
            inverse_state=state,
            val_pre_residual=np.ones((15, 2)),
        )


def test_fit_best_baseline_search_rows_match_inverse_val_metrics() -> None:
    rng = np.random.default_rng(1)
    train_raw = rng.uniform(8.0, 40.0, size=(60, 2))
    val_raw = rng.uniform(8.0, 40.0, size=(32, 2))
    cfg = TargetPreprocessingConfig(
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        enable_residualization=False,
    )
    state, _ = fit_target_preprocessing(
        train_raw, validation_series=val_raw, config=cfg
    )
    tr = apply_target_preprocessing(train_raw, state)
    va = apply_target_preprocessing(val_raw, state)

    args = Namespace(
        seasonal_lags="1,2",
        linear_lag_candidates="1|1,2",
        tree_lag_candidates="1,2|1,2,3",
        tree_max_depths="4,8",
        tree_estimators=10,
        linear_max_samples=2000,
        tree_max_samples=2000,
        random_state=0,
    )

    _fitted, search_rows = fit_best_baseline_models(
        tr.transformed,
        va.transformed,
        args,
        val_series_raw=val_raw,
        train_series_raw=train_raw,
        inverse_state=state,
        val_pre_residual=va.pre_residual,
        history=1,
        horizon=1,
    )

    for row in search_rows:
        model = row["model"]
        if model == "seasonal_naive":
            import json

            lag = json.loads(row["config"])["seasonal_lag"]
            spec = {
                "model": "seasonal_naive",
                "seasonal_lag": lag,
                "config": {"seasonal_lag": lag},
            }
            pred_z = predict_baseline(spec, va.transformed)
            pred = inverse_target_predictions(
                pred_z,
                state=state,
                context_pre_residual=va.pre_residual,
                history=1,
                horizon=1,
            )
            m = compute_metrics(
                val_raw[1:], pred, train_raw
            )
            assert np.isclose(row["validation_wmape"], m["wmape"], atol=1e-9)
            assert np.isclose(row["validation_mae"], m["mae"], atol=1e-9)
