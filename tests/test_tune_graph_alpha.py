from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.models import (
    compute_metrics,
    predict_graph_propagation,
    tune_graph_alpha,
)
from scripts.experiments.preprocessing import (
    TargetPreprocessingConfig,
    apply_target_preprocessing,
    fit_target_preprocessing,
    inverse_target_predictions,
)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    out[mask] = out[mask] / row_sums[mask]
    return out


def test_tune_graph_alpha_inverse_without_raw_raises() -> None:
    train_z = np.ones((20, 2))
    val_z = np.ones((15, 2))
    a = np.eye(2)
    state, _ = fit_target_preprocessing(
        np.ones((20, 2)) * 5.0,
        validation_series=np.ones((15, 2)) * 5.0,
        config=TargetPreprocessingConfig(
            winsor_lower_quantile=0.0,
            winsor_upper_quantile=1.0,
            enable_residualization=False,
        ),
    )
    with pytest.raises(ValueError, match="val_series_raw"):
        tune_graph_alpha(
            train_z,
            val_z,
            a,
            [0.5],
            inverse_state=state,
            val_pre_residual=np.ones((15, 2)),
        )


def test_tune_graph_alpha_original_scale_rows_match_manual_inverse() -> None:
    rng = np.random.default_rng(0)
    train_raw = rng.uniform(10.0, 50.0, size=(80, 3))
    val_raw = rng.uniform(10.0, 50.0, size=(40, 3))
    cfg = TargetPreprocessingConfig(
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        enable_residualization=False,
    )
    state, _ = fit_target_preprocessing(
        train_raw, validation_series=val_raw, config=cfg
    )
    train_app = apply_target_preprocessing(train_raw, state)
    val_app = apply_target_preprocessing(val_raw, state)
    base = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.2, 0.4],
            [0.1, 0.7, 0.2],
        ],
        dtype=float,
    )
    a = _row_normalize(base)

    _best, rows = tune_graph_alpha(
        train_app.transformed,
        val_app.transformed,
        a,
        [0.0, 0.5, 1.0],
        val_series_raw=val_raw,
        train_series_raw=train_raw,
        inverse_state=state,
        val_pre_residual=val_app.pre_residual,
        history=1,
        horizon=1,
    )

    for row in rows:
        alpha = float(row["alpha"])
        pred_z = predict_graph_propagation(val_app.transformed, a, alpha)
        pred_raw = inverse_target_predictions(
            pred_z,
            state=state,
            context_pre_residual=val_app.pre_residual,
            history=1,
            horizon=1,
        )
        m = compute_metrics(
            actual=val_raw[1:],
            pred=pred_raw,
            train_series=train_raw,
        )
        assert np.isclose(row["validation_wmape"], m["wmape"], rtol=0, atol=1e-9)
        assert np.isclose(row["validation_mae"], m["mae"], rtol=0, atol=1e-9)
