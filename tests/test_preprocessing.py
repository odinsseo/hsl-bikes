from __future__ import annotations

from datetime import datetime

import numpy as np

from scripts.experiments.preprocessing import (
    TargetPreprocessingConfig,
    apply_target_preprocessing,
    build_calendar_feature_matrix,
    build_holiday_flags,
    build_preprocessing_metadata,
    build_sparse_activity_features,
    build_static_feature_matrix,
    fit_target_preprocessing,
    inverse_target_predictions,
)


def test_inverse_target_predictions_round_trip_without_residualization() -> None:
    rng = np.random.default_rng(7)
    series = rng.uniform(5.0, 25.0, size=(96, 4))

    cfg = TargetPreprocessingConfig(
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        enable_residualization=False,
    )
    state, _ = fit_target_preprocessing(series, validation_series=series, config=cfg)
    applied = apply_target_preprocessing(series, state)

    history = 6
    horizon = 1
    sample_count = series.shape[0] - history - horizon + 1
    target_idx = np.arange(sample_count) + history + horizon - 1

    model_space_targets = applied.transformed[target_idx]
    recovered = inverse_target_predictions(
        model_space_targets,
        state=state,
        context_pre_residual=applied.pre_residual,
        history=history,
        horizon=horizon,
    )

    assert recovered.shape == model_space_targets.shape
    assert np.allclose(recovered, series[target_idx], atol=1e-6)


def test_fit_target_preprocessing_uses_train_split_only() -> None:
    train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
    )
    validation = np.array(
        [
            [1000.0, 2000.0],
            [900.0, 1800.0],
            [1100.0, 2200.0],
            [1000.0, 2100.0],
        ]
    )

    cfg = TargetPreprocessingConfig(
        winsor_lower_quantile=0.0,
        winsor_upper_quantile=1.0,
        enable_residualization=False,
    )
    state, _ = fit_target_preprocessing(train, validation_series=validation, config=cfg)

    assert np.allclose(state.upper_bounds, train.max(axis=0))
    assert np.allclose(state.lower_bounds, train.min(axis=0))


def test_fit_target_preprocessing_selects_best_residual_lag() -> None:
    t = np.arange(24 * 20, dtype=float)
    base = 50.0 + (0.1 * t) + 4.0 * np.sin(2.0 * np.pi * t / 24.0)

    train = np.column_stack([base, base * 0.8 + 2.0])
    val = np.column_stack([base[: 24 * 10], base[: 24 * 10] * 0.8 + 2.0])

    cfg = TargetPreprocessingConfig(
        enable_residualization=True,
        residual_lag_candidates=(24, 168),
    )
    state, lag_scores = fit_target_preprocessing(
        train, validation_series=val, config=cfg
    )

    assert state.selected_residual_lag == 24
    assert {row.lag for row in lag_scores} == {24, 168}


def test_build_holiday_flags_for_finland_subdivision() -> None:
    timestamps = [
        datetime(2025, 12, 6, 8, 0, 0),
        datetime(2025, 12, 7, 8, 0, 0),
    ]

    flags = build_holiday_flags(
        timestamps,
        country="FI",
        subdivision="18",
    )

    assert flags.shape == (2,)
    assert flags[0] == 1.0


def test_build_calendar_feature_matrix_shape_and_names() -> None:
    timestamps = [
        datetime(2025, 1, 1, 0, 0, 0),
        datetime(2025, 1, 1, 1, 0, 0),
    ]

    matrix, names = build_calendar_feature_matrix(
        timestamps,
        country="FI",
        subdivision="18",
    )

    assert matrix.shape == (2, 6)
    assert names == [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
    ]


def test_build_static_feature_matrix_computes_expected_columns() -> None:
    train = np.array(
        [
            [0.0, 2.0],
            [1.0, 2.0],
            [0.0, 6.0],
        ]
    )

    matrix, names = build_static_feature_matrix(train)

    assert matrix.shape == (2, 3)
    assert names == ["train_mean", "train_variance", "train_zero_rate"]
    assert np.all((matrix[:, :2] >= 0.0) & (matrix[:, :2] <= 1.0))
    assert np.isclose(matrix[0, 0], 0.0)
    assert np.isclose(matrix[1, 0], 1.0)
    assert np.isclose(matrix[0, 1], 0.0)
    assert np.isclose(matrix[1, 1], 1.0)
    assert np.isclose(matrix[0, 2], 2.0 / 3.0)


def test_build_sparse_activity_features_activity_and_zero_run() -> None:
    series = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )

    matrix, names = build_sparse_activity_features(
        series,
        include_activity_mask=True,
        include_zero_run_indicator=True,
        zero_run_length=2,
    )

    assert matrix.shape == (3, 2, 2)
    assert names == ["recent_activity_mask", "long_zero_run_indicator"]
    assert np.isclose(matrix[1, 0, 1], 1.0)


def test_build_preprocessing_metadata_includes_lineage_fields() -> None:
    train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    cfg = TargetPreprocessingConfig(enable_residualization=False)
    state, _ = fit_target_preprocessing(train, validation_series=train, config=cfg)

    metadata = build_preprocessing_metadata(
        state,
        train_time_bounds=("2025-01-01T00:00:00", "2025-01-01T02:00:00"),
        dynamic_feature_definitions=["hour_sin"],
        static_feature_definitions=["train_mean"],
        sparse_feature_definitions=["recent_activity_mask"],
    )

    assert metadata["train_time_bounds"]["start"] == "2025-01-01T00:00:00"
    assert metadata["dynamic_feature_definitions"] == ["hour_sin"]
    assert metadata["static_feature_definitions"] == ["train_mean"]
    assert metadata["sparse_feature_definitions"] == ["recent_activity_mask"]
