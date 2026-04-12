from __future__ import annotations

import numpy as np

from scripts.experiment_runners import (
    aggregate_adjacency_to_groups,
    build_experiment_specs,
    build_one_step_lag_features,
    evaluate_one_step_forecast,
    parse_alpha_grid,
    parse_lag_candidates,
    row_normalize,
)


def test_parse_alpha_grid_is_sorted_and_unique() -> None:
    grid = parse_alpha_grid("1.0,0.5,0.5,0.0")
    assert grid == [0.0, 0.5, 1.0]


def test_parse_lag_candidates_preserves_order_and_uniqueness() -> None:
    candidates = parse_lag_candidates("1|1,24|1|1,2,24")
    assert candidates == [(1,), (1, 24), (1, 2, 24)]


def test_row_normalize_preserves_zero_rows() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 3.0]], dtype=float)
    normalized = row_normalize(matrix)

    assert np.allclose(normalized[0], [0.0, 0.0])
    assert np.allclose(normalized[1], [0.25, 0.75])


def test_build_experiment_specs_covers_all_selected_rqs() -> None:
    specs = build_experiment_specs({"RQ1", "RQ2", "RQ3"})
    rq_values = {spec.rq for spec in specs}

    assert rq_values == {"RQ1", "RQ2", "RQ3"}
    assert len(specs) >= 8


def test_aggregate_adjacency_to_groups_output_is_row_normalized() -> None:
    station_index = ["A", "B", "C"]
    station_to_group = {"A": "G1", "B": "G1", "C": "G2"}
    groups = ["G1", "G2"]

    adjacency = np.array(
        [
            [0.0, 2.0, 4.0],
            [1.0, 0.0, 3.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    grouped = aggregate_adjacency_to_groups(
        adjacency=adjacency,
        station_index=station_index,
        station_to_group=station_to_group,
        groups=groups,
    )

    assert grouped.shape == (2, 2)
    assert np.allclose(grouped.sum(axis=1), [1.0, 1.0])


def test_evaluate_one_step_forecast_returns_finite_metrics() -> None:
    series = np.array(
        [
            [10.0, 5.0],
            [11.0, 6.0],
            [12.0, 7.0],
        ]
    )
    train_series = np.array(
        [
            [8.0, 4.0],
            [9.0, 4.5],
            [10.0, 5.0],
        ]
    )
    adjacency = np.array(
        [
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )

    metrics = evaluate_one_step_forecast(
        series=series,
        adjacency=adjacency,
        alpha=0.5,
        train_series=train_series,
    )

    assert np.isfinite(metrics["wmape"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])


def test_build_one_step_lag_features_shape() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    features = build_one_step_lag_features(series, lags=(1, 2))

    # (T - 1) * N rows and len(lags) columns
    assert features.shape == (6, 2)
