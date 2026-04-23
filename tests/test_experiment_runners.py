from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_runners import (
    aggregate_adjacency_to_groups,
    build_experiment_specs,
    build_one_step_lag_features,
    evaluate_one_step_forecast,
    parse_alpha_grid,
    parse_lag_candidates,
    row_normalize,
)
from scripts.experiments.contracts import REQUIRED_PREPROCESSING_LINEAGE_FIELDS
from scripts.experiments.pipeline import run as run_rq_pipeline


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


def _write_split_csv(
    path: Path, *, start: str, periods: int, stations: list[str]
) -> None:
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    rows = []
    for ts in timestamps:
        for station_idx, station in enumerate(stations):
            repeats = 1 + ((ts.hour + station_idx) % 3)
            for _ in range(repeats):
                rows.append(
                    {
                        "departure": ts.isoformat(),
                        "departure_name": station,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_rq_runner_emits_preprocessing_lineage_and_original_scale_metrics(
    tmp_path: Path,
) -> None:
    stations = ["A", "B"]
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "out"

    _write_split_csv(
        train_path, start="2025-01-01 00:00:00", periods=72, stations=stations
    )
    _write_split_csv(
        val_path, start="2025-01-04 00:00:00", periods=48, stations=stations
    )
    _write_split_csv(
        test_path, start="2025-01-06 00:00:00", periods=48, stations=stations
    )

    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "station_index.txt").write_text("A\nB\n", encoding="utf-8")
    for name in ("SD", "DE", "DC", "ATD"):
        np.save(
            graph_dir / f"{name}.npy", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        )

    args = Namespace(
        train=train_path,
        validation=val_path,
        test=test_path,
        graph_dir=graph_dir,
        communities=tmp_path / "communities.csv",
        output_dir=output_dir,
        rqs="RQ1",
        alpha_grid="0.0,0.5,1.0",
        seasonal_lags="1,24",
        linear_lag_candidates="1|1,24",
        tree_lag_candidates="1,24",
        tree_max_depths="4",
        tree_estimators=10,
        linear_max_samples=5000,
        tree_max_samples=5000,
        preprocess_target=True,
        winsor_lower_quantile=0.005,
        winsor_upper_quantile=0.995,
        preprocess_scaler="robust",
        residualize_target=True,
        residual_lag_candidates="24,168",
        holiday_country="FI",
        holiday_subdivision="18",
        holiday_national_only=False,
        random_state=42,
        progress=False,
        strict_graph_source=False,
        generate_only=False,
    )

    exit_code = run_rq_pipeline(args)
    assert exit_code == 0

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "preprocessing" in metadata
    for field in REQUIRED_PREPROCESSING_LINEAGE_FIELDS:
        assert field in metadata["preprocessing"]

    results = pd.read_csv(output_dir / "results.csv")
    assert "preprocessing_enabled" in results.columns
    assert bool(results["preprocessing_enabled"].all())
    assert "selected_residual_lag" in results.columns
    assert results["selected_residual_lag"].notna().all()
    assert results["test_wmape"].notna().all()
