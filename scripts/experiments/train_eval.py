from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import (
    DATA_DIR,
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    parse_alpha_grid,
)
from .data import (
    build_community_series,
    build_fused_adjacency,
    build_station_series,
    load_communities,
    load_graph_bundle,
    load_split,
)
from .models import (
    compute_metrics,
    fit_best_baseline_models,
    predict_baseline,
    predict_graph_propagation,
    tune_graph_alpha,
)

DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "experiments" / "train_eval_1h"
DEFAULT_STATIONS_DIR = DATA_DIR / "primary" / "stations"
ALLOWED_GRAPH_NAMES = {"SD", "DE", "DC", "ATD"}


def normalize_city_name(value: str | None) -> str:
    if value is None:
        return "Helsinki"
    text = str(value).strip()
    if not text:
        return "Helsinki"

    lowered = text.lower()
    if lowered.startswith("espoo") or lowered.startswith("esbo"):
        return "Espoo"
    return "Helsinki"


def load_station_city_lookup(stations_dir: Path) -> dict[str, str]:
    station_csvs = sorted(stations_dir.glob("*.csv"))
    if not station_csvs:
        raise FileNotFoundError(f"No station CSV found in: {stations_dir}")

    df = pl.read_csv(station_csvs[0], try_parse_dates=False)
    city_col = "Kaupunki" if "Kaupunki" in df.columns else "Stad"
    if city_col not in df.columns:
        raise ValueError("Stations CSV missing city column (Kaupunki/Stad)")

    mapping: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        city = normalize_city_name(row.get(city_col))
        for name_col in ("Nimi", "Name"):
            if name_col not in df.columns:
                continue
            raw_name = row.get(name_col)
            if raw_name is None:
                continue
            name = str(raw_name).strip()
            if name:
                mapping[name] = city

    if not mapping:
        raise ValueError("No station names were parsed from station CSV")
    return mapping


def parse_graph_set(text: str) -> tuple[str, ...]:
    names = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    if not names:
        raise ValueError("graph-set is empty")

    unknown = [name for name in names if name not in ALLOWED_GRAPH_NAMES]
    if unknown:
        raise ValueError(f"Unknown graph(s): {sorted(set(unknown))}")

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return tuple(deduped)


def build_station_cohort_indices(
    train_series: np.ndarray,
    station_index: list[str],
    city_lookup: dict[str, str],
    sparse_quantile: float,
) -> dict[str, np.ndarray]:
    if train_series.ndim != 2:
        raise ValueError("train_series must be 2-D [time, station]")
    if train_series.shape[1] != len(station_index):
        raise ValueError("station_index length does not match train_series width")
    if not (0.0 <= sparse_quantile <= 1.0):
        raise ValueError("sparse_quantile must be in [0, 1]")

    mean_demand = (
        train_series.mean(axis=0)
        if train_series.shape[0] > 0
        else np.zeros(len(station_index))
    )
    sparse_threshold = float(np.quantile(mean_demand, sparse_quantile))
    sparse_mask = mean_demand <= sparse_threshold

    city_labels = np.array(
        [normalize_city_name(city_lookup.get(station)) for station in station_index],
        dtype=object,
    )
    helsinki_mask = city_labels == "Helsinki"
    espoo_mask = city_labels == "Espoo"

    def as_idx(mask: np.ndarray) -> np.ndarray:
        return np.where(mask)[0].astype(int)

    cohorts = {
        "all": as_idx(np.ones(len(station_index), dtype=bool)),
        "helsinki": as_idx(helsinki_mask),
        "espoo": as_idx(espoo_mask),
        "sparse": as_idx(sparse_mask),
        "dense": as_idx(~sparse_mask),
        "sparse_helsinki": as_idx(sparse_mask & helsinki_mask),
        "sparse_espoo": as_idx(sparse_mask & espoo_mask),
    }
    return cohorts


def metrics_for_indices(
    actual: np.ndarray,
    pred: np.ndarray,
    train_series: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    if indices.size == 0:
        return {"wmape": np.nan, "mae": np.nan, "rmse": np.nan, "mase": np.nan}

    return compute_metrics(
        actual=actual[:, indices],
        pred=pred[:, indices],
        train_series=train_series[:, indices],
    )


def evaluate_aggregation(
    aggregation: str,
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    adjacency: np.ndarray,
    alpha_grid: list[float],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    results: list[dict[str, Any]] = []
    alpha_search: list[dict[str, Any]] = []
    baseline_search: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    best_alpha, graph_search = tune_graph_alpha(
        train_series=train_series,
        val_series=val_series,
        adjacency=adjacency,
        alpha_grid=alpha_grid,
    )
    for row in graph_search:
        alpha_search.append({"aggregation": aggregation, **row})

    graph_val_pred = predict_graph_propagation(
        series=val_series,
        adjacency=adjacency,
        alpha=best_alpha,
    )
    graph_test_pred = predict_graph_propagation(
        series=test_series,
        adjacency=adjacency,
        alpha=best_alpha,
    )
    graph_val_metrics = compute_metrics(
        actual=val_series[1:],
        pred=graph_val_pred,
        train_series=train_series,
    )
    graph_test_metrics = compute_metrics(
        actual=test_series[1:],
        pred=graph_test_pred,
        train_series=train_series,
    )
    predictions["graph_propagation"] = (graph_val_pred, graph_test_pred)
    results.append(
        {
            "aggregation": aggregation,
            "model": "graph_propagation",
            "config": json.dumps({"alpha": best_alpha}),
            "n_nodes": int(train_series.shape[1]),
            "n_train_steps": int(train_series.shape[0]),
            "n_validation_steps": int(val_series.shape[0]),
            "n_test_steps": int(test_series.shape[0]),
            "validation_wmape": graph_val_metrics["wmape"],
            "validation_mae": graph_val_metrics["mae"],
            "validation_rmse": graph_val_metrics["rmse"],
            "validation_mase": graph_val_metrics["mase"],
            "test_wmape": graph_test_metrics["wmape"],
            "test_mae": graph_test_metrics["mae"],
            "test_rmse": graph_test_metrics["rmse"],
            "test_mase": graph_test_metrics["mase"],
        }
    )

    fitted_baselines, baseline_search_rows = fit_best_baseline_models(
        train_series=train_series,
        val_series=val_series,
        args=args,
    )
    for row in baseline_search_rows:
        baseline_search.append({"aggregation": aggregation, **row})

    for model_name in ("seasonal_naive", "lagged_linear", "tree_lagged"):
        if model_name not in fitted_baselines:
            continue

        model_spec = fitted_baselines[model_name]
        val_pred = predict_baseline(model_spec, val_series)
        test_pred = predict_baseline(model_spec, test_series)
        predictions[model_name] = (val_pred, test_pred)

        val_metrics = compute_metrics(
            actual=val_series[1:],
            pred=val_pred,
            train_series=train_series,
        )
        test_metrics = compute_metrics(
            actual=test_series[1:],
            pred=test_pred,
            train_series=train_series,
        )
        results.append(
            {
                "aggregation": aggregation,
                "model": model_name,
                "config": json.dumps(model_spec["config"]),
                "n_nodes": int(train_series.shape[1]),
                "n_train_steps": int(train_series.shape[0]),
                "n_validation_steps": int(val_series.shape[0]),
                "n_test_steps": int(test_series.shape[0]),
                "validation_wmape": val_metrics["wmape"],
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mase": val_metrics["mase"],
                "test_wmape": test_metrics["wmape"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mase": test_metrics["mase"],
            }
        )

    return results, alpha_search, baseline_search, predictions


def run(args: argparse.Namespace) -> int:
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    graph_set = parse_graph_set(args.graph_set)

    station_index, matrices = load_graph_bundle(args.graph_dir)
    train_df = load_split(args.train)
    val_df = load_split(args.validation)
    test_df = load_split(args.test)

    train_station = build_station_series(train_df, station_index)
    val_station = build_station_series(val_df, station_index)
    test_station = build_station_series(test_df, station_index)

    station_adj = build_fused_adjacency(
        graph_set=graph_set,
        aggregation="station",
        matrices=matrices,
        station_index=station_index,
        station_to_group=None,
        groups=None,
    )

    results_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    station_results, station_alpha, station_baseline, station_predictions = (
        evaluate_aggregation(
            aggregation="station",
            train_series=train_station,
            val_series=val_station,
            test_series=test_station,
            adjacency=station_adj,
            alpha_grid=alpha_grid,
            args=args,
        )
    )
    results_rows.extend(station_results)
    alpha_rows.extend(station_alpha)
    baseline_rows.extend(station_baseline)

    cohort_rows: list[dict[str, Any]] = []
    city_lookup = load_station_city_lookup(args.stations_dir)
    cohorts = build_station_cohort_indices(
        train_series=train_station,
        station_index=station_index,
        city_lookup=city_lookup,
        sparse_quantile=args.sparse_quantile,
    )
    station_actual_val = val_station[1:]
    station_actual_test = test_station[1:]

    for model_name, (val_pred, test_pred) in station_predictions.items():
        for cohort_name, idx in cohorts.items():
            val_metrics = metrics_for_indices(
                actual=station_actual_val,
                pred=val_pred,
                train_series=train_station,
                indices=idx,
            )
            test_metrics = metrics_for_indices(
                actual=station_actual_test,
                pred=test_pred,
                train_series=train_station,
                indices=idx,
            )
            cohort_rows.append(
                {
                    "aggregation": "station",
                    "model": model_name,
                    "cohort": cohort_name,
                    "n_nodes": int(idx.size),
                    "validation_wmape": val_metrics["wmape"],
                    "validation_mae": val_metrics["mae"],
                    "validation_rmse": val_metrics["rmse"],
                    "validation_mase": val_metrics["mase"],
                    "test_wmape": test_metrics["wmape"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mase": test_metrics["mase"],
                }
            )

    if not args.disable_community:
        station_to_group = load_communities(args.communities, station_index)
        groups = sorted(set(station_to_group.values()))

        train_comm = build_community_series(train_df, station_to_group, groups)
        val_comm = build_community_series(val_df, station_to_group, groups)
        test_comm = build_community_series(test_df, station_to_group, groups)

        comm_adj = build_fused_adjacency(
            graph_set=graph_set,
            aggregation="community",
            matrices=matrices,
            station_index=station_index,
            station_to_group=station_to_group,
            groups=groups,
        )

        comm_results, comm_alpha, comm_baseline, _ = evaluate_aggregation(
            aggregation="community",
            train_series=train_comm,
            val_series=val_comm,
            test_series=test_comm,
            adjacency=comm_adj,
            alpha_grid=alpha_grid,
            args=args,
        )
        results_rows.extend(comm_results)
        alpha_rows.extend(comm_alpha)
        baseline_rows.extend(comm_baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "train_eval_results.csv"
    alpha_path = args.output_dir / "train_eval_alpha_search.csv"
    baseline_path = args.output_dir / "train_eval_baseline_search.csv"
    cohort_path = args.output_dir / "station_cohort_results.csv"
    summary_path = args.output_dir / "summary.json"

    results_df = (
        pl.DataFrame(results_rows).sort(["aggregation", "model", "test_wmape"])
        if results_rows
        else pl.DataFrame()
    )
    alpha_df = (
        pl.DataFrame(alpha_rows).sort(["aggregation", "alpha"])
        if alpha_rows
        else pl.DataFrame()
    )
    baseline_df = (
        pl.DataFrame(baseline_rows).sort(["aggregation", "model"])
        if baseline_rows
        else pl.DataFrame()
    )
    cohort_df = (
        pl.DataFrame(cohort_rows).sort(["model", "cohort"])
        if cohort_rows
        else pl.DataFrame()
    )

    if results_rows:
        results_df.write_csv(results_path)
    else:
        results_path.write_text("", encoding="utf-8")

    if alpha_rows:
        alpha_df.write_csv(alpha_path)
    else:
        alpha_path.write_text("", encoding="utf-8")

    if baseline_rows:
        baseline_df.write_csv(baseline_path)
    else:
        baseline_path.write_text("", encoding="utf-8")

    if cohort_rows:
        cohort_df.write_csv(cohort_path)
    else:
        cohort_path.write_text("", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "graph_set": list(graph_set),
        "alpha_grid": alpha_grid,
        "disable_community": bool(args.disable_community),
        "counts": {
            "results": len(results_rows),
            "alpha_search": len(alpha_rows),
            "baseline_search": len(baseline_rows),
            "cohort_rows": len(cohort_rows),
        },
        "paths": {
            "results": str(results_path),
            "alpha_search": str(alpha_path),
            "baseline_search": str(baseline_path),
            "cohort_results": str(cohort_path),
        },
    }

    if results_rows:
        best_by_agg = (
            results_df.drop_nulls(["test_wmape"])
            .sort(["aggregation", "test_wmape"])
            .group_by("aggregation", maintain_order=True)
            .first()
        )
        summary["best_by_aggregation"] = best_by_agg.to_dicts()

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote train/eval results: {results_path}")
    print(f"Wrote graph alpha search: {alpha_path}")
    print(f"Wrote baseline search: {baseline_path}")
    print(f"Wrote station cohort metrics: {cohort_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1-hour ahead training/evaluation pipeline for station and community levels"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--stations-dir", type=Path, default=DEFAULT_STATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph-set", default="SD,DE,DC,ATD")
    parser.add_argument(
        "--alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    )
    parser.add_argument("--sparse-quantile", type=float, default=0.25)
    parser.add_argument("--disable-community", action="store_true")

    parser.add_argument("--seasonal-lags", default="1,24,168")
    parser.add_argument("--linear-lag-candidates", default="1|1,24|1,2,24")
    parser.add_argument("--tree-lag-candidates", default="1,24|1,2,24")
    parser.add_argument("--tree-max-depths", default="8,12")
    parser.add_argument("--tree-estimators", type=int, default=80)
    parser.add_argument("--linear-max-samples", type=int, default=250000)
    parser.add_argument("--tree-max-samples", type=int, default=120000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
