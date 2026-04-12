from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    build_experiment_specs,
    parse_alpha_grid,
    parse_rqs,
)
from .data import (
    build_community_series,
    build_fused_adjacency,
    build_station_series,
    load_communities,
    load_graph_bundle,
    load_split,
)
from .models import evaluate_baseline_models, evaluate_one_step_forecast


def run(args: argparse.Namespace) -> int:
    selected_rqs = parse_rqs(args.rqs)
    specs = build_experiment_specs(selected_rqs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_df = pd.DataFrame(
        [
            {
                "experiment_id": s.experiment_id,
                "rq": s.rq,
                "aggregation": s.aggregation,
                "graph_set": "+".join(s.graph_set),
                "description": s.description,
            }
            for s in specs
        ]
    )
    matrix_path = args.output_dir / "experiment_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False)

    if args.generate_only:
        print(f"Wrote experiment matrix: {matrix_path}")
        return 0

    station_index, matrices = load_graph_bundle(args.graph_dir)
    station_to_group = None
    groups = None

    if any(spec.aggregation == "community" for spec in specs):
        station_to_group = load_communities(args.communities, station_index)
        groups = sorted(set(station_to_group.values()))

    train_df = load_split(args.train)
    val_df = load_split(args.validation)
    test_df = load_split(args.test)

    series_cache: dict[tuple[str, str], np.ndarray] = {}
    adj_cache: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    baseline_cache: dict[str, list[dict[str, Any]]] = {}

    alpha_grid = parse_alpha_grid(args.alpha_grid)
    alpha_search_rows: list[dict[str, Any]] = []
    baseline_search_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    def get_series(split_name: str, aggregation: str) -> np.ndarray:
        key = (split_name, aggregation)
        if key in series_cache:
            return series_cache[key]

        split_df = {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }[split_name]

        if aggregation == "station":
            series = build_station_series(split_df, station_index)
        elif aggregation == "community":
            if station_to_group is None or groups is None:
                raise ValueError("Community series requested without mapping")
            series = build_community_series(split_df, station_to_group, groups)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        series_cache[key] = series
        return series

    def get_adjacency(graph_set: tuple[str, ...], aggregation: str) -> np.ndarray:
        key = (graph_set, aggregation)
        if key in adj_cache:
            return adj_cache[key]

        adjacency = build_fused_adjacency(
            graph_set=graph_set,
            aggregation=aggregation,
            matrices=matrices,
            station_index=station_index,
            station_to_group=station_to_group,
            groups=groups,
        )
        adj_cache[key] = adjacency
        return adjacency

    for spec in specs:
        train_series = get_series("train", spec.aggregation)
        val_series = get_series("validation", spec.aggregation)
        test_series = get_series("test", spec.aggregation)
        adjacency = get_adjacency(spec.graph_set, spec.aggregation)

        best_alpha = alpha_grid[0]
        best_val = np.inf

        for alpha in alpha_grid:
            val_metrics = evaluate_one_step_forecast(
                series=val_series,
                adjacency=adjacency,
                alpha=alpha,
                train_series=train_series,
            )
            alpha_search_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "rq": spec.rq,
                    "aggregation": spec.aggregation,
                    "graph_set": "+".join(spec.graph_set),
                    "model": "graph_propagation",
                    "alpha": alpha,
                    "validation_wmape": val_metrics["wmape"],
                    "validation_mae": val_metrics["mae"],
                    "validation_rmse": val_metrics["rmse"],
                    "validation_mase": val_metrics["mase"],
                }
            )

            if np.isfinite(val_metrics["wmape"]) and val_metrics["wmape"] < best_val:
                best_val = val_metrics["wmape"]
                best_alpha = alpha

        final_val_metrics = evaluate_one_step_forecast(
            series=val_series,
            adjacency=adjacency,
            alpha=best_alpha,
            train_series=train_series,
        )
        final_test_metrics = evaluate_one_step_forecast(
            series=test_series,
            adjacency=adjacency,
            alpha=best_alpha,
            train_series=train_series,
        )

        result_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "rq": spec.rq,
                "aggregation": spec.aggregation,
                "graph_set": "+".join(spec.graph_set),
                "description": spec.description,
                "model": "graph_propagation",
                "config": json.dumps({"alpha": best_alpha}),
                "n_nodes": int(adjacency.shape[0]),
                "n_train_steps": int(train_series.shape[0]),
                "n_validation_steps": int(val_series.shape[0]),
                "n_test_steps": int(test_series.shape[0]),
                "validation_wmape": final_val_metrics["wmape"],
                "validation_mae": final_val_metrics["mae"],
                "validation_rmse": final_val_metrics["rmse"],
                "validation_mase": final_val_metrics["mase"],
                "test_wmape": final_test_metrics["wmape"],
                "test_mae": final_test_metrics["mae"],
                "test_rmse": final_test_metrics["rmse"],
                "test_mase": final_test_metrics["mase"],
            }
        )

        if spec.aggregation not in baseline_cache:
            baseline_models, baseline_search = evaluate_baseline_models(
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                args=args,
            )
            baseline_cache[spec.aggregation] = baseline_models
            for row in baseline_search:
                baseline_search_rows.append(
                    {
                        "aggregation": spec.aggregation,
                        **row,
                    }
                )

        for base in baseline_cache[spec.aggregation]:
            result_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "rq": spec.rq,
                    "aggregation": spec.aggregation,
                    "graph_set": "+".join(spec.graph_set),
                    "description": spec.description,
                    "model": base["model"],
                    "config": base["config"],
                    "n_nodes": int(train_series.shape[1]),
                    "n_train_steps": int(train_series.shape[0]),
                    "n_validation_steps": int(val_series.shape[0]),
                    "n_test_steps": int(test_series.shape[0]),
                    "validation_wmape": base["validation_wmape"],
                    "validation_mae": base["validation_mae"],
                    "validation_rmse": base["validation_rmse"],
                    "validation_mase": base["validation_mase"],
                    "test_wmape": base["test_wmape"],
                    "test_mae": base["test_mae"],
                    "test_rmse": base["test_rmse"],
                    "test_mase": base["test_mase"],
                }
            )

    alpha_df = pd.DataFrame(alpha_search_rows).sort_values(["experiment_id", "alpha"])
    baseline_df = pd.DataFrame(baseline_search_rows).sort_values(
        ["aggregation", "model"],
        na_position="last",
    )
    results_df = pd.DataFrame(result_rows).sort_values(
        ["rq", "model", "test_wmape"],
        na_position="last",
    )

    alpha_search_path = args.output_dir / "alpha_search.csv"
    baseline_search_path = args.output_dir / "baseline_search.csv"
    results_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"

    alpha_df.to_csv(alpha_search_path, index=False)
    baseline_df.to_csv(baseline_search_path, index=False)
    results_df.to_csv(results_path, index=False)

    best_by_rq = (
        results_df.dropna(subset=["test_wmape"])
        .sort_values(["rq", "test_wmape"])
        .groupby("rq", as_index=False)
        .first()
    )
    best_graph_by_rq = (
        results_df[results_df["model"] == "graph_propagation"]
        .dropna(subset=["test_wmape"])
        .sort_values(["rq", "test_wmape"])
        .groupby("rq", as_index=False)
        .first()
    )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "selected_rqs": sorted(selected_rqs),
        "experiment_count": int(matrix_df.shape[0]),
        "result_rows": int(results_df.shape[0]),
        "alpha_grid": alpha_grid,
        "paths": {
            "experiment_matrix": str(matrix_path),
            "alpha_search": str(alpha_search_path),
            "baseline_search": str(baseline_search_path),
            "results": str(results_path),
        },
        "best_by_rq": best_by_rq.to_dict(orient="records"),
        "best_graph_by_rq": best_graph_by_rq.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote experiment matrix: {matrix_path}")
    print(f"Wrote graph alpha search: {alpha_search_path}")
    print(f"Wrote baseline search: {baseline_search_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ ablation experiment runner")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rqs", default="RQ1,RQ2,RQ3")
    parser.add_argument(
        "--alpha-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--seasonal-lags", default="1,24,168")
    parser.add_argument("--linear-lag-candidates", default="1|1,24|1,2,24")
    parser.add_argument("--tree-lag-candidates", default="1,24|1,2,24")
    parser.add_argument("--tree-max-depths", default="8,12")
    parser.add_argument("--tree-estimators", type=int, default=80)
    parser.add_argument("--linear-max-samples", type=int, default=250000)
    parser.add_argument("--tree-max-samples", type=int, default=120000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
