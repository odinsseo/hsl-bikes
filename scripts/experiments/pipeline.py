from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .config import (
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    build_experiment_specs,
    parse_alpha_grid,
    parse_int_grid,
    parse_rqs,
)
from .data import (
    build_community_series,
    build_fused_adjacency,
    build_hourly_index,
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
from .preprocessing import (
    TargetPreprocessingConfig,
    apply_target_preprocessing,
    build_preprocessing_metadata,
    fit_target_preprocessing,
    inverse_target_predictions,
)
from .provenance import build_run_metadata, write_metadata_sidecar
from .safeguards import assert_train_graph_source


def _iter_with_progress(
    values: list[Any],
    *,
    enabled: bool,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Any:
    if enabled and tqdm is not None:
        return tqdm(values, desc=desc, unit=unit, leave=leave)
    return values


def _resolve_holiday_subdivision(args: argparse.Namespace) -> str | None:
    if getattr(args, "holiday_national_only", False):
        return None
    return args.holiday_subdivision


def run(args: argparse.Namespace) -> int:
    np.random.seed(args.random_state)

    if args.strict_graph_source:
        assert_train_graph_source(
            graph_dir=args.graph_dir,
            train_path=args.train,
            allow_leaky_graph_source=False,
        )

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
    train_hours = build_hourly_index(train_df)
    train_time_bounds = (
        (
            train_hours[0].isoformat(),
            train_hours[-1].isoformat(),
        )
        if train_hours
        else None
    )

    series_cache: dict[tuple[str, str], np.ndarray] = {}
    adj_cache: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    baseline_cache: dict[str, dict[str, dict[str, Any]]] = {}
    preprocessing_cache: dict[str, dict[str, Any]] = {}
    preprocessing_reports: list[dict[str, Any]] = []

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

    def get_preprocessing_bundle(aggregation: str) -> dict[str, Any] | None:
        if not args.preprocess_target:
            return None
        if aggregation in preprocessing_cache:
            return preprocessing_cache[aggregation]

        train_series_raw = get_series("train", aggregation)
        val_series_raw = get_series("validation", aggregation)
        test_series_raw = get_series("test", aggregation)

        preprocessing_config = TargetPreprocessingConfig(
            winsor_lower_quantile=args.winsor_lower_quantile,
            winsor_upper_quantile=args.winsor_upper_quantile,
            enable_log1p=True,
            scaler=args.preprocess_scaler,
            enable_residualization=bool(args.residualize_target),
            residual_lag_candidates=tuple(parse_int_grid(args.residual_lag_candidates)),
            holiday_country=args.holiday_country,
            holiday_subdivision=_resolve_holiday_subdivision(args),
        )
        preprocessing_state, lag_scores = fit_target_preprocessing(
            train_series_raw,
            validation_series=val_series_raw,
            config=preprocessing_config,
        )
        train_applied = apply_target_preprocessing(
            train_series_raw, preprocessing_state
        )
        val_applied = apply_target_preprocessing(val_series_raw, preprocessing_state)
        test_applied = apply_target_preprocessing(test_series_raw, preprocessing_state)

        preprocessing_metadata = build_preprocessing_metadata(
            preprocessing_state,
            train_time_bounds=train_time_bounds,
        )
        preprocessing_metadata["aggregation"] = aggregation
        preprocessing_metadata["residual_lag_scores"] = [
            {"lag": int(row.lag), "score": float(row.score)} for row in lag_scores
        ]

        bundle = {
            "train_series": train_applied.transformed,
            "val_series": val_applied.transformed,
            "test_series": test_applied.transformed,
            "val_pre_residual": val_applied.pre_residual,
            "test_pre_residual": test_applied.pre_residual,
            "state": preprocessing_state,
            "metadata": preprocessing_metadata,
        }
        preprocessing_cache[aggregation] = bundle
        preprocessing_reports.append(preprocessing_metadata)
        return bundle

    spec_iter = _iter_with_progress(
        list(specs),
        enabled=bool(args.progress),
        desc="RQ experiments",
        unit="exp",
    )
    for spec in spec_iter:
        train_series_raw = get_series("train", spec.aggregation)
        val_series_raw = get_series("validation", spec.aggregation)
        test_series_raw = get_series("test", spec.aggregation)

        train_series = train_series_raw
        val_series = val_series_raw
        test_series = test_series_raw
        selected_residual_lag: int | None = None
        preprocess_bundle = get_preprocessing_bundle(spec.aggregation)
        if preprocess_bundle is not None:
            train_series = preprocess_bundle["train_series"]
            val_series = preprocess_bundle["val_series"]
            test_series = preprocess_bundle["test_series"]
            selected_residual_lag = preprocess_bundle["metadata"][
                "selected_residual_lag"
            ]

        adjacency = get_adjacency(spec.graph_set, spec.aggregation)

        tune_kwargs: dict[str, Any] = {}
        if preprocess_bundle is not None:
            tune_kwargs = {
                "val_series_raw": val_series_raw,
                "train_series_raw": train_series_raw,
                "inverse_state": preprocess_bundle["state"],
                "val_pre_residual": preprocess_bundle["val_pre_residual"],
                "history": 1,
                "horizon": 1,
            }

        best_alpha, alpha_search = tune_graph_alpha(
            train_series=train_series,
            val_series=val_series,
            adjacency=adjacency,
            alpha_grid=alpha_grid,
            **tune_kwargs,
        )
        for val_metrics in alpha_search:
            alpha_search_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "rq": spec.rq,
                    "aggregation": spec.aggregation,
                    "graph_set": "+".join(spec.graph_set),
                    "model": "graph_propagation",
                    "alpha": val_metrics["alpha"],
                    "validation_wmape": val_metrics["validation_wmape"],
                    "validation_mae": val_metrics["validation_mae"],
                    "validation_rmse": val_metrics["validation_rmse"],
                    "validation_mase": val_metrics["validation_mase"],
                }
            )

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

        if preprocess_bundle is not None:
            graph_val_pred = inverse_target_predictions(
                graph_val_pred,
                state=preprocess_bundle["state"],
                context_pre_residual=preprocess_bundle["val_pre_residual"],
                history=1,
                horizon=1,
            )
            graph_test_pred = inverse_target_predictions(
                graph_test_pred,
                state=preprocess_bundle["state"],
                context_pre_residual=preprocess_bundle["test_pre_residual"],
                history=1,
                horizon=1,
            )

        final_val_metrics = compute_metrics(
            actual=val_series_raw[1:],
            pred=graph_val_pred,
            train_series=train_series_raw,
        )
        final_test_metrics = compute_metrics(
            actual=test_series_raw[1:],
            pred=graph_test_pred,
            train_series=train_series_raw,
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
                "preprocessing_enabled": bool(args.preprocess_target),
                "selected_residual_lag": selected_residual_lag,
            }
        )

        if spec.aggregation not in baseline_cache:
            baseline_models, baseline_search = fit_best_baseline_models(
                train_series=train_series,
                val_series=val_series,
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

        for model_name in ("seasonal_naive", "lagged_linear", "tree_lagged"):
            if model_name not in baseline_cache[spec.aggregation]:
                continue
            base = baseline_cache[spec.aggregation][model_name]
            val_pred = predict_baseline(base, val_series)
            test_pred = predict_baseline(base, test_series)

            if preprocess_bundle is not None:
                val_pred = inverse_target_predictions(
                    val_pred,
                    state=preprocess_bundle["state"],
                    context_pre_residual=preprocess_bundle["val_pre_residual"],
                    history=1,
                    horizon=1,
                )
                test_pred = inverse_target_predictions(
                    test_pred,
                    state=preprocess_bundle["state"],
                    context_pre_residual=preprocess_bundle["test_pre_residual"],
                    history=1,
                    horizon=1,
                )

            val_metrics = compute_metrics(
                actual=val_series_raw[1:],
                pred=val_pred,
                train_series=train_series_raw,
            )
            test_metrics = compute_metrics(
                actual=test_series_raw[1:],
                pred=test_pred,
                train_series=train_series_raw,
            )

            result_rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "rq": spec.rq,
                    "aggregation": spec.aggregation,
                    "graph_set": "+".join(spec.graph_set),
                    "description": spec.description,
                    "model": model_name,
                    "config": json.dumps(base["config"]),
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
                    "preprocessing_enabled": bool(args.preprocess_target),
                    "selected_residual_lag": selected_residual_lag,
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
    if preprocessing_reports:
        summary["preprocessing"] = preprocessing_reports

    metadata_preprocessing: dict[str, Any] | None = None
    if preprocessing_reports:
        metadata_preprocessing = {
            **preprocessing_reports[0],
            "aggregation_reports": preprocessing_reports,
        }

    metadata = build_run_metadata(
        args=args,
        stage="phase1_rq_runner",
        script="scripts/experiment_runners.py",
        extra={
            "strict_graph_source": bool(args.strict_graph_source),
            "preprocess_target": bool(args.preprocess_target),
            "preprocessing": metadata_preprocessing,
            "preprocessing_reports": preprocessing_reports,
        },
    )
    metadata_path = args.output_dir / "metadata.json"
    write_metadata_sidecar(metadata_path, metadata)
    summary["paths"]["metadata"] = str(metadata_path)
    summary["run_metadata"] = metadata

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
    parser.add_argument("--seasonal-lags", default="1,8,56")
    parser.add_argument("--linear-lag-candidates", default="1|1,8|1,2,8")
    parser.add_argument("--tree-lag-candidates", default="1,8|1,2,8")
    parser.add_argument("--tree-max-depths", default="8,12")
    parser.add_argument("--tree-estimators", type=int, default=80)
    parser.add_argument("--linear-max-samples", type=int, default=250000)
    parser.add_argument("--tree-max-samples", type=int, default=120000)
    parser.add_argument(
        "--preprocess-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train-fitted target preprocessing and inverse-transform metrics to original scale.",
    )
    parser.add_argument("--winsor-lower-quantile", type=float, default=0.005)
    parser.add_argument("--winsor-upper-quantile", type=float, default=0.995)
    parser.add_argument("--preprocess-scaler", choices=["robust"], default="robust")
    parser.add_argument(
        "--residualize-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable residualization with lag selected from --residual-lag-candidates.",
    )
    parser.add_argument("--residual-lag-candidates", default="8,56")
    parser.add_argument("--holiday-country", default="FI")
    parser.add_argument("--holiday-subdivision", default="18")
    parser.add_argument(
        "--holiday-national-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use national-only FI holidays (ignore subdivision) for sensitivity checks.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for experiment and alpha sweeps.",
    )
    parser.add_argument(
        "--strict-graph-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require graph metadata source to match the train split path.",
    )
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
