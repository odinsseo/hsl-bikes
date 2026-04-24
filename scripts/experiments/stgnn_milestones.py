from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from scripts.graph_construction import build_all_graphs, save_graph_bundle

from .config import (
    CSV_SCHEMA_OVERRIDES,
    DATA_DIR,
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    parse_int_grid,
)
from .provenance import build_run_metadata, write_metadata_sidecar
from .stgnn import parse_graph_set
from .stgnn import run as run_stgnn

DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "experiments" / "stgnn_milestones"
DEFAULT_LEAKY_GRAPH_DIR = DATA_DIR / "artifacts" / "graphs" / "leaky_full"
DEFAULT_MERGED = DATA_DIR / "prepared" / "merged" / "trips_merged.csv"


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


def _namespace_for_stgnn(
    args: argparse.Namespace,
    *,
    graph_dir: Path,
    output_dir: Path,
    aggregation: str,
    graph_set: str,
    fusion_mode: str,
    allow_leaky_graph_source: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        train=args.train,
        validation=args.validation,
        test=args.test,
        graph_dir=graph_dir,
        communities=args.communities,
        output_dir=output_dir,
        aggregation=aggregation,
        graph="DE",
        graph_set=graph_set,
        fusion_mode=fusion_mode,
        history=args.history,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_start_epoch=args.early_stop_start_epoch,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_patience=args.lr_decay_patience,
        lr_plateau_threshold=args.lr_plateau_threshold,
        min_learning_rate=args.min_learning_rate,
        max_grad_norm=args.max_grad_norm,
        epoch_progress=args.epoch_progress,
        max_train_windows=args.max_train_windows,
        preprocess_target=args.preprocess_target,
        winsor_lower_quantile=args.winsor_lower_quantile,
        winsor_upper_quantile=args.winsor_upper_quantile,
        preprocess_scaler=args.preprocess_scaler,
        residualize_target=args.residualize_target,
        residual_lag_candidates=args.residual_lag_candidates,
        holiday_country=args.holiday_country,
        holiday_subdivision=args.holiday_subdivision,
        holiday_national_only=args.holiday_national_only,
        include_calendar_covariates=args.include_calendar_covariates,
        include_activity_mask=args.include_activity_mask,
        include_zero_run_indicator=args.include_zero_run_indicator,
        zero_run_length=args.zero_run_length,
        include_static_features=args.include_static_features,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        lazy_windows=args.lazy_windows,
        cache_preprocessed=args.cache_preprocessed,
        refresh_preprocessed_cache=args.refresh_preprocessed_cache,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
        device=args.device,
        random_state=args.random_state,
        strict_graph_source=True,
        allow_leaky_graph_source=allow_leaky_graph_source,
    )


def _preprocessing_lineage_from_args(args: argparse.Namespace) -> dict[str, Any]:
    subdivision = None if bool(args.holiday_national_only) else args.holiday_subdivision

    dynamic_defs: list[str] = []
    if bool(args.include_calendar_covariates):
        dynamic_defs.extend(
            [
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "is_weekend",
                "is_holiday",
            ]
        )
    if bool(args.include_activity_mask):
        dynamic_defs.append("recent_activity_mask")
    if bool(args.include_zero_run_indicator):
        dynamic_defs.append("long_zero_run_indicator")

    static_defs = ["train_mean", "train_variance", "train_zero_rate"]
    if not bool(args.include_static_features):
        static_defs = []

    sparse_defs: list[str] = []
    if bool(args.include_activity_mask):
        sparse_defs.append("recent_activity_mask")
    if bool(args.include_zero_run_indicator):
        sparse_defs.append("long_zero_run_indicator")

    return {
        "preprocessing_version": "v1",
        "fitted_split": "train",
        "train_time_bounds": None,
        "quantile_bounds": {
            "lower": float(args.winsor_lower_quantile),
            "upper": float(args.winsor_upper_quantile),
        },
        "scaler_type": str(args.preprocess_scaler),
        "residual_lag_policy": [
            int(x) for x in parse_int_grid(args.residual_lag_candidates)
        ],
        "selected_residual_lag": None,
        "calendar_source": {
            "library": "holidays",
            "country": str(args.holiday_country),
            "subdivision": subdivision,
        },
        "dynamic_feature_definitions": dynamic_defs,
        "static_feature_definitions": static_defs,
        "sparse_feature_definitions": sparse_defs,
        "train_only_fit": True,
    }


def _read_single_result(result_csv: Path) -> dict[str, Any]:
    frame = pl.read_csv(result_csv)
    if frame.height != 1:
        raise ValueError(f"Expected one result row in {result_csv}, got {frame.height}")
    return frame.to_dicts()[0]


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _optional_delta(lhs: Any, rhs: Any) -> float | None:
    left = _coerce_optional_float(lhs)
    right = _coerce_optional_float(rhs)
    if left is None or right is None:
        return None
    return left - right


def _build_leaky_graphs_if_needed(args: argparse.Namespace) -> None:
    if args.leaky_graph_dir.exists() and (args.leaky_graph_dir / "SD.npy").exists():
        return

    if not args.build_leaky_graphs:
        raise FileNotFoundError(
            f"Leaky graph dir missing and auto-build disabled: {args.leaky_graph_dir}"
        )
    if not args.merged.exists():
        raise FileNotFoundError(
            f"Merged CSV required to build leaky graphs: {args.merged}"
        )

    df = pl.read_csv(
        args.merged,
        try_parse_dates=True,
        schema_overrides=CSV_SCHEMA_OVERRIDES,
    )
    bundle = build_all_graphs(
        df=df,
        k_neighbors=args.sd_k_neighbors,
        sigma_km=args.sd_sigma_km,
        de_min_flow=args.de_min_flow,
    )
    metadata = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": str(args.merged),
        "note": "Leaky full-period graph bundle for sensitivity analysis.",
        "rows_used": int(df.height),
    }
    save_graph_bundle(
        output_dir=args.leaky_graph_dir,
        station_index=bundle["station_index"],
        graphs=bundle["graphs"],
        metadata=metadata,
    )


def _run_config(
    args: argparse.Namespace,
    *,
    graph_dir: Path,
    output_dir: Path,
    aggregation: str,
    graph_set: str,
    fusion_mode: str,
    milestone_tag: str,
    allow_leaky_graph_source: bool = False,
) -> dict[str, Any]:
    stgnn_args = _namespace_for_stgnn(
        args,
        graph_dir=graph_dir,
        output_dir=output_dir,
        aggregation=aggregation,
        graph_set=graph_set,
        fusion_mode=fusion_mode,
        allow_leaky_graph_source=allow_leaky_graph_source,
    )
    run_stgnn(stgnn_args)

    row = _read_single_result(output_dir / "results.csv")
    row["milestone"] = milestone_tag
    row["graph_source"] = str(graph_dir)
    row["run_dir"] = str(output_dir)
    return row


def run(args: argparse.Namespace) -> int:
    single_graphs = parse_graph_set(args.single_graphs)
    fusion_graphs = parse_graph_set(args.fusion_graph_set)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    aggregations = ["station", "community"] if args.include_community else ["station"]

    aggregation_iter = _iter_with_progress(
        list(aggregations),
        enabled=bool(args.progress),
        desc="Milestone aggregations",
        unit="agg",
    )
    for aggregation in aggregation_iter:
        single_iter = _iter_with_progress(
            list(single_graphs),
            enabled=bool(args.progress),
            desc=f"M3.1 single-graph ({aggregation})",
            unit="graph",
            leave=False,
        )
        for graph in single_iter:
            run_dir = runs_dir / f"m31_single_{aggregation}_{graph.lower()}"
            result = _run_config(
                args,
                graph_dir=args.graph_dir,
                output_dir=run_dir,
                aggregation=aggregation,
                graph_set=graph,
                fusion_mode="single",
                milestone_tag="M3.1_single_graph",
            )
            all_results.append(result)

        if len(fusion_graphs) > 1:
            fusion_text = ",".join(fusion_graphs)
            fusion_modes = ["equal", "learned"]
            fusion_iter = _iter_with_progress(
                fusion_modes,
                enabled=bool(args.progress),
                desc=f"M3.2 fusion ({aggregation})",
                unit="mode",
                leave=False,
            )
            for fusion_mode in fusion_iter:
                run_dir = runs_dir / f"m32_fusion_{fusion_mode}_{aggregation}"
                result = _run_config(
                    args,
                    graph_dir=args.graph_dir,
                    output_dir=run_dir,
                    aggregation=aggregation,
                    graph_set=fusion_text,
                    fusion_mode=fusion_mode,
                    milestone_tag="M3.2_multi_graph_fusion",
                )
                all_results.append(result)

    leakage_rows: list[dict[str, Any]] = []
    if args.include_leakage_sensitivity:
        _build_leaky_graphs_if_needed(args)

        leakage_specs = [
            ("single_de_station", "station", "DE", "single"),
        ]
        if len(fusion_graphs) > 1:
            leakage_specs.append(
                (
                    "fusion_learned_station",
                    "station",
                    ",".join(fusion_graphs),
                    "learned",
                )
            )

        leakage_iter = _iter_with_progress(
            list(leakage_specs),
            enabled=bool(args.progress),
            desc="Leakage sensitivity",
            unit="config",
        )
        for tag, aggregation, graph_set, fusion_mode in leakage_iter:
            base_dir = runs_dir / f"leakage_base_{tag}"
            leaky_dir = runs_dir / f"leakage_leaky_{tag}"

            base_row = _run_config(
                args,
                graph_dir=args.graph_dir,
                output_dir=base_dir,
                aggregation=aggregation,
                graph_set=graph_set,
                fusion_mode=fusion_mode,
                milestone_tag="M3.leakage_sensitivity_baseline",
            )
            leaky_row = _run_config(
                args,
                graph_dir=args.leaky_graph_dir,
                output_dir=leaky_dir,
                aggregation=aggregation,
                graph_set=graph_set,
                fusion_mode=fusion_mode,
                milestone_tag="M3.leakage_sensitivity_leaky",
                allow_leaky_graph_source=True,
            )

            baseline_validation_wmape = _coerce_optional_float(
                base_row.get("validation_wmape")
            )
            leaky_validation_wmape = _coerce_optional_float(
                leaky_row.get("validation_wmape")
            )
            baseline_test_wmape = _coerce_optional_float(base_row.get("test_wmape"))
            leaky_test_wmape = _coerce_optional_float(leaky_row.get("test_wmape"))

            all_results.extend([base_row, leaky_row])
            leakage_rows.append(
                {
                    "config": tag,
                    "aggregation": aggregation,
                    "graph_set": graph_set,
                    "fusion_mode": fusion_mode,
                    "baseline_graph_source": str(args.graph_dir),
                    "leaky_graph_source": str(args.leaky_graph_dir),
                    "baseline_validation_wmape": baseline_validation_wmape,
                    "leaky_validation_wmape": leaky_validation_wmape,
                    "delta_validation_wmape": _optional_delta(
                        leaky_validation_wmape,
                        baseline_validation_wmape,
                    ),
                    "baseline_test_wmape": baseline_test_wmape,
                    "leaky_test_wmape": leaky_test_wmape,
                    "delta_test_wmape": _optional_delta(
                        leaky_test_wmape,
                        baseline_test_wmape,
                    ),
                }
            )

    result_df = pl.DataFrame(all_results).sort(
        ["aggregation", "milestone", "test_wmape"]
    )
    result_path = args.output_dir / "milestone_results.csv"
    result_df.write_csv(result_path)

    best_by_milestone_path = args.output_dir / "milestone_best_by_milestone.csv"
    best_by_milestone_df = (
        result_df.drop_nulls(["test_wmape"])
        .sort(["milestone", "aggregation", "test_wmape"])
        .group_by(["milestone", "aggregation"], maintain_order=True)
        .first()
    )
    best_by_milestone_df.write_csv(best_by_milestone_path)

    best_by_graph_set_path = args.output_dir / "milestone_best_by_graph_set.csv"
    best_by_graph_set_df = (
        result_df.drop_nulls(["test_wmape"])
        .sort(["aggregation", "graph_set", "fusion_mode", "test_wmape"])
        .group_by(["aggregation", "graph_set", "fusion_mode"], maintain_order=True)
        .first()
    )
    best_by_graph_set_df.write_csv(best_by_graph_set_path)

    leakage_path = args.output_dir / "leakage_sensitivity.csv"
    if leakage_rows:
        pl.DataFrame(leakage_rows).write_csv(leakage_path)
    else:
        leakage_path.write_text("", encoding="utf-8")

    best_overall = (
        result_df.drop_nulls(["test_wmape"]).sort("test_wmape").head(1).to_dicts()
    )
    best_by_aggregation = (
        result_df.drop_nulls(["test_wmape"])
        .sort(["aggregation", "test_wmape"])
        .group_by("aggregation", maintain_order=True)
        .first()
        .to_dicts()
    )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "single_graphs": list(single_graphs),
        "fusion_graph_set": list(fusion_graphs),
        "include_community": bool(args.include_community),
        "include_leakage_sensitivity": bool(args.include_leakage_sensitivity),
        "preprocess_target": bool(args.preprocess_target),
        "n_runs": int(result_df.height),
        "best_overall": best_overall,
        "best_by_aggregation": best_by_aggregation,
        "paths": {
            "results": str(result_path),
            "best_by_milestone": str(best_by_milestone_path),
            "best_by_graph_set": str(best_by_graph_set_path),
            "leakage_sensitivity": str(leakage_path),
        },
    }

    metadata = build_run_metadata(
        args=args,
        stage="phase3_stgnn_milestones",
        script="scripts/run_stgnn_milestones.py",
        extra={
            "include_leakage_sensitivity": bool(args.include_leakage_sensitivity),
            "preprocess_target": bool(args.preprocess_target),
            "preprocessing": _preprocessing_lineage_from_args(args),
        },
    )
    metadata_path = args.output_dir / "metadata.json"
    write_metadata_sidecar(metadata_path, metadata)
    summary["paths"]["metadata"] = str(metadata_path)
    summary["run_metadata"] = metadata

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote milestone results: {result_path}")
    print(f"Wrote best-by-milestone summary: {best_by_milestone_path}")
    print(f"Wrote best-by-graph-set summary: {best_by_graph_set_path}")
    print(f"Wrote leakage sensitivity: {leakage_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ST-GNN milestone sweeps (M3.1, M3.2, and leakage sensitivity)"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--single-graphs", default="SD,DE,DC,ATD")
    parser.add_argument("--fusion-graph-set", default="SD,DE,DC,ATD")
    parser.add_argument("--include-community", action="store_true")

    parser.add_argument("--include-leakage-sensitivity", action="store_true")
    parser.add_argument("--build-leaky-graphs", action="store_true")
    parser.add_argument("--leaky-graph-dir", type=Path, default=DEFAULT_LEAKY_GRAPH_DIR)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--sd-k-neighbors", type=int, default=12)
    parser.add_argument("--sd-sigma-km", type=float, default=2.5)
    parser.add_argument("--de-min-flow", type=int, default=1)

    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-3,
        help="Minimum absolute WMAPE decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early-stop-start-epoch",
        type=int,
        default=5,
        help="Warm-up epoch threshold; no-improvement counting starts at this epoch.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer for ST-GNN training runs.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="plateau",
        help="Learning-rate scheduler strategy for ST-GNN runs.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="Plateau scheduler decay factor.",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=5,
        help="Plateau scheduler patience in epochs.",
    )
    parser.add_argument(
        "--lr-plateau-threshold",
        type=float,
        default=None,
        help="Absolute WMAPE improvement required by Plateau; defaults to --early-stop-min-delta when omitted.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Minimum learning rate for plateau scheduler.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm; <= 0 disables clipping.",
    )
    parser.add_argument(
        "--epoch-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show per-epoch progress for each ST-GNN training run.",
    )
    parser.add_argument("--max-train-windows", type=int, default=0)
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
    parser.add_argument(
        "--include-calendar-covariates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include cyclical hour/day features and holiday/weekend indicators.",
    )
    parser.add_argument(
        "--include-activity-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include binary recent activity mask as dynamic covariate.",
    )
    parser.add_argument(
        "--include-zero-run-indicator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include sparse long zero-run indicator as dynamic covariate.",
    )
    parser.add_argument("--zero-run-length", type=int, default=2)
    parser.add_argument(
        "--include-static-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include train-derived static node context features.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes for ST-GNN runs.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of prefetched batches per worker when num-workers > 0.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DataLoader pinned memory for ST-GNN runs.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive across epochs when num-workers > 0.",
    )
    parser.add_argument(
        "--lazy-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Construct ST-GNN windows lazily via dataset indexing (default). Use --no-lazy-windows for legacy eager fallback.",
    )
    parser.add_argument(
        "--cache-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save and reuse deterministic preprocessing outputs across ST-GNN runs.",
    )
    parser.add_argument(
        "--refresh-preprocessed-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore existing preprocessing cache and rebuild artifacts.",
    )
    parser.add_argument(
        "--preprocessed-cache-dir",
        type=Path,
        default=DATA_DIR / "artifacts" / "cache" / "stgnn_preprocessed",
        help="Directory for cached preprocessing artifacts shared across runs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for milestone and leakage run loops.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
