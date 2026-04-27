from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .config import (
    DATA_DIR,
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    parse_alpha_grid,
    parse_int_grid,
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
from .preprocessing import (
    TargetPreprocessingConfig,
    apply_target_preprocessing,
    build_preprocessing_metadata,
    fit_target_preprocessing,
    inverse_target_predictions,
)
from .provenance import build_run_metadata, write_metadata_sidecar
from .safeguards import assert_train_graph_source

DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "experiments" / "train_eval_3h"
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


def _resolve_holiday_subdivision(args: argparse.Namespace) -> str | None:
    if getattr(args, "holiday_national_only", False):
        return None
    return args.holiday_subdivision


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


def station_wmape_vector(
    actual: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=float)

    actual_subset = actual[:, indices]
    pred_subset = pred[:, indices]
    numerator = np.abs(actual_subset - pred_subset).sum(axis=0)
    denominator = np.abs(actual_subset).sum(axis=0)

    wmape = np.full(numerator.shape, np.nan, dtype=float)
    valid = denominator > 0.0
    wmape[valid] = numerator[valid] / denominator[valid]
    return wmape


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_level: float,
    batch_size: int = 2048,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return (np.nan, np.nan)
    if clean.size == 1:
        value = float(clean[0])
        return (value, value)

    n = int(clean.size)
    n_bootstrap = max(int(n_bootstrap), 1)
    batch_size = max(int(batch_size), 1)
    alpha = (1.0 - float(ci_level)) / 2.0
    means = np.empty(n_bootstrap, dtype=float)

    written = 0
    while written < n_bootstrap:
        chunk = min(batch_size, n_bootstrap - written)
        sample_idx = rng.integers(0, n, size=(chunk, n))
        means[written : written + chunk] = clean[sample_idx].mean(axis=1)
        written += chunk

    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return (lower, upper)


def paired_sign_permutation_pvalue(
    sample: np.ndarray,
    reference: np.ndarray,
    *,
    rng: np.random.Generator,
    n_permutations: int,
    batch_size: int = 4096,
) -> float:
    sample_arr = np.asarray(sample, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    mask = np.isfinite(sample_arr) & np.isfinite(reference_arr)
    diff = sample_arr[mask] - reference_arr[mask]
    if diff.size == 0:
        return np.nan

    observed = float(np.abs(np.mean(diff)))
    if observed == 0.0:
        return 1.0

    n_permutations = max(int(n_permutations), 1)
    batch_size = max(int(batch_size), 1)
    extreme_count = 0

    processed = 0
    while processed < n_permutations:
        chunk = min(batch_size, n_permutations - processed)
        signs = rng.integers(0, 2, size=(chunk, diff.size), dtype=np.int8)
        signed = signs.astype(np.float32, copy=False)
        signed *= 2.0
        signed -= 1.0

        permuted_mean = np.abs(np.mean(signed * diff, axis=1))
        extreme_count += int(np.sum(permuted_mean >= observed))
        processed += chunk

    return float((extreme_count + 1) / (n_permutations + 1))


def build_station_robustness_rows(
    *,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    cohorts: dict[str, np.ndarray],
    graph_set: tuple[str, ...],
    reference_model: str,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
    progress: bool = False,
    bootstrap_batch_size: int = 2048,
    permutation_batch_size: int = 4096,
) -> list[dict[str, Any]]:
    if reference_model not in predictions:
        available = sorted(predictions.keys())
        raise ValueError(
            f"Reference model '{reference_model}' is not available in predictions: {available}"
        )

    rows: list[dict[str, Any]] = []
    graph_set_text = ",".join(graph_set)

    reference_by_cohort: dict[str, np.ndarray] = {}
    for cohort_name, idx in cohorts.items():
        reference_by_cohort[cohort_name] = station_wmape_vector(
            actual=actual,
            pred=predictions[reference_model],
            indices=idx,
        )

    tasks: list[tuple[str, np.ndarray, str, np.ndarray]] = []
    for cohort_name, idx in cohorts.items():
        for model_name, model_pred in predictions.items():
            tasks.append((cohort_name, idx, model_name, model_pred))

    iterator: Any = tasks
    if progress and tqdm is not None:
        iterator = tqdm(tasks, desc="Robustness rows", unit="row")

    for cohort_name, idx, model_name, model_pred in iterator:
        reference_wmape = reference_by_cohort[cohort_name]
        model_wmape = station_wmape_vector(
            actual=actual,
            pred=model_pred,
            indices=idx,
        )
        wmape_mean = (
            float(np.nanmean(model_wmape)) if np.isfinite(model_wmape).any() else np.nan
        )
        ci_lower, ci_upper = bootstrap_mean_ci(
            model_wmape,
            rng=rng,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            batch_size=bootstrap_batch_size,
        )

        delta_wmape = model_wmape - reference_wmape
        delta_mean = (
            float(np.nanmean(delta_wmape)) if np.isfinite(delta_wmape).any() else np.nan
        )
        delta_ci_lower, delta_ci_upper = bootstrap_mean_ci(
            delta_wmape,
            rng=rng,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            batch_size=bootstrap_batch_size,
        )

        p_value = (
            1.0
            if model_name == reference_model
            else paired_sign_permutation_pvalue(
                model_wmape,
                reference_wmape,
                rng=rng,
                n_permutations=n_permutations,
                batch_size=permutation_batch_size,
            )
        )

        rows.append(
            {
                "aggregation": "station",
                "graph_set": graph_set_text,
                "cohort": cohort_name,
                "model": model_name,
                "reference_model": reference_model,
                "n_nodes": int(idx.size),
                "n_nodes_used": int(np.isfinite(model_wmape).sum()),
                "test_station_wmape_mean": wmape_mean,
                "test_station_wmape_ci_lower": ci_lower,
                "test_station_wmape_ci_upper": ci_upper,
                "delta_station_wmape_vs_reference": delta_mean,
                "delta_ci_lower": delta_ci_lower,
                "delta_ci_upper": delta_ci_upper,
                "paired_sign_permutation_pvalue": p_value,
            }
        )

    return rows


def build_sensitivity_summary_rows(
    *,
    results_df: pl.DataFrame,
    alpha_df: pl.DataFrame,
    baseline_df: pl.DataFrame,
    cohort_df: pl.DataFrame,
    sparse_quantile: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if alpha_df.height > 0:
        for aggregation in alpha_df.get_column("aggregation").unique().to_list():
            subset = alpha_df.filter(
                (pl.col("aggregation") == aggregation)
                & pl.col("validation_wmape").is_not_null()
            )
            if subset.height == 0:
                continue
            best = float(subset.select(pl.col("validation_wmape").min()).item())
            for row in subset.sort("alpha").to_dicts():
                value = float(row["validation_wmape"])
                rows.append(
                    {
                        "sensitivity_axis": "hyperparameter",
                        "scope": "graph_propagation_alpha",
                        "aggregation": aggregation,
                        "model": "graph_propagation",
                        "setting": f"alpha={row['alpha']}",
                        "metric": "validation_wmape",
                        "value": value,
                        "reference_value": best,
                        "delta_vs_reference": value - best,
                    }
                )

    if baseline_df.height > 0:
        clean_baseline = baseline_df.filter(pl.col("validation_wmape").is_not_null())
        if clean_baseline.height > 0:
            best_map: dict[tuple[str, str], float] = {}
            for row in (
                clean_baseline.sort(["aggregation", "model", "validation_wmape"])
                .group_by(["aggregation", "model"], maintain_order=True)
                .first()
                .to_dicts()
            ):
                best_map[(str(row["aggregation"]), str(row["model"]))] = float(
                    row["validation_wmape"]
                )

            for row in clean_baseline.to_dicts():
                aggregation = str(row["aggregation"])
                model = str(row["model"])
                best = best_map[(aggregation, model)]
                value = float(row["validation_wmape"])
                rows.append(
                    {
                        "sensitivity_axis": "hyperparameter",
                        "scope": "baseline_hyperparameters",
                        "aggregation": aggregation,
                        "model": model,
                        "setting": str(row.get("config", "")),
                        "metric": "validation_wmape",
                        "value": value,
                        "reference_value": best,
                        "delta_vs_reference": value - best,
                    }
                )

    if cohort_df.height > 0:
        for model in cohort_df.get_column("model").unique().to_list():
            sparse_row = cohort_df.filter(
                (pl.col("model") == model) & (pl.col("cohort") == "sparse")
            )
            dense_row = cohort_df.filter(
                (pl.col("model") == model) & (pl.col("cohort") == "dense")
            )
            if sparse_row.height == 0 or dense_row.height == 0:
                continue

            sparse_wmape = sparse_row.select("test_wmape").item()
            dense_wmape = dense_row.select("test_wmape").item()
            if sparse_wmape is None or dense_wmape is None:
                continue

            rows.append(
                {
                    "sensitivity_axis": "threshold",
                    "scope": "sparse_vs_dense",
                    "aggregation": "station",
                    "model": str(model),
                    "setting": f"sparse_quantile={sparse_quantile}",
                    "metric": "test_wmape",
                    "value": float(sparse_wmape),
                    "reference_value": float(dense_wmape),
                    "delta_vs_reference": float(sparse_wmape) - float(dense_wmape),
                }
            )

    if results_df.height > 0:
        for model in results_df.get_column("model").unique().to_list():
            station_rows = results_df.filter(
                (pl.col("model") == model)
                & (pl.col("aggregation") == "station")
                & pl.col("test_wmape").is_not_null()
            )
            community_rows = results_df.filter(
                (pl.col("model") == model)
                & (pl.col("aggregation") == "community")
                & pl.col("test_wmape").is_not_null()
            )
            if station_rows.height == 0 or community_rows.height == 0:
                continue

            station_best = float(station_rows.select(pl.col("test_wmape").min()).item())
            community_best = float(
                community_rows.select(pl.col("test_wmape").min()).item()
            )
            rows.append(
                {
                    "sensitivity_axis": "resolution",
                    "scope": "station_vs_community",
                    "aggregation": "station_community",
                    "model": str(model),
                    "setting": "aggregation_resolution",
                    "metric": "test_wmape",
                    "value": station_best,
                    "reference_value": community_best,
                    "delta_vs_reference": station_best - community_best,
                }
            )

    return rows


def evaluate_aggregation(
    aggregation: str,
    graph_set: tuple[str, ...],
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
    dict[str, Any] | None,
]:
    results: list[dict[str, Any]] = []
    alpha_search: list[dict[str, Any]] = []
    baseline_search: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    graph_set_text = ",".join(graph_set)

    train_series_raw = train_series
    val_series_raw = val_series
    test_series_raw = test_series

    preprocessing_metadata: dict[str, Any] | None = None
    preprocessing_state = None
    val_pre_residual = val_series_raw
    test_pre_residual = test_series_raw

    if args.preprocess_target:
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
        train_series = train_applied.transformed
        val_series = val_applied.transformed
        test_series = test_applied.transformed
        val_pre_residual = val_applied.pre_residual
        test_pre_residual = test_applied.pre_residual
        preprocessing_metadata = build_preprocessing_metadata(preprocessing_state)
        preprocessing_metadata["aggregation"] = aggregation
        preprocessing_metadata["residual_lag_scores"] = [
            {"lag": int(row.lag), "score": float(row.score)} for row in lag_scores
        ]

    tune_kwargs: dict[str, Any] = {}
    if args.preprocess_target:
        if preprocessing_state is None or val_pre_residual is None:
            raise RuntimeError("preprocess_target requires fitted preprocessing state")
        tune_kwargs = {
            "val_series_raw": val_series_raw,
            "train_series_raw": train_series_raw,
            "inverse_state": preprocessing_state,
            "val_pre_residual": val_pre_residual,
            "history": 1,
            "horizon": 1,
        }

    best_alpha, graph_search = tune_graph_alpha(
        train_series=train_series,
        val_series=val_series,
        adjacency=adjacency,
        alpha_grid=alpha_grid,
        **tune_kwargs,
    )
    for row in graph_search:
        alpha_search.append(
            {"aggregation": aggregation, "graph_set": graph_set_text, **row}
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
    if args.preprocess_target:
        graph_val_pred = inverse_target_predictions(
            graph_val_pred,
            state=preprocessing_state,
            context_pre_residual=val_pre_residual,
            history=1,
            horizon=1,
        )
        graph_test_pred = inverse_target_predictions(
            graph_test_pred,
            state=preprocessing_state,
            context_pre_residual=test_pre_residual,
            history=1,
            horizon=1,
        )

    graph_val_metrics = compute_metrics(
        actual=val_series_raw[1:],
        pred=graph_val_pred,
        train_series=train_series_raw,
    )
    graph_test_metrics = compute_metrics(
        actual=test_series_raw[1:],
        pred=graph_test_pred,
        train_series=train_series_raw,
    )
    predictions["graph_propagation"] = (graph_val_pred, graph_test_pred)
    results.append(
        {
            "aggregation": aggregation,
            "graph_set": graph_set_text,
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
            "preprocessing_enabled": bool(args.preprocess_target),
            "selected_residual_lag": (
                preprocessing_metadata["selected_residual_lag"]
                if preprocessing_metadata is not None
                else None
            ),
        }
    )

    fitted_baselines, baseline_search_rows = fit_best_baseline_models(
        train_series=train_series,
        val_series=val_series,
        args=args,
    )
    for row in baseline_search_rows:
        baseline_search.append(
            {"aggregation": aggregation, "graph_set": graph_set_text, **row}
        )

    for model_name in ("seasonal_naive", "lagged_linear", "tree_lagged"):
        if model_name not in fitted_baselines:
            continue

        model_spec = fitted_baselines[model_name]
        val_pred = predict_baseline(model_spec, val_series)
        test_pred = predict_baseline(model_spec, test_series)

        if args.preprocess_target:
            val_pred = inverse_target_predictions(
                val_pred,
                state=preprocessing_state,
                context_pre_residual=val_pre_residual,
                history=1,
                horizon=1,
            )
            test_pred = inverse_target_predictions(
                test_pred,
                state=preprocessing_state,
                context_pre_residual=test_pre_residual,
                history=1,
                horizon=1,
            )

        predictions[model_name] = (val_pred, test_pred)

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
        results.append(
            {
                "aggregation": aggregation,
                "graph_set": graph_set_text,
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
                "preprocessing_enabled": bool(args.preprocess_target),
                "selected_residual_lag": (
                    preprocessing_metadata["selected_residual_lag"]
                    if preprocessing_metadata is not None
                    else None
                ),
            }
        )

    return results, alpha_search, baseline_search, predictions, preprocessing_metadata


def run(args: argparse.Namespace) -> int:
    np.random.seed(args.random_state)
    rng = np.random.default_rng(args.random_state)

    if not (0.0 < args.ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")

    if args.strict_graph_source:
        assert_train_graph_source(
            graph_dir=args.graph_dir,
            train_path=args.train,
            allow_leaky_graph_source=False,
        )

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
    preprocessing_reports: list[dict[str, Any]] = []

    (
        station_results,
        station_alpha,
        station_baseline,
        station_predictions,
        station_preprocessing,
    ) = evaluate_aggregation(
        aggregation="station",
        graph_set=graph_set,
        train_series=train_station,
        val_series=val_station,
        test_series=test_station,
        adjacency=station_adj,
        alpha_grid=alpha_grid,
        args=args,
    )
    results_rows.extend(station_results)
    alpha_rows.extend(station_alpha)
    baseline_rows.extend(station_baseline)
    if station_preprocessing is not None:
        preprocessing_reports.append(station_preprocessing)

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
                    "graph_set": ",".join(graph_set),
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

        comm_results, comm_alpha, comm_baseline, _, comm_preprocessing = (
            evaluate_aggregation(
                aggregation="community",
                graph_set=graph_set,
                train_series=train_comm,
                val_series=val_comm,
                test_series=test_comm,
                adjacency=comm_adj,
                alpha_grid=alpha_grid,
                args=args,
            )
        )
        results_rows.extend(comm_results)
        alpha_rows.extend(comm_alpha)
        baseline_rows.extend(comm_baseline)
        if comm_preprocessing is not None:
            preprocessing_reports.append(comm_preprocessing)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "train_eval_results.csv"
    alpha_path = args.output_dir / "train_eval_alpha_search.csv"
    baseline_path = args.output_dir / "train_eval_baseline_search.csv"
    cohort_path = args.output_dir / "station_cohort_results.csv"
    robustness_path = args.output_dir / "station_robustness_statistics.csv"
    sensitivity_path = args.output_dir / "sensitivity_summary.csv"
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

    robustness_rows = build_station_robustness_rows(
        actual=station_actual_test,
        predictions={name: preds[1] for name, preds in station_predictions.items()},
        cohorts=cohorts,
        graph_set=graph_set,
        reference_model=args.robustness_reference_model,
        rng=rng,
        n_bootstrap=args.bootstrap_resamples,
        n_permutations=args.permutation_resamples,
        ci_level=args.ci_level,
        progress=bool(args.progress),
        bootstrap_batch_size=args.bootstrap_batch_size,
        permutation_batch_size=args.permutation_batch_size,
    )
    robustness_df = (
        pl.DataFrame(robustness_rows).sort(["cohort", "model"])
        if robustness_rows
        else pl.DataFrame()
    )

    sensitivity_rows = build_sensitivity_summary_rows(
        results_df=results_df,
        alpha_df=alpha_df,
        baseline_df=baseline_df,
        cohort_df=cohort_df,
        sparse_quantile=args.sparse_quantile,
    )
    sensitivity_df = (
        pl.DataFrame(sensitivity_rows).sort(["sensitivity_axis", "model", "setting"])
        if sensitivity_rows
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

    if robustness_rows:
        robustness_df.write_csv(robustness_path)
    else:
        robustness_path.write_text("", encoding="utf-8")

    if sensitivity_rows:
        sensitivity_df.write_csv(sensitivity_path)
    else:
        sensitivity_path.write_text("", encoding="utf-8")

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
            "robustness_rows": len(robustness_rows),
            "sensitivity_rows": len(sensitivity_rows),
        },
        "paths": {
            "results": str(results_path),
            "alpha_search": str(alpha_path),
            "baseline_search": str(baseline_path),
            "cohort_results": str(cohort_path),
            "robustness_statistics": str(robustness_path),
            "sensitivity_summary": str(sensitivity_path),
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
        stage="phase1_train_eval",
        script="scripts/train_eval_pipeline.py",
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

    print(f"Wrote train/eval results: {results_path}")
    print(f"Wrote graph alpha search: {alpha_path}")
    print(f"Wrote baseline search: {baseline_path}")
    print(f"Wrote station cohort metrics: {cohort_path}")
    print(f"Wrote station robustness statistics: {robustness_path}")
    print(f"Wrote sensitivity summary: {sensitivity_path}")
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
    parser.add_argument("--robustness-reference-model", default="graph_propagation")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--permutation-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=2048)
    parser.add_argument("--permutation-batch-size", type=int, default=4096)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars for long-running robustness calculations.",
    )
    parser.add_argument(
        "--strict-graph-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require graph metadata source to match the train split path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
