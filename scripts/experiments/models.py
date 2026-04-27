from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .config import parse_int_grid, parse_lag_candidates
from .preprocessing import TargetPreprocessingState, inverse_target_predictions


def compute_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    train_series: np.ndarray,
) -> dict[str, float]:
    err = actual - pred
    abs_err = np.abs(err)

    denom = float(np.abs(actual).sum())
    wmape = float(abs_err.sum() / denom) if denom > 0 else np.nan
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err**2)))

    if train_series.shape[0] >= 2:
        mase_denom = float(np.mean(np.abs(np.diff(train_series, axis=0))))
        mase = float(mae / mase_denom) if mase_denom > 0 else np.nan
    else:
        mase = np.nan

    return {"wmape": wmape, "mae": mae, "rmse": rmse, "mase": mase}


def _baseline_validation_metrics(
    pred_model_space: np.ndarray,
    *,
    val_series: np.ndarray,
    train_series: np.ndarray,
    val_series_raw: np.ndarray | None,
    train_series_raw: np.ndarray | None,
    inverse_state: TargetPreprocessingState | None,
    val_pre_residual: np.ndarray | None,
    history: int = 1,
    horizon: int = 1,
) -> dict[str, float]:
    """Validation metrics for one-step baseline preds; inverse to counts when ``inverse_state`` is set."""
    if inverse_state is not None:
        if val_series_raw is None or train_series_raw is None or val_pre_residual is None:
            raise ValueError(
                "val_series_raw, train_series_raw, and val_pre_residual are required when "
                "inverse_state is set (original-scale validation metrics)"
            )
        pred = inverse_target_predictions(
            pred_model_space,
            state=inverse_state,
            context_pre_residual=val_pre_residual,
            history=history,
            horizon=horizon,
        )
        return compute_metrics(
            actual=val_series_raw[1:],
            pred=pred,
            train_series=train_series_raw,
        )
    return compute_metrics(
        actual=val_series[1:],
        pred=pred_model_space,
        train_series=train_series,
    )


def evaluate_one_step_forecast(
    series: np.ndarray,
    adjacency: np.ndarray,
    alpha: float,
    train_series: np.ndarray,
) -> dict[str, float]:
    if series.shape[0] < 2:
        return {"wmape": np.nan, "mae": np.nan, "rmse": np.nan, "mase": np.nan}

    prev = series[:-1]
    actual = series[1:]
    pred = alpha * prev + (1.0 - alpha) * (prev @ adjacency)
    pred = np.clip(pred, a_min=0.0, a_max=None)

    return compute_metrics(actual=actual, pred=pred, train_series=train_series)


def predict_graph_propagation(
    series: np.ndarray,
    adjacency: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if series.shape[0] < 2:
        return np.zeros((0, series.shape[1]))

    prev = series[:-1]
    pred = alpha * prev + (1.0 - alpha) * (prev @ adjacency)
    return np.clip(pred, a_min=0.0, a_max=None)


def tune_graph_alpha(
    train_series: np.ndarray,
    val_series: np.ndarray,
    adjacency: np.ndarray,
    alpha_grid: list[float],
    *,
    val_series_raw: np.ndarray | None = None,
    train_series_raw: np.ndarray | None = None,
    inverse_state: TargetPreprocessingState | None = None,
    val_pre_residual: np.ndarray | None = None,
    history: int = 1,
    horizon: int = 1,
) -> tuple[float, list[dict[str, float]]]:
    if not alpha_grid:
        raise ValueError("alpha_grid cannot be empty")

    use_original_scale = inverse_state is not None
    if use_original_scale:
        if val_series_raw is None or train_series_raw is None or val_pre_residual is None:
            raise ValueError(
                "val_series_raw, train_series_raw, and val_pre_residual are required when "
                "inverse_state is set (original-scale α tuning)"
            )

    best_alpha = alpha_grid[0]
    best_wmape = np.inf
    search_rows: list[dict[str, float]] = []

    for alpha in alpha_grid:
        pred_val = predict_graph_propagation(
            series=val_series,
            adjacency=adjacency,
            alpha=alpha,
        )
        if use_original_scale:
            pred_val = inverse_target_predictions(
                pred_val,
                state=inverse_state,
                context_pre_residual=val_pre_residual,
                history=history,
                horizon=horizon,
            )
            metrics = compute_metrics(
                actual=val_series_raw[1:],
                pred=pred_val,
                train_series=train_series_raw,
            )
        else:
            metrics = compute_metrics(
                actual=val_series[1:],
                pred=pred_val,
                train_series=train_series,
            )
        search_rows.append(
            {
                "alpha": alpha,
                "validation_wmape": metrics["wmape"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_mase": metrics["mase"],
            }
        )
        if np.isfinite(metrics["wmape"]) and metrics["wmape"] < best_wmape:
            best_wmape = metrics["wmape"]
            best_alpha = alpha

    return best_alpha, search_rows


def build_one_step_lag_features(
    series: np.ndarray, lags: tuple[int, ...]
) -> np.ndarray:
    """Build lagged features for one-step prediction using in-split history only."""
    if series.shape[0] < 2:
        return np.zeros((0, len(lags)))

    t_idx = np.arange(1, series.shape[0])
    feature_blocks: list[np.ndarray] = []

    for lag in lags:
        src_idx = np.where((t_idx - lag) >= 0, t_idx - lag, t_idx - 1)
        feature_blocks.append(series[src_idx])

    stacked = np.stack(feature_blocks, axis=-1)
    return stacked.reshape(-1, len(lags))


def evaluate_seasonal_naive(
    series: np.ndarray,
    seasonal_lag: int,
    train_series: np.ndarray,
) -> dict[str, float]:
    if series.shape[0] < 2:
        return {"wmape": np.nan, "mae": np.nan, "rmse": np.nan, "mase": np.nan}

    t_idx = np.arange(1, series.shape[0])
    src_idx = np.where((t_idx - seasonal_lag) >= 0, t_idx - seasonal_lag, t_idx - 1)
    pred = series[src_idx]
    actual = series[1:]
    pred = np.clip(pred, a_min=0.0, a_max=None)

    return compute_metrics(actual=actual, pred=pred, train_series=train_series)


def sample_lagged_training(
    series: np.ndarray,
    lags: tuple[int, ...],
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample lagged (X, y) pairs from a matrix time series."""
    max_lag = max(lags)
    t_count = series.shape[0] - max_lag
    if t_count <= 0 or series.shape[1] == 0:
        return np.zeros((0, len(lags))), np.zeros((0,))

    n_nodes = series.shape[1]
    total = t_count * n_nodes
    sample_size = min(max_samples, total)

    rng = np.random.default_rng(random_state)
    flat = rng.choice(total, size=sample_size, replace=False)
    t_idx = (flat // n_nodes) + max_lag
    node_idx = flat % n_nodes

    features = np.column_stack([series[t_idx - lag, node_idx] for lag in lags])
    target = series[t_idx, node_idx]
    return features.astype(float), target.astype(float)


def predict_with_lagged_model(
    model: Any,
    series: np.ndarray,
    lags: tuple[int, ...],
) -> np.ndarray:
    if series.shape[0] < 2:
        return np.zeros((0, series.shape[1]))

    x = build_one_step_lag_features(series, lags)
    pred = model.predict(x).reshape(series.shape[0] - 1, series.shape[1])
    return np.clip(pred, a_min=0.0, a_max=None)


def fit_best_baseline_models(
    train_series: np.ndarray,
    val_series: np.ndarray,
    args: Any,
    *,
    val_series_raw: np.ndarray | None = None,
    train_series_raw: np.ndarray | None = None,
    inverse_state: TargetPreprocessingState | None = None,
    val_pre_residual: np.ndarray | None = None,
    history: int = 1,
    horizon: int = 1,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Tune baseline model configs on validation and return fitted best models."""
    search_rows: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any]] = {}

    seasonal_lags = parse_int_grid(args.seasonal_lags)
    linear_candidates = parse_lag_candidates(args.linear_lag_candidates)
    tree_candidates = parse_lag_candidates(args.tree_lag_candidates)
    tree_depths = parse_int_grid(args.tree_max_depths)

    best_seasonal_lag = seasonal_lags[0]
    best_seasonal_val = np.inf
    for lag in seasonal_lags:
        if val_series.shape[0] < 2:
            val_metrics = {
                "wmape": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "mase": np.nan,
            }
        else:
            t_idx = np.arange(1, val_series.shape[0])
            src_idx = np.where((t_idx - lag) >= 0, t_idx - lag, t_idx - 1)
            pred_val = np.clip(val_series[src_idx], a_min=0.0, a_max=None)
            val_metrics = _baseline_validation_metrics(
                pred_val,
                val_series=val_series,
                train_series=train_series,
                val_series_raw=val_series_raw,
                train_series_raw=train_series_raw,
                inverse_state=inverse_state,
                val_pre_residual=val_pre_residual,
                history=history,
                horizon=horizon,
            )
        search_rows.append(
            {
                "model": "seasonal_naive",
                "config": json.dumps({"seasonal_lag": lag}),
                "validation_wmape": val_metrics["wmape"],
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mase": val_metrics["mase"],
            }
        )
        if (
            np.isfinite(val_metrics["wmape"])
            and val_metrics["wmape"] < best_seasonal_val
        ):
            best_seasonal_val = val_metrics["wmape"]
            best_seasonal_lag = lag

    fitted["seasonal_naive"] = {
        "model": "seasonal_naive",
        "seasonal_lag": best_seasonal_lag,
        "config": {"seasonal_lag": best_seasonal_lag},
    }

    best_linear: dict[str, Any] | None = None
    best_linear_wmape = np.inf
    for lags in linear_candidates:
        x_train, y_train = sample_lagged_training(
            series=train_series,
            lags=lags,
            max_samples=args.linear_max_samples,
            random_state=args.random_state,
        )
        if x_train.shape[0] == 0:
            continue

        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        pred_val = predict_with_lagged_model(model=model, series=val_series, lags=lags)
        val_metrics = _baseline_validation_metrics(
            pred_val,
            val_series=val_series,
            train_series=train_series,
            val_series_raw=val_series_raw,
            train_series_raw=train_series_raw,
            inverse_state=inverse_state,
            val_pre_residual=val_pre_residual,
            history=history,
            horizon=horizon,
        )
        cfg = {"lags": list(lags), "max_samples": args.linear_max_samples}
        search_rows.append(
            {
                "model": "lagged_linear",
                "config": json.dumps(cfg),
                "validation_wmape": val_metrics["wmape"],
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mase": val_metrics["mase"],
            }
        )
        if (
            np.isfinite(val_metrics["wmape"])
            and val_metrics["wmape"] < best_linear_wmape
        ):
            best_linear_wmape = val_metrics["wmape"]
            best_linear = {"model": model, "lags": lags, "config": cfg}

    if best_linear is not None:
        fitted["lagged_linear"] = {
            "model": "lagged_linear",
            "estimator": best_linear["model"],
            "lags": best_linear["lags"],
            "config": best_linear["config"],
        }

    best_tree: dict[str, Any] | None = None
    best_tree_wmape = np.inf
    for lags in tree_candidates:
        x_train, y_train = sample_lagged_training(
            series=train_series,
            lags=lags,
            max_samples=args.tree_max_samples,
            random_state=args.random_state,
        )
        if x_train.shape[0] == 0:
            continue

        for depth in tree_depths:
            model = RandomForestRegressor(
                n_estimators=args.tree_estimators,
                max_depth=depth,
                random_state=args.random_state,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            pred_val = predict_with_lagged_model(
                model=model, series=val_series, lags=lags
            )
            val_metrics = _baseline_validation_metrics(
                pred_val,
                val_series=val_series,
                train_series=train_series,
                val_series_raw=val_series_raw,
                train_series_raw=train_series_raw,
                inverse_state=inverse_state,
                val_pre_residual=val_pre_residual,
                history=history,
                horizon=horizon,
            )
            cfg = {
                "lags": list(lags),
                "max_depth": depth,
                "n_estimators": args.tree_estimators,
                "max_samples": args.tree_max_samples,
            }
            search_rows.append(
                {
                    "model": "tree_lagged",
                    "config": json.dumps(cfg),
                    "validation_wmape": val_metrics["wmape"],
                    "validation_mae": val_metrics["mae"],
                    "validation_rmse": val_metrics["rmse"],
                    "validation_mase": val_metrics["mase"],
                }
            )
            if (
                np.isfinite(val_metrics["wmape"])
                and val_metrics["wmape"] < best_tree_wmape
            ):
                best_tree_wmape = val_metrics["wmape"]
                best_tree = {"model": model, "lags": lags, "config": cfg}

    if best_tree is not None:
        fitted["tree_lagged"] = {
            "model": "tree_lagged",
            "estimator": best_tree["model"],
            "lags": best_tree["lags"],
            "config": best_tree["config"],
        }

    return fitted, search_rows


def predict_baseline(
    fitted_model: dict[str, Any],
    series: np.ndarray,
) -> np.ndarray:
    name = fitted_model["model"]
    if name == "seasonal_naive":
        lag = int(fitted_model["seasonal_lag"])
        if series.shape[0] < 2:
            return np.zeros((0, series.shape[1]))
        t_idx = np.arange(1, series.shape[0])
        src_idx = np.where((t_idx - lag) >= 0, t_idx - lag, t_idx - 1)
        pred = series[src_idx]
        return np.clip(pred, a_min=0.0, a_max=None)

    if name in {"lagged_linear", "tree_lagged"}:
        return predict_with_lagged_model(
            model=fitted_model["estimator"],
            series=series,
            lags=tuple(fitted_model["lags"]),
        )

    raise ValueError(f"Unknown baseline model type: {name}")


def evaluate_baseline_models(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    args: Any,
    *,
    val_series_raw: np.ndarray | None = None,
    test_series_raw: np.ndarray | None = None,
    train_series_raw: np.ndarray | None = None,
    inverse_state: TargetPreprocessingState | None = None,
    val_pre_residual: np.ndarray | None = None,
    test_pre_residual: np.ndarray | None = None,
    history: int = 1,
    horizon: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tune and evaluate seasonal, linear, and tree baselines once per aggregation."""
    fitted, search_rows = fit_best_baseline_models(
        train_series=train_series,
        val_series=val_series,
        args=args,
        val_series_raw=val_series_raw,
        train_series_raw=train_series_raw,
        inverse_state=inverse_state,
        val_pre_residual=val_pre_residual,
        history=history,
        horizon=horizon,
    )

    use_raw = inverse_state is not None
    if use_raw:
        if (
            val_series_raw is None
            or test_series_raw is None
            or train_series_raw is None
            or val_pre_residual is None
            or test_pre_residual is None
        ):
            raise ValueError(
                "val_series_raw, test_series_raw, train_series_raw, val_pre_residual, "
                "and test_pre_residual are required when inverse_state is set"
            )

    model_rows: list[dict[str, Any]] = []
    for name in ("seasonal_naive", "lagged_linear", "tree_lagged"):
        if name not in fitted:
            continue
        model_spec = fitted[name]
        pred_val = predict_baseline(model_spec, val_series)
        pred_test = predict_baseline(model_spec, test_series)
        if use_raw:
            pred_val = inverse_target_predictions(
                pred_val,
                state=inverse_state,
                context_pre_residual=val_pre_residual,
                history=history,
                horizon=horizon,
            )
            pred_test = inverse_target_predictions(
                pred_test,
                state=inverse_state,
                context_pre_residual=test_pre_residual,
                history=history,
                horizon=horizon,
            )
            val_metrics = compute_metrics(
                actual=val_series_raw[1:],
                pred=pred_val,
                train_series=train_series_raw,
            )
            test_metrics = compute_metrics(
                actual=test_series_raw[1:],
                pred=pred_test,
                train_series=train_series_raw,
            )
        else:
            val_metrics = compute_metrics(
                actual=val_series[1:],
                pred=pred_val,
                train_series=train_series,
            )
            test_metrics = compute_metrics(
                actual=test_series[1:],
                pred=pred_test,
                train_series=train_series,
            )
        model_rows.append(
            {
                "model": name,
                "config": json.dumps(model_spec["config"]),
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

    return model_rows, search_rows
