from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import holidays
import numpy as np

_EPS = 1e-8


@dataclass(frozen=True)
class TargetPreprocessingConfig:
    winsor_lower_quantile: float = 0.005
    winsor_upper_quantile: float = 0.995
    enable_log1p: bool = True
    scaler: str = "robust"
    enable_residualization: bool = True
    residual_lag_candidates: tuple[int, ...] = (24, 168)
    holiday_country: str = "FI"
    holiday_subdivision: str | None = "18"


@dataclass(frozen=True)
class TargetPreprocessingState:
    version: str
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    selected_residual_lag: int | None
    config: TargetPreprocessingConfig


@dataclass(frozen=True)
class AppliedTargetPreprocessing:
    transformed: np.ndarray
    pre_residual: np.ndarray


@dataclass(frozen=True)
class ResidualLagCandidateScore:
    lag: int
    score: float


def _validate_quantiles(lower: float, upper: float) -> None:
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")


def _validate_lags(lags: tuple[int, ...]) -> tuple[int, ...]:
    if not lags:
        raise ValueError("residual_lag_candidates cannot be empty")
    if any(lag <= 0 for lag in lags):
        raise ValueError("all residual lag candidates must be > 0")

    deduped: list[int] = []
    seen: set[int] = set()
    for lag in lags:
        if lag in seen:
            continue
        deduped.append(int(lag))
        seen.add(int(lag))
    return tuple(deduped)


def _validate_series(name: str, series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [time, node]")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    if (arr < 0.0).any():
        raise ValueError(
            f"{name} contains negative counts; expected non-negative demand values"
        )
    return arr


def _winsorize(series: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(series, lower, upper)


def _seasonal_baseline(series: np.ndarray, lag: int) -> np.ndarray:
    t_idx = np.arange(series.shape[0])
    src_idx = np.where((t_idx - lag) >= 0, t_idx - lag, np.maximum(t_idx - 1, 0))
    return series[src_idx]


def _apply_base_transform(
    series: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    enable_log1p: bool,
) -> np.ndarray:
    out = _winsorize(series, lower_bounds, upper_bounds)
    if enable_log1p:
        out = np.log1p(np.clip(out, a_min=0.0, a_max=None))
    return (out - center) / scale


def _score_residual_lag(base_series: np.ndarray, lag: int) -> float:
    baseline = _seasonal_baseline(base_series, lag)
    residual = base_series - baseline
    tail = residual[1:] if residual.shape[0] > 1 else residual
    return float(np.mean(np.abs(tail)))


def fit_target_preprocessing(
    train_series: np.ndarray,
    *,
    validation_series: np.ndarray | None = None,
    config: TargetPreprocessingConfig | None = None,
) -> tuple[
    TargetPreprocessingState,
    list[ResidualLagCandidateScore],
]:
    cfg = config or TargetPreprocessingConfig()
    _validate_quantiles(cfg.winsor_lower_quantile, cfg.winsor_upper_quantile)
    lag_candidates = _validate_lags(cfg.residual_lag_candidates)

    if cfg.scaler != "robust":
        raise ValueError(f"Unsupported scaler: {cfg.scaler}")

    train = _validate_series("train_series", train_series)
    val = None
    if validation_series is not None:
        val = _validate_series("validation_series", validation_series)

    lower_bounds = np.quantile(train, cfg.winsor_lower_quantile, axis=0)
    upper_bounds = np.quantile(train, cfg.winsor_upper_quantile, axis=0)

    train_wins = _winsorize(train, lower_bounds, upper_bounds)
    train_scaled_source = (
        np.log1p(np.clip(train_wins, a_min=0.0, a_max=None))
        if cfg.enable_log1p
        else train_wins
    )

    center = np.median(train_scaled_source, axis=0)
    q75 = np.quantile(train_scaled_source, 0.75, axis=0)
    q25 = np.quantile(train_scaled_source, 0.25, axis=0)
    scale = np.where(np.abs(q75 - q25) > _EPS, q75 - q25, 1.0)

    selected_residual_lag: int | None = None
    lag_scores: list[ResidualLagCandidateScore] = []

    if cfg.enable_residualization:
        if val is None:
            selected_residual_lag = int(lag_candidates[0])
        else:
            val_base = _apply_base_transform(
                val,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                center=center,
                scale=scale,
                enable_log1p=cfg.enable_log1p,
            )
            best_lag = int(lag_candidates[0])
            best_score = np.inf
            for lag in lag_candidates:
                score = _score_residual_lag(val_base, lag)
                lag_scores.append(ResidualLagCandidateScore(lag=int(lag), score=score))
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_lag = int(lag)
            selected_residual_lag = best_lag

    state = TargetPreprocessingState(
        version="v1",
        lower_bounds=lower_bounds.astype(float),
        upper_bounds=upper_bounds.astype(float),
        center=center.astype(float),
        scale=scale.astype(float),
        selected_residual_lag=selected_residual_lag,
        config=TargetPreprocessingConfig(
            winsor_lower_quantile=float(cfg.winsor_lower_quantile),
            winsor_upper_quantile=float(cfg.winsor_upper_quantile),
            enable_log1p=bool(cfg.enable_log1p),
            scaler=str(cfg.scaler),
            enable_residualization=bool(cfg.enable_residualization),
            residual_lag_candidates=lag_candidates,
            holiday_country=str(cfg.holiday_country),
            holiday_subdivision=(
                str(cfg.holiday_subdivision)
                if cfg.holiday_subdivision is not None
                else None
            ),
        ),
    )
    return state, lag_scores


def apply_target_preprocessing(
    series: np.ndarray,
    state: TargetPreprocessingState,
) -> AppliedTargetPreprocessing:
    arr = _validate_series("series", series)
    base = _apply_base_transform(
        arr,
        lower_bounds=state.lower_bounds,
        upper_bounds=state.upper_bounds,
        center=state.center,
        scale=state.scale,
        enable_log1p=state.config.enable_log1p,
    )

    transformed = base
    if state.config.enable_residualization and state.selected_residual_lag is not None:
        transformed = base - _seasonal_baseline(base, state.selected_residual_lag)

    return AppliedTargetPreprocessing(
        transformed=transformed.astype(float),
        pre_residual=base.astype(float),
    )


def _target_baseline_for_windows(
    pre_residual_series: np.ndarray,
    *,
    history: int,
    horizon: int,
    lag: int,
) -> np.ndarray:
    if pre_residual_series.ndim != 2:
        raise ValueError("pre_residual_series must be 2-D [time, node]")

    sample_count = pre_residual_series.shape[0] - history - horizon + 1
    if sample_count <= 0:
        return np.zeros((0, pre_residual_series.shape[1]), dtype=float)

    target_idx = np.arange(sample_count) + history + horizon - 1
    src_idx = np.where(
        (target_idx - lag) >= 0,
        target_idx - lag,
        np.maximum(target_idx - 1, 0),
    )
    return pre_residual_series[src_idx]


def inverse_target_predictions(
    predictions: np.ndarray,
    *,
    state: TargetPreprocessingState,
    context_pre_residual: np.ndarray,
    history: int,
    horizon: int,
) -> np.ndarray:
    pred = np.asarray(predictions, dtype=float)
    if pred.ndim != 2:
        raise ValueError("predictions must be 2-D [samples, node]")

    base = pred.copy()
    if state.config.enable_residualization and state.selected_residual_lag is not None:
        baseline = _target_baseline_for_windows(
            context_pre_residual,
            history=history,
            horizon=horizon,
            lag=state.selected_residual_lag,
        )
        if baseline.shape != base.shape:
            raise ValueError(
                "Baseline shape does not match predictions shape for inverse residualization"
            )
        base = base + baseline

    out = (base * state.scale) + state.center
    if state.config.enable_log1p:
        out = np.expm1(out)

    return np.clip(out, a_min=0.0, a_max=None).astype(float)


def build_preprocessing_metadata(
    state: TargetPreprocessingState,
    *,
    fitted_split: str = "train",
    train_time_bounds: tuple[str, str] | None = None,
    dynamic_feature_definitions: list[str] | None = None,
    static_feature_definitions: list[str] | None = None,
    sparse_feature_definitions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "preprocessing_version": state.version,
        "fitted_split": fitted_split,
        "train_time_bounds": (
            {
                "start": train_time_bounds[0],
                "end": train_time_bounds[1],
            }
            if train_time_bounds is not None
            else None
        ),
        "quantile_bounds": {
            "lower": float(state.config.winsor_lower_quantile),
            "upper": float(state.config.winsor_upper_quantile),
        },
        "scaler_type": state.config.scaler,
        "residual_lag_policy": list(state.config.residual_lag_candidates),
        "selected_residual_lag": state.selected_residual_lag,
        "calendar_source": {
            "library": "holidays",
            "country": state.config.holiday_country,
            "subdivision": state.config.holiday_subdivision,
        },
        "dynamic_feature_definitions": list(dynamic_feature_definitions or []),
        "static_feature_definitions": list(static_feature_definitions or []),
        "sparse_feature_definitions": list(sparse_feature_definitions or []),
        "train_only_fit": True,
    }


def build_calendar_feature_matrix(
    timestamps: list[datetime],
    *,
    country: str = "FI",
    subdivision: str | None = "18",
) -> tuple[np.ndarray, list[str]]:
    if not timestamps:
        return np.zeros((0, 0), dtype=float), []

    hour = np.asarray([ts.hour for ts in timestamps], dtype=float)
    weekday = np.asarray([ts.weekday() for ts in timestamps], dtype=float)
    weekend = np.asarray([1.0 if ts.weekday() >= 5 else 0.0 for ts in timestamps])
    holiday = build_holiday_flags(
        timestamps,
        country=country,
        subdivision=subdivision,
    )

    hour_angle = 2.0 * np.pi * hour / 24.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0

    matrix = np.column_stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            weekend,
            holiday,
        ]
    ).astype(float)
    names = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
    ]
    return matrix, names


def build_static_feature_matrix(
    train_series_raw: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    train = _validate_series("train_series_raw", train_series_raw)
    if train.shape[0] == 0:
        return np.zeros((train.shape[1], 0), dtype=float), []

    mean = train.mean(axis=0)
    var = train.var(axis=0)
    zero_rate = (train <= 0.0).mean(axis=0)

    # Keep static channels on comparable scales to avoid one channel dominating input projections.
    mean_min = float(np.min(mean))
    mean_max = float(np.max(mean))
    mean_denom = max(mean_max - mean_min, _EPS)
    mean_scaled = (mean - mean_min) / mean_denom

    var_min = float(np.min(var))
    var_max = float(np.max(var))
    var_denom = max(var_max - var_min, _EPS)
    var_scaled = (var - var_min) / var_denom

    matrix = np.column_stack([mean_scaled, var_scaled, zero_rate]).astype(float)
    names = ["train_mean", "train_variance", "train_zero_rate"]
    return matrix, names


def build_sparse_activity_features(
    series_raw: np.ndarray,
    *,
    include_activity_mask: bool = True,
    include_zero_run_indicator: bool = False,
    zero_run_length: int = 6,
) -> tuple[np.ndarray, list[str]]:
    arr = _validate_series("series_raw", series_raw)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 0), dtype=float), []

    if zero_run_length <= 0:
        raise ValueError("zero_run_length must be > 0")

    blocks: list[np.ndarray] = []
    names: list[str] = []

    if include_activity_mask:
        blocks.append((arr > 0.0).astype(float)[..., None])
        names.append("recent_activity_mask")

    if include_zero_run_indicator:
        zero_mask = (arr <= 0.0).astype(np.int32)
        prefix = np.vstack(
            [
                np.zeros((1, zero_mask.shape[1]), dtype=np.int32),
                np.cumsum(zero_mask, axis=0, dtype=np.int32),
            ]
        )
        end_idx = np.arange(1, arr.shape[0] + 1, dtype=np.int32)
        start_idx = np.maximum(end_idx - int(zero_run_length), 0)
        window_sum = prefix[end_idx, :] - prefix[start_idx, :]
        zero_run = (window_sum == int(zero_run_length)).astype(float)
        blocks.append(zero_run[..., None])
        names.append("long_zero_run_indicator")

    if not blocks:
        return np.zeros((arr.shape[0], arr.shape[1], 0), dtype=float), []

    return np.concatenate(blocks, axis=-1).astype(float), names


def build_holiday_flags(
    timestamps: list[datetime],
    *,
    country: str = "FI",
    subdivision: str | None = "18",
) -> np.ndarray:
    if not timestamps:
        return np.zeros((0,), dtype=float)

    years = sorted({ts.year for ts in timestamps})
    calendar = holidays.country_holidays(country, years=years, subdiv=subdivision)
    flags = [
        1.0 if date(ts.year, ts.month, ts.day) in calendar else 0.0 for ts in timestamps
    ]
    return np.asarray(flags, dtype=float)
