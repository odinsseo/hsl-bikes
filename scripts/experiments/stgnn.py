from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
    parse_int_grid,
)
from .data import (
    aggregate_adjacency_to_groups,
    build_community_series,
    build_hourly_index,
    build_station_series,
    load_communities,
    load_graph_bundle,
    load_split,
)
from .models import compute_metrics
from .preprocessing import (
    TargetPreprocessingConfig,
    TargetPreprocessingState,
    apply_target_preprocessing,
    build_calendar_feature_matrix,
    build_preprocessing_metadata,
    build_sparse_activity_features,
    build_static_feature_matrix,
    fit_target_preprocessing,
    inverse_target_predictions,
)
from .provenance import build_run_metadata, write_metadata_sidecar
from .safeguards import assert_train_graph_source

ALLOWED_GRAPH_NAMES = {"SD", "DE", "DC", "ATD"}
DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "experiments" / "stgnn_single_graph"
DEFAULT_PREPROCESSED_CACHE_DIR = DATA_DIR / "artifacts" / "cache" / "stgnn_preprocessed"


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, max(cpu_count - 1, 1))


def _path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_preprocessed_cache_key(
    args: argparse.Namespace,
    *,
    holiday_subdivision: str | None,
) -> tuple[str, dict[str, Any]]:
    key_payload: dict[str, Any] = {
        "cache_schema_version": "stgnn-preprocessed-v1",
        "aggregation": str(args.aggregation),
        "splits": {
            "train": _path_identity(args.train),
            "validation": _path_identity(args.validation),
            "test": _path_identity(args.test),
        },
        "graph_dir": _path_identity(args.graph_dir / "station_index.txt"),
        "communities": (
            _path_identity(args.communities)
            if str(args.aggregation) == "community"
            else None
        ),
        "settings": {
            "history": int(args.history),
            "horizon": int(args.horizon),
            "max_train_windows": int(args.max_train_windows),
            "random_state": int(args.random_state),
            "preprocess_target": bool(args.preprocess_target),
            "winsor_lower_quantile": float(args.winsor_lower_quantile),
            "winsor_upper_quantile": float(args.winsor_upper_quantile),
            "preprocess_scaler": str(args.preprocess_scaler),
            "residualize_target": bool(args.residualize_target),
            "residual_lag_candidates": str(args.residual_lag_candidates),
            "holiday_country": str(args.holiday_country),
            "holiday_subdivision": holiday_subdivision,
            "holiday_national_only": bool(args.holiday_national_only),
            "include_calendar_covariates": bool(args.include_calendar_covariates),
            "include_activity_mask": bool(args.include_activity_mask),
            "include_zero_run_indicator": bool(args.include_zero_run_indicator),
            "zero_run_length": int(args.zero_run_length),
            "include_static_features": bool(args.include_static_features),
        },
    }
    payload_json = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:20]
    return digest, key_payload


def _serialize_preprocessing_state(
    state: TargetPreprocessingState | None,
) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "version": state.version,
        "lower_bounds": state.lower_bounds.tolist(),
        "upper_bounds": state.upper_bounds.tolist(),
        "center": state.center.tolist(),
        "scale": state.scale.tolist(),
        "selected_residual_lag": state.selected_residual_lag,
        "config": asdict(state.config),
    }


def _deserialize_preprocessing_state(
    payload: dict[str, Any] | None,
) -> TargetPreprocessingState | None:
    if payload is None:
        return None

    cfg = TargetPreprocessingConfig(**payload["config"])
    return TargetPreprocessingState(
        version=str(payload["version"]),
        lower_bounds=np.asarray(payload["lower_bounds"], dtype=float),
        upper_bounds=np.asarray(payload["upper_bounds"], dtype=float),
        center=np.asarray(payload["center"], dtype=float),
        scale=np.asarray(payload["scale"], dtype=float),
        selected_residual_lag=(
            int(payload["selected_residual_lag"])
            if payload["selected_residual_lag"] is not None
            else None
        ),
        config=cfg,
    )


def _load_preprocessed_cache(
    *,
    cache_root: Path,
    cache_key: str,
) -> dict[str, Any] | None:
    cache_dir = cache_root / cache_key
    manifest_path = cache_dir / "manifest.json"
    arrays_path = cache_dir / "arrays.npz"
    if not manifest_path.exists() or not arrays_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        cached_arrays = {name: arrays[name] for name in arrays.files}

    sample_indices = cached_arrays.get("train_sample_indices")
    sample_indices_present = bool(manifest.get("sample_indices_present", False))
    train_sample_indices = None
    if sample_indices_present and sample_indices is not None:
        train_sample_indices = np.asarray(sample_indices, dtype=int)

    return {
        "manifest": manifest,
        "train_series_raw": np.asarray(cached_arrays["train_series_raw"], dtype=float),
        "val_series_raw": np.asarray(cached_arrays["val_series_raw"], dtype=float),
        "test_series_raw": np.asarray(cached_arrays["test_series_raw"], dtype=float),
        "train_series": np.asarray(cached_arrays["train_series"], dtype=float),
        "val_series": np.asarray(cached_arrays["val_series"], dtype=float),
        "test_series": np.asarray(cached_arrays["test_series"], dtype=float),
        "train_dynamic_covariates": np.asarray(
            cached_arrays["train_dynamic_covariates"], dtype=float
        ),
        "val_dynamic_covariates": np.asarray(
            cached_arrays["val_dynamic_covariates"], dtype=float
        ),
        "test_dynamic_covariates": np.asarray(
            cached_arrays["test_dynamic_covariates"], dtype=float
        ),
        "static_covariates": np.asarray(
            cached_arrays["static_covariates"], dtype=float
        ),
        "val_pre_residual": np.asarray(cached_arrays["val_pre_residual"], dtype=float),
        "test_pre_residual": np.asarray(
            cached_arrays["test_pre_residual"], dtype=float
        ),
        "train_sample_indices": train_sample_indices,
        "train_time_bounds": (
            tuple(manifest["train_time_bounds"])
            if manifest.get("train_time_bounds") is not None
            else None
        ),
        "dynamic_feature_names": list(manifest.get("dynamic_feature_names", [])),
        "sparse_feature_names": list(manifest.get("sparse_feature_names", [])),
        "static_feature_names": list(manifest.get("static_feature_names", [])),
        "preprocessing_metadata": manifest.get("preprocessing_metadata"),
        "preprocessing_state": _deserialize_preprocessing_state(
            manifest.get("preprocessing_state")
        ),
        "residual_lag_scores": list(manifest.get("residual_lag_scores", [])),
    }


def _save_preprocessed_cache(
    *,
    cache_root: Path,
    cache_key: str,
    cache_key_payload: dict[str, Any],
    train_series_raw: np.ndarray,
    val_series_raw: np.ndarray,
    test_series_raw: np.ndarray,
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    train_dynamic_covariates: np.ndarray,
    val_dynamic_covariates: np.ndarray,
    test_dynamic_covariates: np.ndarray,
    static_covariates: np.ndarray,
    val_pre_residual: np.ndarray,
    test_pre_residual: np.ndarray,
    train_sample_indices: np.ndarray | None,
    train_time_bounds: tuple[str, str] | None,
    dynamic_feature_names: list[str],
    sparse_feature_names: list[str],
    static_feature_names: list[str],
    preprocessing_metadata: dict[str, Any] | None,
    preprocessing_state: TargetPreprocessingState | None,
    residual_lag_scores: list[dict[str, float]],
) -> None:
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_indices_present = train_sample_indices is not None
    indices_array = (
        np.asarray(train_sample_indices, dtype=int)
        if sample_indices_present
        else np.zeros((0,), dtype=int)
    )

    arrays_path = cache_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        train_series_raw=np.asarray(train_series_raw, dtype=float),
        val_series_raw=np.asarray(val_series_raw, dtype=float),
        test_series_raw=np.asarray(test_series_raw, dtype=float),
        train_series=np.asarray(train_series, dtype=float),
        val_series=np.asarray(val_series, dtype=float),
        test_series=np.asarray(test_series, dtype=float),
        train_dynamic_covariates=np.asarray(train_dynamic_covariates, dtype=float),
        val_dynamic_covariates=np.asarray(val_dynamic_covariates, dtype=float),
        test_dynamic_covariates=np.asarray(test_dynamic_covariates, dtype=float),
        static_covariates=np.asarray(static_covariates, dtype=float),
        val_pre_residual=np.asarray(val_pre_residual, dtype=float),
        test_pre_residual=np.asarray(test_pre_residual, dtype=float),
        train_sample_indices=indices_array,
    )

    manifest = {
        "cache_schema_version": "stgnn-preprocessed-v1",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "key_payload": cache_key_payload,
        "sample_indices_present": sample_indices_present,
        "train_time_bounds": list(train_time_bounds) if train_time_bounds else None,
        "dynamic_feature_names": list(dynamic_feature_names),
        "sparse_feature_names": list(sparse_feature_names),
        "static_feature_names": list(static_feature_names),
        "preprocessing_metadata": preprocessing_metadata,
        "preprocessing_state": _serialize_preprocessing_state(preprocessing_state),
        "residual_lag_scores": residual_lag_scores,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _iter_epochs(total_epochs: int, *, enabled: bool) -> Any:
    epoch_range = range(1, total_epochs + 1)
    if enabled and tqdm is not None:
        return tqdm(epoch_range, desc="ST-GNN epochs", unit="epoch")
    return epoch_range


def parse_graph_set(value: str) -> tuple[str, ...]:
    parts = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("graph set is empty")

    unknown = [name for name in parts if name not in ALLOWED_GRAPH_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown graph(s) '{sorted(set(unknown))}'. Allowed values: {sorted(ALLOWED_GRAPH_NAMES)}"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for name in parts:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return tuple(deduped)


def parse_graph_name(value: str) -> str:
    graph_set = parse_graph_set(value)
    if len(graph_set) != 1:
        raise ValueError(
            "Expected a single graph name. Use one of: "
            f"{sorted(ALLOWED_GRAPH_NAMES)}"
        )
    return graph_set[0]


def _resolve_holiday_subdivision(args: argparse.Namespace) -> str | None:
    if getattr(args, "holiday_national_only", False):
        return None
    return args.holiday_subdivision


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_adjacency_for_gcn(
    adjacency: np.ndarray,
    add_self_loops: bool = True,
) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    out = matrix.copy()
    if add_self_loops:
        out = out + np.eye(out.shape[0], dtype=float)

    degree = out.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask] ** -0.5

    return inv_sqrt[:, None] * out * inv_sqrt[None, :]


def build_supervised_windows(
    series: np.ndarray,
    history: int,
    horizon: int,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if series.ndim != 2:
        raise ValueError("series must be 2-D [time, nodes]")
    if history <= 0:
        raise ValueError("history must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    time_steps, nodes = series.shape
    sample_count = time_steps - history - horizon + 1
    if sample_count <= 0:
        return (
            np.zeros((0, history, nodes), dtype=float),
            np.zeros((0, nodes), dtype=float),
        )

    if sample_indices is None:
        sample_idx = np.arange(sample_count, dtype=int)
    else:
        sample_idx = np.asarray(sample_indices, dtype=int)
        if sample_idx.ndim != 1:
            raise ValueError("sample_indices must be 1-D when provided")
        if sample_idx.size == 0:
            return (
                np.zeros((0, history, nodes), dtype=float),
                np.zeros((0, nodes), dtype=float),
            )
        if (sample_idx < 0).any() or (sample_idx >= sample_count).any():
            raise ValueError("sample_indices contains out-of-range window indices")

    features = np.stack(
        [series[i : i + history] for i in sample_idx],
        axis=0,
    )
    target_idx = sample_idx + history + horizon - 1
    target = series[target_idx]
    return features, target


def build_stgnn_windows_with_covariates(
    series: np.ndarray,
    history: int,
    horizon: int,
    *,
    dynamic_covariates: np.ndarray | None = None,
    static_covariates: np.ndarray | None = None,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_x, base_y = build_supervised_windows(
        series=series,
        history=history,
        horizon=horizon,
        sample_indices=sample_indices,
    )
    sample_count = base_x.shape[0]
    nodes = series.shape[1]

    dynamic_dim = 0
    if dynamic_covariates is not None:
        if dynamic_covariates.ndim != 3:
            raise ValueError("dynamic_covariates must be 3-D [time, nodes, features]")
        if (
            dynamic_covariates.shape[0] != series.shape[0]
            or dynamic_covariates.shape[1] != nodes
        ):
            raise ValueError("dynamic_covariates shape is incompatible with series")
        dynamic_dim = int(dynamic_covariates.shape[2])

    static_dim = 0
    if static_covariates is not None:
        if static_covariates.ndim != 2:
            raise ValueError("static_covariates must be 2-D [nodes, features]")
        if static_covariates.shape[0] != nodes:
            raise ValueError("static_covariates node dimension must match series")
        static_dim = int(static_covariates.shape[1])

    total_dim = 1 + dynamic_dim + static_dim
    if sample_count <= 0:
        return np.zeros((0, history, nodes, total_dim), dtype=float), base_y

    blocks = [base_x[..., None]]

    if dynamic_covariates is not None and dynamic_dim > 0:
        if sample_indices is None:
            dynamic_idx = np.arange(sample_count, dtype=int)
        else:
            dynamic_idx = np.asarray(sample_indices, dtype=int)
        dynamic_windows = np.stack(
            [dynamic_covariates[i : i + history] for i in dynamic_idx],
            axis=0,
        )
        blocks.append(dynamic_windows)

    if static_covariates is not None and static_dim > 0:
        static_block = np.broadcast_to(
            static_covariates[None, None, :, :],
            (sample_count, history, nodes, static_dim),
        )
        blocks.append(static_block)

    return np.concatenate(blocks, axis=-1), base_y


def build_dynamic_covariates(
    series_raw: np.ndarray,
    timestamps: list[datetime],
    *,
    include_calendar_covariates: bool,
    include_activity_mask: bool,
    include_zero_run_indicator: bool,
    zero_run_length: int,
    holiday_country: str,
    holiday_subdivision: str | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if series_raw.ndim != 2:
        raise ValueError("series_raw must be 2-D [time, nodes]")

    time_steps, nodes = series_raw.shape
    if timestamps and len(timestamps) != time_steps:
        raise ValueError(
            "timestamps length must match series_raw time axis when timestamps are provided"
        )

    blocks: list[np.ndarray] = []
    dynamic_names: list[str] = []

    if include_calendar_covariates:
        if len(timestamps) != time_steps:
            raise ValueError("calendar covariates require one timestamp per time step")
        calendar_matrix, calendar_names = build_calendar_feature_matrix(
            timestamps,
            country=holiday_country,
            subdivision=holiday_subdivision,
        )
        calendar_block = np.broadcast_to(
            calendar_matrix[:, None, :],
            (time_steps, nodes, calendar_matrix.shape[1]),
        )
        blocks.append(calendar_block)
        dynamic_names.extend(calendar_names)

    sparse_block, sparse_names = build_sparse_activity_features(
        series_raw,
        include_activity_mask=include_activity_mask,
        include_zero_run_indicator=include_zero_run_indicator,
        zero_run_length=zero_run_length,
    )
    if sparse_block.shape[2] > 0:
        blocks.append(sparse_block)
        dynamic_names.extend(sparse_names)

    if not blocks:
        return np.zeros((time_steps, nodes, 0), dtype=float), [], []
    return np.concatenate(blocks, axis=-1), dynamic_names, sparse_names


def _time_bounds(timestamps: list[datetime]) -> tuple[str, str] | None:
    if not timestamps:
        return None
    return timestamps[0].isoformat(), timestamps[-1].isoformat()


def maybe_subsample_windows(
    features: np.ndarray,
    target: np.ndarray,
    max_windows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_windows <= 0 or features.shape[0] <= max_windows:
        return features, target

    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(features.shape[0], size=max_windows, replace=False))
    return features[idx], target[idx]


def maybe_subsample_window_indices(
    sample_count: int,
    max_windows: int,
    random_state: int,
) -> np.ndarray | None:
    if max_windows <= 0 or sample_count <= max_windows:
        return None
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(sample_count, size=max_windows, replace=False))


class STGNNWindowDataset(Dataset[Any]):
    def __init__(
        self,
        *,
        series: np.ndarray,
        history: int,
        horizon: int,
        dynamic_covariates: np.ndarray | None = None,
        static_covariates: np.ndarray | None = None,
        sample_indices: np.ndarray | None = None,
    ) -> None:
        if history <= 0:
            raise ValueError("history must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        self.series = np.asarray(series, dtype=np.float32)
        if self.series.ndim != 2:
            raise ValueError("series must be 2-D [time, nodes]")

        self.history = int(history)
        self.horizon = int(horizon)
        self.nodes = int(self.series.shape[1])
        self.dynamic_covariates = None
        self.static_covariates = None
        self.dynamic_dim = 0
        self.static_dim = 0

        if dynamic_covariates is not None:
            dyn = np.asarray(dynamic_covariates, dtype=np.float32)
            if dyn.ndim != 3:
                raise ValueError(
                    "dynamic_covariates must be 3-D [time, nodes, features]"
                )
            if dyn.shape[0] != self.series.shape[0] or dyn.shape[1] != self.nodes:
                raise ValueError("dynamic_covariates shape is incompatible with series")
            self.dynamic_covariates = dyn
            self.dynamic_dim = int(dyn.shape[2])

        if static_covariates is not None:
            stat = np.asarray(static_covariates, dtype=np.float32)
            if stat.ndim != 2:
                raise ValueError("static_covariates must be 2-D [nodes, features]")
            if stat.shape[0] != self.nodes:
                raise ValueError("static_covariates node dimension must match series")
            self.static_covariates = stat
            self.static_dim = int(stat.shape[1])

        total = self.series.shape[0] - self.history - self.horizon + 1
        if total <= 0:
            self.sample_indices = np.zeros((0,), dtype=int)
        elif sample_indices is None:
            self.sample_indices = np.arange(total, dtype=int)
        else:
            idx = np.asarray(sample_indices, dtype=int)
            if idx.ndim != 1:
                raise ValueError("sample_indices must be 1-D when provided")
            if (idx < 0).any() or (idx >= total).any():
                raise ValueError("sample_indices contains out-of-range window indices")
            self.sample_indices = idx

        self.input_channels = 1 + self.dynamic_dim + self.static_dim
        self._static_history_block = None
        if self.static_covariates is not None and self.static_dim > 0:
            self._static_history_block = np.broadcast_to(
                self.static_covariates[None, :, :],
                (self.history, self.nodes, self.static_dim),
            )

    def __len__(self) -> int:
        return int(self.sample_indices.size)

    def __getitem__(self, index: int) -> Any:
        start = int(self.sample_indices[index])
        end = start + self.history

        base = self.series[start:end][..., None].astype(np.float32, copy=False)
        dynamic = None
        if self.dynamic_covariates is not None and self.dynamic_dim > 0:
            dynamic = self.dynamic_covariates[start:end].astype(np.float32, copy=False)
        static = None
        if self._static_history_block is not None:
            static = self._static_history_block.astype(np.float32, copy=False)

        target_idx = start + self.history + self.horizon - 1
        target = self.series[target_idx]
        return (
            base,
            dynamic,
            static,
            target.astype(np.float32, copy=False),
        )

    def targets_array(self) -> np.ndarray:
        if self.sample_indices.size == 0:
            return np.zeros((0, self.nodes), dtype=np.float32)
        target_idx = self.sample_indices + self.history + self.horizon - 1
        return self.series[target_idx].astype(np.float32, copy=False)


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device_arg not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    return device_arg


class A3TGCNGraphFusion(nn.Module):
    """A3T-GCN-style baseline with fixed or learned fusion over graph views."""

    def __init__(
        self,
        adjacency_stack: np.ndarray,
        input_channels: int,
        hidden_dim: int,
        dropout: float,
        learnable_fusion: bool,
    ) -> None:
        super().__init__()
        if adjacency_stack.ndim != 3:
            raise ValueError("adjacency_stack must be 3-D [graphs, nodes, nodes]")

        adj_tensor = torch.tensor(adjacency_stack, dtype=torch.float32)
        self.register_buffer("adjacency_stack", adj_tensor)
        self.learnable_fusion = bool(learnable_fusion)

        n_graphs = int(adjacency_stack.shape[0])
        if self.learnable_fusion and n_graphs > 1:
            self.fusion_logits = nn.Parameter(torch.zeros(n_graphs))
            self.register_buffer("fusion_weights_fixed", torch.zeros(0))
        else:
            fixed = np.full(n_graphs, 1.0 / float(n_graphs), dtype=float)
            self.register_buffer(
                "fusion_weights_fixed",
                torch.tensor(fixed, dtype=torch.float32),
            )

        if input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.temporal_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def _fusion_weights(self) -> torch.Tensor:
        if self.learnable_fusion and hasattr(self, "fusion_logits"):
            return torch.softmax(self.fusion_logits, dim=0)
        return self.fusion_weights_fixed

    def get_fusion_weights(self) -> np.ndarray:
        return self._fusion_weights().detach().cpu().numpy().astype(float)

    def forward(self, x: Any) -> Any:
        # x shape: [batch, history, nodes] or [batch, history, nodes, channels]
        if x.ndim == 3:
            x4 = x.unsqueeze(-1)
            residual = x[:, -1, :]
        elif x.ndim == 4:
            x4 = x
            residual = x[:, -1, :, 0]
        else:
            raise ValueError("Input tensor must be 3-D or 4-D")

        fused_adjacency = torch.einsum(
            "g,gij->ij",
            self._fusion_weights(),
            self.adjacency_stack,
        )
        propagated = torch.einsum("ij,btjf->btif", fused_adjacency, x4)
        hidden = torch.tanh(self.input_proj(propagated))
        hidden = self.dropout(hidden)

        attn_logits = self.temporal_score(hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1)

        out = self.output_proj(context).squeeze(-1)
        return out + residual


def _to_loader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    random_state: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    collate_fn: Any | None = None,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(random_state)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (persistent_workers if num_workers > 0 else False),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _stgnn_window_collate(
    batch: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    base_parts, dynamic_parts, static_parts, targets = zip(*batch)

    batch_size = len(batch)
    history = int(base_parts[0].shape[0])
    nodes = int(base_parts[0].shape[1])
    dynamic_dim = int(dynamic_parts[0].shape[2]) if dynamic_parts[0] is not None else 0
    static_dim = int(static_parts[0].shape[2]) if static_parts[0] is not None else 0
    total_dim = 1 + dynamic_dim + static_dim

    # Preallocate final tensor to avoid stack+concatenate temporaries in workers.
    features = np.empty((batch_size, history, nodes, total_dim), dtype=np.float32)

    dynamic_start = 1
    static_start = dynamic_start + dynamic_dim
    for i in range(batch_size):
        features[i, :, :, 0:1] = base_parts[i]
        if dynamic_dim > 0:
            assert dynamic_parts[i] is not None
            features[i, :, :, dynamic_start:static_start] = dynamic_parts[i]
        if static_dim > 0:
            assert static_parts[i] is not None
            features[i, :, :, static_start:] = static_parts[i]

    target_nodes = int(targets[0].shape[0])
    targets_batch = np.empty((batch_size, target_nodes), dtype=np.float32)
    for i in range(batch_size):
        targets_batch[i, :] = targets[i]

    return (
        torch.from_numpy(features),
        torch.from_numpy(targets_batch),
    )


def _shutdown_loader_workers(loader: Any | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown_fn = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_fn):
        try:
            shutdown_fn()
        except Exception:
            pass


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str,
    non_blocking: bool,
    max_grad_norm: float,
) -> float:
    model.train()
    losses: list[float] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else np.nan


def _eval_loss(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str,
    non_blocking: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            pred = model(x_batch)
            losses.append(float(loss_fn(pred, y_batch).detach().cpu()))

    return float(np.mean(losses)) if losses else np.nan


def _eval_loss_and_predict(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str,
    non_blocking: bool,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            pred = model(x_batch)
            losses.append(float(loss_fn(pred, y_batch).detach().cpu()))
            outputs.append(pred.detach().cpu().numpy())

    preds = (
        np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=float)
    )
    return (float(np.mean(losses)) if losses else np.nan), preds


def _predict(model: Any, loader: Any, device: str, non_blocking: bool) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            pred = (
                model(x_batch.to(device, non_blocking=non_blocking))
                .detach()
                .cpu()
                .numpy()
            )
            outputs.append(pred)

    if not outputs:
        return np.zeros((0, 0), dtype=float)
    return np.concatenate(outputs, axis=0)


def run(args: argparse.Namespace) -> int:
    set_random_seed(args.random_state)
    device = _resolve_device(args.device)

    if args.strict_graph_source:
        assert_train_graph_source(
            graph_dir=args.graph_dir,
            train_path=args.train,
            allow_leaky_graph_source=bool(args.allow_leaky_graph_source),
        )

    graph_set_text = args.graph_set if args.graph_set else args.graph
    graph_set = parse_graph_set(graph_set_text)

    if args.fusion_mode == "single" and len(graph_set) != 1:
        raise ValueError("fusion-mode=single requires exactly one graph in graph-set")

    station_index, matrices = load_graph_bundle(args.graph_dir)
    holiday_subdivision = _resolve_holiday_subdivision(args)

    station_to_group: dict[str, str] | None = None
    groups: list[str] | None = None
    if args.aggregation == "community":
        station_to_group = load_communities(args.communities, station_index)
        groups = sorted(set(station_to_group.values()))

    if args.aggregation == "station":
        adjacency_mats = [np.asarray(matrices[name], dtype=float) for name in graph_set]
    else:
        assert station_to_group is not None
        assert groups is not None
        adjacency_mats = [
            aggregate_adjacency_to_groups(
                adjacency=np.asarray(matrices[name], dtype=float),
                station_index=station_index,
                station_to_group=station_to_group,
                groups=groups,
            )
            for name in graph_set
        ]

    cache_enabled = bool(getattr(args, "cache_preprocessed", True))
    cache_refresh = bool(getattr(args, "refresh_preprocessed_cache", False))
    cache_root = Path(
        getattr(args, "preprocessed_cache_dir", DEFAULT_PREPROCESSED_CACHE_DIR)
    )
    cache_key, cache_key_payload = _build_preprocessed_cache_key(
        args,
        holiday_subdivision=holiday_subdivision,
    )
    cache_hit = False

    preprocessing_metadata: dict[str, Any] | None = None
    preprocessing_state: TargetPreprocessingState | None = None
    residual_lag_scores: list[dict[str, float]] = []

    cache_payload = None
    if cache_enabled and not cache_refresh:
        cache_payload = _load_preprocessed_cache(
            cache_root=cache_root, cache_key=cache_key
        )

    if cache_payload is not None:
        cache_hit = True
        train_series_raw = cache_payload["train_series_raw"]
        val_series_raw = cache_payload["val_series_raw"]
        test_series_raw = cache_payload["test_series_raw"]

        train_series = cache_payload["train_series"]
        val_series = cache_payload["val_series"]
        test_series = cache_payload["test_series"]

        train_dynamic_covariates = cache_payload["train_dynamic_covariates"]
        val_dynamic_covariates = cache_payload["val_dynamic_covariates"]
        test_dynamic_covariates = cache_payload["test_dynamic_covariates"]
        static_covariates = cache_payload["static_covariates"]

        val_pre_residual = cache_payload["val_pre_residual"]
        test_pre_residual = cache_payload["test_pre_residual"]
        train_sample_indices = cache_payload["train_sample_indices"]

        train_time_bounds = cache_payload["train_time_bounds"]
        dynamic_feature_names = cache_payload["dynamic_feature_names"]
        sparse_feature_names = cache_payload["sparse_feature_names"]
        static_feature_names = cache_payload["static_feature_names"]

        preprocessing_metadata = cache_payload["preprocessing_metadata"]
        preprocessing_state = cache_payload["preprocessing_state"]
        residual_lag_scores = cache_payload["residual_lag_scores"]
        print(f"Loaded preprocessed cache: {cache_root / cache_key}")
    else:
        train_df = load_split(args.train)
        val_df = load_split(args.validation)
        test_df = load_split(args.test)

        if args.aggregation == "station":
            train_series_raw = build_station_series(train_df, station_index)
            val_series_raw = build_station_series(val_df, station_index)
            test_series_raw = build_station_series(test_df, station_index)
        else:
            assert station_to_group is not None
            assert groups is not None
            train_series_raw = build_community_series(
                train_df, station_to_group, groups
            )
            val_series_raw = build_community_series(val_df, station_to_group, groups)
            test_series_raw = build_community_series(test_df, station_to_group, groups)

        train_hours = build_hourly_index(train_df)
        val_hours = build_hourly_index(val_df)
        test_hours = build_hourly_index(test_df)
        train_time_bounds = _time_bounds(train_hours)

        for split_name, series_split, hours_split in (
            ("train", train_series_raw, train_hours),
            ("validation", val_series_raw, val_hours),
            ("test", test_series_raw, test_hours),
        ):
            if len(hours_split) != int(series_split.shape[0]):
                raise ValueError(
                    f"{split_name} hourly index length ({len(hours_split)}) does not match series rows ({series_split.shape[0]})"
                )

        train_dynamic_covariates, dynamic_feature_names, sparse_feature_names = (
            build_dynamic_covariates(
                train_series_raw,
                train_hours,
                include_calendar_covariates=bool(args.include_calendar_covariates),
                include_activity_mask=bool(args.include_activity_mask),
                include_zero_run_indicator=bool(args.include_zero_run_indicator),
                zero_run_length=args.zero_run_length,
                holiday_country=args.holiday_country,
                holiday_subdivision=holiday_subdivision,
            )
        )
        val_dynamic_covariates, val_dynamic_feature_names, _ = build_dynamic_covariates(
            val_series_raw,
            val_hours,
            include_calendar_covariates=bool(args.include_calendar_covariates),
            include_activity_mask=bool(args.include_activity_mask),
            include_zero_run_indicator=bool(args.include_zero_run_indicator),
            zero_run_length=args.zero_run_length,
            holiday_country=args.holiday_country,
            holiday_subdivision=holiday_subdivision,
        )
        test_dynamic_covariates, test_dynamic_feature_names, _ = (
            build_dynamic_covariates(
                test_series_raw,
                test_hours,
                include_calendar_covariates=bool(args.include_calendar_covariates),
                include_activity_mask=bool(args.include_activity_mask),
                include_zero_run_indicator=bool(args.include_zero_run_indicator),
                zero_run_length=args.zero_run_length,
                holiday_country=args.holiday_country,
                holiday_subdivision=holiday_subdivision,
            )
        )

        if (
            dynamic_feature_names != val_dynamic_feature_names
            or dynamic_feature_names != test_dynamic_feature_names
        ):
            raise ValueError(
                "Dynamic covariate feature definitions are inconsistent across splits"
            )

        if args.include_static_features:
            static_covariates, static_feature_names = build_static_feature_matrix(
                train_series_raw
            )
        else:
            static_covariates = np.zeros((train_series_raw.shape[1], 0), dtype=float)
            static_feature_names = []

        val_pre_residual = val_series_raw
        test_pre_residual = test_series_raw
        if args.preprocess_target:
            preprocessing_config = TargetPreprocessingConfig(
                winsor_lower_quantile=args.winsor_lower_quantile,
                winsor_upper_quantile=args.winsor_upper_quantile,
                enable_log1p=True,
                scaler=args.preprocess_scaler,
                enable_residualization=bool(args.residualize_target),
                residual_lag_candidates=tuple(
                    parse_int_grid(args.residual_lag_candidates)
                ),
                holiday_country=args.holiday_country,
                holiday_subdivision=holiday_subdivision,
            )
            preprocessing_state, lag_scores = fit_target_preprocessing(
                train_series_raw,
                validation_series=val_series_raw,
                config=preprocessing_config,
            )
            train_applied = apply_target_preprocessing(
                train_series_raw, preprocessing_state
            )
            val_applied = apply_target_preprocessing(
                val_series_raw, preprocessing_state
            )
            test_applied = apply_target_preprocessing(
                test_series_raw,
                preprocessing_state,
            )

            train_series = train_applied.transformed
            val_series = val_applied.transformed
            test_series = test_applied.transformed
            val_pre_residual = val_applied.pre_residual
            test_pre_residual = test_applied.pre_residual

            preprocessing_metadata = build_preprocessing_metadata(
                preprocessing_state,
                train_time_bounds=train_time_bounds,
                dynamic_feature_definitions=dynamic_feature_names,
                static_feature_definitions=static_feature_names,
                sparse_feature_definitions=sparse_feature_names,
            )
            residual_lag_scores = [
                {"lag": int(row.lag), "score": float(row.score)} for row in lag_scores
            ]
        else:
            train_series = train_series_raw
            val_series = val_series_raw
            test_series = test_series_raw

        train_total_windows = train_series.shape[0] - args.history - args.horizon + 1
        train_sample_indices = maybe_subsample_window_indices(
            sample_count=train_total_windows,
            max_windows=args.max_train_windows,
            random_state=args.random_state,
        )

        if cache_enabled:
            _save_preprocessed_cache(
                cache_root=cache_root,
                cache_key=cache_key,
                cache_key_payload=cache_key_payload,
                train_series_raw=train_series_raw,
                val_series_raw=val_series_raw,
                test_series_raw=test_series_raw,
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                train_dynamic_covariates=train_dynamic_covariates,
                val_dynamic_covariates=val_dynamic_covariates,
                test_dynamic_covariates=test_dynamic_covariates,
                static_covariates=static_covariates,
                val_pre_residual=val_pre_residual,
                test_pre_residual=test_pre_residual,
                train_sample_indices=train_sample_indices,
                train_time_bounds=train_time_bounds,
                dynamic_feature_names=dynamic_feature_names,
                sparse_feature_names=sparse_feature_names,
                static_feature_names=static_feature_names,
                preprocessing_metadata=preprocessing_metadata,
                preprocessing_state=preprocessing_state,
                residual_lag_scores=residual_lag_scores,
            )
            print(f"Saved preprocessed cache: {cache_root / cache_key}")

    if cache_hit and args.preprocess_target and preprocessing_state is None:
        raise RuntimeError(
            "Cached preprocessing state missing while preprocess_target is enabled"
        )

    use_lazy_windows = bool(getattr(args, "lazy_windows", True))
    lazy_collate_fn: Any | None = _stgnn_window_collate if use_lazy_windows else None
    if use_lazy_windows:
        train_dataset = STGNNWindowDataset(
            series=train_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=train_dynamic_covariates,
            static_covariates=static_covariates,
            sample_indices=train_sample_indices,
        )
        val_dataset = STGNNWindowDataset(
            series=val_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=val_dynamic_covariates,
            static_covariates=static_covariates,
        )
        test_dataset = STGNNWindowDataset(
            series=test_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=test_dynamic_covariates,
            static_covariates=static_covariates,
        )
        val_y = val_dataset.targets_array()
        input_channels = int(train_dataset.input_channels)
        train_windows = len(train_dataset)
        validation_windows = len(val_dataset)
        test_windows = len(test_dataset)
    else:
        warnings.warn(
            "Using legacy eager window materialization path (--no-lazy-windows). "
            "This fallback is retained temporarily and may be removed once rollout is complete.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_x, train_y = build_stgnn_windows_with_covariates(
            series=train_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=train_dynamic_covariates,
            static_covariates=static_covariates,
            sample_indices=train_sample_indices,
        )
        val_x, val_y = build_stgnn_windows_with_covariates(
            series=val_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=val_dynamic_covariates,
            static_covariates=static_covariates,
        )
        test_x, test_y = build_stgnn_windows_with_covariates(
            series=test_series,
            history=args.history,
            horizon=args.horizon,
            dynamic_covariates=test_dynamic_covariates,
            static_covariates=static_covariates,
        )
        input_channels = int(train_x.shape[-1])
        train_windows = int(train_x.shape[0])
        validation_windows = int(val_x.shape[0])
        test_windows = int(test_x.shape[0])

        train_dataset = TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        )
        test_dataset = TensorDataset(
            torch.tensor(test_x, dtype=torch.float32),
            torch.tensor(test_y, dtype=torch.float32),
        )

    _, val_y_raw = build_supervised_windows(
        series=val_series_raw,
        history=args.history,
        horizon=args.horizon,
    )
    _, test_y_raw = build_supervised_windows(
        series=test_series_raw,
        history=args.history,
        horizon=args.horizon,
    )

    if train_windows == 0 or validation_windows == 0 or test_windows == 0:
        raise ValueError(
            "Insufficient time steps for selected history/horizon on one of the splits"
        )

    norm_adj_stack = np.stack(
        [
            normalize_adjacency_for_gcn(adjacency, add_self_loops=True)
            for adjacency in adjacency_mats
        ],
        axis=0,
    )

    learnable_fusion = args.fusion_mode == "learned" and len(graph_set) > 1

    num_workers = max(int(getattr(args, "num_workers", 0)), 0)
    pin_memory = bool(getattr(args, "pin_memory", True)) and device == "cuda"
    prefetch_factor = max(int(getattr(args, "prefetch_factor", 2)), 1)
    persistent_workers = bool(getattr(args, "persistent_workers", True))
    non_blocking = bool(pin_memory)

    train_loader = _to_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        random_state=args.random_state,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=lazy_collate_fn,
    )
    # Validation/test use single-process loading to avoid worker-side peak RAM spikes
    # from collate-time batch assembly during long milestone sweeps.
    eval_num_workers = 0
    val_loader = _to_loader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        collate_fn=lazy_collate_fn,
    )
    test_loader = _to_loader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
        num_workers=eval_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        collate_fn=lazy_collate_fn,
    )

    model = A3TGCNGraphFusion(
        adjacency_stack=norm_adj_stack,
        input_channels=input_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learnable_fusion=learnable_fusion,
    ).to(device)

    optimizer_name = str(getattr(args, "optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = str(getattr(args, "lr_scheduler", "plateau")).lower()
    early_stop_patience = max(1, int(args.patience))
    early_stop_min_delta = max(0.0, float(getattr(args, "early_stop_min_delta", 1e-3)))
    early_stop_start_epoch = max(1, int(getattr(args, "early_stop_start_epoch", 5)))

    lr_decay_patience_requested = int(getattr(args, "lr_decay_patience", 5))
    lr_decay_patience = min(
        max(0, lr_decay_patience_requested),
        max(0, early_stop_patience - 1),
    )
    lr_plateau_threshold_raw = getattr(args, "lr_plateau_threshold", None)
    lr_plateau_threshold = (
        early_stop_min_delta
        if lr_plateau_threshold_raw is None
        else max(0.0, float(lr_plateau_threshold_raw))
    )

    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if scheduler_name == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "lr_decay_factor", 0.5)),
            patience=lr_decay_patience,
            threshold=lr_plateau_threshold,
            threshold_mode="abs",
            min_lr=float(getattr(args, "min_learning_rate", 1e-5)),
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported lr-scheduler: {scheduler_name}")

    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
    huber_delta = float(getattr(args, "huber_delta", 1.0))
    loss_fn = nn.HuberLoss(delta=huber_delta)

    history_rows: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_wmape = np.inf
    best_val_huber = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    epoch_progress = bool(getattr(args, "epoch_progress", False))
    epoch_iter = _iter_epochs(args.epochs, enabled=epoch_progress)

    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            non_blocking=non_blocking,
            max_grad_norm=max_grad_norm,
        )
        val_loss, val_pred_model_space = _eval_loss_and_predict(
            model,
            val_loader,
            loss_fn,
            device,
            non_blocking=non_blocking,
        )
        if args.preprocess_target:
            val_pred = inverse_target_predictions(
                val_pred_model_space,
                state=preprocessing_state,
                context_pre_residual=val_pre_residual,
                history=args.history,
                horizon=args.horizon,
            )
        else:
            val_pred = val_pred_model_space
        val_metrics_epoch = compute_metrics(
            actual=val_y_raw,
            pred=val_pred,
            train_series=train_series_raw,
        )
        val_wmape = float(val_metrics_epoch["wmape"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_huber": train_loss,
                "validation_huber": val_loss,
                "validation_wmape": val_wmape,
                # Backward-compatible aliases consumed by historical diagnostics notebook.
                "train_mse": train_loss,
                "validation_mse": val_loss,
            }
        )

        if (
            lr_scheduler is not None
            and epoch >= early_stop_start_epoch
            and np.isfinite(val_wmape)
        ):
            lr_scheduler.step(val_wmape)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - epoch_start
        if epoch_progress and tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                {
                    "train_huber": f"{train_loss:.5f}",
                    "val_huber": f"{val_loss:.5f}",
                    "val_wmape": f"{val_wmape:.5f}",
                    "best_wmape": f"{best_val_wmape:.5f}",
                    "lr": f"{current_lr:.2e}",
                    "sec": f"{epoch_time:.1f}",
                }
            )
        elif epoch_progress and tqdm is None:
            print(
                f"[epoch {epoch}/{args.epochs}] "
                f"train_huber={train_loss:.5f} val_huber={val_loss:.5f} "
                f"val_wmape={val_wmape:.5f} best_wmape={best_val_wmape:.5f} "
                f"lr={current_lr:.2e} sec={epoch_time:.1f}"
            )

        if np.isfinite(val_wmape) and val_wmape < (
            best_val_wmape - early_stop_min_delta
        ):
            best_val_wmape = val_wmape
            best_val_huber = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        elif epoch >= early_stop_start_epoch:
            epochs_without_improvement += 1

        if (
            epoch >= early_stop_start_epoch
            and epochs_without_improvement >= early_stop_patience
        ):
            break

    model.load_state_dict(best_state)

    val_pred_model_space = _predict(
        model,
        val_loader,
        device,
        non_blocking=non_blocking,
    )
    test_pred_model_space = _predict(
        model,
        test_loader,
        device,
        non_blocking=non_blocking,
    )

    if args.preprocess_target:
        val_pred = inverse_target_predictions(
            val_pred_model_space,
            state=preprocessing_state,
            context_pre_residual=val_pre_residual,
            history=args.history,
            horizon=args.horizon,
        )
        test_pred = inverse_target_predictions(
            test_pred_model_space,
            state=preprocessing_state,
            context_pre_residual=test_pre_residual,
            history=args.history,
            horizon=args.horizon,
        )
    else:
        val_pred = val_pred_model_space
        test_pred = test_pred_model_space

    val_metrics = compute_metrics(
        actual=val_y_raw,
        pred=val_pred,
        train_series=train_series_raw,
    )
    test_metrics = compute_metrics(
        actual=test_y_raw,
        pred=test_pred,
        train_series=train_series_raw,
    )
    transformed_metrics = compute_metrics(
        actual=val_y,
        pred=val_pred_model_space,
        train_series=train_series,
    )
    fusion_weights = model.get_fusion_weights().tolist()
    fusion_weights_by_graph = {
        graph: float(weight) for graph, weight in zip(graph_set, fusion_weights)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = args.output_dir / "training_curve.csv"
    results_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"

    pl.DataFrame(history_rows).write_csv(curve_path)
    pl.DataFrame(
        [
            {
                "aggregation": args.aggregation,
                "graph": graph_set[0] if len(graph_set) == 1 else "+".join(graph_set),
                "graph_set": "+".join(graph_set),
                "fusion_mode": args.fusion_mode,
                "model": "A3T_GCN_graph_fusion",
                "history": args.history,
                "horizon": args.horizon,
                "input_channels": input_channels,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "epochs_requested": args.epochs,
                "epochs_ran": len(history_rows),
                "best_epoch": best_epoch,
                "train_windows": train_windows,
                "validation_windows": validation_windows,
                "test_windows": test_windows,
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
                "fusion_weights": json.dumps(fusion_weights_by_graph),
            }
        ]
    ).write_csv(results_path)

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "aggregation": args.aggregation,
        "graph": graph_set[0] if len(graph_set) == 1 else "+".join(graph_set),
        "graph_set": list(graph_set),
        "fusion_mode": args.fusion_mode,
        "model": "A3T_GCN_graph_fusion",
        "device": device,
        "history": args.history,
        "horizon": args.horizon,
        "input_channels": input_channels,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "persistent_workers": (persistent_workers if num_workers > 0 else False),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_function": {
            "name": "huber",
            "delta": huber_delta,
            "masked": False,
            "space": "transformed",
        },
        "selection_metric": {
            "name": "validation_wmape",
            "space": "original",
            "mode": "min",
            "min_delta": early_stop_min_delta,
        },
        "early_stopping": {
            "monitor": "validation_wmape",
            "mode": "min",
            "patience": early_stop_patience,
            "min_delta": early_stop_min_delta,
            "start_epoch": early_stop_start_epoch,
        },
        "optimizer": optimizer_name,
        "lr_scheduler": scheduler_name,
        "max_grad_norm": max_grad_norm,
        "epoch_progress": epoch_progress,
        "lr_decay_factor": (
            float(getattr(args, "lr_decay_factor", 0.5))
            if scheduler_name == "plateau"
            else None
        ),
        "lr_decay_patience": (
            lr_decay_patience if scheduler_name == "plateau" else None
        ),
        "lr_decay_patience_requested": (
            lr_decay_patience_requested if scheduler_name == "plateau" else None
        ),
        "lr_plateau_threshold": (
            lr_plateau_threshold if scheduler_name == "plateau" else None
        ),
        "lr_plateau_threshold_mode": ("abs" if scheduler_name == "plateau" else None),
        "lr_plateau_start_epoch": (
            early_stop_start_epoch if scheduler_name == "plateau" else None
        ),
        "min_learning_rate": (
            float(getattr(args, "min_learning_rate", 1e-5))
            if scheduler_name == "plateau"
            else None
        ),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history_rows),
        "best_epoch": best_epoch,
        "best_validation_wmape": best_val_wmape,
        "best_validation_huber": best_val_huber,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "fusion_weights": fusion_weights_by_graph,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_model_space": transformed_metrics,
        },
        "covariates": {
            "enabled": bool(dynamic_feature_names or static_feature_names),
            "dynamic_feature_definitions": dynamic_feature_names,
            "static_feature_definitions": static_feature_names,
            "sparse_feature_definitions": sparse_feature_names,
            "train_time_bounds": (
                {
                    "start": train_time_bounds[0],
                    "end": train_time_bounds[1],
                }
                if train_time_bounds is not None
                else None
            ),
        },
        "preprocessing_cache": {
            "enabled": cache_enabled,
            "hit": cache_hit,
            "key": cache_key if cache_enabled else None,
            "path": str(cache_root / cache_key) if cache_enabled else None,
        },
        "paths": {
            "training_curve": str(curve_path),
            "results": str(results_path),
        },
    }

    metadata = build_run_metadata(
        args=args,
        stage="phase3_stgnn",
        script="scripts/train_stgnn_pipeline.py",
        extra={
            "strict_graph_source": bool(args.strict_graph_source),
            "allow_leaky_graph_source": bool(args.allow_leaky_graph_source),
            "preprocessing_cache": summary["preprocessing_cache"],
            "preprocessing": preprocessing_metadata,
            "residual_lag_scores": residual_lag_scores,
            "covariates": summary["covariates"],
        },
    )
    metadata_path = args.output_dir / "metadata.json"
    write_metadata_sidecar(metadata_path, metadata)
    summary["paths"]["metadata"] = str(metadata_path)
    summary["run_metadata"] = metadata
    if preprocessing_metadata is not None:
        summary["preprocessing"] = preprocessing_metadata
        summary["preprocessing"]["residual_lag_scores"] = residual_lag_scores

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote training curve: {curve_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")

    # Explicitly release large arrays/loaders between milestone configs.
    _shutdown_loader_workers(train_loader)
    _shutdown_loader_workers(val_loader)
    _shutdown_loader_workers(test_loader)
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    del model, optimizer
    if lr_scheduler is not None:
        del lr_scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate single-graph A3T-GCN-style baseline"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--communities", type=Path, default=DEFAULT_COMMUNITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--aggregation", choices=["station", "community"], default="station"
    )
    parser.add_argument(
        "--graph",
        default="DE",
        help="Single graph shortcut (deprecated in favor of --graph-set)",
    )
    parser.add_argument(
        "--graph-set",
        default=None,
        help="Comma-separated graph set, e.g. DE or SD,DE,DC,ATD",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["single", "equal", "learned"],
        default="single",
        help="single: one graph only, equal: fixed mean over graph-set, learned: trainable fusion weights",
    )

    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
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
        help="Optimizer for ST-GNN training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="plateau",
        help="Learning-rate scheduler strategy.",
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
        help="Show per-epoch progress (loss, lr, and epoch time) during training.",
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
    parser.add_argument("--residual-lag-candidates", default="24,168")
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
    parser.add_argument("--zero-run-length", type=int, default=6)
    parser.add_argument(
        "--include-static-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include train-derived static node context features.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched per worker when num-workers > 0.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DataLoader pinned memory for faster host->GPU transfers.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive between epochs when num-workers > 0.",
    )
    parser.add_argument(
        "--lazy-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Construct windows lazily via dataset indexing (default). Use --no-lazy-windows for legacy eager fallback.",
    )
    parser.add_argument(
        "--cache-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save and reuse deterministic preprocessing outputs across runs.",
    )
    parser.add_argument(
        "--refresh-preprocessed-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore any cache hit and rebuild preprocessing artifacts.",
    )
    parser.add_argument(
        "--preprocessed-cache-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED_CACHE_DIR,
        help="Directory for cached preprocessing artifacts.",
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--strict-graph-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require graph metadata source to match the train split path.",
    )
    parser.add_argument(
        "--allow-leaky-graph-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow leaky graph metadata source (for leakage-sensitivity runs only).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
