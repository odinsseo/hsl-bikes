from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DATA_DIR,
    DEFAULT_COMMUNITIES,
    DEFAULT_GRAPH_DIR,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
)
from .data import (
    aggregate_adjacency_to_groups,
    build_community_series,
    build_station_series,
    load_communities,
    load_graph_bundle,
    load_split,
)
from .models import compute_metrics

ALLOWED_GRAPH_NAMES = {"SD", "DE", "DC", "ATD"}
DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "experiments" / "stgnn_single_graph"


def parse_graph_name(value: str) -> str:
    graph = value.strip().upper()
    if graph not in ALLOWED_GRAPH_NAMES:
        raise ValueError(
            f"Unknown graph '{value}'. Allowed values: {sorted(ALLOWED_GRAPH_NAMES)}"
        )
    return graph


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

    features = np.stack(
        [series[i : i + history] for i in range(sample_count)],
        axis=0,
    ).astype(float)
    target = np.stack(
        [series[i + history + horizon - 1] for i in range(sample_count)],
        axis=0,
    ).astype(float)
    return features, target


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


class A3TGCNSingleGraph(nn.Module):
    """Lightweight A3T-GCN-style baseline with temporal attention over graph-propagated inputs."""

    def __init__(
        self,
        adjacency: np.ndarray,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        adj_tensor = torch.tensor(adjacency, dtype=torch.float32)
        self.register_buffer("adjacency", adj_tensor)

        self.input_proj = nn.Linear(1, hidden_dim)
        self.temporal_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: Any) -> Any:
        # x shape: [batch, history, nodes]
        x4 = x.unsqueeze(-1)
        propagated = torch.einsum("ij,btjf->btif", self.adjacency, x4)
        hidden = torch.tanh(self.input_proj(propagated))
        hidden = self.dropout(hidden)

        attn_logits = self.temporal_score(hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = (attn_weights.unsqueeze(-1) * hidden).sum(dim=1)

        out = self.output_proj(context).squeeze(-1)
        residual = x[:, -1, :]
        return torch.relu(out + residual)


def _to_loader(
    features: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_state: int,
) -> Any:
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(target, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(random_state)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str,
) -> float:
    model.train()
    losses: list[float] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else np.nan


def _eval_loss(
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: str,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            losses.append(float(loss_fn(pred, y_batch).detach().cpu()))

    return float(np.mean(losses)) if losses else np.nan


def _predict(model: Any, loader: Any, device: str) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            pred = model(x_batch.to(device)).detach().cpu().numpy()
            outputs.append(pred)

    if not outputs:
        return np.zeros((0, 0), dtype=float)
    return np.concatenate(outputs, axis=0)


def run(args: argparse.Namespace) -> int:
    graph_name = parse_graph_name(args.graph)
    set_random_seed(args.random_state)
    device = _resolve_device(args.device)

    station_index, matrices = load_graph_bundle(args.graph_dir)
    train_df = load_split(args.train)
    val_df = load_split(args.validation)
    test_df = load_split(args.test)

    if args.aggregation == "station":
        train_series = build_station_series(train_df, station_index)
        val_series = build_station_series(val_df, station_index)
        test_series = build_station_series(test_df, station_index)
        adjacency = np.asarray(matrices[graph_name], dtype=float)
    else:
        station_to_group = load_communities(args.communities, station_index)
        groups = sorted(set(station_to_group.values()))

        train_series = build_community_series(train_df, station_to_group, groups)
        val_series = build_community_series(val_df, station_to_group, groups)
        test_series = build_community_series(test_df, station_to_group, groups)
        adjacency = aggregate_adjacency_to_groups(
            adjacency=np.asarray(matrices[graph_name], dtype=float),
            station_index=station_index,
            station_to_group=station_to_group,
            groups=groups,
        )

    train_x, train_y = build_supervised_windows(
        series=train_series,
        history=args.history,
        horizon=args.horizon,
    )
    val_x, val_y = build_supervised_windows(
        series=val_series,
        history=args.history,
        horizon=args.horizon,
    )
    test_x, test_y = build_supervised_windows(
        series=test_series,
        history=args.history,
        horizon=args.horizon,
    )

    if train_x.shape[0] == 0 or val_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError(
            "Insufficient time steps for selected history/horizon on one of the splits"
        )

    train_x, train_y = maybe_subsample_windows(
        features=train_x,
        target=train_y,
        max_windows=args.max_train_windows,
        random_state=args.random_state,
    )

    norm_adj = normalize_adjacency_for_gcn(adjacency, add_self_loops=True)

    train_loader = _to_loader(
        features=train_x,
        target=train_y,
        batch_size=args.batch_size,
        shuffle=True,
        random_state=args.random_state,
    )
    val_loader = _to_loader(
        features=val_x,
        target=val_y,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
    )
    test_loader = _to_loader(
        features=test_x,
        target=test_y,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=args.random_state,
    )

    model = A3TGCNSingleGraph(
        adjacency=norm_adj,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    history_rows: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = _eval_loss(model, val_loader, loss_fn, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": val_loss,
            }
        )

        if np.isfinite(val_loss) and val_loss < (best_val_loss - 1e-10):
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(best_state)

    val_pred = _predict(model, val_loader, device)
    test_pred = _predict(model, test_loader, device)

    val_metrics = compute_metrics(
        actual=val_y, pred=val_pred, train_series=train_series
    )
    test_metrics = compute_metrics(
        actual=test_y, pred=test_pred, train_series=train_series
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = args.output_dir / "training_curve.csv"
    results_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"

    pl.DataFrame(history_rows).write_csv(curve_path)
    pl.DataFrame(
        [
            {
                "aggregation": args.aggregation,
                "graph": graph_name,
                "model": "A3T_GCN_single_graph",
                "history": args.history,
                "horizon": args.horizon,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "epochs_requested": args.epochs,
                "epochs_ran": len(history_rows),
                "best_epoch": best_epoch,
                "train_windows": int(train_x.shape[0]),
                "validation_windows": int(val_x.shape[0]),
                "test_windows": int(test_x.shape[0]),
                "validation_wmape": val_metrics["wmape"],
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mase": val_metrics["mase"],
                "test_wmape": test_metrics["wmape"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mase": test_metrics["mase"],
            }
        ]
    ).write_csv(results_path)

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "aggregation": args.aggregation,
        "graph": graph_name,
        "model": "A3T_GCN_single_graph",
        "device": device,
        "history": args.history,
        "horizon": args.horizon,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history_rows),
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_loss,
        "train_windows": int(train_x.shape[0]),
        "validation_windows": int(val_x.shape[0]),
        "test_windows": int(test_x.shape[0]),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "paths": {
            "training_curve": str(curve_path),
            "results": str(results_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote training curve: {curve_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")
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
    parser.add_argument("--graph", default="DE")

    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train-windows", type=int, default=0)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args)
