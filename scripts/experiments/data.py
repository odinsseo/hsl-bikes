from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .config import CSV_SCHEMA_OVERRIDES, row_normalize

# Canonical resolution for station/community demand tensors and time indices.
DEMAND_TIME_BUCKET = "3h"


def load_split(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True, schema_overrides=CSV_SCHEMA_OVERRIDES)
    if "departure" not in df.columns:
        raise ValueError(f"Required column missing in {path}: departure")

    return df.with_columns(
        pl.col("departure")
        .cast(pl.String, strict=False)
        .str.strptime(pl.Datetime, strict=False)
        .alias("departure_ts")
    ).drop_nulls(["departure_name", "departure_ts"])


def load_graph_bundle(graph_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    station_index_path = graph_dir / "station_index.txt"
    matrix_paths = {
        "SD": graph_dir / "SD.npy",
        "DE": graph_dir / "DE.npy",
        "DC": graph_dir / "DC.npy",
        "ATD": graph_dir / "ATD.npy",
    }

    missing = [
        str(p) for p in [station_index_path, *matrix_paths.values()] if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing graph artifacts: {missing}")

    station_index = [
        line.strip()
        for line in station_index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrices = {name: np.load(path) for name, path in matrix_paths.items()}

    n = len(station_index)
    for name, matrix in matrices.items():
        if matrix.shape != (n, n):
            raise ValueError(
                f"{name} shape {matrix.shape} does not match station size {n}"
            )

    return station_index, matrices


def load_communities(path: Path, station_index: list[str]) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Community mapping not found: {path}")

    df = pd.read_csv(path)
    if "station_name" not in df.columns or "community" not in df.columns:
        raise ValueError(
            "Communities CSV must contain station_name and community columns"
        )

    mapping = {
        str(row["station_name"]): str(row["community"])
        for _, row in df.iterrows()
        if pd.notna(row["station_name"]) and pd.notna(row["community"])
    }

    for station in station_index:
        if station not in mapping:
            mapping[station] = f"missing::{station}"

    return mapping


def _dense_demand_bucket_pivot(
    df: pl.DataFrame, group_col: str, group_values: list[str]
) -> np.ndarray:
    """Aggregate trips into ``DEMAND_TIME_BUCKET`` bins and return a dense [time, group] matrix."""
    bucketed = (
        df.with_columns(
            pl.col("departure_ts").dt.truncate(DEMAND_TIME_BUCKET).alias("bucket_start")
        )
        .group_by(["bucket_start", group_col])
        .len()
        .rename({"len": "demand"})
    )

    if bucketed.height == 0:
        return np.zeros((0, len(group_values)), dtype=float)

    pivot = (
        bucketed.pivot(
            values="demand",
            index="bucket_start",
            on=group_col,
            aggregate_function="sum",
        )
        .sort("bucket_start")
        .fill_null(0)
    )

    missing_groups = [value for value in group_values if value not in pivot.columns]
    if missing_groups:
        pivot = pivot.with_columns([pl.lit(0).alias(value) for value in missing_groups])

    start = pivot.get_column("bucket_start").min()
    end = pivot.get_column("bucket_start").max()
    full_index = pl.DataFrame(
        {
            "bucket_start": pl.datetime_range(
                start, end, interval=DEMAND_TIME_BUCKET, eager=True
            )
        }
    )
    dense = full_index.join(pivot, on="bucket_start", how="left").fill_null(0)

    return dense.select(group_values).to_numpy().astype(float)


def build_station_series(df: pl.DataFrame, station_index: list[str]) -> np.ndarray:
    return _dense_demand_bucket_pivot(
        df,
        group_col="departure_name",
        group_values=station_index,
    )


def build_community_series(
    df: pl.DataFrame,
    station_to_group: dict[str, str],
    groups: list[str],
) -> np.ndarray:
    mapping_df = pl.DataFrame(
        {
            "departure_name": list(station_to_group.keys()),
            "community": list(station_to_group.values()),
        }
    )
    with_group = (
        df.join(mapping_df, on="departure_name", how="left")
        .drop_nulls(["community"])
        .with_columns(pl.col("community").cast(pl.String, strict=False))
    )
    return _dense_demand_bucket_pivot(with_group, group_col="community", group_values=groups)


def build_hourly_index(df: pl.DataFrame) -> list[datetime]:
    """Return dense timestamps for each demand row (one per ``DEMAND_TIME_BUCKET`` step)."""
    bucketed = df.with_columns(
        pl.col("departure_ts").dt.truncate(DEMAND_TIME_BUCKET).alias("bucket_start")
    )
    buckets = bucketed.get_column("bucket_start")
    if buckets is None or len(buckets) == 0:
        return []

    start = buckets.min()
    end = buckets.max()
    return (
        pl.datetime_range(start, end, interval=DEMAND_TIME_BUCKET, eager=True)
        .to_list()
    )


def aggregate_adjacency_to_groups(
    adjacency: np.ndarray,
    station_index: list[str],
    station_to_group: dict[str, str],
    groups: list[str],
) -> np.ndarray:
    station_to_idx = {station: i for i, station in enumerate(station_index)}
    group_to_indices: dict[str, list[int]] = {g: [] for g in groups}

    for station in station_index:
        grp = station_to_group.get(station)
        if grp is None:
            continue
        group_to_indices[grp].append(station_to_idx[station])

    g = len(groups)
    out = np.zeros((g, g), dtype=float)

    for i, src_group in enumerate(groups):
        src_idx = group_to_indices[src_group]
        if not src_idx:
            continue
        for j, dst_group in enumerate(groups):
            dst_idx = group_to_indices[dst_group]
            if not dst_idx:
                continue
            block = adjacency[np.ix_(src_idx, dst_idx)]
            out[i, j] = float(block.mean()) if block.size else 0.0

    np.fill_diagonal(out, 0.0)
    return row_normalize(out)


def build_fused_adjacency(
    graph_set: tuple[str, ...],
    aggregation: str,
    matrices: dict[str, np.ndarray],
    station_index: list[str],
    station_to_group: dict[str, str] | None,
    groups: list[str] | None,
) -> np.ndarray:
    selected = [
        row_normalize(np.asarray(matrices[name], dtype=float)) for name in graph_set
    ]

    if aggregation == "station":
        fused = np.mean(selected, axis=0)
        return row_normalize(fused)

    if aggregation == "community":
        if station_to_group is None or groups is None:
            raise ValueError(
                "Community aggregation requested without community mapping"
            )
        aggregated = [
            aggregate_adjacency_to_groups(mat, station_index, station_to_group, groups)
            for mat in selected
        ]
        fused = np.mean(aggregated, axis=0)
        return row_normalize(fused)

    raise ValueError(f"Unknown aggregation type: {aggregation}")
