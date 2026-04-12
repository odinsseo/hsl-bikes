"""Construct SD, DE, DC, and ATD graph views from city-bike OD data.

Usage:
    python scripts/graph_construction.py --input data/prepared/splits/train/train.csv --output-dir data/artifacts/graphs/train
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "prepared" / "splits" / "train" / "train.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "graphs" / "train"
CSV_SCHEMA_OVERRIDES = {
    "departure_id": pl.String,
    "return_id": pl.String,
    "departure_name": pl.String,
    "return_name": pl.String,
}


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize adjacency matrix while keeping all-zero rows unchanged."""
    matrix = matrix.astype(float, copy=True)
    row_sums = matrix.sum(axis=1, keepdims=True)
    non_zero = row_sums.squeeze() > 0
    matrix[non_zero] = matrix[non_zero] / row_sums[non_zero]
    return matrix


def build_station_index(
    df: pl.DataFrame,
    departure_col: str = "departure_name",
    return_col: str = "return_name",
) -> list[str]:
    dep = set(
        df.select(pl.col(departure_col).cast(pl.String))
        .get_column(departure_col)
        .drop_nulls()
        .to_list()
    )
    ret = set(
        df.select(pl.col(return_col).cast(pl.String))
        .get_column(return_col)
        .drop_nulls()
        .to_list()
    )
    stations = dep | ret
    return sorted(stations)


def extract_station_coordinates(
    df: pl.DataFrame,
    departure_name_col: str = "departure_name",
    return_name_col: str = "return_name",
    departure_lat_col: str = "departure_latitude",
    departure_lon_col: str = "departure_longitude",
    return_lat_col: str = "return_latitude",
    return_lon_col: str = "return_longitude",
) -> pl.DataFrame:
    """Create a station->(lat, lon) table from trip rows."""
    dep = df.select(
        [
            pl.col(departure_name_col).alias("station_name"),
            pl.col(departure_lat_col).alias("latitude"),
            pl.col(departure_lon_col).alias("longitude"),
        ]
    )
    ret = df.select(
        [
            pl.col(return_name_col).alias("station_name"),
            pl.col(return_lat_col).alias("latitude"),
            pl.col(return_lon_col).alias("longitude"),
        ]
    )

    coords = (
        pl.concat([dep, ret], how="vertical_relaxed")
        .drop_nulls(["station_name", "latitude", "longitude"])
        .unique(subset=["station_name"], keep="first")
        .with_columns(pl.col("station_name").cast(pl.String))
    )
    return coords


def haversine_matrix(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Compute pairwise haversine distance matrix in kilometers."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_sd_adjacency(
    station_index: list[str],
    station_coords: pl.DataFrame,
    k_neighbors: int = 12,
    sigma_km: float = 2.5,
    row_normalized: bool = True,
) -> np.ndarray:
    """Build Spatial Distance (SD) graph using gaussian distance kernel + kNN pruning."""
    n = len(station_index)
    adjacency = np.zeros((n, n), dtype=float)

    coord_lookup = {
        row["station_name"]: (float(row["latitude"]), float(row["longitude"]))
        for row in station_coords.iter_rows(named=True)
    }
    valid_names = [name for name in station_index if name in coord_lookup]
    if not valid_names:
        return adjacency

    valid_indices = np.array(
        [station_index.index(name) for name in valid_names], dtype=int
    )
    lat = np.array([coord_lookup[name][0] for name in valid_names], dtype=float)
    lon = np.array([coord_lookup[name][1] for name in valid_names], dtype=float)
    distances = haversine_matrix(lat, lon)

    kernel = np.exp(-(distances**2) / (2.0 * sigma_km**2))
    np.fill_diagonal(kernel, 0.0)

    if k_neighbors > 0:
        keep_mask = np.zeros_like(kernel, dtype=bool)
        for i in range(kernel.shape[0]):
            nearest = np.argsort(distances[i])[1 : k_neighbors + 1]
            keep_mask[i, nearest] = True
        kernel = np.where(keep_mask, kernel, 0.0)

    kernel = np.maximum(kernel, kernel.T)
    adjacency[np.ix_(valid_indices, valid_indices)] = kernel

    if row_normalized:
        adjacency = row_normalize(adjacency)
    return adjacency


def _empty_adjacency(station_index: list[str]) -> np.ndarray:
    return np.zeros((len(station_index), len(station_index)), dtype=float)


def build_de_adjacency(
    df: pl.DataFrame,
    station_index: list[str],
    departure_col: str = "departure_name",
    return_col: str = "return_name",
    min_flow: int = 1,
    row_normalized: bool = True,
) -> np.ndarray:
    """Build Demand Edge (DE) directed flow adjacency from OD trip counts."""
    station_to_idx = {station: i for i, station in enumerate(station_index)}
    adjacency = _empty_adjacency(station_index)

    flow = df.group_by([departure_col, return_col]).len().rename({"len": "flow"})
    if min_flow > 1:
        flow = flow.filter(pl.col("flow") >= min_flow)

    for row in flow.iter_rows(named=True):
        i = station_to_idx.get(str(row[departure_col]))
        j = station_to_idx.get(str(row[return_col]))
        if i is None or j is None or i == j:
            continue
        adjacency[i, j] = float(row["flow"])

    if row_normalized:
        adjacency = row_normalize(adjacency)
    return adjacency


def build_atd_adjacency(
    df: pl.DataFrame,
    station_index: list[str],
    departure_col: str = "departure_name",
    return_col: str = "return_name",
    duration_col: str = "duration_sec",
    as_similarity: bool = True,
    row_normalized: bool = True,
) -> np.ndarray:
    """Build Average Trip Duration (ATD) adjacency from mean OD duration."""
    if duration_col not in df.columns:
        if "duration" in df.columns:
            duration_col = "duration"
        else:
            raise ValueError("ATD requires duration column (duration_sec or duration).")

    station_to_idx = {station: i for i, station in enumerate(station_index)}
    adjacency = _empty_adjacency(station_index)

    atd = df.group_by([departure_col, return_col]).agg(
        pl.col(duration_col).mean().alias("avg_duration")
    )

    for row in atd.iter_rows(named=True):
        i = station_to_idx.get(str(row[departure_col]))
        j = station_to_idx.get(str(row[return_col]))
        if i is None or j is None or i == j:
            continue
        duration_value = float(row["avg_duration"])
        if duration_value <= 0:
            continue

        if as_similarity:
            weight = 1.0 / (1.0 + duration_value / 60.0)
        else:
            weight = duration_value
        adjacency[i, j] = weight

    if row_normalized:
        adjacency = row_normalize(adjacency)
    return adjacency


def build_dc_adjacency(
    df: pl.DataFrame,
    station_index: list[str],
    departure_col: str = "departure_name",
    timestamp_col: str = "departure",
    row_normalized: bool = False,
) -> np.ndarray:
    """Build Demand Correlation (DC) graph from hourly outflow correlations."""
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' missing from dataframe")

    tmp = (
        df.select([departure_col, timestamp_col])
        .with_columns(
            pl.col(timestamp_col)
            .cast(pl.String, strict=False)
            .str.strptime(pl.Datetime, strict=False)
            .alias(timestamp_col)
        )
        .drop_nulls([departure_col, timestamp_col])
        .with_columns(pl.col(timestamp_col).dt.truncate("1h").alias("hour_start"))
    )

    hourly = tmp.group_by(["hour_start", departure_col]).len().rename({"len": "count"})
    if hourly.height == 0:
        return _empty_adjacency(station_index)

    pivot = hourly.pivot(
        values="count",
        index="hour_start",
        on=departure_col,
        aggregate_function="sum",
    ).fill_null(0)

    for station in station_index:
        if station not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(station))

    matrix = pivot.sort("hour_start").select(station_index).to_numpy()
    if matrix.shape[0] < 2:
        corr = np.zeros((len(station_index), len(station_index)), dtype=float)
    else:
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    np.fill_diagonal(corr, 0.0)

    corr = np.clip(corr, a_min=0.0, a_max=None)

    if row_normalized:
        corr = row_normalize(corr)
    return corr


def filter_by_end_time(
    df: pl.DataFrame,
    end_time: str | None,
    timestamp_col: str = "departure",
) -> pl.DataFrame:
    """Filter rows to [min_time, end_time) for leakage-safe graph construction."""
    if end_time is None:
        return df
    parsed = pl.Series("ts", [end_time]).str.strptime(pl.Datetime, strict=False).item()
    if parsed is None:
        raise ValueError(f"Invalid cutoff datetime: {end_time}")

    return df.with_columns(
        pl.col(timestamp_col)
        .cast(pl.String, strict=False)
        .str.strptime(pl.Datetime, strict=False)
        .alias(timestamp_col)
    ).filter(pl.col(timestamp_col) < parsed)


def build_all_graphs(
    df: pl.DataFrame,
    k_neighbors: int = 12,
    sigma_km: float = 2.5,
    de_min_flow: int = 1,
) -> dict[str, Any]:
    station_index = build_station_index(df)
    station_coords = extract_station_coordinates(df)

    sd = build_sd_adjacency(
        station_index=station_index,
        station_coords=station_coords,
        k_neighbors=k_neighbors,
        sigma_km=sigma_km,
        row_normalized=True,
    )
    de = build_de_adjacency(
        df=df,
        station_index=station_index,
        min_flow=de_min_flow,
        row_normalized=True,
    )
    dc = build_dc_adjacency(df=df, station_index=station_index, row_normalized=False)
    atd = build_atd_adjacency(
        df=df,
        station_index=station_index,
        as_similarity=True,
        row_normalized=True,
    )

    return {
        "station_index": station_index,
        "graphs": {
            "SD": sd,
            "DE": de,
            "DC": dc,
            "ATD": atd,
        },
    }


def save_graph_bundle(
    output_dir: Path,
    station_index: list[str],
    graphs: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, matrix in graphs.items():
        np.save(output_dir / f"{name}.npy", matrix)

    (output_dir / "station_index.txt").write_text(
        "\n".join(station_index), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct SD/DE/DC/ATD graphs")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cutoff-end",
        default=None,
        help="Optional end timestamp (exclusive) for leakage-safe filtering.",
    )
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--sigma-km", type=float, default=2.5)
    parser.add_argument("--de-min-flow", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pl.read_csv(
        args.input,
        try_parse_dates=True,
        schema_overrides=CSV_SCHEMA_OVERRIDES,
    )
    filtered = filter_by_end_time(
        df=df, end_time=args.cutoff_end, timestamp_col="departure"
    )

    bundle = build_all_graphs(
        df=filtered,
        k_neighbors=args.k_neighbors,
        sigma_km=args.sigma_km,
        de_min_flow=args.de_min_flow,
    )

    metadata = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_csv": str(args.input),
        "rows_used": int(len(filtered)),
        "rows_total": int(len(df)),
        "cutoff_end": args.cutoff_end,
        "station_count": int(len(bundle["station_index"])),
        "graph_shapes": {
            name: list(matrix.shape) for name, matrix in bundle["graphs"].items()
        },
    }

    save_graph_bundle(
        output_dir=args.output_dir,
        station_index=bundle["station_index"],
        graphs=bundle["graphs"],
        metadata=metadata,
    )

    print(f"Saved graph bundle to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
