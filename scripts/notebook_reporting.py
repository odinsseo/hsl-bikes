from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scripts.experiments.data import build_station_series, load_graph_bundle, load_split
from scripts.experiments.train_eval import (
    DEFAULT_STATIONS_DIR,
    build_station_cohort_indices,
    load_station_city_lookup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "artifacts" / "experiments"


def canon_graph_set(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).replace(",", "+")
    parts = [part.strip().upper() for part in text.split("+") if part.strip()]
    return "+".join(parts)


def require_csv(
    relative_path: str,
    *,
    required_columns: tuple[str, ...] = (),
    root: Path = EXPERIMENTS_DIR,
) -> pl.DataFrame:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")

    frame = pl.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Artifact {path} is missing required columns: {missing}")
    return frame


def optional_csv(
    relative_path: str,
    *,
    required_columns: tuple[str, ...] = (),
    root: Path = EXPERIMENTS_DIR,
) -> pl.DataFrame | None:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return None
    frame = pl.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Artifact {path} is missing required columns: {missing}")
    return frame


def relative_change(new_value: float, reference_value: float) -> float:
    if not np.isfinite(reference_value) or reference_value == 0.0:
        return np.nan
    return float((new_value - reference_value) / reference_value)


def parse_fusion_weights(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, (list, tuple)):
        return [float(v) for v in parsed]
    return []


def add_graph_set_column(frame: pl.DataFrame, *, fallback: str = "") -> pl.DataFrame:
    if "graph_set" in frame.columns:
        return frame.with_columns(pl.col("graph_set").cast(pl.Utf8).alias("graph_set"))
    return frame.with_columns(pl.lit(fallback).alias("graph_set"))


# Pre-specified primary analysis cohort per RQ (aligns with docs/statistical_inference_rq.md).
PRIMARY_COHORT_BY_RQ: dict[str, str] = {
    "RQ1": "all",
    "RQ2": "all",
    "RQ3": "sparse_espoo",
}

# One headline contrast per RQ for distribution plots (must exist in rq_hypothesis_tests.csv).
HEADLINE_CONTRAST_BY_RQ: dict[str, str] = {
    "RQ1": "SD_vs_DC",
    "RQ2": "all_view_vs_DE_DC",
    "RQ3": "station_vs_community_DE_DC",
}


def load_station_wmape_vector(scores_dir: Path, experiment_id: str) -> np.ndarray:
    """Load per-station test WMAPE vector from pipeline `station_scores/{experiment_id}.npz`."""
    path = scores_dir / f"{experiment_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing station score file: {path}")
    data = np.load(path)
    if "wmape_by_station" not in data:
        raise ValueError(f"Missing wmape_by_station in {path}")
    return np.asarray(data["wmape_by_station"], dtype=float)


def cohort_station_indices(
    cohort: str,
    *,
    train_csv: Path,
    graph_dir: Path,
    stations_dir: Path,
    sparse_quantile: float = 0.25,
) -> np.ndarray:
    """Station column indices for a named cohort (same definition as rq_hypothesis_tests)."""
    station_index, _ = load_graph_bundle(graph_dir)
    train_df = load_split(train_csv)
    train_station = build_station_series(train_df, station_index)
    city_lookup = load_station_city_lookup(stations_dir)
    cohorts = build_station_cohort_indices(
        train_station,
        station_index,
        city_lookup,
        float(sparse_quantile),
    )
    if cohort not in cohorts:
        raise KeyError(f"Unknown cohort {cohort!r}; known: {sorted(cohorts)}")
    return np.asarray(cohorts[cohort], dtype=int)


def paired_station_wmape_diff(
    scores_dir: Path,
    experiment_a: str,
    experiment_b: str,
    cohort_indices: np.ndarray,
) -> np.ndarray:
    """Paired differences (A − B) on stations in ``cohort_indices`` with finite A and B."""
    va = load_station_wmape_vector(scores_dir, experiment_a)[cohort_indices]
    vb = load_station_wmape_vector(scores_dir, experiment_b)[cohort_indices]
    mask = np.isfinite(va) & np.isfinite(vb)
    return (va - vb)[mask]


def load_rq_inference_geo(
    artifact_root: Path,
    *,
    stations_dir: Path | None = None,
) -> dict[str, Any]:
    """Paths and sparse quantile read from rq_runner artifacts (for notebook plots)."""
    meta_path = artifact_root / "rq_runner" / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    args = meta.get("args", {})
    summary_path = artifact_root / "rq_runner" / "rq_hypothesis_summary.json"
    if summary_path.is_file():
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        sparse_q = float(summ.get("sparse_quantile", 0.25))
    else:
        sparse_q = float(args.get("sparse_quantile", 0.25))

    return {
        "train_csv": Path(args["train"]),
        "graph_dir": Path(args["graph_dir"]),
        "stations_dir": stations_dir or DEFAULT_STATIONS_DIR,
        "sparse_quantile": sparse_q,
        "scores_dir": artifact_root / "rq_runner" / "station_scores",
    }
