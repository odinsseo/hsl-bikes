from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
