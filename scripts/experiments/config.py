from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

DEFAULT_TRAIN = PREPARED_DIR / "splits" / "train" / "train.csv"
DEFAULT_VALIDATION = PREPARED_DIR / "splits" / "validation" / "validation.csv"
DEFAULT_TEST = PREPARED_DIR / "splits" / "test" / "test.csv"
DEFAULT_GRAPH_DIR = ARTIFACTS_DIR / "graphs" / "train"
DEFAULT_COMMUNITIES = ARTIFACTS_DIR / "network" / "notebook" / "communities.csv"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "rq_runner"

CSV_SCHEMA_OVERRIDES = {
    "departure_id": pl.String,
    "return_id": pl.String,
    "departure_name": pl.String,
    "return_name": pl.String,
}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    rq: str
    aggregation: str
    graph_set: tuple[str, ...]
    description: str


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix while preserving all-zero rows."""
    out = np.asarray(matrix, dtype=float).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    out[mask] = out[mask] / row_sums[mask]
    return out


def parse_alpha_grid(alpha_grid: str) -> list[float]:
    values = [float(x.strip()) for x in alpha_grid.split(",") if x.strip()]
    if not values:
        raise ValueError("Alpha grid is empty")
    for val in values:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Alpha value out of range [0, 1]: {val}")
    return sorted(set(values))


def parse_int_grid(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Integer grid is empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"All integer grid values must be > 0: {values}")
    return sorted(set(values))


def parse_lag_candidates(text: str) -> list[tuple[int, ...]]:
    """Parse lag candidates from format '1|1,24|1,2,24'."""
    candidates: list[tuple[int, ...]] = []
    for raw in text.split("|"):
        lags = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
        if not lags:
            continue
        if any(v <= 0 for v in lags):
            raise ValueError(f"Lag values must be > 0: {lags}")
        candidates.append(lags)
    if not candidates:
        raise ValueError("No lag candidates were parsed")
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for cand in candidates:
        if cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out


def build_experiment_specs(selected_rqs: set[str]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []

    if "RQ1" in selected_rqs:
        specs.extend(
            [
                ExperimentSpec(
                    "RQ1_SD_STATION",
                    "RQ1",
                    "station",
                    ("SD",),
                    "Geometric baseline: SD only",
                ),
                ExperimentSpec(
                    "RQ1_DE_STATION",
                    "RQ1",
                    "station",
                    ("DE",),
                    "Functional baseline: DE only",
                ),
                ExperimentSpec(
                    "RQ1_DC_STATION",
                    "RQ1",
                    "station",
                    ("DC",),
                    "Functional baseline: DC only",
                ),
                ExperimentSpec(
                    "RQ1_DE_DC_STATION",
                    "RQ1",
                    "station",
                    ("DE", "DC"),
                    "Functional fusion: DE + DC",
                ),
            ]
        )

    if "RQ2" in selected_rqs:
        specs.extend(
            [
                ExperimentSpec(
                    "RQ2_SD_DE_STATION",
                    "RQ2",
                    "station",
                    ("SD", "DE"),
                    "Two-view fusion: SD + DE",
                ),
                ExperimentSpec(
                    "RQ2_SD_DC_STATION",
                    "RQ2",
                    "station",
                    ("SD", "DC"),
                    "Two-view fusion: SD + DC",
                ),
                ExperimentSpec(
                    "RQ2_DE_DC_STATION",
                    "RQ2",
                    "station",
                    ("DE", "DC"),
                    "Two-view fusion: DE + DC",
                ),
                ExperimentSpec(
                    "RQ2_ALL_STATION",
                    "RQ2",
                    "station",
                    ("SD", "DE", "DC", "ATD"),
                    "All-view fusion: SD + DE + DC + ATD",
                ),
            ]
        )

    if "RQ3" in selected_rqs:
        specs.extend(
            [
                ExperimentSpec(
                    "RQ3_ALL_STATION",
                    "RQ3",
                    "station",
                    ("SD", "DE", "DC", "ATD"),
                    "All-view station-level baseline",
                ),
                ExperimentSpec(
                    "RQ3_ALL_COMMUNITY",
                    "RQ3",
                    "community",
                    ("SD", "DE", "DC", "ATD"),
                    "All-view community-level baseline",
                ),
                ExperimentSpec(
                    "RQ3_FUNCTIONAL_STATION",
                    "RQ3",
                    "station",
                    ("DE", "DC"),
                    "Functional station-level baseline",
                ),
                ExperimentSpec(
                    "RQ3_FUNCTIONAL_COMMUNITY",
                    "RQ3",
                    "community",
                    ("DE", "DC"),
                    "Functional community-level baseline",
                ),
            ]
        )

    return specs


def parse_rqs(rq_text: str) -> set[str]:
    allowed = {"RQ1", "RQ2", "RQ3"}
    selected = {part.strip().upper() for part in rq_text.split(",") if part.strip()}
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"Unknown RQ(s): {sorted(unknown)}")
    if not selected:
        raise ValueError("No research questions selected")
    return selected
