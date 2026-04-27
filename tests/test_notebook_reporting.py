from __future__ import annotations

from pathlib import Path

import polars as pl

import numpy as np

from scripts.notebook_reporting import (
    HEADLINE_CONTRAST_BY_RQ,
    PRIMARY_COHORT_BY_RQ,
    canon_graph_set,
    load_station_wmape_vector,
    optional_csv,
    paired_station_wmape_diff,
    parse_fusion_weights,
    relative_change,
    require_csv,
)


def test_canon_graph_set_normalizes_delimiters() -> None:
    assert canon_graph_set("sd,de") == "SD+DE"
    assert canon_graph_set("SD+ DC") == "SD+DC"


def test_parse_fusion_weights_from_literal_string() -> None:
    weights = parse_fusion_weights("[0.1, 0.2, 0.7]")
    assert weights == [0.1, 0.2, 0.7]


def test_relative_change_computes_fraction() -> None:
    delta = relative_change(new_value=0.9, reference_value=1.0)
    assert abs(delta + 0.1) < 1e-12


def test_require_csv_and_optional_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo.csv"
    pl.DataFrame([{"a": 1, "b": 2}]).write_csv(csv_path)

    frame = require_csv("demo.csv", required_columns=("a",), root=tmp_path)
    assert frame.height == 1

    maybe_missing = optional_csv("missing.csv", root=tmp_path)
    assert maybe_missing is None


def test_primary_cohort_and_headline_constants() -> None:
    assert PRIMARY_COHORT_BY_RQ["RQ3"] == "sparse_espoo"
    assert HEADLINE_CONTRAST_BY_RQ["RQ1"] == "SD_vs_DC"


def test_load_station_wmape_vector_and_paired_diff(tmp_path: Path) -> None:
    scores = tmp_path / "station_scores"
    scores.mkdir(parents=True)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.5, 2.5, 2.0], dtype=np.float64)
    np.savez_compressed(scores / "exp_a.npz", wmape_by_station=a)
    np.savez_compressed(scores / "exp_b.npz", wmape_by_station=b)
    assert np.allclose(load_station_wmape_vector(scores, "exp_a"), a)
    idx = np.array([0, 1, 2], dtype=int)
    diff = paired_station_wmape_diff(scores, "exp_a", "exp_b", idx)
    assert np.allclose(diff, a - b)
