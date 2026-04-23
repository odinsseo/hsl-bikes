from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.notebook_reporting import (
    canon_graph_set,
    optional_csv,
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
