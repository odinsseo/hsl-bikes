from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.safeguards import assert_train_graph_source


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assert_train_graph_source_accepts_matching_source(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs" / "train"
    train_csv = tmp_path / "prepared" / "splits" / "train" / "train.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_csv.write_text("departure,departure_name\n", encoding="utf-8")

    _write_metadata(
        graph_dir / "metadata.json",
        {"input_csv": str(train_csv)},
    )

    metadata = assert_train_graph_source(
        graph_dir=graph_dir,
        train_path=train_csv,
        allow_leaky_graph_source=False,
    )

    assert metadata["input_csv"] == str(train_csv)


def test_assert_train_graph_source_rejects_mismatched_source(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs" / "train"
    train_csv = tmp_path / "prepared" / "splits" / "train" / "train.csv"
    other_csv = tmp_path / "prepared" / "merged" / "trips_merged.csv"

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    other_csv.parent.mkdir(parents=True, exist_ok=True)
    train_csv.write_text("departure,departure_name\n", encoding="utf-8")
    other_csv.write_text("departure,departure_name\n", encoding="utf-8")

    _write_metadata(graph_dir / "metadata.json", {"input_csv": str(other_csv)})

    with pytest.raises(ValueError, match="does not match train split"):
        assert_train_graph_source(
            graph_dir=graph_dir,
            train_path=train_csv,
            allow_leaky_graph_source=False,
        )


def test_assert_train_graph_source_blocks_leaky_note_by_default(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graphs" / "leaky"
    train_csv = tmp_path / "prepared" / "splits" / "train" / "train.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_csv.write_text("departure,departure_name\n", encoding="utf-8")

    _write_metadata(
        graph_dir / "metadata.json",
        {
            "source": str(tmp_path / "prepared" / "merged" / "trips_merged.csv"),
            "note": "Leaky full-period graph bundle for sensitivity analysis.",
        },
    )

    with pytest.raises(ValueError, match="Leaky graph source detected"):
        assert_train_graph_source(
            graph_dir=graph_dir,
            train_path=train_csv,
            allow_leaky_graph_source=False,
        )


def test_assert_train_graph_source_allows_leaky_when_explicit(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs" / "leaky"
    train_csv = tmp_path / "prepared" / "splits" / "train" / "train.csv"
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_csv.write_text("departure,departure_name\n", encoding="utf-8")

    _write_metadata(
        graph_dir / "metadata.json",
        {
            "source": str(tmp_path / "prepared" / "merged" / "trips_merged.csv"),
            "note": "Leaky full-period graph bundle for sensitivity analysis.",
        },
    )

    metadata = assert_train_graph_source(
        graph_dir=graph_dir,
        train_path=train_csv,
        allow_leaky_graph_source=True,
    )

    assert "source" in metadata
