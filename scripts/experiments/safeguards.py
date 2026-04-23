from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def read_graph_metadata(graph_dir: Path) -> dict[str, Any]:
    metadata_path = graph_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Graph metadata missing: {metadata_path}. Rebuild graphs with scripts/graph_construction.py"
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_source_path(raw_path: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    source = Path(raw_path)
    if source.is_absolute():
        return source.resolve()
    return (project_root / source).resolve()


def assert_train_graph_source(
    *,
    graph_dir: Path,
    train_path: Path,
    allow_leaky_graph_source: bool = False,
) -> dict[str, Any]:
    metadata = read_graph_metadata(graph_dir)

    note = str(metadata.get("note", "")).lower()
    if "leaky" in note and not allow_leaky_graph_source:
        raise ValueError(
            "Leaky graph source detected in metadata note. "
            "Use non-leaky train graphs or explicitly allow leaky graph source for sensitivity runs."
        )

    source_raw = metadata.get("input_csv") or metadata.get("source")
    if source_raw is None and not allow_leaky_graph_source:
        raise ValueError("Graph metadata must include input_csv/source in strict mode.")

    if source_raw is not None and not allow_leaky_graph_source:
        source_path = _resolve_source_path(str(source_raw))
        expected_train = train_path.resolve()
        if source_path != expected_train:
            raise ValueError(
                "Graph source does not match train split in strict mode: "
                f"source={source_path}, expected_train={expected_train}"
            )

    return metadata
