from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from scripts.experiments.contracts import (
    CANONICAL_ARTIFACT_SPECS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_PREPROCESSING_LINEAGE_FIELDS,
    ArtifactSpec,
    validate_artifact_output_dir,
    validate_canonical_experiment_artifacts,
    validate_metadata_sidecar,
    validate_results_schema,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _required_metadata() -> dict[str, Any]:
    return {
        "generated_at_utc": "2025-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "command": "python scripts/some_runner.py",
        "script": "scripts/some_runner.py",
        "args": {"foo": "bar"},
        "stage": "phase_test",
    }


def _write_result_csv(path: Path, required_columns: tuple[str, ...]) -> None:
    row = {column: 1 for column in required_columns}
    pl.DataFrame([row]).write_csv(path)


def _write_empty_result_csv(path: Path, required_columns: tuple[str, ...]) -> None:
    frame = pl.DataFrame(schema=[(column, pl.Utf8) for column in required_columns])
    frame.write_csv(path)


def _write_artifact_dir(
    artifacts_root: Path,
    *,
    spec: ArtifactSpec,
    include_summary: bool = True,
    include_metadata: bool = True,
    result_columns: tuple[str, ...] | None = None,
) -> Path:
    output_dir = artifacts_root / spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if include_summary:
        _write_json(output_dir / "summary.json", {"ok": True})

    if include_metadata:
        _write_json(output_dir / "metadata.json", _required_metadata())

    _write_result_csv(
        output_dir / spec.results_file,
        result_columns if result_columns is not None else spec.required_columns,
    )

    for auxiliary in spec.auxiliary_results:
        _write_result_csv(output_dir / auxiliary.file, auxiliary.required_columns)

    return output_dir


def test_validate_metadata_sidecar_accepts_required_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    _write_json(metadata_path, _required_metadata())

    metadata = validate_metadata_sidecar(metadata_path)
    for field in REQUIRED_METADATA_FIELDS:
        assert field in metadata


def test_validate_metadata_sidecar_rejects_missing_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    payload = _required_metadata()
    payload.pop("stage")
    _write_json(metadata_path, payload)

    with pytest.raises(ValueError, match="missing required field"):
        validate_metadata_sidecar(metadata_path)


def test_validate_metadata_sidecar_rejects_missing_preprocessing_lineage(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    _write_json(metadata_path, _required_metadata())

    with pytest.raises(ValueError, match="missing preprocessing lineage block"):
        validate_metadata_sidecar(
            metadata_path,
            require_preprocessing_lineage=True,
        )


def test_validate_metadata_sidecar_accepts_preprocessing_lineage(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    payload = _required_metadata()
    payload["preprocessing"] = {
        "preprocessing_version": "v1",
        "fitted_split": "train",
        "train_time_bounds": {
            "start": "2025-01-01T00:00:00",
            "end": "2025-01-02T00:00:00",
        },
        "quantile_bounds": {"lower": 0.01, "upper": 0.99},
        "scaler_type": "robust",
        "residual_lag_policy": [8, 56],
        "selected_residual_lag": 8,
        "calendar_source": {"country": "FI", "subdivision": "18"},
        "dynamic_feature_definitions": ["hour_sin"],
        "static_feature_definitions": ["train_mean"],
        "sparse_feature_definitions": ["recent_activity_mask"],
        "train_only_fit": True,
    }
    _write_json(metadata_path, payload)

    metadata = validate_metadata_sidecar(
        metadata_path,
        require_preprocessing_lineage=True,
    )
    for field in REQUIRED_PREPROCESSING_LINEAGE_FIELDS:
        assert field in metadata["preprocessing"]


def test_validate_results_schema_rejects_missing_columns(tmp_path: Path) -> None:
    result_path = tmp_path / "results.csv"
    pl.DataFrame([{"a": 1}]).write_csv(result_path)

    with pytest.raises(ValueError, match="missing required column"):
        validate_results_schema(result_path, required_columns=("a", "b"))


def test_validate_results_schema_rejects_empty_when_min_rows_required(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results.csv"
    _write_empty_result_csv(result_path, ("a", "b"))

    with pytest.raises(ValueError, match="expected at least"):
        validate_results_schema(result_path, required_columns=("a", "b"), min_rows=1)


def test_validate_artifact_output_dir_requires_summary(tmp_path: Path) -> None:
    spec = ArtifactSpec(
        name="demo",
        output_dir="experiments/demo",
        results_file="results.csv",
        required_columns=("x",),
    )
    _write_artifact_dir(tmp_path, spec=spec, include_summary=False)

    with pytest.raises(FileNotFoundError, match="summary.json"):
        validate_artifact_output_dir(tmp_path, spec)


def test_validate_artifact_output_dir_happy_path(tmp_path: Path) -> None:
    spec = ArtifactSpec(
        name="demo",
        output_dir="experiments/demo",
        results_file="results.csv",
        required_columns=("x", "y"),
    )
    _write_artifact_dir(tmp_path, spec=spec)

    report = validate_artifact_output_dir(tmp_path, spec)
    assert report["name"] == "demo"
    assert report["results_rows"] == 1


def test_validate_canonical_experiment_artifacts_all_pass(tmp_path: Path) -> None:
    for spec in CANONICAL_ARTIFACT_SPECS:
        _write_artifact_dir(tmp_path, spec=spec)

    reports = validate_canonical_experiment_artifacts(tmp_path)
    assert len(reports) == len(CANONICAL_ARTIFACT_SPECS)
    assert all(report["results_rows"] == 1 for report in reports)


def test_validate_canonical_experiment_artifacts_can_allow_empty_results(
    tmp_path: Path,
) -> None:
    for spec in CANONICAL_ARTIFACT_SPECS:
        output_dir = tmp_path / spec.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "summary.json", {"ok": True})
        _write_json(output_dir / "metadata.json", _required_metadata())
        _write_empty_result_csv(output_dir / spec.results_file, spec.required_columns)
        for auxiliary in spec.auxiliary_results:
            _write_empty_result_csv(
                output_dir / auxiliary.file, auxiliary.required_columns
            )

    reports = validate_canonical_experiment_artifacts(
        tmp_path,
        allow_empty_results=True,
    )
    assert len(reports) == len(CANONICAL_ARTIFACT_SPECS)
    assert all(report["results_rows"] == 0 for report in reports)
    assert all(report["min_rows"] == 0 for report in reports)
