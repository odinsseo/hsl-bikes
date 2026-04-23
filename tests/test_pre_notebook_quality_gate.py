from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from scripts.experiments.contracts import CANONICAL_ARTIFACT_SPECS
from scripts.pre_notebook_quality_gate import run


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _required_metadata() -> dict[str, object]:
    return {
        "generated_at_utc": "2025-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "command": "python scripts/some_runner.py",
        "script": "scripts/some_runner.py",
        "args": {"foo": "bar"},
        "stage": "phase_test",
    }


def _required_preprocessing_lineage() -> dict[str, object]:
    return {
        "preprocessing_version": "v1",
        "fitted_split": "train",
        "train_time_bounds": {
            "start": "2025-01-01T00:00:00",
            "end": "2025-01-02T00:00:00",
        },
        "quantile_bounds": {"lower": 0.005, "upper": 0.995},
        "scaler_type": "robust",
        "residual_lag_policy": [24, 168],
        "selected_residual_lag": 24,
        "calendar_source": {
            "library": "holidays",
            "country": "FI",
            "subdivision": "18",
        },
        "dynamic_feature_definitions": [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "is_holiday",
        ],
        "static_feature_definitions": [
            "train_mean",
            "train_variance",
            "train_zero_rate",
        ],
        "sparse_feature_definitions": ["recent_activity_mask"],
        "train_only_fit": True,
    }


def _write_artifacts(
    artifacts_root: Path, *, include_preprocessing: bool = True
) -> None:
    for spec in CANONICAL_ARTIFACT_SPECS:
        output_dir = artifacts_root / spec.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "summary.json", {"ok": True})
        metadata = _required_metadata()
        if include_preprocessing:
            metadata["preprocessing"] = _required_preprocessing_lineage()
        _write_json(output_dir / "metadata.json", metadata)
        row = {column: "x" for column in spec.required_columns}
        pl.DataFrame([row]).write_csv(output_dir / spec.results_file)
        for auxiliary in spec.auxiliary_results:
            auxiliary_row = {column: "x" for column in auxiliary.required_columns}
            pl.DataFrame([auxiliary_row]).write_csv(output_dir / auxiliary.file)


def test_pre_notebook_quality_gate_passes_on_valid_artifacts(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    exit_code = run(
        Namespace(
            artifacts_root=tmp_path,
            allow_empty_results=False,
            require_preprocessing_lineage=True,
        )
    )

    assert exit_code == 0


def test_pre_notebook_quality_gate_rejects_missing_lineage_by_default(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path, include_preprocessing=False)

    with pytest.raises(ValueError, match="preprocessing lineage"):
        run(
            Namespace(
                artifacts_root=tmp_path,
                allow_empty_results=False,
                require_preprocessing_lineage=True,
            )
        )


def test_pre_notebook_quality_gate_can_disable_lineage_requirement(
    tmp_path: Path,
) -> None:
    _write_artifacts(tmp_path, include_preprocessing=False)

    exit_code = run(
        Namespace(
            artifacts_root=tmp_path,
            allow_empty_results=False,
            require_preprocessing_lineage=False,
        )
    )

    assert exit_code == 0
