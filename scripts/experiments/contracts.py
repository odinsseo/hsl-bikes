from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .config import ARTIFACTS_DIR

REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "generated_at_utc",
    "git_commit",
    "command",
    "script",
    "args",
    "stage",
)

REQUIRED_PREPROCESSING_LINEAGE_FIELDS: tuple[str, ...] = (
    "preprocessing_version",
    "fitted_split",
    "train_time_bounds",
    "quantile_bounds",
    "scaler_type",
    "residual_lag_policy",
    "selected_residual_lag",
    "calendar_source",
    "dynamic_feature_definitions",
    "static_feature_definitions",
    "sparse_feature_definitions",
    "train_only_fit",
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    output_dir: str
    results_file: str
    required_columns: tuple[str, ...]
    min_rows: int = 1
    auxiliary_results: tuple["AuxiliaryResultSpec", ...] = ()


@dataclass(frozen=True)
class AuxiliaryResultSpec:
    file: str
    required_columns: tuple[str, ...]
    min_rows: int = 1


CANONICAL_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="rq_runner",
        output_dir="experiments/rq_runner",
        results_file="results.csv",
        required_columns=(
            "experiment_id",
            "rq",
            "aggregation",
            "graph_set",
            "model",
            "validation_wmape",
            "test_wmape",
        ),
    ),
    ArtifactSpec(
        name="train_eval_3h",
        output_dir="experiments/train_eval_3h",
        results_file="train_eval_results.csv",
        required_columns=(
            "aggregation",
            "graph_set",
            "model",
            "validation_wmape",
            "test_wmape",
        ),
        auxiliary_results=(
            AuxiliaryResultSpec(
                file="station_cohort_results.csv",
                required_columns=(
                    "aggregation",
                    "graph_set",
                    "model",
                    "cohort",
                    "test_wmape",
                ),
            ),
            AuxiliaryResultSpec(
                file="station_robustness_statistics.csv",
                required_columns=(
                    "aggregation",
                    "graph_set",
                    "cohort",
                    "model",
                    "reference_model",
                    "test_station_wmape_mean",
                    "delta_station_wmape_vs_reference",
                    "paired_sign_permutation_pvalue",
                ),
            ),
            AuxiliaryResultSpec(
                file="sensitivity_summary.csv",
                required_columns=(
                    "sensitivity_axis",
                    "scope",
                    "aggregation",
                    "model",
                    "setting",
                    "metric",
                    "value",
                    "reference_value",
                    "delta_vs_reference",
                ),
            ),
        ),
    ),
    ArtifactSpec(
        name="stgnn_single_graph",
        output_dir="experiments/stgnn_single_graph",
        results_file="results.csv",
        required_columns=(
            "aggregation",
            "graph_set",
            "fusion_mode",
            "model",
            "validation_wmape",
            "test_wmape",
            "fusion_weights",
        ),
    ),
    ArtifactSpec(
        name="stgnn_milestones",
        output_dir="experiments/stgnn_milestones",
        results_file="milestone_results.csv",
        required_columns=(
            "milestone",
            "aggregation",
            "graph_source",
            "fusion_mode",
            "validation_wmape",
            "test_wmape",
        ),
        auxiliary_results=(
            AuxiliaryResultSpec(
                file="milestone_best_by_milestone.csv",
                required_columns=(
                    "milestone",
                    "aggregation",
                    "test_wmape",
                ),
            ),
            AuxiliaryResultSpec(
                file="milestone_best_by_graph_set.csv",
                required_columns=(
                    "aggregation",
                    "graph_set",
                    "fusion_mode",
                    "test_wmape",
                ),
            ),
        ),
    ),
)


def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def _read_json_dict(path: Path) -> dict[str, Any]:
    _require_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def validate_metadata_sidecar(
    metadata_path: Path,
    *,
    required_fields: Sequence[str] = REQUIRED_METADATA_FIELDS,
    require_preprocessing_lineage: bool = False,
) -> dict[str, Any]:
    metadata = _read_json_dict(metadata_path)
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise ValueError(
            f"Metadata sidecar {metadata_path} missing required field(s): {missing}"
        )

    if require_preprocessing_lineage:
        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError(
                f"Metadata sidecar {metadata_path} missing preprocessing lineage block"
            )
        missing_lineage = [
            field
            for field in REQUIRED_PREPROCESSING_LINEAGE_FIELDS
            if field not in preprocessing
        ]
        if missing_lineage:
            raise ValueError(
                f"Metadata sidecar {metadata_path} missing preprocessing lineage field(s): {missing_lineage}"
            )

    return metadata


def validate_results_schema(
    result_path: Path,
    *,
    required_columns: Sequence[str],
    min_rows: int = 1,
) -> pl.DataFrame:
    _require_file(result_path)
    frame = pl.read_csv(result_path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Result file {result_path} missing required column(s): {missing}"
        )

    if frame.height < int(min_rows):
        raise ValueError(
            f"Result file {result_path} has {frame.height} row(s), "
            f"expected at least {min_rows}"
        )

    return frame


def validate_artifact_output_dir(
    artifacts_root: Path,
    spec: ArtifactSpec,
    *,
    require_preprocessing_lineage: bool = False,
) -> dict[str, Any]:
    output_dir = artifacts_root / spec.output_dir

    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"
    result_path = output_dir / spec.results_file

    _read_json_dict(summary_path)
    validate_metadata_sidecar(
        metadata_path,
        require_preprocessing_lineage=require_preprocessing_lineage,
    )
    results = validate_results_schema(
        result_path,
        required_columns=spec.required_columns,
        min_rows=spec.min_rows,
    )

    auxiliary_reports: list[dict[str, Any]] = []
    for auxiliary in spec.auxiliary_results:
        auxiliary_path = output_dir / auxiliary.file
        auxiliary_frame = validate_results_schema(
            auxiliary_path,
            required_columns=auxiliary.required_columns,
            min_rows=auxiliary.min_rows,
        )
        auxiliary_reports.append(
            {
                "file": auxiliary.file,
                "min_rows": int(auxiliary.min_rows),
                "rows": int(auxiliary_frame.height),
                "columns": list(auxiliary_frame.columns),
            }
        )

    return {
        "name": spec.name,
        "output_dir": str(output_dir),
        "min_rows": int(spec.min_rows),
        "results_rows": int(results.height),
        "results_columns": list(results.columns),
        "auxiliary_results": auxiliary_reports,
    }


def validate_canonical_experiment_artifacts(
    artifacts_root: Path = ARTIFACTS_DIR,
    *,
    allow_empty_results: bool = False,
    require_preprocessing_lineage: bool = False,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for spec in CANONICAL_ARTIFACT_SPECS:
        if allow_empty_results:
            effective_spec = replace(
                spec,
                min_rows=0,
                auxiliary_results=tuple(
                    replace(auxiliary, min_rows=0)
                    for auxiliary in spec.auxiliary_results
                ),
            )
        else:
            effective_spec = spec
        reports.append(
            validate_artifact_output_dir(
                artifacts_root,
                effective_spec,
                require_preprocessing_lineage=require_preprocessing_lineage,
            )
        )
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical experiment artifacts, schemas, and metadata sidecars"
    )
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--allow-empty-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow zero-row canonical result CSVs (default: strict non-empty checks).",
    )
    parser.add_argument(
        "--require-preprocessing-lineage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require strict preprocessing lineage fields in metadata sidecars.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    reports = validate_canonical_experiment_artifacts(
        args.artifacts_root,
        allow_empty_results=bool(args.allow_empty_results),
        require_preprocessing_lineage=bool(args.require_preprocessing_lineage),
    )
    print(f"Validated {len(reports)} canonical artifact output directories")
    for report in reports:
        print(
            "- "
            f"{report['name']}: rows={report['results_rows']} "
            f"(min={report['min_rows']}), "
            f"dir={report['output_dir']}"
        )
    return 0


def main() -> int:
    args = parse_args()
    return run(args)
