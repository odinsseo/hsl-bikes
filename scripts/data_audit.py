"""Data quality and split-integrity audit for HSL city-bike datasets.

Usage:
    python scripts/data_audit.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

DEFAULT_MERGED = PREPARED_DIR / "merged" / "trips_merged.csv"
DEFAULT_TRAIN = PREPARED_DIR / "splits" / "train" / "train.csv"
DEFAULT_VALIDATION = PREPARED_DIR / "splits" / "validation" / "validation.csv"
DEFAULT_TEST = PREPARED_DIR / "splits" / "test" / "test.csv"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "audit" / "data_audit_report.json"
CSV_SCHEMA_OVERRIDES = {
    "departure_id": pl.String,
    "return_id": pl.String,
    "departure_name": pl.String,
    "return_name": pl.String,
}


def _path_str(path: Path) -> str:
    """Prefer repository-relative paths in reports to avoid leaking local machine paths."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        return str(path)


def _parse_datetime(value: str) -> datetime:
    parsed = pl.Series("ts", [value]).str.strptime(pl.Datetime, strict=False).item()
    if parsed is None:
        raise ValueError(f"Could not parse datetime string: {value}")
    return parsed


def _to_datetime_expr(timestamp_col: str) -> pl.Expr:
    return (
        pl.col(timestamp_col)
        .cast(pl.String, strict=False)
        .str.strptime(pl.Datetime, strict=False)
    )


def _timestamp_series(df: pl.DataFrame, timestamp_col: str) -> pl.Series:
    if timestamp_col not in df.columns:
        return pl.Series(name=timestamp_col, values=[], dtype=pl.Datetime)
    ts = df.select(_to_datetime_expr(timestamp_col).alias("_ts")).get_column("_ts")
    return ts.drop_nulls()


def summarize_dataset(
    df: pl.DataFrame, timestamp_col: str = "departure"
) -> dict[str, Any]:
    """Return descriptive stats for one dataset split."""
    row_count = int(df.height)
    null_counts = df.null_count().row(0, named=True)
    null_rates = {
        col: (round((float(count) / row_count) * 100.0, 4) if row_count else 0.0)
        for col, count in null_counts.items()
    }
    duplicate_rows = int(df.height - df.unique().height)

    parsed_ts = None
    valid_ts = pl.Series(name="_ts", values=[], dtype=pl.Datetime)
    if timestamp_col in df.columns:
        parsed_ts = df.select(_to_datetime_expr(timestamp_col).alias("_ts")).get_column(
            "_ts"
        )
        valid_ts = parsed_ts.drop_nulls()

    timestamp_min = valid_ts.min()
    timestamp_max = valid_ts.max()

    summary: dict[str, Any] = {
        "rows": int(row_count),
        "columns": int(df.width),
        "column_names": list(df.columns),
        "null_rate_percent": {k: float(v) for k, v in null_rates.items()},
        "duplicate_rows": duplicate_rows,
        "timestamp_column": timestamp_col,
        "timestamp_missing": (
            int(parsed_ts.is_null().sum()) if parsed_ts is not None else None
        ),
        "timestamp_min": (
            timestamp_min.isoformat() if timestamp_min is not None else None
        ),
        "timestamp_max": (
            timestamp_max.isoformat() if timestamp_max is not None else None
        ),
        "timestamp_is_monotonic_increasing": (
            bool(valid_ts.is_sorted()) if valid_ts.len() > 0 else None
        ),
    }
    return summary


def evaluate_split_boundaries(
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_end: str,
    validation_end: str,
    timestamp_col: str = "departure",
) -> dict[str, bool]:
    """Check each split respects configured time boundaries."""
    train_end_ts = _parse_datetime(train_end)
    validation_end_ts = _parse_datetime(validation_end)

    train_ts = _timestamp_series(train_df, timestamp_col)
    val_ts = _timestamp_series(validation_df, timestamp_col)
    test_ts = _timestamp_series(test_df, timestamp_col)

    checks = {
        "train_before_train_end": (
            bool((train_ts < train_end_ts).all()) if train_ts.len() > 0 else True
        ),
        "validation_in_range": (
            bool(((val_ts >= train_end_ts) & (val_ts < validation_end_ts)).all())
            if val_ts.len() > 0
            else True
        ),
        "test_after_validation_end": (
            bool((test_ts >= validation_end_ts).all()) if test_ts.len() > 0 else True
        ),
    }
    checks["all_passed"] = bool(all(checks.values()))
    return checks


def evaluate_temporal_overlap(
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
    test_df: pl.DataFrame,
    timestamp_col: str = "departure",
) -> dict[str, bool | str | None]:
    """Check no temporal overlap between adjacent splits."""
    train_ts = _timestamp_series(train_df, timestamp_col)
    val_ts = _timestamp_series(validation_df, timestamp_col)
    test_ts = _timestamp_series(test_df, timestamp_col)

    train_max = train_ts.max() if train_ts.len() > 0 else None
    val_min = val_ts.min() if val_ts.len() > 0 else None
    val_max = val_ts.max() if val_ts.len() > 0 else None
    test_min = test_ts.min() if test_ts.len() > 0 else None

    train_val_non_overlap = True
    val_test_non_overlap = True

    if train_max is not None and val_min is not None:
        train_val_non_overlap = bool(train_max < val_min)
    if val_max is not None and test_min is not None:
        val_test_non_overlap = bool(val_max < test_min)

    return {
        "train_max": train_max.isoformat() if train_max is not None else None,
        "validation_min": val_min.isoformat() if val_min is not None else None,
        "validation_max": val_max.isoformat() if val_max is not None else None,
        "test_min": test_min.isoformat() if test_min is not None else None,
        "train_validation_non_overlap": train_val_non_overlap,
        "validation_test_non_overlap": val_test_non_overlap,
        "all_passed": bool(train_val_non_overlap and val_test_non_overlap),
    }


def _load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides=CSV_SCHEMA_OVERRIDES,
    )


def run_audit(
    merged_path: Path = DEFAULT_MERGED,
    train_path: Path = DEFAULT_TRAIN,
    validation_path: Path = DEFAULT_VALIDATION,
    test_path: Path = DEFAULT_TEST,
    train_end: str = "2022-01-01",
    validation_end: str = "2023-01-01",
    timestamp_col: str = "departure",
) -> dict[str, Any]:
    """Run full audit and return a JSON-serializable report."""
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "inputs": {
            "merged": _path_str(merged_path),
            "train": _path_str(train_path),
            "validation": _path_str(validation_path),
            "test": _path_str(test_path),
            "timestamp_col": timestamp_col,
            "train_end": train_end,
            "validation_end": validation_end,
        },
        "datasets": {},
        "checks": {},
        "warnings": [],
        "errors": [],
        "status": "pass",
    }

    required = {
        "merged": merged_path,
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }

    missing = {
        name: _path_str(path) for name, path in required.items() if not path.exists()
    }
    if missing:
        report["errors"].append({"type": "missing_files", "details": missing})
        report["status"] = "fail"
        return report

    merged_df = _load_csv(merged_path)
    train_df = _load_csv(train_path)
    validation_df = _load_csv(validation_path)
    test_df = _load_csv(test_path)

    report["datasets"]["merged"] = summarize_dataset(
        merged_df, timestamp_col=timestamp_col
    )
    report["datasets"]["train"] = summarize_dataset(
        train_df, timestamp_col=timestamp_col
    )
    report["datasets"]["validation"] = summarize_dataset(
        validation_df, timestamp_col=timestamp_col
    )
    report["datasets"]["test"] = summarize_dataset(test_df, timestamp_col=timestamp_col)

    report["checks"]["split_boundaries"] = evaluate_split_boundaries(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        train_end=train_end,
        validation_end=validation_end,
        timestamp_col=timestamp_col,
    )
    report["checks"]["temporal_overlap"] = evaluate_temporal_overlap(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        timestamp_col=timestamp_col,
    )

    for split_name in ("merged", "train", "validation", "test"):
        duplicate_rows = report["datasets"][split_name]["duplicate_rows"]
        if duplicate_rows > 0:
            report["warnings"].append(
                {
                    "type": "duplicate_rows",
                    "split": split_name,
                    "count": int(duplicate_rows),
                }
            )

    all_checks = []
    for check_values in report["checks"].values():
        if isinstance(check_values, dict) and "all_passed" in check_values:
            all_checks.append(bool(check_values["all_passed"]))

    if not all(all_checks):
        report["status"] = "fail"

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit merged/split city-bike datasets"
    )
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timestamp-col", default="departure")
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--validation-end", default="2023-01-01")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when audit fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(
        merged_path=args.merged,
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        train_end=args.train_end,
        validation_end=args.validation_end,
        timestamp_col=args.timestamp_col,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Audit status: {report['status']}")
    print(f"Audit report written to: {args.output}")

    if args.strict and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
