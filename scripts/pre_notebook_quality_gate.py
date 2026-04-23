"""Run publication pre-notebook artifact quality checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.config import ARTIFACTS_DIR  # noqa: E402
from scripts.experiments.contracts import (
    validate_canonical_experiment_artifacts,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate canonical experiment artifacts before notebook reporting"
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
        default=True,
        help="Require strict preprocessing lineage fields in metadata sidecars.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    reports = validate_canonical_experiment_artifacts(
        artifacts_root=args.artifacts_root,
        allow_empty_results=bool(args.allow_empty_results),
        require_preprocessing_lineage=bool(args.require_preprocessing_lineage),
    )

    print("Pre-notebook quality gate passed")
    print(f"Checked {len(reports)} canonical artifact output directories")
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


if __name__ == "__main__":
    raise SystemExit(main())
