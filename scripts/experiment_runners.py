"""Run RQ1/RQ2/RQ3 ablation experiments with graph and non-graph baselines.

This module is the stable CLI/public API facade for the experiment runner. The
implementation is split into smaller modules under ``scripts.experiments``.

After a successful run, post-hoc paired tests (Holm-adjusted) can be generated with::

    python -m scripts.experiments.rq_hypothesis_tests --output-dir data/artifacts/experiments/rq_runner

See ``docs/statistical_inference_rq.md`` and the repository README for details.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support direct execution via `python scripts/experiment_runners.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.config import (
    ExperimentSpec,  # noqa: E402
    build_experiment_specs,
    parse_alpha_grid,
    parse_lag_candidates,
    row_normalize,
)
from scripts.experiments.data import aggregate_adjacency_to_groups  # noqa: E402
from scripts.experiments.models import (  # noqa: E402
    build_one_step_lag_features,
    evaluate_one_step_forecast,
)
from scripts.experiments.pipeline import main, parse_args, run  # noqa: E402

__all__ = [
    "ExperimentSpec",
    "aggregate_adjacency_to_groups",
    "build_experiment_specs",
    "build_one_step_lag_features",
    "evaluate_one_step_forecast",
    "main",
    "parse_alpha_grid",
    "parse_args",
    "parse_lag_candidates",
    "row_normalize",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
