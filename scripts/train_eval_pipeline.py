"""CLI facade for the 1-hour ahead training/evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.train_eval import main, parse_args, run  # noqa: E402

__all__ = ["main", "parse_args", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
