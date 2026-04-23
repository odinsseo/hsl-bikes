"""CLI facade for ST-GNN milestone orchestration (M3.1/M3.2/leakage)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.stgnn_milestones import main, parse_args, run  # noqa: E402

__all__ = ["main", "parse_args", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
