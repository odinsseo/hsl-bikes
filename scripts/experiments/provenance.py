from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def current_git_commit(project_root: Path = PROJECT_ROOT) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = output.strip()
    return commit or None


def build_run_metadata(
    *,
    args: Any,
    stage: str,
    script: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "command": " ".join(sys.argv),
        "script": script,
        "stage": stage,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "args": _to_jsonable(vars(args)),
    }
    if extra:
        metadata.update(_to_jsonable(extra))
    return metadata


def write_metadata_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
