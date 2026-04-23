from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "notebooks" / "executed"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "artifacts" / "notebooks" / "validation_manifest.json"
)

NOTEBOOK_PATHS: tuple[Path, ...] = (
    NOTEBOOK_DIR / "rq1_functional_vs_geometric.ipynb",
    NOTEBOOK_DIR / "rq2_fusion_heterogeneity.ipynb",
    NOTEBOOK_DIR / "rq3_cluster_robustness.ipynb",
    NOTEBOOK_DIR / "rq_synthesis.ipynb",
)


def build_nbconvert_command(
    notebook_path: Path,
    *,
    output_dir: Path,
    timeout_seconds: int,
) -> list[str]:
    output_name = f"executed_{notebook_path.name}"
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={int(timeout_seconds)}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]


def execute_notebook(
    notebook_path: Path,
    *,
    output_dir: Path,
    timeout_seconds: int,
) -> tuple[int, str, str]:
    output_path = output_dir / f"executed_{notebook_path.name}"
    native_error = ""
    try:
        import nbformat
        from nbclient import NotebookClient

        with notebook_path.open("r", encoding="utf-8-sig") as handle:
            notebook = nbformat.read(handle, as_version=4)

        client = NotebookClient(
            notebook,
            timeout=int(timeout_seconds),
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        client.execute()

        with output_path.open("w", encoding="utf-8") as handle:
            nbformat.write(notebook, handle)
        return 0, f"Executed with nbclient: {notebook_path}", ""
    except Exception:
        native_error = traceback.format_exc()

    # Fallback path for environments where nbclient execution is unavailable.
    command = build_nbconvert_command(
        notebook_path,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    stderr_combined = result.stderr
    if native_error:
        stderr_combined = (
            f"nbclient_error:\n{native_error}\n\nnbconvert_error:\n{stderr_combined}"
        )
    return result.returncode, result.stdout, stderr_combined


def run(args: argparse.Namespace) -> int:
    missing = [path for path in NOTEBOOK_PATHS if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Notebook(s) not found:\n{missing_text}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    overall_success = True

    for notebook_path in NOTEBOOK_PATHS:
        started_at = datetime.now(tz=timezone.utc).isoformat()
        if args.check_only:
            checks.append(
                {
                    "notebook": str(notebook_path),
                    "status": "exists",
                    "started_at_utc": started_at,
                    "finished_at_utc": started_at,
                    "return_code": 0,
                }
            )
            continue

        return_code, stdout_text, stderr_text = execute_notebook(
            notebook_path,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        )
        finished_at = datetime.now(tz=timezone.utc).isoformat()

        status = "passed" if return_code == 0 else "failed"
        overall_success = overall_success and return_code == 0
        checks.append(
            {
                "notebook": str(notebook_path),
                "executed_output": str(output_dir / f"executed_{notebook_path.name}"),
                "status": status,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "return_code": int(return_code),
                "stdout_tail": "\n".join(stdout_text.splitlines()[-20:]),
                "stderr_tail": "\n".join(stderr_text.splitlines()[-20:]),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "check_only": bool(args.check_only),
        "timeout_seconds": int(args.timeout_seconds),
        "output_dir": str(output_dir),
        "notebooks": checks,
        "overall_status": "passed" if overall_success else "failed",
    }

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote notebook validation manifest: {manifest_path}")
    return 0 if overall_success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate publication notebook suite using clean-kernel execution"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
