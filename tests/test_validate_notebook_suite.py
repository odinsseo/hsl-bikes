from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts import validate_notebook_suite


def _write_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_build_nbconvert_command_contains_timeout(tmp_path: Path) -> None:
    notebook_path = tmp_path / "demo.ipynb"
    command = validate_notebook_suite.build_nbconvert_command(
        notebook_path,
        output_dir=tmp_path / "executed",
        timeout_seconds=123,
    )

    assert "nbconvert" in command
    assert "--ExecutePreprocessor.timeout=123" in command


def test_run_check_only_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    notebook_paths = [tmp_path / "n1.ipynb", tmp_path / "n2.ipynb"]
    for path in notebook_paths:
        _write_notebook(path)

    monkeypatch.setattr(
        validate_notebook_suite, "NOTEBOOK_PATHS", tuple(notebook_paths)
    )

    output_dir = tmp_path / "executed"
    manifest_path = tmp_path / "manifest.json"
    exit_code = validate_notebook_suite.run(
        Namespace(
            output_dir=output_dir,
            manifest_path=manifest_path,
            timeout_seconds=60,
            check_only=True,
        )
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["overall_status"] == "passed"
    assert len(manifest["notebooks"]) == 2
