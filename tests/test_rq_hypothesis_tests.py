"""Tests for RQ hypothesis helpers and Holm adjustment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.pipeline import _broadcast_community_pred_to_stations
from scripts.experiments.rq_hypothesis_tests import (
    build_contrast_specs,
    holm_bonferroni,
    run_one_contrast,
    ContrastSpec,
)
from scripts.experiments.train_eval import paired_sign_permutation_pvalue


def test_holm_bonferroni_toy_matches_step_up() -> None:
    # Sorted p 0.01, 0.03, 0.04 -> cumulative max of (m-j+1)*p_(j) in 0-based w[j]=(m-j)*sp[j]
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]


def test_holm_single_p() -> None:
    assert holm_bonferroni([0.2]) == [0.2]
    assert holm_bonferroni([0.99]) == [0.99]


def test_paired_sign_permutation_detects_shift() -> None:
    rng = np.random.default_rng(0)
    ref = np.ones(40, dtype=float)
    sample = ref + 0.15
    p_small = paired_sign_permutation_pvalue(
        sample, ref, rng=rng, n_permutations=2000
    )
    rng2 = np.random.default_rng(1)
    p_same = paired_sign_permutation_pvalue(
        ref.copy(), ref.copy(), rng=rng2, n_permutations=2000
    )
    assert p_small < 0.05
    assert p_same == 1.0


def test_broadcast_community_pred_to_stations_shape() -> None:
    station_index = ["s1", "s2", "s3"]
    station_to_group = {"s1": "A", "s2": "A", "s3": "B"}
    groups = ["A", "B"]
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    out = _broadcast_community_pred_to_stations(
        pred,
        station_index=station_index,
        station_to_group=station_to_group,
        groups=groups,
    )
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out[:, 0], pred[:, 0])
    np.testing.assert_array_equal(out[:, 1], pred[:, 0])
    np.testing.assert_array_equal(out[:, 2], pred[:, 1])


def test_build_contrast_specs_rq1_only() -> None:
    rows = [
        {
            "experiment_id": "RQ1_SD_STATION",
            "rq": "RQ1",
            "aggregation": "station",
            "graph_set": "SD",
            "model": "graph_propagation",
            "validation_wmape": 1.0,
            "test_wmape": 1.0,
        },
        {
            "experiment_id": "RQ1_DE_STATION",
            "rq": "RQ1",
            "aggregation": "station",
            "graph_set": "DE",
            "model": "graph_propagation",
            "validation_wmape": 1.0,
            "test_wmape": 1.0,
        },
        {
            "experiment_id": "RQ1_DC_STATION",
            "rq": "RQ1",
            "aggregation": "station",
            "graph_set": "DC",
            "model": "graph_propagation",
            "validation_wmape": 1.0,
            "test_wmape": 1.0,
        },
        {
            "experiment_id": "RQ1_DE_DC_STATION",
            "rq": "RQ1",
            "aggregation": "station",
            "graph_set": "DE+DC",
            "model": "graph_propagation",
            "validation_wmape": 1.0,
            "test_wmape": 1.0,
        },
    ]
    df = pd.DataFrame(rows)
    specs = build_contrast_specs(df, rqs={"RQ1"})
    assert len(specs) == 3


def test_run_one_contrast_empty_cohort(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    spec = ContrastSpec(
        rq="RQ1",
        contrast_id="t",
        h0="h",
        experiment_a="a",
        experiment_b="b",
        label_a="A",
        label_b="B",
    )
    va = np.array([1.0, 2.0])
    vb = np.array([1.1, 2.1])
    np.savez_compressed(tmp_path / "a.npz", wmape_by_station=va)
    np.savez_compressed(tmp_path / "b.npz", wmape_by_station=vb)
    row = run_one_contrast(
        spec=spec,
        cohort="empty",
        cohort_idx=np.array([], dtype=int),
        scores_dir=tmp_path,
        rng=rng,
        n_permutations=100,
        n_bootstrap=50,
        ci_level=0.95,
        alpha=0.05,
        two_sided=True,
    )
    assert row["n_stations_used"] == 0
    assert row["note"] == "empty_cohort"
