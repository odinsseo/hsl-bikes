from __future__ import annotations

import numpy as np

from scripts.experiments.train_eval import (
    bootstrap_mean_ci,
    build_station_cohort_indices,
    build_station_robustness_rows,
    metrics_for_indices,
    normalize_city_name,
    paired_sign_permutation_pvalue,
    parse_graph_set,
    station_wmape_vector,
)


def test_parse_graph_set_deduplicates_and_normalizes() -> None:
    graph_set = parse_graph_set("sd,DE,dc,DE")
    assert graph_set == ("SD", "DE", "DC")


def test_normalize_city_name_defaults_to_helsinki() -> None:
    assert normalize_city_name(None) == "Helsinki"
    assert normalize_city_name("") == "Helsinki"
    assert normalize_city_name("Esbo") == "Espoo"


def test_build_station_cohort_indices_shapes() -> None:
    train_series = np.array(
        [
            [10.0, 2.0, 1.0, 5.0],
            [11.0, 1.0, 0.0, 4.0],
            [12.0, 2.0, 0.0, 5.0],
        ]
    )
    station_index = ["A", "B", "C", "D"]
    city_lookup = {"A": "Helsinki", "B": "Espoo", "C": "Espoo", "D": "Helsinki"}

    cohorts = build_station_cohort_indices(
        train_series=train_series,
        station_index=station_index,
        city_lookup=city_lookup,
        sparse_quantile=0.25,
    )

    assert set(cohorts.keys()) == {
        "all",
        "helsinki",
        "espoo",
        "sparse",
        "dense",
        "sparse_helsinki",
        "sparse_espoo",
    }
    assert cohorts["all"].size == 4


def test_metrics_for_indices_handles_empty_indices() -> None:
    actual = np.array([[1.0, 2.0], [2.0, 3.0]])
    pred = np.array([[1.0, 2.0], [2.0, 3.0]])
    train = np.array([[0.0, 0.0], [1.0, 1.0]])

    metrics = metrics_for_indices(
        actual=actual,
        pred=pred,
        train_series=train,
        indices=np.array([], dtype=int),
    )

    assert np.isnan(metrics["wmape"])
    assert np.isnan(metrics["mae"])


def test_station_wmape_vector_computes_per_station() -> None:
    actual = np.array([[2.0, 4.0], [2.0, 2.0]])
    pred = np.array([[1.0, 4.0], [3.0, 1.0]])

    wmape = station_wmape_vector(actual=actual, pred=pred, indices=np.array([0, 1]))

    assert wmape.shape == (2,)
    assert np.isclose(wmape[0], 0.5)
    assert np.isclose(wmape[1], 1.0 / 6.0)


def test_bootstrap_mean_ci_contains_empirical_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(42)

    lower, upper = bootstrap_mean_ci(
        values,
        rng=rng,
        n_bootstrap=300,
        ci_level=0.95,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert lower <= float(values.mean()) <= upper


def test_paired_sign_permutation_pvalue_detects_gap() -> None:
    rng = np.random.default_rng(42)
    sample = np.full(10, 2.0)
    reference = np.full(10, 1.0)

    p_value = paired_sign_permutation_pvalue(
        sample,
        reference,
        rng=rng,
        n_permutations=2000,
    )

    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.05


def test_build_station_robustness_rows_includes_reference_rows() -> None:
    rng = np.random.default_rng(7)
    actual = np.array(
        [
            [10.0, 5.0],
            [12.0, 4.0],
            [11.0, 3.0],
        ]
    )
    predictions = {
        "graph_propagation": np.array(
            [
                [10.0, 5.0],
                [12.0, 4.0],
                [11.0, 3.0],
            ]
        ),
        "seasonal_naive": np.array(
            [
                [9.0, 4.0],
                [11.0, 3.0],
                [12.0, 4.0],
            ]
        ),
    }
    cohorts = {
        "all": np.array([0, 1], dtype=int),
        "first": np.array([0], dtype=int),
    }

    rows = build_station_robustness_rows(
        actual=actual,
        predictions=predictions,
        cohorts=cohorts,
        graph_set=("SD", "DE"),
        reference_model="graph_propagation",
        rng=rng,
        n_bootstrap=200,
        n_permutations=500,
        ci_level=0.95,
    )

    assert len(rows) == 4
    ref_rows = [row for row in rows if row["model"] == "graph_propagation"]
    assert len(ref_rows) == 2
    assert all(
        np.isclose(row["delta_station_wmape_vs_reference"], 0.0) for row in ref_rows
    )
    assert all(
        np.isclose(row["paired_sign_permutation_pvalue"], 1.0) for row in ref_rows
    )
