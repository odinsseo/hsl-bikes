from __future__ import annotations

import numpy as np

from scripts.experiments.train_eval import (
    build_station_cohort_indices,
    metrics_for_indices,
    normalize_city_name,
    parse_graph_set,
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
