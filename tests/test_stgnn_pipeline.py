from __future__ import annotations

import numpy as np

from scripts.experiments.stgnn import (
    build_supervised_windows,
    normalize_adjacency_for_gcn,
    parse_graph_name,
)


def test_parse_graph_name_normalizes_case() -> None:
    assert parse_graph_name("de") == "DE"


def test_parse_graph_name_rejects_unknown() -> None:
    try:
        parse_graph_name("XYZ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown graph")


def test_build_supervised_windows_shape() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    x, y = build_supervised_windows(series=series, history=2, horizon=1)

    assert x.shape == (2, 2, 2)
    assert y.shape == (2, 2)
    assert np.allclose(y[0], [5.0, 6.0])


def test_build_supervised_windows_returns_empty_when_too_short() -> None:
    series = np.array([[1.0, 2.0], [3.0, 4.0]])

    x, y = build_supervised_windows(series=series, history=4, horizon=1)

    assert x.shape[0] == 0
    assert y.shape[0] == 0


def test_normalize_adjacency_for_gcn_is_finite() -> None:
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ]
    )

    norm = normalize_adjacency_for_gcn(adjacency)

    assert norm.shape == (3, 3)
    assert np.isfinite(norm).all()
    assert np.all(norm >= 0.0)
