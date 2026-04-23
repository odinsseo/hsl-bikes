from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import torch

from scripts.experiments.stgnn import (
    A3TGCNGraphFusion,
    STGNNWindowDataset,
    _stgnn_window_collate,
    build_dynamic_covariates,
    build_stgnn_windows_with_covariates,
    build_supervised_windows,
    normalize_adjacency_for_gcn,
    parse_graph_name,
    parse_graph_set,
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


def test_parse_graph_set_deduplicates_and_normalizes() -> None:
    parsed = parse_graph_set("sd,DE,dc,DE")
    assert parsed == ("SD", "DE", "DC")


def test_parse_graph_set_rejects_empty() -> None:
    try:
        parse_graph_set(" , ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty graph set")


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


def test_build_supervised_windows_respects_sample_indices() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    x, y = build_supervised_windows(
        series=series,
        history=2,
        horizon=1,
        sample_indices=np.array([1]),
    )

    assert x.shape == (1, 2, 2)
    assert y.shape == (1, 2)
    assert np.allclose(x[0], np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert np.allclose(y[0], np.array([7.0, 8.0]))


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


def test_graph_fusion_fixed_weights_are_uniform() -> None:
    adjacency_stack = np.stack(
        [np.eye(3, dtype=float), np.ones((3, 3), dtype=float)],
        axis=0,
    )
    model = A3TGCNGraphFusion(
        adjacency_stack=adjacency_stack,
        input_channels=1,
        hidden_dim=4,
        dropout=0.0,
        learnable_fusion=False,
    )

    weights = model.get_fusion_weights()
    assert np.allclose(weights, [0.5, 0.5])
    assert np.isclose(weights.sum(), 1.0)


def test_graph_fusion_learned_weights_sum_to_one() -> None:
    adjacency_stack = np.stack(
        [np.eye(2, dtype=float), np.eye(2, dtype=float) * 2.0, np.ones((2, 2))],
        axis=0,
    )
    model = A3TGCNGraphFusion(
        adjacency_stack=adjacency_stack,
        input_channels=1,
        hidden_dim=4,
        dropout=0.0,
        learnable_fusion=True,
    )

    with torch.no_grad():
        model.fusion_logits.copy_(torch.tensor([0.0, 1.0, 2.0]))

    weights = model.get_fusion_weights()
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_build_stgnn_windows_with_covariates_has_expected_channels() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    dynamic = np.ones((4, 2, 2), dtype=float)
    static = np.array([[10.0], [20.0]], dtype=float)

    x, y = build_stgnn_windows_with_covariates(
        series,
        history=2,
        horizon=1,
        dynamic_covariates=dynamic,
        static_covariates=static,
    )

    assert x.shape == (2, 2, 2, 4)
    assert y.shape == (2, 2)
    assert np.allclose(x[0, :, :, 0], series[:2])
    assert np.allclose(x[:, :, :, -1], np.array([[[10.0, 20.0], [10.0, 20.0]]] * 2))


def test_build_stgnn_windows_with_covariates_respects_sample_indices() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    dynamic = np.ones((4, 2, 2), dtype=float)
    static = np.array([[10.0], [20.0]], dtype=float)

    x, y = build_stgnn_windows_with_covariates(
        series,
        history=2,
        horizon=1,
        dynamic_covariates=dynamic,
        static_covariates=static,
        sample_indices=np.array([1]),
    )

    assert x.shape == (1, 2, 2, 4)
    assert y.shape == (1, 2)
    assert np.allclose(x[0, :, :, 0], np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert np.allclose(y[0], np.array([7.0, 8.0]))


def test_build_dynamic_covariates_includes_calendar_and_sparse_features() -> None:
    series = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 2.0],
        ]
    )
    start = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i) for i in range(series.shape[0])]

    covariates, names, sparse_names = build_dynamic_covariates(
        series,
        timestamps,
        include_calendar_covariates=True,
        include_activity_mask=True,
        include_zero_run_indicator=True,
        zero_run_length=2,
        holiday_country="FI",
        holiday_subdivision="18",
    )

    assert covariates.shape == (3, 2, 8)
    assert names[:6] == [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
    ]
    assert sparse_names == ["recent_activity_mask", "long_zero_run_indicator"]


def test_graph_fusion_accepts_multichannel_input() -> None:
    adjacency_stack = np.stack([np.eye(2, dtype=float)], axis=0)
    model = A3TGCNGraphFusion(
        adjacency_stack=adjacency_stack,
        input_channels=3,
        hidden_dim=4,
        dropout=0.0,
        learnable_fusion=False,
    )
    x = torch.ones((5, 4, 2, 3), dtype=torch.float32)

    out = model(x)

    assert out.shape == (5, 2)


def test_graph_fusion_allows_negative_predictions() -> None:
    adjacency_stack = np.stack([np.eye(2, dtype=float)], axis=0)
    model = A3TGCNGraphFusion(
        adjacency_stack=adjacency_stack,
        input_channels=1,
        hidden_dim=2,
        dropout=0.0,
        learnable_fusion=False,
    )
    for param in model.parameters():
        param.data.zero_()

    x = torch.full((3, 4, 2, 1), -2.0, dtype=torch.float32)
    out = model(x)

    assert out.shape == (3, 2)
    assert torch.all(out < 0.0)


def test_stgnn_window_dataset_matches_expected_shapes() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    dynamic = np.ones((4, 2, 2), dtype=float)
    static = np.array([[10.0], [20.0]], dtype=float)

    dataset = STGNNWindowDataset(
        series=series,
        history=2,
        horizon=1,
        dynamic_covariates=dynamic,
        static_covariates=static,
    )

    assert len(dataset) == 2
    assert dataset.input_channels == 4
    x_base, x_dynamic, x_static, y = dataset[0]
    assert x_base.shape == (2, 2, 1)
    assert x_dynamic is not None
    assert x_dynamic.shape == (2, 2, 2)
    assert x_static is not None
    assert x_static.shape == (2, 2, 1)
    assert y.shape == (2,)
    x = np.concatenate([x_base, x_dynamic, x_static], axis=-1)
    assert x.shape == (2, 2, 4)
    assert np.allclose(x[:, :, 0], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(y, np.array([5.0, 6.0]))


def test_stgnn_window_dataset_respects_sample_indices() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    dataset = STGNNWindowDataset(
        series=series,
        history=2,
        horizon=1,
        sample_indices=np.array([1]),
    )

    assert len(dataset) == 1
    x_base, x_dynamic, x_static, y = dataset[0]
    assert x_dynamic is None
    assert x_static is None
    assert np.allclose(x_base[:, :, 0], np.array([[3.0, 4.0], [5.0, 6.0]]))
    assert np.allclose(y, np.array([7.0, 8.0]))


def test_stgnn_window_collate_concatenates_channels_per_batch() -> None:
    series = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    dynamic = np.ones((4, 2, 1), dtype=float)
    static = np.array([[10.0], [20.0]], dtype=float)
    dataset = STGNNWindowDataset(
        series=series,
        history=2,
        horizon=1,
        dynamic_covariates=dynamic,
        static_covariates=static,
    )

    x_batch, y_batch = _stgnn_window_collate([dataset[0], dataset[1]])

    assert x_batch.shape == (2, 2, 2, 3)
    assert y_batch.shape == (2, 2)
    assert torch.allclose(x_batch[:, :, :, 1], torch.ones((2, 2, 2)))
