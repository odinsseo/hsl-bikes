from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path

from scripts.experiments.contracts import REQUIRED_PREPROCESSING_LINEAGE_FIELDS
from scripts.experiments.stgnn_milestones import (
    _coerce_optional_float,
    _namespace_for_stgnn,
    _optional_delta,
    _preprocessing_lineage_from_args,
)


def _base_args(tmp_path: Path) -> Namespace:
    return Namespace(
        train=tmp_path / "train.csv",
        validation=tmp_path / "validation.csv",
        test=tmp_path / "test.csv",
        communities=tmp_path / "communities.csv",
        history=24,
        horizon=1,
        hidden_dim=32,
        dropout=0.1,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=2,
        early_stop_min_delta=1e-3,
        early_stop_start_epoch=5,
        optimizer="adamw",
        lr_scheduler="plateau",
        lr_decay_factor=0.5,
        lr_decay_patience=5,
        lr_plateau_threshold=1e-3,
        min_learning_rate=1e-5,
        max_grad_norm=1.0,
        epoch_progress=False,
        max_train_windows=0,
        preprocess_target=True,
        winsor_lower_quantile=0.005,
        winsor_upper_quantile=0.995,
        preprocess_scaler="robust",
        residualize_target=True,
        residual_lag_candidates="24,168",
        holiday_country="FI",
        holiday_subdivision="18",
        holiday_national_only=False,
        include_calendar_covariates=True,
        include_activity_mask=True,
        include_zero_run_indicator=True,
        zero_run_length=6,
        include_static_features=True,
        num_workers=0,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        lazy_windows=False,
        cache_preprocessed=True,
        refresh_preprocessed_cache=False,
        preprocessed_cache_dir=tmp_path / "cache",
        device="cpu",
        random_state=42,
    )


def test_namespace_for_stgnn_forwards_preprocessing_and_covariate_flags(
    tmp_path: Path,
) -> None:
    args = _base_args(tmp_path)

    namespace = _namespace_for_stgnn(
        args,
        graph_dir=tmp_path / "graphs",
        output_dir=tmp_path / "out",
        aggregation="station",
        graph_set="SD,DE",
        fusion_mode="learned",
        allow_leaky_graph_source=False,
    )

    assert namespace.preprocess_target is True
    assert namespace.residualize_target is True
    assert namespace.residual_lag_candidates == "24,168"
    assert namespace.include_calendar_covariates is True
    assert namespace.include_activity_mask is True
    assert namespace.include_zero_run_indicator is True
    assert namespace.include_static_features is True
    assert namespace.optimizer == "adamw"
    assert namespace.lr_scheduler == "plateau"
    assert math.isclose(namespace.early_stop_min_delta, 1e-3)
    assert namespace.early_stop_start_epoch == 5
    assert math.isclose(namespace.lr_decay_factor, 0.5)
    assert namespace.lr_decay_patience == 5
    assert math.isclose(namespace.lr_plateau_threshold, 1e-3)
    assert math.isclose(namespace.min_learning_rate, 1e-5)
    assert math.isclose(namespace.max_grad_norm, 1.0)
    assert namespace.epoch_progress is False
    assert namespace.num_workers == 0
    assert namespace.prefetch_factor == 2
    assert namespace.pin_memory is True
    assert namespace.persistent_workers is True
    assert namespace.lazy_windows is False
    assert namespace.cache_preprocessed is True
    assert namespace.refresh_preprocessed_cache is False
    assert namespace.preprocessed_cache_dir == tmp_path / "cache"


def test_preprocessing_lineage_from_args_has_required_fields(tmp_path: Path) -> None:
    args = _base_args(tmp_path)

    lineage = _preprocessing_lineage_from_args(args)

    for field in REQUIRED_PREPROCESSING_LINEAGE_FIELDS:
        assert field in lineage
    assert lineage["calendar_source"]["country"] == "FI"
    assert lineage["calendar_source"]["subdivision"] == "18"


def test_coerce_optional_float_handles_missing_or_bad_values() -> None:
    assert _coerce_optional_float(None) is None
    assert _coerce_optional_float("") is None
    assert _coerce_optional_float("  ") is None
    assert _coerce_optional_float("abc") is None
    assert _coerce_optional_float(float("inf")) is None
    assert _coerce_optional_float(float("nan")) is None
    assert _coerce_optional_float("0.123") == 0.123


def test_optional_delta_returns_none_when_inputs_are_missing() -> None:
    assert _optional_delta(None, 0.1) is None
    assert _optional_delta(0.1, None) is None
    assert _optional_delta("bad", 0.1) is None
    assert math.isclose(_optional_delta(0.3, 0.1), 0.2)
