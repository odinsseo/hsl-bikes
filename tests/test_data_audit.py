from __future__ import annotations

import polars as pl

from scripts.data_audit import (
    evaluate_split_boundaries,
    evaluate_temporal_overlap,
    summarize_dataset,
)


def _df_with_departures(values: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"departure": values, "x": list(range(len(values)))})


def test_summarize_dataset_exposes_core_fields() -> None:
    df = pl.DataFrame(
        {
            "departure": ["2021-01-01 10:00:00", "2021-01-01 11:00:00", None],
            "station": ["A", "A", "B"],
        }
    )

    summary = summarize_dataset(df, timestamp_col="departure")

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["timestamp_missing"] == 1
    assert "departure" in summary["null_rate_percent"]


def test_split_boundaries_pass_for_valid_split() -> None:
    train = _df_with_departures(["2021-12-31 23:00:00"])
    validation = _df_with_departures(["2022-06-01 10:00:00"])
    test = _df_with_departures(["2023-04-01 10:00:00"])

    checks = evaluate_split_boundaries(
        train_df=train,
        validation_df=validation,
        test_df=test,
        train_end="2022-01-01",
        validation_end="2023-01-01",
    )

    assert checks["all_passed"] is True


def test_split_boundaries_fail_when_validation_outside_range() -> None:
    train = _df_with_departures(["2021-12-31 23:00:00"])
    validation = _df_with_departures(["2023-03-01 00:00:00"])
    test = _df_with_departures(["2023-04-01 10:00:00"])

    checks = evaluate_split_boundaries(
        train_df=train,
        validation_df=validation,
        test_df=test,
        train_end="2022-01-01",
        validation_end="2023-01-01",
    )

    assert checks["validation_in_range"] is False
    assert checks["all_passed"] is False


def test_temporal_overlap_detects_overlap() -> None:
    train = _df_with_departures(["2022-05-01 00:00:00"])
    validation = _df_with_departures(["2022-04-30 23:00:00"])
    test = _df_with_departures(["2023-01-01 00:00:00"])

    checks = evaluate_temporal_overlap(train, validation, test)

    assert checks["train_validation_non_overlap"] is False
    assert checks["all_passed"] is False
