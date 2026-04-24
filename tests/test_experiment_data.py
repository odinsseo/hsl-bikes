from __future__ import annotations

from datetime import datetime

import polars as pl

from scripts.experiments.data import (
    DEMAND_TIME_BUCKET,
    build_hourly_index,
    build_station_series,
)


def test_demand_time_bucket_is_three_hours() -> None:
    assert DEMAND_TIME_BUCKET == "3h"


def test_build_hourly_index_matches_station_series_length() -> None:
    df = pl.DataFrame(
        {
            "departure_name": ["S1", "S1", "S1"],
            "departure_ts": [
                datetime(2022, 6, 1, 1, 0, 0),
                datetime(2022, 6, 1, 4, 30, 0),
                datetime(2022, 6, 1, 7, 0, 0),
            ],
        }
    )
    stations = ["S1"]
    idx = build_hourly_index(df)
    series = build_station_series(df, stations)

    assert len(idx) == series.shape[0]
    assert series.shape == (3, 1)
    assert series.sum() == 3.0


def test_dense_grid_has_no_false_zero_gaps_between_buckets() -> None:
    """Regression: index grid interval must match aggregation truncate."""
    df = pl.DataFrame(
        {
            "departure_name": ["A", "A"],
            "departure_ts": [
                datetime(2022, 6, 1, 0, 0, 0),
                datetime(2022, 6, 1, 3, 0, 0),
            ],
        }
    )
    series = build_station_series(df, ["A"])
    assert series.shape[0] == 2
    assert float(series.sum()) == 2.0
