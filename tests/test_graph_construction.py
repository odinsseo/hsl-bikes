from __future__ import annotations

import numpy as np
import polars as pl

from scripts.graph_construction import (
    build_dc_adjacency,
    build_de_adjacency,
    build_sd_adjacency,
    build_station_index,
)


def _sample_trips() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "departure_name": ["A", "A", "B", "C", "B"],
            "return_name": ["B", "C", "C", "A", "A"],
            "departure": [
                "2022-01-01 10:05:00",
                "2022-01-01 10:20:00",
                "2022-01-01 11:00:00",
                "2022-01-01 11:40:00",
                "2022-01-01 12:00:00",
            ],
            "departure_latitude": [60.0, 60.0, 60.1, 60.2, 60.1],
            "departure_longitude": [24.9, 24.9, 24.95, 25.0, 24.95],
            "return_latitude": [60.1, 60.2, 60.2, 60.0, 60.0],
            "return_longitude": [24.95, 25.0, 25.0, 24.9, 24.9],
            "duration_sec": [600, 700, 800, 900, 650],
        }
    )


def test_de_row_normalization() -> None:
    df = _sample_trips()
    stations = build_station_index(df)

    de = build_de_adjacency(df=df, station_index=stations, row_normalized=True)

    non_zero_rows = de.sum(axis=1) > 0
    assert np.allclose(de[non_zero_rows].sum(axis=1), 1.0)


def test_sd_matrix_is_symmetric_after_knn_symmetrization() -> None:
    df = _sample_trips()
    stations = build_station_index(df)
    coords = pl.DataFrame(
        {
            "station_name": ["A", "B", "C"],
            "latitude": [60.0, 60.1, 60.2],
            "longitude": [24.9, 24.95, 25.0],
        }
    )

    sd = build_sd_adjacency(
        station_index=stations,
        station_coords=coords,
        k_neighbors=2,
        sigma_km=2.0,
        row_normalized=False,
    )

    assert np.allclose(sd, sd.T)
    assert np.all(np.diag(sd) == 0.0)


def test_dc_is_square_and_non_negative() -> None:
    df = _sample_trips()
    stations = build_station_index(df)

    dc = build_dc_adjacency(df=df, station_index=stations)

    assert dc.shape == (len(stations), len(stations))
    assert np.all(dc >= 0.0)
