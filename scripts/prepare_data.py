"""
Load and prepare Helsinki/Espoo city bike OD trip data from data/primary/trips (by year/month),
merge with station coordinates from data/primary/stations, and optionally split into
train/validation/test by time (time-series best practice).

Drops trips whose departure falls in HSL inactive months (November–March); see
`HSL_CITY_BIKE_INACTIVE_MONTHS` in config/constants.py.

Output: data/prepared/merged/trips_merged.csv and data/prepared/splits/*.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

# Ensure project root is on path for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.constants import (
    HSL_CITY_BIKE_INACTIVE_MONTHS,
    RENAMED_STATIONS,
    STATIONS_TO_DROP_PREFIXES,
)

DATA_DIR = PROJECT_ROOT / "data"
PRIMARY_DIR = DATA_DIR / "primary"
PREPARED_DIR = DATA_DIR / "prepared"

TRIPS_BASE = PRIMARY_DIR / "trips"
STATIONS_DIR = PRIMARY_DIR / "stations"
MERGED_DIR = PREPARED_DIR / "merged"
SPLITS_DIR = PREPARED_DIR / "splits"
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "validation"
TEST_DIR = SPLITS_DIR / "test"

# Column name normalization (OD CSV headers)
COL_RENAME = {
    "Departure": "departure",
    "Return": "return",
    "Departure station id": "departure_id",
    "Departure station name": "departure_name",
    "Return station id": "return_id",
    "Return station name": "return_name",
    "Covered distance (m)": "distance_m",
    "Duration (sec.)": "duration_sec",
}
CSV_SCHEMA_OVERRIDES = {
    "Departure station id": pl.String,
    "Return station id": pl.String,
    "Departure station name": pl.String,
    "Return station name": pl.String,
    "Covered distance (m)": pl.Float64,
    "Duration (sec.)": pl.Float64,
    "departure_id": pl.String,
    "return_id": pl.String,
    "departure_name": pl.String,
    "return_name": pl.String,
    "distance_m": pl.Float64,
    "duration_sec": pl.Float64,
}


def get_trip_csv_paths():
    """Return list of (path, year, month) for all trip CSVs under data/primary/trips."""
    paths = []
    if not TRIPS_BASE.exists():
        return paths
    for year_dir in sorted(TRIPS_BASE.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.glob("*.csv")):
            # e.g. 2024-08.csv
            stem = f.stem
            if len(stem) == 7 and stem[4] == "-":
                try:
                    y, m = int(stem[:4]), int(stem[5:7])
                    paths.append((f, y, m))
                except ValueError:
                    pass
    return paths


def load_stations(stations_dir: Path) -> pl.DataFrame:
    """
    Load station coordinates from data/primary/stations (one CSV). Returns raw DataFrame
    with columns Nimi, Name, x (longitude), y (latitude) for building coords map.
    """
    csvs = list(stations_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {stations_dir}")
    df = pl.read_csv(csvs[0], try_parse_dates=True)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Stations CSV must have x, y. Found: {list(df.columns)}")
    return df


def build_station_coordinates(stations_df: pl.DataFrame) -> pl.DataFrame:
    """
    Build station_name -> (lat, lon) table from stations DataFrame.
    Uses both Nimi and Name so trip data matches whether it uses Finnish or English names.
    """
    rows: list[dict[str, object]] = []
    for row in stations_df.iter_rows(named=True):
        if row.get("x") is None or row.get("y") is None:
            continue
        lon, lat = float(row["x"]), float(row["y"])
        for col in ("Nimi", "Name"):
            if col in stations_df.columns and row.get(col) is not None:
                name = str(row[col]).strip()
                if name and name.lower() != "nan":
                    rows.append(
                        {
                            "station_name": name,
                            "latitude": lat,
                            "longitude": lon,
                        }
                    )

    if not rows:
        raise ValueError("No valid station names with coordinates found.")

    return pl.DataFrame(rows).unique(subset=["station_name"], keep="first")


def load_and_rename_one(path: Path) -> pl.DataFrame:
    """Load a single trip CSV with dtypes/parse_dates and normalize column names."""
    df = pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides=CSV_SCHEMA_OVERRIDES,
    )

    rename = {k: v for k, v in COL_RENAME.items() if k in df.columns}
    if rename:
        df = df.rename(rename)

    cast_expr = []
    for col in ("departure_id", "return_id", "departure_name", "return_name"):
        if col in df.columns:
            cast_expr.append(pl.col(col).cast(pl.String, strict=False).alias(col))
    if cast_expr:
        df = df.with_columns(cast_expr)

    return df


def clean_trips(df: pl.DataFrame) -> pl.DataFrame:
    """Apply renames, drop maintenance stations, parse datetimes, add speed."""
    for col in ("departure_name", "return_name"):
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.String)
                .str.strip_chars()
                .map_elements(
                    lambda x: RENAMED_STATIONS.get(x, x) if x is not None else None,
                    return_dtype=pl.String,
                )
                .alias(col)
            )

    # Drop maintenance / non-public stations.
    for col in ("departure_name", "return_name"):
        if col in df.columns:
            drop_expr = pl.lit(False)
            for prefix in STATIONS_TO_DROP_PREFIXES:
                drop_expr = drop_expr | pl.col(col).str.starts_with(prefix)
            df = df.filter(~drop_expr)

    critical = [
        c
        for c in ("departure", "return", "departure_name", "return_name")
        if c in df.columns
    ]
    if critical:
        df = df.drop_nulls(critical)

    for col in ("departure", "return"):
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.String, strict=False)
                .str.strptime(pl.Datetime, strict=False)
                .alias(col)
            )

    datetime_cols = [c for c in ("departure", "return") if c in df.columns]
    if datetime_cols:
        df = df.drop_nulls(datetime_cols)

    for col in ("distance_m", "duration_sec"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    if "duration_sec" in df.columns:
        df = df.filter(pl.col("duration_sec") > 0)
    if "distance_m" in df.columns:
        df = df.filter(pl.col("distance_m") >= 0)

    if "distance_m" in df.columns and "duration_sec" in df.columns:
        df = df.with_columns(
            [
                ((pl.col("distance_m") / pl.col("duration_sec")) * 3.6).alias(
                    "speed_kmh"
                ),
                (pl.col("duration_sec") / 60.0).alias("duration_min"),
            ]
        )

    return df


def add_station_coordinates(
    df: pl.DataFrame, station_coords: pl.DataFrame
) -> pl.DataFrame:
    """Add departure/return coordinates by joining station lookup table."""
    dep = station_coords.rename(
        {
            "station_name": "departure_name",
            "latitude": "departure_latitude",
            "longitude": "departure_longitude",
        }
    )
    ret = station_coords.rename(
        {
            "station_name": "return_name",
            "latitude": "return_latitude",
            "longitude": "return_longitude",
        }
    )
    return df.join(dep, on="departure_name", how="left").join(
        ret, on="return_name", how="left"
    )


def _load_and_clean_one(path: Path, year: int, month: int) -> pl.DataFrame:
    """Load one trip CSV, rename, clean per-file; used by parallel workers."""
    df = load_and_rename_one(path)
    df = df.with_columns([pl.lit(year).alias("_year"), pl.lit(month).alias("_month")])
    return clean_trips(df)


def _parse_datetime(value: str):
    parsed = pl.Series("ts", [value]).str.strptime(pl.Datetime, strict=False).item()
    if parsed is None:
        raise ValueError(f"Could not parse datetime: {value}")
    return parsed


def run(
    save_merged: bool = True,
    save_splits: bool = True,
    train_end_date: str = "2022-01-01",
    val_end_date: str = "2023-01-01",
):
    """
    Load all trip CSVs, clean, merge with stations, save merged and optionally train/val/test.
    Time split: train = departure < train_end_date, val = train_end_date <= departure < val_end_date, test = rest.
    """
    paths = get_trip_csv_paths()
    if not paths:
        raise FileNotFoundError(f"No trip CSVs found under {TRIPS_BASE}")
    print(f"Found {len(paths)} trip CSV files under {TRIPS_BASE}")

    # Load stations once and build name -> (lat, lon) from it
    print("Loading stations...", end=" ", flush=True)
    stations_df = load_stations(STATIONS_DIR)
    station_coords = build_station_coordinates(stations_df)
    print(f"done ({station_coords.height:,} station names with coordinates).")

    # Load and clean trip files in parallel (per-file clean), then concat
    max_workers = min(len(paths), max(1, (os.cpu_count() or 2) - 1))
    print(f"Loading and cleaning trip files (workers={max_workers})...")
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_load_and_clean_one, path, y, m): path
            for path, y, m in paths
        }
        for future in tqdm(
            as_completed(future_to_path),
            total=len(paths),
            desc="Trip files",
            unit="file",
        ):
            path = future_to_path[future]
            try:
                frames.append(future.result())
            except Exception as e:
                tqdm.write(f"Skip {path.name}: {e}")

    if not frames:
        raise RuntimeError("No trip files could be loaded successfully.")

    print(f"Concatenating {len(frames)} DataFrames...", flush=True)
    df = pl.concat(frames, how="vertical_relaxed")
    df = df.with_columns(
        pl.col("departure")
        .cast(pl.String, strict=False)
        .str.strptime(pl.Datetime, strict=False)
        .alias("departure")
    ).drop_nulls(["departure"])
    n_before_season = df.height
    inactive = sorted(HSL_CITY_BIKE_INACTIVE_MONTHS)
    df = df.filter(~pl.col("departure").dt.month().is_in(inactive))
    dropped_season = n_before_season - df.height
    if dropped_season:
        print(
            f"Excluded {dropped_season:,} trips in HSL inactive months "
            f"(months {inactive})..."
        )
    print(f"Adding station coordinates ({df.height:,} rows)...", flush=True)
    df = add_station_coordinates(df, station_coords)

    # Optional: drop rows with missing coordinates (stations not in our list)
    coord_cols = [
        "departure_latitude",
        "departure_longitude",
        "return_latitude",
        "return_longitude",
    ]
    n_before = df.height
    df = df.drop_nulls(coord_cols)
    dropped = n_before - df.height
    if dropped > 0:
        print(f"Dropped {dropped:,} rows with missing station coordinates.")

    print("Sorting by departure time...", flush=True)
    df = df.sort("departure")

    if save_merged:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MERGED_DIR / "trips_merged.csv"
        print(f"Writing merged CSV to {out_path}...", flush=True)
        df.write_csv(out_path)
        print(f"  Saved merged data: {out_path} ({df.height:,} rows)")

    if save_splits:
        train_end = _parse_datetime(train_end_date)
        val_end = _parse_datetime(val_end_date)
        df = df.with_columns(
            pl.col("departure")
            .cast(pl.String, strict=False)
            .str.strptime(pl.Datetime, strict=False)
            .alias("departure")
        ).drop_nulls(["departure"])
        train_df = df.filter(pl.col("departure") < train_end)
        val_df = df.filter(
            (pl.col("departure") >= train_end) & (pl.col("departure") < val_end)
        )
        test_df = df.filter(pl.col("departure") >= val_end)
        print("Writing train/validation/test splits...")
        for name, part, dir_path in [
            ("train", train_df, TRAIN_DIR),
            ("validation", val_df, VAL_DIR),
            ("test", test_df, TEST_DIR),
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            fname = dir_path / f"{name}.csv"
            part.write_csv(fname)
            print(f"  {name}: {fname} ({part.height:,} rows)")
    print("Done.")

    return df


if __name__ == "__main__":
    run(save_merged=True, save_splits=True)
