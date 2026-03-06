"""
Load and prepare Helsinki/Espoo city bike OD trip data from data/trips (by year/month),
merge with station coordinates from data/stations, and optionally split into
train/validation/test by time (time-series best practice).
Output: data/merged/trips_merged.csv and data/train, data/validation, data/test.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import sys
import warnings

import pandas as pd
from tqdm import tqdm

# Ensure project root is on path for config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.constants import RENAMED_STATIONS, STATIONS_TO_DROP_PREFIXES
DATA_DIR = PROJECT_ROOT / "data"
TRIPS_BASE = DATA_DIR / "trips"
STATIONS_DIR = DATA_DIR / "stations"
MERGED_DIR = DATA_DIR / "merged"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

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


def get_trip_csv_paths():
    """Return list of (path, year, month) for all trip CSVs under data/trips."""
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


def load_stations(stations_dir: Path) -> pd.DataFrame:
    """
    Load station coordinates from data/stations (one CSV). Returns raw DataFrame
    with columns Nimi, Name, x (longitude), y (latitude) for building coords map.
    """
    csvs = list(stations_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {stations_dir}")
    df = pd.read_csv(csvs[0])
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Stations CSV must have x, y. Found: {list(df.columns)}")
    return df


def build_name_to_coords_from_df(stations_df: pd.DataFrame) -> dict:
    """
    Build dict station name -> (lat, lon) from stations DataFrame.
    Uses both Nimi and Name so trip data matches whether it uses Finnish or English names.
    """
    coords = {}
    for _, row in stations_df.iterrows():
        lon, lat = float(row["x"]), float(row["y"])
        for col in ("Nimi", "Name"):
            if col in stations_df.columns and pd.notna(row[col]):
                name = str(row[col]).strip()
                if name and name.lower() != "nan":
                    coords[name] = (lat, lon)
    return coords


def load_and_rename_one(path: Path) -> pd.DataFrame:
    """Load a single trip CSV with dtypes/parse_dates and normalize column names."""
    kwargs = {"low_memory": False}
    # Speed up read: specify dtypes and parse_dates (only for columns that exist)
    kwargs["dtype"] = {
        "Departure station id": str,
        "Return station id": str,
        "Departure station name": str,
        "Return station name": str,
    }
    kwargs["parse_dates"] = ["Departure", "Return"]
    df = pd.read_csv(path, **kwargs)
    rename = {k: v for k, v in COL_RENAME.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    if "Covered distance (m)" in df.columns and "distance_m" not in df.columns:
        df = df.rename(columns={"Covered distance (m)": "distance_m"})
    if "Duration (sec.)" in df.columns and "duration_sec" not in df.columns:
        df = df.rename(columns={"Duration (sec.)": "duration_sec"})
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Apply renames, drop maintenance stations, parse datetimes, add speed."""
    df = df.copy()
    # Strip station names
    for col in ("departure_name", "return_name"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Replace renamed stations
    df["departure_name"] = df["departure_name"].replace(RENAMED_STATIONS)
    df["return_name"] = df["return_name"].replace(RENAMED_STATIONS)
    # Drop maintenance / non-public stations
    for col in ("departure_name", "return_name"):
        if col in df.columns:
            for prefix in STATIONS_TO_DROP_PREFIXES:
                df = df[~df[col].astype(str).str.startswith(prefix)]
    # Drop rows with missing critical fields
    df = df.dropna(subset=["departure", "return", "departure_name", "return_name"], how="any")
    # Parse datetimes (allow ISO and space format)
    df["departure"] = pd.to_datetime(df["departure"], errors="coerce")
    df["return"] = pd.to_datetime(df["return"], errors="coerce")
    df = df.dropna(subset=["departure", "return"])
    # Numeric distance/duration
    for col, dtype in [("distance_m", float), ("duration_sec", float)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop invalid duration/distance (e.g. negative or zero duration)
    if "duration_sec" in df.columns:
        df = df[df["duration_sec"] > 0]
    if "distance_m" in df.columns:
        df = df[df["distance_m"] >= 0]
    # Average speed km/h: (distance_m / 1000) / (duration_sec / 3600) = distance_m / duration_sec * 3.6
    df["speed_kmh"] = (df["distance_m"] / df["duration_sec"]) * 3.6
    # Duration in minutes for convenience
    df["duration_min"] = df["duration_sec"] / 60.0
    return df


def add_station_coordinates(df: pd.DataFrame, name_to_coords: dict) -> pd.DataFrame:
    """Add departure/return lat/lon via vectorized .map() from name_to_lat/lon dicts."""
    name_to_lat = {k: v[0] for k, v in name_to_coords.items()}
    name_to_lon = {k: v[1] for k, v in name_to_coords.items()}
    df = df.copy()
    df["departure_latitude"] = df["departure_name"].map(name_to_lat)
    df["departure_longitude"] = df["departure_name"].map(name_to_lon)
    df["return_latitude"] = df["return_name"].map(name_to_lat)
    df["return_longitude"] = df["return_name"].map(name_to_lon)
    return df


def _load_and_clean_one(path: Path, year: int, month: int) -> pd.DataFrame:
    """Load one trip CSV, rename, clean per-file; used by parallel workers."""
    df = load_and_rename_one(path)
    df["_year"] = year
    df["_month"] = month
    return clean_trips(df)


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
    name_to_coords = build_name_to_coords_from_df(stations_df)
    print(f"done ({len(name_to_coords)} station names with coordinates).")

    # Load and clean trip files in parallel (per-file clean), then concat
    max_workers = min(len(paths), (os.cpu_count() or 2) - 1) or 1
    print(f"Loading and cleaning trip files (workers={max_workers})...")
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
        except (PermissionError, OSError):
            executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
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
        finally:
            executor.shutdown(wait=True)
    print(f"Concatenating {len(frames)} DataFrames...", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"Adding station coordinates ({len(df):,} rows)...", flush=True)
    df = add_station_coordinates(df, name_to_coords)

    # Optional: drop rows with missing coordinates (stations not in our list)
    missing_dep = df["departure_latitude"].isna()
    missing_ret = df["return_latitude"].isna()
    if missing_dep.any() or missing_ret.any():
        n_before = len(df)
        df = df.dropna(subset=["departure_latitude", "departure_longitude", "return_latitude", "return_longitude"])
        print(f"Dropped {n_before - len(df):,} rows with missing station coordinates.")
    print("Sorting by departure time...", flush=True)
    df = df.sort_values("departure").reset_index(drop=True)

    if save_merged:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MERGED_DIR / "trips_merged.csv"
        print(f"Writing merged CSV to {out_path}...", flush=True)
        df.to_csv(out_path, index=False)
        print(f"  Saved merged data: {out_path} ({len(df):,} rows)")

    if save_splits:
        train_end = pd.Timestamp(train_end_date)
        val_end = pd.Timestamp(val_end_date)
        train_df = df[df["departure"] < train_end]
        val_df = df[(df["departure"] >= train_end) & (df["departure"] < val_end)]
        test_df = df[df["departure"] >= val_end]
        print("Writing train/validation/test splits...")
        for name, part, dir_path in [
            ("train", train_df, TRAIN_DIR),
            ("validation", val_df, VAL_DIR),
            ("test", test_df, TEST_DIR),
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            fname = dir_path / f"{name}.csv"
            part.to_csv(fname, index=False)
            print(f"  {name}: {fname} ({len(part):,} rows)")
    print("Done.")

    return df


if __name__ == "__main__":
    run(save_merged=True, save_splits=True)
