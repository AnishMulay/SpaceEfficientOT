from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnMapping:
    pickup_time: str
    dropoff_time: str
    pickup_lon: str
    pickup_lat: str
    dropoff_lon: str
    dropoff_lat: str


# Widened slightly to reduce accidental exclusion of valid outer-edges trips
# around the NYC metro area while still filtering obvious outliers.
DEFAULT_LAT_RANGE = (39.8, 41.2)
DEFAULT_LON_RANGE = (-75.2, -72.8)


def _resolve_columns(df: pd.DataFrame) -> ColumnMapping:
    cols = {c.lower(): c for c in df.columns}

    def find(patterns: list[str]) -> Optional[str]:
        for pattern in patterns:
            if pattern.lower() in cols:
                return cols[pattern.lower()]
        return None

    pickup_time = find(["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"])
    dropoff_time = find(["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"])
    pickup_lon = find(["pickup_longitude", "pickup_lon", "pickup_long"])
    pickup_lat = find(["pickup_latitude", "pickup_lat"])
    dropoff_lon = find(["dropoff_longitude", "dropoff_lon", "dropoff_long"])
    dropoff_lat = find(["dropoff_latitude", "dropoff_lat"])

    mapping = {
        "pickup_time": pickup_time,
        "dropoff_time": dropoff_time,
        "pickup_lon": pickup_lon,
        "pickup_lat": pickup_lat,
        "dropoff_lon": dropoff_lon,
        "dropoff_lat": dropoff_lat,
    }

    missing = [name for name, col in mapping.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    return ColumnMapping(**mapping)  # type: ignore[arg-type]


def _load_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def load_day(
    path: str | Path,
    *,
    date: str,
    n: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 1,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    logger: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, ColumnMapping]:
    """Load NYC taxi trips for a single day and return filtered data + column mapping."""

    def _log(message: str) -> None:
        if logger is not None:
            logger(message)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    _log(f"Loading NYC taxi data from {path}")
    df = _load_file(path)
    _log(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    mapping = _resolve_columns(df)
    _log(
        "Resolved columns: "
        f"pickup_time={mapping.pickup_time}, dropoff_time={mapping.dropoff_time}, "
        f"pickup_lon={mapping.pickup_lon}, pickup_lat={mapping.pickup_lat}, "
        f"dropoff_lon={mapping.dropoff_lon}, dropoff_lat={mapping.dropoff_lat}"
    )

    df = df.copy()
    df[mapping.pickup_time] = pd.to_datetime(df[mapping.pickup_time])
    df[mapping.dropoff_time] = pd.to_datetime(df[mapping.dropoff_time])

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    mask = (df[mapping.pickup_time] >= day_start) & (df[mapping.pickup_time] < day_end)
    df = df.loc[mask].copy()
    _log(f"Rows after filtering to date {date}: {len(df)}")

    if df.empty:
        raise ValueError(f"No trips found for date {date}")

    coord_cols = [
        mapping.pickup_lat,
        mapping.pickup_lon,
        mapping.dropoff_lat,
        mapping.dropoff_lon,
    ]

    df = df.dropna(subset=coord_cols)
    _log(f"Rows after dropping NA coordinates: {len(df)}")

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    within_bounds = (
        (df[mapping.pickup_lat].between(lat_min, lat_max))
        & (df[mapping.dropoff_lat].between(lat_min, lat_max))
        & (df[mapping.pickup_lon].between(lon_min, lon_max))
        & (df[mapping.dropoff_lon].between(lon_min, lon_max))
    )

    df = df.loc[within_bounds].copy()
    _log(f"Rows within coordinate bounds lat={lat_range}, lon={lon_range}: {len(df)}")

    if df.empty:
        raise ValueError("No trips remain after coordinate filtering")

    # Sampling
    if n is not None and n > 0 and n < len(df):
        if random_sample:
            df = df.sample(n=n, random_state=seed)
            _log(f"Randomly sampled {n} trips (seed={seed})")
        else:
            df = df.iloc[:n]
            _log(f"Selected first {n} trips (sorted by pickup time)")
    elif n is not None and n >= len(df):
        _log(f"Requested n={n}, using all {len(df)} trips")

    df = df.sort_values(mapping.pickup_time).reset_index(drop=True)
    _log(f"Final dataset size after sorting: {len(df)} trips")

    return df, mapping
