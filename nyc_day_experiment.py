#!/usr/bin/env python3
"""
nyc_day_experiment.py

- Reads a month-wide NYC Yellow Taxi 2014 file (Parquet or CSV).
- Filters to ONE local calendar day (e.g., 2014-06-15) in America/New_York.
- Prints the number of entries for that day.
- Builds bipartite tensors on GPU:
    xA:[N,2]  dropoff XY (A/left)
    xB:[N,2]  pickup  XY (B/right)
    tA:[N], tB:[N]  int64 seconds; here tA == tB == pickup_time
- Calls spef_matching_2(...) from spef_matching_nyc.py (same directory).

Run:
  python nyc_day_experiment.py --input ./data/yellow_tripdata_2014-06.parquet --date 2014-06-15 --n 100000 --tile_k 4096 --C 32 --delta 1.0 --seed 1
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch

# Hard-coded: import solver from the same folder
import spef_matching_nyc as solver


# ---------------------------
#  Projection: lon/lat -> local XY (meters) around NYC
# ---------------------------
def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray,
                 lon0: float = -74.0, lat0: float = 40.7) -> Tuple[np.ndarray, np.ndarray]:
    """Fast local equirectangular projection (meters)."""
    R = 6371000.0  # Earth radius in meters
    lonr = np.deg2rad(lon.astype(np.float64))
    latr = np.deg2rad(lat.astype(np.float64))
    lon0r = np.deg2rad(lon0)
    lat0r = np.deg2rad(lat0)
    x = (lonr - lon0r) * np.cos(lat0r) * R
    y = (latr - lat0r) * R
    return x.astype(np.float32), y.astype(np.float32)


# ---------------------------
#  IO
# ---------------------------
def load_day(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def normalize_columns(df: pd.DataFrame):
    """Return canonical column names for pickup/dropoff timestamps and coords."""
    # Datetime columns: prefer 2014 names; fall back to tpep_* if present.
    if "pickup_datetime" in df.columns and "dropoff_datetime" in df.columns:
        pu_col, do_col = "pickup_datetime", "dropoff_datetime"
    else:
        pu_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
        do_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"

    # Coordinate columns (2014 schema has exact coords)
    required = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}. Expected {required} in the 2014 dataset.")
    return pu_col, do_col, "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"


def ensure_local_tz(series: pd.Series, tz: str) -> pd.Series:
    """Parse to datetime and ensure tz-aware in the given local timezone."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize(tz)
    return dt.dt.tz_convert(tz)


def filter_by_date_local(df: pd.DataFrame, pu_col: str, do_col: str, date_str: str, tz: str) -> pd.DataFrame:
    """
    Keep only rows whose pickup time falls on the given LOCAL calendar day (tz).
    Returns a new DF where both pickup/dropoff are tz-aware in `tz`.
    """
    dt_pu_local = ensure_local_tz(df[pu_col], tz)
    dt_do_local = ensure_local_tz(df[do_col], tz)

    start = pd.Timestamp(date_str, tz=tz)
    end = start + pd.Timedelta(days=1)
    mask = (dt_pu_local >= start) & (dt_pu_local < end)

    out = df.loc[mask].copy()
    out[pu_col] = dt_pu_local[mask]
    out[do_col] = dt_do_local[mask]
    return out


def clean_and_sort(df: pd.DataFrame, pu_col: str, do_col: str,
                   pulo: str, pula: str, dolo: str, dola: str) -> pd.DataFrame:
    """Minimal cleaning and sort by pickup time (tz-aware)."""
    df = df.dropna(subset=[pu_col, do_col]).copy()

    # Basic time sanity: non-negative durations
    df = df[df[do_col] >= df[pu_col]]

    # NYC bounding box filter for obvious junk
    lon_ok = df[pulo].between(-75.5, -72.5) & df[dolo].between(-75.5, -72.5)
    lat_ok = df[pula].between(40.0, 41.2) & df[dola].between(40.0, 41.2)
    df = df[lon_ok & lat_ok]

    # Sort by pickup time
    df = df.sort_values(pu_col).reset_index(drop=True)
    return df


def take_first_n(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None:
        return df
    if n <= 0:
        raise ValueError("--n must be positive if provided")
    if n >= len(df):
        return df
    return df.iloc[:n].reset_index(drop=True)


def to_epoch_seconds_utc(dt_series: pd.Series) -> np.ndarray:
    """
    Convert a tz-aware datetime series to int64 UNIX seconds in UTC.
    If tz-naive sneaks in, treat it as UTC to avoid errors.
    """
    if dt_series.dt.tz is None:
        dt_utc = dt_series.dt.tz_localize("UTC")
    else:
        dt_utc = dt_series.dt.tz_convert("UTC")
    return (dt_utc.view("int64") // 10**9).astype(np.int64)


def build_tensors(df: pd.DataFrame, pu_col: str, do_col: str,
                  pulo: str, pula: str, dolo: str, dola: str,
                  device: torch.device):
    """
    Build xA, xB, tA, tB per current experiment:
      - tA == tB == pickup_time (int64 seconds, UTC)
      - xB from pickup coords, xA from dropoff coords
    """
    # Times: use pickup time for both tA and tB
    t_pick_np = to_epoch_seconds_utc(df[pu_col])  # [N]
    tA_np = t_pick_np.copy()
    tB_np = t_pick_np.copy()

    # Coordinates -> XY (meters)
    xB_x, xB_y = lonlat_to_xy(df[pulo].to_numpy(), df[pula].to_numpy())
    xA_x, xA_y = lonlat_to_xy(df[dolo].to_numpy(), df[dola].to_numpy())

    xB_np = np.stack([xB_x, xB_y], axis=1).astype(np.float32)  # [N,2]
    xA_np = np.stack([xA_x, xA_y], axis=1).astype(np.float32)  # [N,2]

    # Torch tensors on chosen device (hard-coded below)
    xA = torch.from_numpy(xA_np).to(device=device, dtype=torch.float32)
    xB = torch.from_numpy(xB_np).to(device=device, dtype=torch.float32)
    tA = torch.from_numpy(tA_np).to(device=device, dtype=torch.int64)
    tB = torch.from_numpy(tB_np).to(device=device, dtype=torch.int64)

    return xA, xB, tA, tB


def main():
    ap = argparse.ArgumentParser(description="Prepare NYC 2014 single-day tensors and run spef matching.")
    ap.add_argument("--input", required=True, help="Path to a month-wide Yellow Taxi 2014 file (Parquet or CSV).")
    ap.add_argument("--date", required=True, help="Single LOCAL date to keep, e.g., 2014-06-15.")
    ap.add_argument("--tz", default="America/New_York",
                    help="Timezone for --date and timestamp interpretation (default: America/New_York).")
    ap.add_argument("--n", type=int, default=None,
                    help="If provided, take only the first n requests after sorting on that day.")
    # Keep solver-related knobs you might want to tune:
    ap.add_argument("--tile_k", type=int, default=4096, help="Tile size parameter if your solver expects it.")
    ap.add_argument("--C", type=int, default=32, help="C parameter (as in your solver).")
    ap.add_argument("--delta", type=float, default=1.0, help="delta parameter (as in your solver).")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (if applicable).")

    args = ap.parse_args()

    # Hard-coded device (no CLI flag): prefer CUDA if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # Load and prep data
    df = load_day(args.input)
    pu_col, do_col, pulo, pula, dolo, dola = normalize_columns(df)

    # Filter to one local day first (makes day boundary correct), then minimal clean/sort
    df = filter_by_date_local(df, pu_col, do_col, args.date, tz=args.tz)
    df = clean_and_sort(df, pu_col, do_col, pulo, pula, dolo, dola)
    df = take_first_n(df, args.n)

    N = len(df)
    print(f"[info] entries on {args.date}: {N}")

    # Build bipartite tensors on GPU
    xA, xB, tA, tB = build_tensors(df, pu_col, do_col, pulo, pula, dolo, dola, device)

    # Call your NYC-masked solver (same-folder import)
    if not hasattr(solver, "spef_matching_2"):
        raise AttributeError("spef_matching_nyc.py does not expose spef_matching_2. Please add it or adjust the call below.")

    out = solver.spef_matching_2(
        xA=xA,
        xB=xB,
        C=args.C,
        k=args.tile_k,
        delta=args.delta,
        device=device,
        seed=args.seed,
        tA=tA,
        tB=tB,
    )

    print("[done] spef_matching_2 finished.")
    # Minimal summary without pulling big tensors to CPU
    if isinstance(out, dict):
        summary = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__) for k, v in out.items()}
        print("[summary]", summary)
    else:
        print("[summary] return type:", type(out).__name__)


if __name__ == "__main__":
    main()
