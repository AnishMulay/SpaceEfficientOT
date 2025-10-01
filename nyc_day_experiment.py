#!/usr/bin/env python3
"""
nyc_day_experiment.py

Load one day of NYC Yellow Taxi 2014 data, optionally take the first N requests,
project lon/lat to a local Euclidean plane, build tensors:
  xA:[N,2]  dropoff XY  (A, left set)
  xB:[N,2]  pickup  XY  (B, right set)
  tA:[N]    int64 seconds
  tB:[N]    int64 seconds

Per user's current modeling choice here:
 - We sort by pickup time.
 - We set tA == tB == pickup_time (so the forward-in-time mask becomes j >= i).
 - Each request is split into one B node (pickup) and one A node (dropoff), sharing the same time.

Finally, we call the solver's spef_matching_2(...) entrypoint and pass (xA, xB, tA, tB).

Usage:
  python nyc_day_experiment.py --input ./data/yellow_tripdata_2014-01-15.csv --n 100000 --device cuda --solver_path /mnt/data/spef_matching_nyc.py
"""

import os
import argparse
import importlib.util
from typing import Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------
#  Projection: lon/lat -> local XY (meters) around NYC
# ---------------------------
def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, lon0: float = -74.0, lat0: float = 40.7) -> Tuple[np.ndarray, np.ndarray]:
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
    # Datetime columns: prefer 'pickup_datetime' / 'dropoff_datetime' (2014 style),
    # fall back to 'tpep_*' if present.
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


def clean_and_sort(df: pd.DataFrame, pu_col: str, do_col: str, pulo: str, pula: str, dolo: str, dola: str) -> pd.DataFrame:
    """Minimal cleaning and sort by pickup time."""
    df = df.copy()
    df[pu_col] = pd.to_datetime(df[pu_col], utc=True, errors="coerce")
    df[do_col] = pd.to_datetime(df[do_col], utc=True, errors="coerce")
    df = df.dropna(subset=[pu_col, do_col])

    # Basic time sanity: non-negative duration
    df = df[df[do_col] >= df[pu_col]]

    # NYC bounding box filter
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
    # Ensure timezone-aware UTC, then convert to int64 seconds
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize("UTC")
    else:
        dt_series = dt_series.dt.tz_convert("UTC")
    return (dt_series.view("int64") // 10**9).astype(np.int64)


def build_tensors(df: pd.DataFrame, pu_col: str, do_col: str, pulo: str, pula: str, dolo: str, dola: str, device: torch.device):
    """Build xA, xB, tA, tB according to the user's experiment spec.
    - Sort by pickup time already done upstream.
    - Set tA == tB == pickup_time (int64 seconds).
    - xB uses pickup coords; xA uses dropoff coords.
    """
    # Times: use pickup time for both tA and tB
    t_pick_np = to_epoch_seconds_utc(df[pu_col])  # [N]
    tA_np = t_pick_np.copy()
    tB_np = t_pick_np.copy()

    # Coordinates -> XY
    xB_x, xB_y = lonlat_to_xy(df[pulo].to_numpy(), df[pula].to_numpy())
    xA_x, xA_y = lonlat_to_xy(df[dolo].to_numpy(), df[dola].to_numpy())

    xB_np = np.stack([xB_x, xB_y], axis=1).astype(np.float32)  # [N,2]
    xA_np = np.stack([xA_x, xA_y], axis=1).astype(np.float32)  # [N,2]

    # Torch tensors on the chosen device
    xA = torch.from_numpy(xA_np).to(device=device, dtype=torch.float32)
    xB = torch.from_numpy(xB_np).to(device=device, dtype=torch.float32)
    tA = torch.from_numpy(tA_np).to(device=device, dtype=torch.int64)
    tB = torch.from_numpy(tB_np).to(device=device, dtype=torch.int64)

    return xA, xB, tA, tB


def import_solver(path: str):
    spec = importlib.util.spec_from_file_location("spef_solver_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load solver module at: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Prepare NYC 2014 day tensors and run spef matching.")
    ap.add_argument("--input", required=True, help="Relative path to one day's Yellow Taxi 2014 file (CSV or Parquet).")
    ap.add_argument("--n", type=int, default=None, help="If provided, take only the first n requests after sorting by pickup time.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available).")
    ap.add_argument("--solver_path", default="/mnt/data/spef_matching_nyc.py", help="Path to solver Python file (default: NYC-masked solver).")
    ap.add_argument("--tile_k", type=int, default=4096, help="Tile size parameter if your solver expects it.")
    ap.add_argument("--C", type=int, default=32, help="C parameter as in your solver (if applicable).")
    ap.add_argument("--delta", type=float, default=1.0, help="delta parameter as in your solver (if applicable).")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (if applicable).")

    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"[info] device: {device}")

    # Load and prep data
    df = load_day(args.input)
    pu_col, do_col, pulo, pula, dolo, dola = normalize_columns(df)
    df = clean_and_sort(df, pu_col, do_col, pulo, pula, dolo, dola)
    df = take_first_n(df, args.n)
    N = len(df)
    print(f"[info] requests: {N}")

    # Build tensors
    xA, xB, tA, tB = build_tensors(df, pu_col, do_col, pulo, pula, dolo, dola, device)

    # Import solver and call entrypoint
    solver = import_solver(args.solver_path)

    if not hasattr(solver, "spef_matching_2"):
        raise AttributeError("Solver module does not expose spef_matching_2. Please adjust the call below to your entrypoint.")

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
    if isinstance(out, dict):
        summary = {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__) for k, v in out.items()}
        print("[summary]", summary)
    else:
        print("[summary] return type:", type(out).__name__)


if __name__ == "__main__":
    main()
