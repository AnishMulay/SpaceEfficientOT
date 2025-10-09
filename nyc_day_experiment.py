#!/usr/bin/env python3
"""
nyc_day_experiment.py  —  NYC Taxi: zone-centroid → Euclidean bipartite builder

- Reads a month-wide 2014 file (Parquet/CSV).
- Filters to ONE local calendar day (e.g., 2014-01-10) in America/New_York (configurable).
- Loads Taxi Zone shapefile and maps LocationID -> centroid(lon,lat).
- Builds bipartite tensors on GPU:
    xA:[N,2]  dropoff XY (A/left)
    xB:[N,2]  pickup  XY (B/right)
    tA:[N], tB:[N]  int64 seconds; here tA == tB == pickup_time (per current experiment)
- Computes C automatically (if not provided) like experiments.py:
    pick 1 random B, compute max Euclidean distance to all A, then square.
- Calls spef_matching_2(...) (imported directly) and prints an experiment-style summary
  (wall time, cost, iterations, key timing metrics) WITHOUT saving to disk.
"""

import os
import argparse
from typing import Tuple, Optional
import time

import numpy as np
import pandas as pd
import torch
import geopandas as gpd

# Import the matching entrypoint directly (same directory as this script)
from spef_matching_nyc import spef_matching_2


# ---------------------------
#  Projection: lon/lat -> local XY (meters) around NYC
# ---------------------------
def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray,
                 lon0: float = -74.0, lat0: float = 40.7) -> Tuple[np.ndarray, np.ndarray]:
    """Fast local equirectangular projection (meters)."""
    R = 6371000.0
    lonr = np.deg2rad(lon.astype(np.float64))
    latr = np.deg2rad(lat.astype(np.float64))
    lon0r = np.deg2rad(lon0); lat0r = np.deg2rad(lat0)
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
    """Resolve timestamp + location-id columns with case-insensitive matching."""
    cmap = {c.lower(): c for c in df.columns}

    def pick(*candidates) -> Optional[str]:
        for name in candidates:
            if name in cmap:
                return cmap[name]
        return None

    # Datetimes (2014: pickup_datetime or tpep_*)
    pu_col = pick("pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime")
    do_col = pick("dropoff_datetime", "tpep_dropoff_datetime", "lpep_dropoff_datetime")
    if not pu_col or not do_col:
        raise ValueError(f"Missing pickup/dropoff timestamps; columns present: {list(df.columns)}")

    # Location IDs (zones)
    pu_id = pick("pulocationid", "pickup_location_id")
    do_id = pick("dolocationid", "dropoff_location_id")
    if not pu_id or not do_id:
        raise ValueError(
            "This file does not have zone IDs (PULocationID/DOLocationID or pickup_location_id/dropoff_location_id). "
            f"Columns present: {list(df.columns)}"
        )

    return pu_col, do_col, pu_id, do_id


def ensure_local_tz(series: pd.Series, tz: str) -> pd.Series:
    """Parse to datetime and ensure tz-aware in the given local timezone."""
    dt = pd.to_datetime(series, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize(tz)
    return dt.dt.tz_convert(tz)


def filter_by_date_local(df: pd.DataFrame, pu_col: str, do_col: str, date_str: str, tz: str) -> pd.DataFrame:
    """Keep rows whose pickup time falls on the given LOCAL day (tz)."""
    dt_pu_local = ensure_local_tz(df[pu_col], tz)
    dt_do_local = ensure_local_tz(df[do_col], tz)
    start = pd.Timestamp(date_str, tz=tz)
    end = start + pd.Timedelta(days=1)
    mask = (dt_pu_local >= start) & (dt_pu_local < end)
    out = df.loc[mask].copy()
    out[pu_col] = dt_pu_local[mask]
    out[do_col] = dt_do_local[mask]
    return out


def clean_and_sort(df: pd.DataFrame, pu_col: str, do_col: str) -> pd.DataFrame:
    """Minimal cleaning and sort by pickup time (tz-aware)."""
    df = df.dropna(subset=[pu_col, do_col]).copy()
    df = df[df[do_col] >= df[pu_col]]
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
    Convert tz-aware datetime series to int64 UNIX seconds in UTC and
    return a NumPy array (not a pandas Series).
    """
    if dt_series.dt.tz is None:
        dt_utc = dt_series.dt.tz_localize("UTC")
    else:
        dt_utc = dt_series.dt.tz_convert("UTC")
    return ((dt_utc.astype("int64") // 10**9).to_numpy(dtype=np.int64))


# ---------------------------
#  Zones → centroids
# ---------------------------
def load_zone_centroids(zones_path: str) -> dict[int, tuple[float, float]]:
    """
    Read Taxi Zones shapefile, compute centroids in a projected CRS, then
    transform to EPSG:4326 to get (lon, lat). Return LocationID -> (lon, lat).
    """
    gdf = gpd.read_file(zones_path)
    print(f"[zones] loaded: {zones_path}")
    print(f"[zones] crs: {gdf.crs}")
    print(f"[zones] columns: {list(gdf.columns)}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("[zones] head:"); print(gdf.head(3))

    # Resolve LocationID column (case-insensitive)
    cmap = {c.lower(): c for c in gdf.columns}
    loc_col = cmap.get("locationid") or cmap.get("location_id")
    if not loc_col:
        raise ValueError("Zones file missing 'LocationID' column.")

    # If CRS is missing, assume 4326; otherwise use the provided CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Compute centroids in a projected CRS (good for geometry ops)
    if str(gdf.crs).upper().endswith("4326"):
        # Source is geographic -> project to NYSP 2263 for centroid, then back to 4326
        gdf_proj = gdf.to_crs("EPSG:2263")
        cents_local = gdf_proj.geometry.centroid
        cents_ll = gpd.GeoSeries(cents_local, crs="EPSG:2263").to_crs("EPSG:4326")
    else:
        # Source is already projected (e.g., EPSG:2263)
        cents_local = gdf.geometry.centroid
        cents_ll = gpd.GeoSeries(cents_local, crs=gdf.crs).to_crs("EPSG:4326")

    cent_map = {int(loc): (pt.x, pt.y) for loc, pt in zip(gdf[loc_col].astype(int), cents_ll)}
    return cent_map


def ids_to_lonlat(df: pd.DataFrame, pu_id_col: str, do_id_col: str, cent_map: dict[int, tuple[float, float]]):
    """Map PULocationID/DOLocationID to (lon,lat) via centroid dict. Drops rows with unknown IDs."""
    pu_ids = df[pu_id_col].astype("Int64")  # support NA
    do_ids = df[do_id_col].astype("Int64")

    pu_ll = pu_ids.map(lambda z: cent_map.get(int(z)) if pd.notna(z) else None)
    do_ll = do_ids.map(lambda z: cent_map.get(int(z)) if pd.notna(z) else None)

    mask = pu_ll.notna() & do_ll.notna()
    dropped = (~mask).sum()
    if dropped:
        print(f"[zones] dropping {int(dropped)} rows with unknown zone IDs")
    df2 = df.loc[mask].reset_index(drop=True)

    pu_arr = np.array(list(pu_ll[mask]), dtype="object")
    do_arr = np.array(list(do_ll[mask]), dtype="object")

    # Split tuples into float arrays
    pu_lon = np.array([t[0] for t in pu_arr], dtype="float32")
    pu_lat = np.array([t[1] for t in pu_arr], dtype="float32")
    do_lon = np.array([t[0] for t in do_arr], dtype="float32")
    do_lat = np.array([t[1] for t in do_arr], dtype="float32")

    return df2, (pu_lon, pu_lat), (do_lon, do_lat)


# ---------------------------
#  Build tensors
# ---------------------------
def build_tensors_from_centroids(df: pd.DataFrame,
                                 pu_col: str, do_col: str,
                                 pu_id_col: str, do_id_col: str,
                                 centroids_path: str,
                                 device: torch.device):
    """Use zone centroids to create xA, xB, tA, tB."""
    cent_map = load_zone_centroids(centroids_path)
    df2, (pu_lon, pu_lat), (do_lon, do_lat) = ids_to_lonlat(df, pu_id_col, do_id_col, cent_map)

    # Times: per current experiment use pickup time for both tA and tB
    t_pick_np = to_epoch_seconds_utc(df2[pu_col])  # [N]
    tA_np = t_pick_np.copy()
    tB_np = t_pick_np.copy()

    # Project to XY (meters)
    xB_x, xB_y = lonlat_to_xy(pu_lon, pu_lat)
    xA_x, xA_y = lonlat_to_xy(do_lon, do_lat)

    xB_np = np.stack([xB_x, xB_y], axis=1).astype(np.float32)
    xA_np = np.stack([xA_x, xA_y], axis=1).astype(np.float32)

    # Torch tensors on device
    xA = torch.from_numpy(xA_np).to(device=device, dtype=torch.float32)
    xB = torch.from_numpy(xB_np).to(device=device, dtype=torch.float32)
    tA = torch.from_numpy(tA_np).to(device=device, dtype=torch.int64)
    tB = torch.from_numpy(tB_np).to(device=device, dtype=torch.int64)

    return df2, xA, xB, tA, tB


# ---------------------------
#  Auto-compute C (like experiments.py)
# ---------------------------
def compute_C_auto(xA: torch.Tensor, xB: torch.Tensor, device: torch.device, seed: int) -> torch.Tensor:
    """
    Pick 1 random B row, compute max Euclidean distance to all A, then square it.
    Returns a scalar torch tensor on `device`.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    nB = xB.shape[0]
    rb = torch.randint(0, nB, (1,), device=device, generator=gen)
    bpt = xB.index_select(0, rb)           # [1, d]
    dists = torch.cdist(bpt, xA)           # [1, nA]
    C = dists.max() ** 2                   # scalar tensor
    return C


# ---------------------------
#  Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="NYC 2014 single-day tensors via zone centroids; run spef matching.")
    ap.add_argument("--input", required=True, help="Path to month-wide Yellow Taxi file (Parquet/CSV).")
    ap.add_argument("--zones", required=True, help="Path to Taxi Zones shapefile (e.g., ./nyc_data/taxi_zones.shp).")
    ap.add_argument("--date", required=True, help="Single LOCAL date to keep, e.g., 2014-01-10.")
    ap.add_argument("--tz", default="America/New_York", help="Timezone for --date (default: America/New_York).")
    ap.add_argument("--n", type=int, default=None, help="If provided, take only the first n requests after sorting.")
    ap.add_argument("--tile_k", type=int, default=4096, help="Tile size (if your solver expects it).")
    ap.add_argument("--C", type=float, default=None, help="Scaling factor. If omitted, compute from data on GPU.")
    ap.add_argument("--delta", type=float, default=1.0, help="delta parameter (as in your solver).")
    ap.add_argument("--cmax", type=int, default=None, help="Cap INTEGERIZED edge cost; costs >= cmax become cmax (default: no cap).")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (used by solver and C auto).")

    args = ap.parse_args()

    # Hard-coded device: prefer CUDA when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # Load trips and normalize columns
    df = load_day(args.input)
    print(f"[debug] loaded trips columns ({len(df.columns)}): {list(df.columns)}")
    pu_col, do_col, pu_id, do_id = normalize_columns(df)

    # Filter to one local day, clean, sort, cap to first n
    df = filter_by_date_local(df, pu_col, do_col, args.date, tz=args.tz)
    df = clean_and_sort(df, pu_col, do_col)
    df = take_first_n(df, args.n)

    N = len(df)
    print(f"[info] entries on {args.date}: {N}")
    if N == 0:
        print("[warn] no rows for that date after filtering; check --date/--tz and input file.")
        return

    # Build tensors from zone centroids
    df2, xA, xB, tA, tB = build_tensors_from_centroids(
        df, pu_col, do_col, pu_id, do_id, args.zones, device
    )
    N2 = len(df2)
    print(f"[info] entries after dropping unknown zone IDs: {N2}")

    # Sanity preview: first 5 bipartite rows
    K = min(5, N2)
    print("[sanity] head(df2) key columns:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df2[[pu_col, do_col, pu_id, do_id]].head(K))

    xA5 = xA[:K].detach().cpu().numpy()
    xB5 = xB[:K].detach().cpu().numpy()
    t5  = tA[:K].detach().cpu().numpy()
    print(f"\n[preview] first {K} bipartite rows (i, t_sec_utc, B=(pickup_x,pickup_y), A=(dropoff_x,dropoff_y))")
    for i in range(K):
        print(f"  {i:4d}  t={int(t5[i])}  B=({xB5[i,0]:.1f},{xB5[i,1]:.1f})  A=({xA5[i,0]:.1f},{xA5[i,1]:.1f})")

    # ---- C: manual OR auto (experiments.py method) ----
    if args.C is not None:
        C_tensor = torch.as_tensor(args.C, device=device, dtype=torch.float32)
        print(f"[info] C (manual): {float(C_tensor):.6g}")
    else:
        C_tensor = compute_C_auto(xA, xB, device, args.seed)
        print(f"[info] C (auto, max-dist^2 from one random B): {float(C_tensor):.6g}")

    # ----- Run solver and print experiment-style summary (no JSON save) -----
    print("\n[run] spef_matching_2 ...")
    t0 = time.perf_counter()
    out = spef_matching_2(
        xA=xA, xB=xB,
        C=C_tensor, k=args.tile_k, delta=args.delta,
        device=device, seed=args.seed,
        tA=tA, tB=tB,
        cmax_int=args.cmax,
    )
    t1 = time.perf_counter()
    wall = t1 - t0
    print("[done] spef_matching_2 finished.")

    # --- Robust summary printing (tuple/dict/other)
    def _preview_vec(name, t, k=10):
        if isinstance(t, torch.Tensor):
            td = t.detach().cpu()
            print(f"  {name}: Tensor(shape={tuple(td.shape)}, dtype={td.dtype}, device={t.device})")
            n = min(k, td.shape[0])
            print(f"    head[{n}]:", td[:n].tolist())
        else:
            print(f"  {name}: {type(t).__name__}")

    if isinstance(out, tuple) and len(out) == 6:
        Mb, yA, yB, matching_cost, iteration, metrics = out
        print("\n[summary]")
        print(f"  N={N2}, k(tile)={args.tile_k}, delta={args.delta}, seed={args.seed}")
        if args.cmax is not None:
            print(f"  cmax={args.cmax}")
        print(f"  wall_time_s={wall:.3f}")
        try:
            print(f"  matching_cost={float(matching_cost):.6g}")
        except Exception:
            print(f"  matching_cost={matching_cost}")
        print(f"  iterations={iteration}")

        # Key metrics if present
        if isinstance(metrics, dict):
            for k in ("slack_compute_total", "slack_compute_avg",
                      "tile_updates_total", "tile_updates_avg",
                      "final_fill_time", "cost_calc_time", "inner_loops_count"):
                if k in metrics:
                    print(f"  {k}={metrics[k]}")

        # Tiny previews of vectors
        _preview_vec("Mb (match for B→A)", Mb, k=10)
        _preview_vec("yA", yA, k=10)
        _preview_vec("yB", yB, k=10)

    elif isinstance(out, dict):
        print("\n[summary] dict keys:", list(out.keys()))
        if "matching_cost" in out:
            try:
                print(f"  matching_cost={float(out['matching_cost']):.6g}")
            except Exception:
                print(f"  matching_cost={out['matching_cost']}")
        print(f"  wall_time_s={wall:.3f}")
        if args.cmax is not None:
            print(f"  cmax={args.cmax}")
        for k in out:
            if isinstance(out[k], torch.Tensor):
                _preview_vec(k, out[k], k=10)
    else:
        print("\n[summary] return type:", type(out).__name__)
        print("  wall_time_s=", f"{wall:.3f}")
        if args.cmax is not None:
            print(f"  cmax={args.cmax}")
        print("  value:", out)


if __name__ == "__main__":
    main()
