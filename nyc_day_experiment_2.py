#!/usr/bin/env python3
"""NYC Taxi one-day experiment using lat/lon Haversine costs."""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from spef_matching_nyc_2 import spef_matching_2


def load_one_day_latlon(
    path,                     # Parquet or CSV with exact coords (2014)
    day="2014-01-10",         # local calendar day (America/New_York)
    tz="America/New_York",
    pickup_cols=("pickup_datetime","pickup_longitude","pickup_latitude"),
    dropoff_cols=("dropoff_datetime","dropoff_longitude","dropoff_latitude"),
    device="cuda",
    sample_n: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 1,
):
    """
    Returns:
      xA: [N,2] dropoff (lon,lat)   (A/left)
      xB: [N,2] pickup  (lon,lat)   (B/right)
      tA: [N]   int64 seconds-since-epoch (dropoff time)
      tB: [N]   int64 seconds-since-epoch (pickup  time)
    Sorted so that tA and tB are nondecreasing.
    """

    # Read file
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Handle column name variants from 2014 releases
    # Common alternatives: tpep_pickup_datetime/tpep_dropoff_datetime (Green/Yellow later years),
    # trip_pickup_datetime/trip_dropoff_datetime (older), etc.
    # Prefer given names; otherwise fall back to common variants.
    def choose(c, alts):
        if c in df.columns: return c
        for a in alts:
            if a in df.columns: return a
        raise KeyError(f"None of {c, *alts} present in columns: {df.columns.tolist()}")

    p_time = choose(pickup_cols[0], ["tpep_pickup_datetime","trip_pickup_datetime"])
    p_lon  = choose(pickup_cols[1], ["pickup_longitude","start_lon","pickup_long"])
    p_lat  = choose(pickup_cols[2], ["pickup_latitude","start_lat","pickup_lat"])

    d_time = choose(dropoff_cols[0], ["tpep_dropoff_datetime","trip_dropoff_datetime"])
    d_lon  = choose(dropoff_cols[1], ["dropoff_longitude","end_lon","dropoff_long"])
    d_lat  = choose(dropoff_cols[2], ["dropoff_latitude","end_lat","dropoff_lat"])

    # Parse times in local tz, filter the single day
    # Ensure timestamps are TZ-aware in tz, then convert to integer seconds.
    df[p_time] = pd.to_datetime(df[p_time], errors="coerce").dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")
    df[d_time] = pd.to_datetime(df[d_time], errors="coerce").dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")

    # Some rows might already be tz-aware; ensure tz conversion
    df[p_time] = df[p_time].dt.tz_convert(tz)
    df[d_time] = df[d_time].dt.tz_convert(tz)

    day_start = pd.Timestamp(day, tz=tz)
    day_end   = day_start + pd.Timedelta(days=1)
    mask_day  = (df[p_time] >= day_start) & (df[p_time] < day_end) & df[p_time].notna() & df[d_time].notna()

    cols = [p_time, p_lon, p_lat, d_time, d_lon, d_lat]
    df2 = df.loc[mask_day, cols].dropna().copy()
    df2.sort_values(by=p_time, inplace=True, kind="mergesort")

    if sample_n is not None:
        if sample_n <= 0:
            raise ValueError("--n must be a positive integer")
    if sample_n is not None and sample_n > 0 and len(df2) > sample_n:
        if random_sample:
            df2 = df2.sample(n=sample_n, random_state=seed).sort_values(by=p_time, kind="mergesort")
        else:
            df2 = df2.head(sample_n)

    # Build tensors: note we keep (lon,lat) ordering consistently
    xB = torch.tensor(df2[[p_lon, p_lat]].to_numpy(np.float32), device=device)  # pickups (B/right)
    xA = torch.tensor(df2[[d_lon, d_lat]].to_numpy(np.float32), device=device)  # dropoffs (A/left)

    # Integer seconds (Unix-like)
    tB = torch.tensor((df2[p_time].view(np.int64) // 10**9).to_numpy(), dtype=torch.int64, device=device)
    tA = torch.tensor((df2[d_time].view(np.int64) // 10**9).to_numpy(), dtype=torch.int64, device=device)

    # Sort A and B independently by time (as you do now)
    # This ensures the solver can rely on monotonic tA and tB.
    A_idx = torch.argsort(tA)
    B_idx = torch.argsort(tB)

    xA, tA = xA[A_idx], tA[A_idx]
    xB, tB = xB[B_idx], tB[B_idx]

    return xA, xB, tA, tB, len(df2)


EARTH_RADIUS_METERS = 6_371_000.0
_SOLVER_SIGNATURE = inspect.signature(spef_matching_2)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _sanitize_metrics_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_metrics_dict(value)
        elif isinstance(value, (int, float, str, bool)):
            sanitized[key] = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            sanitized[key] = value.item()
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
    return sanitized


def haversine_pairwise_blockwise(
    xB: torch.Tensor,
    xA: torch.Tensor,
    block_rows: int,
    block_cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute pairwise Haversine distances in meters, block by block on the GPU."""
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_rows and block_cols must be positive integers")

    xA = xA.to(device=device, dtype=dtype)
    xB = xB.to(device=device, dtype=dtype)

    m = xB.shape[0]
    n = xA.shape[0]
    out = torch.empty((m, n), device=device, dtype=dtype)
    if m == 0 or n == 0:
        return out

    xA_rad = torch.deg2rad(xA)
    xB_rad = torch.deg2rad(xB)

    latA = xA_rad[:, 1]
    lonA = xA_rad[:, 0]
    latB = xB_rad[:, 1]
    lonB = xB_rad[:, 0]

    cos_latA = torch.cos(latA)
    cos_latB = torch.cos(latB)

    radius = torch.tensor(EARTH_RADIUS_METERS, device=device, dtype=dtype)

    for row_start in range(0, m, block_rows):
        row_end = min(row_start + block_rows, m)
        lat_b = latB[row_start:row_end].unsqueeze(1)
        lon_b = lonB[row_start:row_end].unsqueeze(1)
        cos_lat_b = cos_latB[row_start:row_end].unsqueeze(1)

        for col_start in range(0, n, block_cols):
            col_end = min(col_start + block_cols, n)
            lat_a = latA[col_start:col_end].unsqueeze(0)
            lon_a = lonA[col_start:col_end].unsqueeze(0)
            cos_lat_a = cos_latA[col_start:col_end].unsqueeze(0)

            dlat = lat_a - lat_b
            dlon = lon_a - lon_b
            sin_dlat = torch.sin(dlat * 0.5)
            sin_dlon = torch.sin(dlon * 0.5)
            a = sin_dlat.square() + cos_lat_b * cos_lat_a * sin_dlon.square()
            a = torch.clamp(a, min=0.0, max=1.0)
            sqrt_a = torch.sqrt(a)
            sqrt_one_minus_a = torch.sqrt(torch.clamp(1.0 - a, min=0.0, max=1.0))
            c = 2.0 * torch.atan2(sqrt_a, sqrt_one_minus_a)
            out[row_start:row_end, col_start:col_end] = radius * c

    return out


def _extract_solver_metrics(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) >= 5:
        matching_cost = result[-2]
        iterations = result[-1]
        return {
            "matching_cost": matching_cost,
            "iterations": iterations,
        }
    raise TypeError(f"Unsupported solver return type: {type(result).__name__}")


def run_one_day(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    if "cost_matrix" not in _SOLVER_SIGNATURE.parameters:
        raise RuntimeError("spef_matching_2 does not accept cost_matrix=...; please update the solver.")

    if args.block_rows <= 0 or args.block_cols <= 0:
        raise ValueError("--block_rows and --block_cols must be positive integers")

    t_start = time.perf_counter()

    try:
        load_start = time.perf_counter()
        xA, xB, tA, tB, n_rows = load_one_day_latlon(
            args.input,
            day=args.day,
            tz=args.tz,
            device=device,
            sample_n=args.n,
            random_sample=args.random_sample,
            seed=args.seed,
        )
        _sync_if_cuda(device)
        load_end = time.perf_counter()
    except KeyError as exc:
        raise RuntimeError(f"Missing required column while loading data: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load day slice: {exc}") from exc

    dist_start = time.perf_counter()
    D = haversine_pairwise_blockwise(
        xB,
        xA,
        args.block_rows,
        args.block_cols,
        device=device,
        dtype=torch.float32,
    )
    _sync_if_cuda(device)
    dist_end = time.perf_counter()

    candidate_kwargs = (
        ("xA", xA),
        ("xB", xB),
        ("tA", tA),
        ("tB", tB),
        ("cost_matrix", D),
        ("Cmax", args.cmax_meters),
        ("scale", args.scale),
        ("use_int_costs", args.int_costs),
        ("device", device),
        ("seed", args.seed),
    )

    solver_kwargs: Dict[str, Any] = {}
    for key, value in candidate_kwargs:
        if key not in _SOLVER_SIGNATURE.parameters:
            continue
        if key == "Cmax" and value is None:
            continue
        solver_kwargs[key] = value

    _sync_if_cuda(device)
    solve_start = time.perf_counter()
    solver_result = spef_matching_2(**solver_kwargs)
    _sync_if_cuda(device)
    solve_end = time.perf_counter()

    metrics_raw = _extract_solver_metrics(solver_result)
    if "matching_cost" not in metrics_raw or ("iterations" not in metrics_raw and "iteration" not in metrics_raw):
        raise RuntimeError("Solver result missing matching_cost or iterations field.")

    matching_cost = _to_float(metrics_raw["matching_cost"])
    iterations_value = metrics_raw.get("iterations", metrics_raw.get("iteration"))
    iterations = _to_int(iterations_value)

    extra_metrics = {
        k: v
        for k, v in metrics_raw.items()
        if k not in {"matching_cost", "iterations", "iteration"}
    }
    sanitized_extra = _sanitize_metrics_dict(extra_metrics) if extra_metrics else {}

    _sync_if_cuda(device)
    total_end = time.perf_counter()

    metrics: Dict[str, Any] = {
        "input": str(args.input),
        "day": args.day,
        "tz": args.tz,
        "device": str(device),
        "n": int(n_rows),
        "load_seconds": load_end - load_start,
        "distance_seconds": dist_end - dist_start,
        "solve_seconds": solve_end - solve_start,
        "total_seconds": total_end - t_start,
        "cmax_meters": float(args.cmax_meters) if args.cmax_meters is not None else None,
        "scale": float(args.scale),
        "int_costs": bool(args.int_costs),
        "block_rows": int(args.block_rows),
        "block_cols": int(args.block_cols),
        "matching_cost": matching_cost,
        "iterations": iterations,
    }

    if sanitized_extra:
        metrics["solver_extra"] = sanitized_extra

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-day NYC taxi matching experiment with Haversine costs."
    )
    parser.add_argument("--input", required=True, help="Path to NYC taxi parquet or CSV file with exact coordinates.")
    parser.add_argument("--day", default="2014-01-10", help="Local day to run (default: 2014-01-10).")
    parser.add_argument("--tz", default="America/New_York", help="Timezone of timestamps (default: America/New_York).")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    parser.add_argument("--cmax_meters", type=float, default=None, help="Clamp costs to at most this many meters (after masking).")
    parser.add_argument("--scale", type=float, default=1.0, help="Cost scaling factor before integer cast.")
    parser.add_argument("--int_costs", action="store_true", help="Use integer-cost path in solver.")
    parser.add_argument("--block_rows", type=int, default=4096, help="Tile rows for pairwise Haversine.")
    parser.add_argument("--block_cols", type=int, default=4096, help="Tile cols for pairwise Haversine.")
    parser.add_argument("--n", type=int, default=None, help="If provided, limit to this many requests after filtering.")
    parser.add_argument("--random_sample", action="store_true", help="If set with --n, draw a random sample of requests before sorting.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling and solver.")
    parser.add_argument("--output_json", default=None, help="Optional path to write run metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        device = torch.device(args.device)
    except Exception as exc:
        print(f"[error] Invalid device '{args.device}': {exc}", file=sys.stderr)
        sys.exit(1)

    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[error] --device={args.device} requested CUDA but no CUDA device is available.", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = run_one_day(args, device)
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[records] n={metrics['n']}")
    print(
        "[timing] load={load:.3f}s distance={distance:.3f}s solve={solve:.3f}s total={total:.3f}s".format(
            load=metrics["load_seconds"],
            distance=metrics["distance_seconds"],
            solve=metrics["solve_seconds"],
            total=metrics["total_seconds"],
        )
    )
    print(
        "[config] day={day} tz={tz} device={device} cmax={cmax} scale={scale} int_costs={int_costs}".format(
            day=metrics["day"],
            tz=metrics["tz"],
            device=metrics["device"],
            cmax=metrics["cmax_meters"],
            scale=metrics["scale"],
            int_costs=metrics["int_costs"],
        )
    )
    print(f"[distance] block_rows={metrics['block_rows']} block_cols={metrics['block_cols']}")
    print(f"[result] matching_cost={metrics['matching_cost']:.6g} iterations={metrics['iterations']}")

    solver_extra = metrics.get("solver_extra")
    if isinstance(solver_extra, dict) and solver_extra:
        print("[solver] extra metrics:")
        for key, value in solver_extra.items():
            print(f"  {key}={value}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"[json] wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
