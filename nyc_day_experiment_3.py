#!/usr/bin/env python3
"""
Simple NYC taxi experiment with lat/lon coordinates.
Load data, sample n requests, sort by pickup time, prepare for solver.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import time


def load_and_filter_data(file_path, date_str, n=None, random_sample=False, seed=1):
    """
    Load NYC taxi data with lat/lon coordinates, filter to one day, 
    sample n requests, and sort by pickup time.
    
    Returns: DataFrame with pickup/dropoff coordinates and times
    """
    # Load data
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} total records")
    print(f"Columns: {list(df.columns)}")
    
    # Find coordinate columns (case insensitive)
    cols = {c.lower(): c for c in df.columns}
    
    # Common column name patterns
    pickup_time_col = None
    pickup_lat_col = None  
    pickup_lon_col = None
    dropoff_time_col = None
    dropoff_lat_col = None
    dropoff_lon_col = None
    
    # Find pickup time
    for pattern in ['pickup_datetime', 'tpep_pickup_datetime', 'lpep_pickup_datetime']:
        if pattern.lower() in cols:
            pickup_time_col = cols[pattern.lower()]
            break
    
    # Find dropoff time  
    for pattern in ['dropoff_datetime', 'tpep_dropoff_datetime', 'lpep_dropoff_datetime']:
        if pattern.lower() in cols:
            dropoff_time_col = cols[pattern.lower()]
            break
            
    # Find pickup coordinates
    for pattern in ['pickup_latitude', 'pickup_lat']:
        if pattern.lower() in cols:
            pickup_lat_col = cols[pattern.lower()]
            break
    for pattern in ['pickup_longitude', 'pickup_lon', 'pickup_long']:
        if pattern.lower() in cols:
            pickup_lon_col = cols[pattern.lower()]
            break
            
    # Find dropoff coordinates
    for pattern in ['dropoff_latitude', 'dropoff_lat']:
        if pattern.lower() in cols:
            dropoff_lat_col = cols[pattern.lower()]
            break
    for pattern in ['dropoff_longitude', 'dropoff_lon', 'dropoff_long']:
        if pattern.lower() in cols:
            dropoff_lon_col = cols[pattern.lower()]
            break
    
    # Check we found all required columns
    required_cols = [pickup_time_col, pickup_lat_col, pickup_lon_col, 
                    dropoff_time_col, dropoff_lat_col, dropoff_lon_col]
    if any(col is None for col in required_cols):
        raise ValueError(f"Missing required columns. Found: {[col for col in required_cols if col is not None]}")
    
    print(f"Using columns: pickup_time={pickup_time_col}, pickup_lat={pickup_lat_col}, pickup_lon={pickup_lon_col}")
    print(f"               dropoff_time={dropoff_time_col}, dropoff_lat={dropoff_lat_col}, dropoff_lon={dropoff_lon_col}")
    
    # Convert times to datetime
    df[pickup_time_col] = pd.to_datetime(df[pickup_time_col])
    df[dropoff_time_col] = pd.to_datetime(df[dropoff_time_col])
    
    # Filter to specific date (assuming UTC for simplicity)
    date_start = pd.Timestamp(date_str)
    date_end = date_start + pd.Timedelta(days=1)
    
    mask = (df[pickup_time_col] >= date_start) & (df[pickup_time_col] < date_end)
    df_day = df[mask].copy()
    print(f"Records for {date_str}: {len(df_day)}")
    
    if len(df_day) == 0:
        raise ValueError(f"No data found for date {date_str}")
    
    # Remove rows with missing coordinates
    coord_cols = [pickup_lat_col, pickup_lon_col, dropoff_lat_col, dropoff_lon_col]
    df_day = df_day.dropna(subset=coord_cols)
    print(f"Records after removing missing coordinates: {len(df_day)}")
    
    # Remove invalid coordinates (basic sanity check for NYC area)
    valid_mask = (
        (df_day[pickup_lat_col] >= 40.0) & (df_day[pickup_lat_col] <= 41.0) &
        (df_day[pickup_lon_col] >= -75.0) & (df_day[pickup_lon_col] <= -73.0) &
        (df_day[dropoff_lat_col] >= 40.0) & (df_day[dropoff_lat_col] <= 41.0) &
        (df_day[dropoff_lon_col] >= -75.0) & (df_day[dropoff_lon_col] <= -73.0)
    )
    df_day = df_day[valid_mask]
    print(f"Records after coordinate validation: {len(df_day)}")
    
    # Sample n requests if specified
    if n is not None and n > 0:
        if n >= len(df_day):
            print(f"Requested {n} samples but only {len(df_day)} available, using all")
        else:
            if random_sample:
                df_day = df_day.sample(n=n, random_state=seed)
                print(f"Randomly sampled {n} requests")
            else:
                df_day = df_day.head(n)
                print(f"Took first {n} requests")
    
    # Sort by pickup time
    df_day = df_day.sort_values(pickup_time_col).reset_index(drop=True)
    print(f"Final dataset: {len(df_day)} requests sorted by pickup time")
    
    return df_day, {
        'pickup_time': pickup_time_col,
        'pickup_lat': pickup_lat_col,
        'pickup_lon': pickup_lon_col,
        'dropoff_time': dropoff_time_col,
        'dropoff_lat': dropoff_lat_col,
        'dropoff_lon': dropoff_lon_col
    }


def prepare_solver_inputs(df, col_mapping, device='cuda'):
    """
    Convert DataFrame to tensors ready for the solver.
    
    Returns: xA (dropoff coords), xB (pickup coords), tA (dropoff times), tB (pickup times)
    """
    # Extract coordinates as numpy arrays
    pickup_coords = df[[col_mapping['pickup_lon'], col_mapping['pickup_lat']]].values.astype(np.float32)
    dropoff_coords = df[[col_mapping['dropoff_lon'], col_mapping['dropoff_lat']]].values.astype(np.float32)
    
    # Convert times to Unix timestamps (seconds since epoch)
    pickup_times = df[col_mapping['pickup_time']].astype('int64') // 10**9  # nanoseconds to seconds
    dropoff_times = df[col_mapping['dropoff_time']].astype('int64') // 10**9
    
    # Convert to tensors
    xB = torch.from_numpy(pickup_coords).to(device)    # [N, 2] pickup coordinates (B/right side)
    xA = torch.from_numpy(dropoff_coords).to(device)   # [N, 2] dropoff coordinates (A/left side)  
    tB = torch.from_numpy(pickup_times.values).to(device, dtype=torch.int64)   # [N] pickup times
    tA = torch.from_numpy(dropoff_times.values).to(device, dtype=torch.int64)  # [N] dropoff times
    
    print(f"Prepared tensors: xA={xA.shape}, xB={xB.shape}, tA={tA.shape}, tB={tB.shape}")
    print(f"Device: {device}")
    
    return xA, xB, tA, tB


def main():
    parser = argparse.ArgumentParser(description="NYC taxi experiment with lat/lon coordinates")
    parser.add_argument("--input", required=True, help="Path to NYC taxi data file (CSV or Parquet)")
    parser.add_argument("--date", default="2014-01-01", help="Date to filter (YYYY-MM-DD)")
    parser.add_argument("--n", type=int, default=None, help="Number of requests to sample")
    parser.add_argument("--random_sample", action="store_true", help="Random sample instead of first n")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    parser.add_argument("--k", type=int, default=500, help="Tile size")
    parser.add_argument("--delta", type=float, default=0.01, help="Approximation parameter")
    parser.add_argument("--cmax", type=int, default=None, help="Clamp integerized costs at this value (default: no clamp)")
    parser.add_argument(
        "--stopping_condition",
        type=int,
        default=None,
        help="Stop once free B requests fall to this count (default: run to convergence)",
    )
    
    args = parser.parse_args()
    
    print(f"Loading NYC taxi data for {args.date}")
    print(f"Input file: {args.input}")
    if args.n:
        print(f"Sampling: {args.n} requests ({'random' if args.random_sample else 'first'})")
    if args.cmax is not None:
        print(f"Cost clamp (cmax): {args.cmax}")
    if args.stopping_condition is not None:
        print(f"Stopping condition (free B threshold): {args.stopping_condition}")
    
    # Load and prepare data
    df, col_mapping = load_and_filter_data(
        args.input, 
        args.date, 
        n=args.n, 
        random_sample=args.random_sample, 
        seed=args.seed
    )
    
    # Prepare solver inputs
    device = torch.device(args.device)
    xA, xB, tA, tB = prepare_solver_inputs(df, col_mapping, device)
    
    # Preview first few records
    print("\nFirst 5 records:")
    print("Index | Pickup Time | Pickup (lon,lat) | Dropoff (lon,lat)")
    for i in range(min(5, len(df))):
        pickup_time = df.iloc[i][col_mapping['pickup_time']]
        pickup_coords = f"({xB[i,0]:.6f}, {xB[i,1]:.6f})"
        dropoff_coords = f"({xA[i,0]:.6f}, {xA[i,1]:.6f})"
        print(f"{i:5d} | {pickup_time} | {pickup_coords:20s} | {dropoff_coords}")
    
    print(f"\nData ready for solver!")
    print(f"Call solver with: xA={xA.shape}, xB={xB.shape}, tA={tA.shape}, tB={tB.shape}")
    
    # Call solver with parameters
    from spef_matching_nyc_3 import spef_matching_2
    from haversine_utils import haversine_distance_cpu
    
    # Estimate C using sample-based approach
    import random
    random.seed(args.seed)
    sample_idx = random.randint(0, len(xB) - 1)
    lat_b, lon_b = xB[sample_idx, 1].cpu(), xB[sample_idx, 0].cpu()
    
    max_dist = 0.0
    for i in range(len(xA)):
        lat_a, lon_a = xA[i, 1].cpu(), xA[i, 0].cpu()
        dist = haversine_distance_cpu(lat_b, lon_b, lat_a, lon_a)
        max_dist = max(max_dist, dist.item())
    
    C = 4.0 * max_dist
    
    print(f"\nEstimated C={C:.2f} from sample distances")
    print(f"Calling solver with C={C:.2f}, k={args.k}, delta={args.delta}")
    solver_start = time.perf_counter()
    result = spef_matching_2(
        xA=xA, xB=xB, C=C, k=args.k, delta=args.delta, device=device,
        tA=tA, tB=tB, seed=args.seed,
        cmax_int=args.cmax,
        stopping_condition=args.stopping_condition,
    )
    solver_elapsed = time.perf_counter() - solver_start
    
    Mb, yA, yB, matching_cost, iterations, timing_metrics = result
    print(f"Solver completed in {iterations} iterations")
    print(f"Total matching cost: {matching_cost:.2f}")
    print(f"Timing metrics: {timing_metrics}")
    print()
    print()

    # Additional human-readable summaries
    total_cost_m = float(matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = timing_metrics.get("feasible_matches", 0)
    free_b = timing_metrics.get("free_B", 0)

    print(f"Solver elapsed time: {solver_elapsed:.2f} s")

    if feasible_matches > 0:
        avg_cost_m = total_cost_m / feasible_matches
        avg_cost_km = avg_cost_m / 1000.0
        print(f"Total matching cost (km): {total_cost_km:.2f}")
        print(
            f"Average matching cost per feasible edge: {avg_cost_m:.2f} m ({avg_cost_km:.4f} km)"
        )
    else:
        print(f"Total matching cost (km): {total_cost_km:.2f}")
        print("Average matching cost per feasible edge: undefined (no feasible matches)")

    print(f"Feasible matches: {feasible_matches}, Free B nodes: {free_b}")


if __name__ == "__main__":
    main()
