#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Tuple

# Determinism and precision controls BEFORE importing torch
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch

# Ensure local package imports work without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

NYC_TAXI_DIR = EXPERIMENT_DIR.parent / "nyc_taxi"
if str(NYC_TAXI_DIR) not in sys.path:
    sys.path.insert(0, str(NYC_TAXI_DIR))

# Highest precision matmul; disable TF32 paths
torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
except Exception:
    pass

import spef_ot.kernels.euclidean_speed  # noqa: F401 - ensure kernel registration
from spef_ot import MatchResult, match  # noqa: E402

from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402


EARTH_RADIUS_METERS = 6_371_000.0


def _project_lonlat_to_meters(
    xA_deg: torch.Tensor,
    xB_deg: torch.Tensor,
    *,
    origin_lon: float | None = None,
    origin_lat: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    # Choose origin as medians if not provided
    if origin_lon is None or origin_lat is None:
        lonA, latA = xA_deg[:, 0], xA_deg[:, 1]
        lonB, latB = xB_deg[:, 0], xB_deg[:, 1]
        lon0 = torch.median(torch.cat((lonA, lonB))).item()
        lat0 = torch.median(torch.cat((latA, latB))).item()
    else:
        lon0, lat0 = float(origin_lon), float(origin_lat)

    # Convert degrees to meters in local tangent plane
    deg2rad = torch.tensor(torch.pi / 180.0, device=xA_deg.device, dtype=torch.float32)
    lat0_rad = torch.tensor(lat0 * (torch.pi / 180.0), device=xA_deg.device, dtype=torch.float32)
    scale_x = float(EARTH_RADIUS_METERS) * torch.cos(lat0_rad)
    scale_y = float(EARTH_RADIUS_METERS)

    def _proj(coords: torch.Tensor) -> torch.Tensor:
        lon = coords[:, 0] - float(lon0)
        lat = coords[:, 1] - float(lat0)
        x = lon.to(dtype=torch.float32) * deg2rad * scale_x
        y = lat.to(dtype=torch.float32) * deg2rad * scale_y
        return torch.stack((x, y), dim=1)

    return _proj(xA_deg), _proj(xB_deg), lon0, lat0


@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20141014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 100000
    random_sample: bool = True
    seed: int = 1
    device: str | None = "cuda"
    k: int = 512
    delta: float = 0.001
    stopping_condition: int | None = 1000
    c_sample: int = 1
    C: float | None = 100000.0
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 100000.0
    future_only: bool = True
    fill_policy: str = "none"
    preview_count: int = 5
    verbose: bool = True
    out: str | None = None
    no_warmup: bool = False
    origin_lon: float | None = None
    origin_lat: float | None = None
    # Optional routes post-processing
    routes: bool = False
    routes_json: str | None = None
    near_thresh_frac: float = 0.9


DEFAULT_CONFIG = ExperimentConfig()


def _load_config_from_json(path: Path, base: ExperimentConfig) -> ExperimentConfig:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object at the top level")
    merged = asdict(base)
    for key, value in data.items():
        if key in merged:
            merged[key] = value
    return ExperimentConfig(**merged)


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates: dict[str, Any] = {}
    for field in fields(ExperimentConfig):
        name = field.name
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        updates[name] = value
    if not updates:
        return config
    return replace(config, **updates)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NYC taxi experiment using the Euclidean speed slack kernel",
        epilog=(
            "Example: python experiments/nyc_taxi_speed_euclidean/run.py "
            "--input ./data/nyc.csv --date 2014-10-14 --device cuda "
            "--k 512 --delta 0.001 --speed-mps 8.0 --stopping-condition 1000 --fill-policy none"
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--input", type=str, default=None, help="Path to NYC taxi CSV/Parquet")
    parser.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    parser.add_argument("--n", type=int, default=None, help="Number of trips to keep")
    parser.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    parser.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--k", type=int, default=None, help="Tile size (K)")
    parser.add_argument("--delta", type=float, default=None, help="Scaling delta")
    parser.add_argument("--stopping-condition", dest="stopping_condition", type=int, default=None)
    parser.add_argument("--c-sample", dest="c_sample", type=int, default=None)
    parser.add_argument("--C", dest="C", type=float, default=None, help="Scaling constant C (meters)")
    parser.add_argument("--speed-mps", dest="speed_mps", type=float, default=None)
    parser.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=None)
    parser.add_argument("--future-only", dest="future_only", action="store_true", default=None)
    parser.add_argument("--no-future-only", dest="future_only", action="store_false", default=None)
    parser.add_argument("--fill-policy", dest="fill_policy", choices=("greedy", "none"), default=None)
    parser.add_argument("--preview-count", dest="preview_count", type=int, default=None)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=None)
    parser.add_argument("--origin-lon", dest="origin_lon", type=float, default=None)
    parser.add_argument("--origin-lat", dest="origin_lat", type=float, default=None)
    # Optional routes post-processing flags
    parser.add_argument("--routes", dest="routes", action="store_true", default=None,
                        help="Enable post-processing to reconstruct and analyze taxi routes")
    parser.add_argument("--no-routes", dest="routes", action="store_false", default=None,
                        help="Disable routes post-processing (default)")
    parser.add_argument("--routes-json", dest="routes_json", type=str, default=None,
                        help="Optional path to write routes statistics JSON (separate from --out)")
    parser.add_argument("--near-thresh-frac", dest="near_thresh_frac", type=float, default=None,
                        help="Fraction of y_max to treat as near-threshold for A->B edges (default: 0.9)")
    return parser


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    config = DEFAULT_CONFIG
    if args.config:
        config_path = Path(args.config)
        config = _load_config_from_json(config_path, config)
    config = _apply_overrides(config, args)
    return config


def _resolve_paths(config: ExperimentConfig) -> ExperimentConfig:
    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()

    out_path = config.out
    if out_path is not None:
        out_obj = Path(out_path)
        if not out_obj.is_absolute():
            out_obj = (EXPERIMENT_DIR / out_obj).resolve()
        out_path = str(out_obj)

    routes_json = config.routes_json
    if routes_json is not None:
        routes_obj = Path(routes_json)
        if not routes_obj.is_absolute():
            routes_obj = (EXPERIMENT_DIR / routes_obj).resolve()
        routes_json = str(routes_obj)

    return replace(config, input=str(input_path), out=out_path, routes_json=routes_json)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _estimate_c_euclidean(
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
    *,
    sample_size: int = 64,
    seed: int = 1,
) -> float:
    device = xA_m.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    m = xB_m.shape[0]
    sample_size = min(max(1, sample_size), m)
    idx = torch.randperm(m, device=device, generator=generator)[:sample_size]
    xb = xB_m.index_select(0, idx)  # [S,2]
    xa_T = xA_m.transpose(0, 1)
    xa2 = (xA_m.square()).sum(dim=1)
    # Distances for the sample vs all A
    xb2 = (xb.square()).sum(dim=1)
    prod = xb @ xa_T
    d2 = xb2.unsqueeze(1) + xa2.unsqueeze(0) - 2.0 * prod
    d2 = torch.clamp(d2, min=0.0)
    dist = torch.sqrt(d2)
    max_dist = float(dist.max().item())
    return 4.0 * max_dist


def _euclidean_distance_pairs(x_from_m: torch.Tensor, x_to_m: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distances (meters) for aligned pairs of points.

    Expects tensors of shape [N,2] in meters; returns shape [N].
    """
    if x_from_m.shape != x_to_m.shape:
        raise ValueError("Input tensors must have the same shape for pairwise distances")
    if x_from_m.numel() == 0:
        return x_from_m.new_empty((0,), dtype=torch.float64)
    diff = (x_to_m.to(dtype=torch.float64) - x_from_m.to(dtype=torch.float64))
    return torch.linalg.vector_norm(diff, dim=1)


def _compute_routes_stats(
    *,
    Mb: torch.Tensor,
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
    tA: torch.Tensor,
    tB: torch.Tensor,
    y_max_meters: float | None,
    near_thresh_frac: float,
) -> dict[str, Any]:
    """Reconstruct routes from Mb mapping and compute validation + stats (Euclidean)."""
    Mb_cpu = Mb.to(dtype=torch.int64).cpu()
    xA_cpu = xA_m.to(dtype=torch.float32).cpu()
    xB_cpu = xB_m.to(dtype=torch.float32).cpu()
    tA_cpu = tA.to(dtype=torch.int64).cpu()
    tB_cpu = tB.to(dtype=torch.int64).cpu()

    nA = int(xA_cpu.shape[0])
    nB = int(xB_cpu.shape[0])
    N = min(nA, nB)

    # Build inverse map Ma from Mb
    Ma_cpu = torch.full((nA,), -1, dtype=torch.int64)
    matched_mask = Mb_cpu != -1
    if bool(matched_mask.any()):
        rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
        cols = Mb_cpu.index_select(0, rows)
        Ma_cpu[cols] = rows

    # Precompute within-request distances for aligned B[i] -> A[i]
    within_dists = _euclidean_distance_pairs(xB_cpu[:N], xA_cpu[:N])

    # Route starts are unmatched B in the analyzed range
    starts = torch.nonzero(Mb_cpu[:N] == -1, as_tuple=False).squeeze(1).tolist()

    routes: list[list[int]] = []
    route_invalid_flags: list[bool] = []
    route_invalid_edges: list[int] = []
    ab_edges_a: list[int] = []
    ab_edges_b: list[int] = []

    for b0 in starts:
        route: list[int] = []
        invalid_route = False
        invalid_edges_in_route = 0
        visited_b_local = set()

        i = int(b0)
        while True:
            if i in visited_b_local:
                invalid_route = True
                break
            visited_b_local.add(i)
            route.append(i)

            a_idx = i
            if a_idx >= nA:
                break
            b_next = int(Ma_cpu[a_idx].item())
            if b_next < 0 or b_next >= N:
                break

            # Validate A->B transition (distance and time)
            xa = xA_cpu[a_idx].unsqueeze(0)
            xb = xB_cpu[b_next].unsqueeze(0)
            dist_ab = float(_euclidean_distance_pairs(xa, xb)[0].item())
            time_ok = bool(int(tA_cpu[a_idx].item()) <= int(tB_cpu[b_next].item()))
            edge_ok = True
            if y_max_meters is not None and y_max_meters > 0.0:
                edge_ok = edge_ok and (dist_ab < float(y_max_meters))
            if not time_ok or not edge_ok:
                invalid_edges_in_route += 1
                invalid_route = True

            ab_edges_a.append(a_idx)
            ab_edges_b.append(b_next)

            i = b_next

        routes.append(route)
        route_invalid_flags.append(bool(invalid_route))
        route_invalid_edges.append(int(invalid_edges_in_route))

    num_routes = len(routes)
    route_lengths = (
        torch.tensor([len(r) for r in routes], dtype=torch.int64)
        if routes
        else torch.tensor([], dtype=torch.int64)
    )

    within_total_per_route: list[float] = []
    reposition_total_per_route: list[float] = []
    for r in routes:
        if not r:
            within_total_per_route.append(0.0)
            reposition_total_per_route.append(0.0)
            continue
        idx = torch.tensor(r, dtype=torch.int64)
        wsum = within_dists.index_select(0, idx.clamp_max(N - 1)).sum(dtype=torch.float64)
        within_total_per_route.append(float(wsum.item()))

        if len(r) <= 1:
            reposition_total_per_route.append(0.0)
        else:
            a_idx_list = torch.tensor(r[:-1], dtype=torch.int64)
            b_idx_list = torch.tensor(r[1:], dtype=torch.int64)
            xa = xA_cpu.index_select(0, a_idx_list)
            xb = xB_cpu.index_select(0, b_idx_list)
            d = _euclidean_distance_pairs(xa, xb).sum(dtype=torch.float64)
            reposition_total_per_route.append(float(d.item()))

    total_per_route = [w + r for w, r in zip(within_total_per_route, reposition_total_per_route)]

    def _min_max_mean(vals: list[float]) -> tuple[float | None, float | None, float | None]:
        if not vals:
            return None, None, None
        t = torch.tensor(vals, dtype=torch.float64)
        return float(t.min().item()), float(t.max().item()), float(t.mean().item())

    # A->B edges collected across all routes
    if ab_edges_a:
        a_idx_tensor = torch.tensor(ab_edges_a, dtype=torch.int64)
        b_idx_tensor = torch.tensor(ab_edges_b, dtype=torch.int64)
        xa_all = xA_cpu.index_select(0, a_idx_tensor)
        xb_all = xB_cpu.index_select(0, b_idx_tensor)
        ab_dists = _euclidean_distance_pairs(xa_all, xb_all)
        ab_dists_list = [float(x) for x in ab_dists.tolist()]
    else:
        ab_dists_list = []

    near_frac = float(near_thresh_frac)
    near_threshold_fraction = None
    if y_max_meters is not None and y_max_meters > 0.0 and ab_dists_list:
        y_max = float(y_max_meters)
        lower = near_frac * y_max
        num_near = sum(1 for d in ab_dists_list if (d >= lower and d < y_max))
        near_threshold_fraction = num_near / max(1, len(ab_dists_list))

    requests_unserved = int(((Mb_cpu[:N] == -1) & (Ma_cpu[:N] == -1)).sum().item())

    min_len = int(route_lengths.min().item()) if num_routes > 0 else None
    max_len = int(route_lengths.max().item()) if num_routes > 0 else None
    mean_len = float(route_lengths.to(dtype=torch.float64).mean().item()) if num_routes > 0 else None

    min_m, max_m, mean_m = _min_max_mean(total_per_route)
    avg_km = (mean_m / 1000.0) if mean_m is not None else None
    min_km = (min_m / 1000.0) if min_m is not None else None
    max_km = (max_m / 1000.0) if max_m is not None else None

    return {
        "num_routes": int(num_routes),
        "invalid_routes": int(sum(1 for f in route_invalid_flags if f)),
        "invalid_edges": int(sum(route_invalid_edges)),
        "requests_unserved": requests_unserved,
        "avg_route_len": mean_len,
        "min_route_len": min_len,
        "max_route_len": max_len,
        "avg_total_dist_km": avg_km,
        "min_total_dist_km": min_km,
        "max_total_dist_km": max_km,
        "near_thresh_frac": near_thresh_frac,
    }

def _run_solver(
    *,
    xA: torch.Tensor,
    xB: torch.Tensor,
    C: float,
    k: int,
    delta: float,
    device: torch.device,
    seed: int,
    times_A: torch.Tensor,
    times_B: torch.Tensor,
    stopping_condition: int | None,
    speed_mps: float | None,
    y_max_meters: float | None,
    future_only: bool,
    fill_policy: str,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[MatchResult, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = match(
        xA,
        xB,
        kernel="euclidean_speed",
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        times_A=times_A,
        times_B=times_B,
        stopping_condition=stopping_condition,
        speed_mps=speed_mps,
        y_max_meters=y_max_meters,
        future_only=future_only,
        fill_policy=fill_policy,
        progress_callback=progress_callback,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])
    config = _resolve_paths(_resolve_config(args))

    # Seeds for determinism
    _seed_all(int(config.seed))

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def log(msg: str) -> None:
        print(msg)

    log("=== NYC Taxi Euclidean Speed Experiment ===")
    log(f"Input file      : {config.input}")
    log(f"Date            : {config.date}")
    log(f"Requested trips : {config.n} ({'random' if config.random_sample else 'first'})")
    log(f"Speed constraint: {config.speed_mps} m/s")
    log(f"Y max clamp     : {config.y_max_meters} m")
    log(f"Future-only     : {config.future_only}")
    log(f"Fill policy     : {config.fill_policy}")
    log(f"Device          : {device}")

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=log,
    )
    log(f"Loader returned {len(df)} trips after filtering.")

    xA_deg, xB_deg, tA, tB = prepare_tensors(df, mapping, device=device)

    # Project lon/lat to planar meters (local tangent plane)
    xA_m, xB_m, lon0, lat0 = _project_lonlat_to_meters(
        xA_deg, xB_deg, origin_lon=config.origin_lon, origin_lat=config.origin_lat
    )
    log(
        f"Projection origin (lon,lat): ({lon0:.6f}, {lat0:.6f}); "
        f"xA{tuple(xA_m.shape)} xB{tuple(xB_m.shape)}"
    )

    # C handling: prefer provided; else estimate in planar metric
    if config.C is not None:
        if config.C <= 0:
            raise ValueError("C must be positive when provided")
        C = float(config.C)
        log(f"Using provided C value: C={C:.4f}")
    else:
        C = _estimate_c_euclidean(xA_m, xB_m, sample_size=config.c_sample, seed=config.seed)
        log(f"Estimated C value (Euclidean): C={C:.4f} (sample_size={config.c_sample})")

    log(
        "Solver parameters: "
        f"k={config.k}, delta={config.delta}, stopping_condition={config.stopping_condition}"
    )

    warmup_time = 0.0
    if not config.no_warmup:
        _, warmup_time = _run_solver(
            xA=xA_m,
            xB=xB_m,
            C=C,
            k=config.k,
            delta=config.delta,
            device=device,
            seed=config.seed,
            times_A=tA,
            times_B=tB,
            stopping_condition=config.stopping_condition,
            speed_mps=config.speed_mps,
            y_max_meters=config.y_max_meters,
            future_only=bool(config.future_only),
            fill_policy=config.fill_policy,
            progress_callback=None,
        )
        log(f"Warmup run completed in {warmup_time:.4f} s")
    else:
        log("Skipping warm-up run (--no-warmup)")

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        if event == "iteration":
            print(
                f"[Iter {payload['iteration']}] "
                f"free_B={payload['free_b']} matched_B={payload['matched_b']} "
                f"f={payload['objective_gap']:.3f} threshold={payload['threshold']:.3f}"
            )

    result, runtime = _run_solver(
        xA=xA_m,
        xB=xB_m,
        C=C,
        k=config.k,
        delta=config.delta,
        device=device,
        seed=config.seed,
        times_A=tA,
        times_B=tB,
        stopping_condition=config.stopping_condition,
        speed_mps=config.speed_mps,
        y_max_meters=config.y_max_meters,
        future_only=bool(config.future_only),
        fill_policy=config.fill_policy,
        progress_callback=progress_callback if config.verbose else None,
    )
    print(f"Measured run completed in {runtime:.4f} s over {int(result.iterations)} iterations")

    total_cost_m = float(result.matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))
    removed_by_future = float(result.metrics.get("removed_by_future", 0.0))
    removed_by_speed = float(result.metrics.get("removed_by_speed", 0.0))
    removed_by_ymax = float(result.metrics.get("removed_by_ymax", 0.0))

    avg_cost_m = total_cost_m / feasible_matches if feasible_matches > 0 else None
    avg_cost_km = avg_cost_m / 1000.0 if avg_cost_m is not None else None
    print("\n=== Match Summary ===")
    print(f"Feasible matches : {feasible_matches}")
    print(f"Free B nodes     : {free_B}")
    print(f"Removed by future: {removed_by_future}")
    print(f"Removed by speed : {removed_by_speed}")
    print(f"Removed by y_max : {removed_by_ymax}")
    print(f"Total cost (m)   : {total_cost_m:.4f}")
    print(f"Total cost (km)  : {total_cost_km:.6f}")
    if avg_cost_m is not None:
        print(f"Average cost (m) : {avg_cost_m:.4f}")
        print(f"Average cost (km): {avg_cost_km:.6f}")
    else:
        print("Average cost     : undefined (no feasible matches)")

    output = {
        "params": {
            "input": str(Path(config.input).resolve()),
            "date": config.date,
            "n_requested": config.n,
            "n_used": len(df),
            "random_sample": bool(config.random_sample),
            "seed": config.seed,
            "device": str(device),
            "k": config.k,
            "delta": config.delta,
            "stopping_condition": config.stopping_condition,
            "C_estimate": C,
            "c_sample": config.c_sample,
            "speed_mps": config.speed_mps,
            "y_max_meters": config.y_max_meters,
            "future_only": bool(config.future_only),
            "fill_policy": config.fill_policy,
            "origin_lon": lon0,
            "origin_lat": lat0,
        },
        "performance": {
            "warmup_runtime_sec": warmup_time,
            "runtime_sec": runtime,
            "iterations": int(result.iterations),
            "timing_metrics": result.metrics,
        },
        "metrics": {
            "matching_cost_m": total_cost_m,
            "matching_cost_km": total_cost_km,
            "avg_cost_m": avg_cost_m,
            "avg_cost_km": avg_cost_km,
            "feasible_matches": feasible_matches,
            "free_B": free_B,
            "removed_by_future": removed_by_future,
            "removed_by_speed": removed_by_speed,
            "removed_by_ymax": removed_by_ymax,
        },
    }

    # Optional routes post-processing
    if bool(config.routes):
        print("\n=== Routes Post-Processing (enabled) ===")
        try:
            routes_summary = _compute_routes_stats(
                Mb=result.Mb,
                xA_m=xA_m,
                xB_m=xB_m,
                tA=tA,
                tB=tB,
                y_max_meters=config.y_max_meters,
                near_thresh_frac=float(config.near_thresh_frac),
            )
            def _fmt(v: Any) -> str:
                return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

            print("Routes Summary:")
            print(f"  num_routes              : {_fmt(routes_summary.get('num_routes'))}")
            print(f"  invalid_routes          : {_fmt(routes_summary.get('invalid_routes'))}")
            print(f"  invalid_edges           : {_fmt(routes_summary.get('invalid_edges'))}")
            print(f"  requests_unserved       : {_fmt(routes_summary.get('requests_unserved'))}")
            print(f"  avg_route_len           : {_fmt(routes_summary.get('avg_route_len'))}")
            print(f"  min_route_len (edges)   : {_fmt(routes_summary.get('min_route_len'))}")
            print(f"  max_route_len (edges)   : {_fmt(routes_summary.get('max_route_len'))}")
            print(f"  avg_total_dist_km       : {_fmt(routes_summary.get('avg_total_dist_km'))}")
            print(f"  min_total_dist_km       : {_fmt(routes_summary.get('min_total_dist_km'))}")
            print(f"  max_total_dist_km       : {_fmt(routes_summary.get('max_total_dist_km'))}")
            print(f"  near_thresh_frac        : {_fmt(routes_summary.get('near_thresh_frac'))}")

            output["routes"] = routes_summary

            if config.routes_json:
                rpath = Path(config.routes_json)
                rpath.parent.mkdir(parents=True, exist_ok=True)
                with rpath.open("w", encoding="utf-8") as f:
                    json.dump(routes_summary, f, indent=2)
                print(f"Routes stats written to {rpath}")
        except Exception as ex:
            print(f"Routes post-processing failed: {type(ex).__name__}: {ex}")

    print(json.dumps(output, indent=2))

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote results to {out_path}")
    else:
        print("\nNo output path provided; results printed to stdout.")


if __name__ == "__main__":
    main()
