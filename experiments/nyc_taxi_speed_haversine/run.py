#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

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

torch.set_float32_matmul_precision("high")

import spef_ot.kernels.haversine_speed  # noqa: F401 - ensure kernel registration
from spef_ot import MatchResult, match  # noqa: E402

from estimate_c import estimate_c  # noqa: E402
from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402


@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv"
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
        kernel="haversine_speed",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NYC taxi experiment using the Haversine speed slack kernel",
        epilog=(
            "Example: python experiments/NYC_Taxi_Speed_Haversine/run.py "
            "--input ./data/nyc_taxi_day.parquet --date 2014-10-14 "
            "--n 10000 --device cuda --k 512 --delta 0.01 "
            "--speed-mps 8.0 --y-max-meters 3000 --stopping-condition 1000 --fill-policy none"
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--input", type=str, default=None, help="Path to NYC taxi CSV/Parquet")
    parser.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    parser.add_argument("--n", type=int, default=None, help="Number of trips to keep")
    parser.add_argument(
        "--random-sample",
        dest="random_sample",
        action="store_true",
        default=None,
        help="Sample n trips uniformly at random",
    )
    parser.add_argument(
        "--no-random-sample",
        dest="random_sample",
        action="store_false",
        default=None,
        help="Disable random sampling",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu/cuda)")
    parser.add_argument("--k", type=int, default=None, help="Tile size for solver batches")
    parser.add_argument("--delta", type=float, default=None, help="Scaling delta")
    parser.add_argument(
        "--stopping-condition",
        dest="stopping_condition",
        type=int,
        default=None,
        help="Stop early when free B nodes fall to this count",
    )
    parser.add_argument(
        "--c-sample",
        dest="c_sample",
        type=int,
        default=None,
        help="Sample size for estimating C (used only if --C is not provided)",
    )
    parser.add_argument(
        "--C",
        dest="C",
        type=float,
        default=None,
        help="Provide scaling constant C directly (meters); skips estimation",
    )
    parser.add_argument(
        "--speed-mps",
        dest="speed_mps",
        type=float,
        default=None,
        help="Maximum average speed allowed for matches (meters per second)",
    )
    parser.add_argument(
        "--y-max-meters",
        dest="y_max_meters",
        type=float,
        default=None,
        help="Clamp Haversine distance to this threshold before integerization",
    )
    parser.add_argument(
        "--future-only",
        dest="future_only",
        action="store_true",
        default=None,
        help="Enforce pickup times to follow drop-off times",
    )
    parser.add_argument(
        "--no-future-only",
        dest="future_only",
        action="store_false",
        default=None,
        help="Allow matches where pickup precedes drop-off",
    )
    parser.add_argument(
        "--fill-policy",
        dest="fill_policy",
        choices=("greedy", "none"),
        default=None,
        help="Final fill policy passed to the solver",
    )
    parser.add_argument(
        "--preview-count",
        dest="preview_count",
        type=int,
        default=None,
        help="Number of example trips to print before solving",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=None,
        help="Enable detailed progress logging (per-iteration + tiles)",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--no-warmup",
        dest="no_warmup",
        action="store_true",
        default=None,
        help="Skip the initial warm-up run",
    )
    # Optional routes post-processing flags
    parser.add_argument(
        "--routes",
        dest="routes",
        action="store_true",
        default=None,
        help="Enable post-processing to reconstruct and analyze taxi routes",
    )
    parser.add_argument(
        "--no-routes",
        dest="routes",
        action="store_false",
        default=None,
        help="Disable routes post-processing (default)",
    )
    parser.add_argument(
        "--routes-json",
        dest="routes_json",
        type=str,
        default=None,
        help="Optional path to write routes statistics JSON (separate from --out)",
    )
    parser.add_argument(
        "--near-thresh-frac",
        dest="near_thresh_frac",
        type=float,
        default=None,
        help="Fraction of y_max to treat as near-threshold for A->B edges (default: 0.9)",
    )
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


def _haversine_distance_pairs(x_from_deg: torch.Tensor, x_to_deg: torch.Tensor) -> torch.Tensor:
    """Compute Haversine distances (meters) for aligned pairs of points.

    Expects `x_from_deg` and `x_to_deg` to be shape [N, 2] in degrees (lon, lat),
    and returns a tensor of shape [N] with distances in meters.
    """
    if x_from_deg.shape != x_to_deg.shape:
        raise ValueError("Input tensors must have the same shape for pairwise distances")
    if x_from_deg.numel() == 0:
        return x_from_deg.new_empty((0,), dtype=torch.float64)

    lon1 = torch.deg2rad(x_from_deg[:, 0].to(dtype=torch.float64))
    lat1 = torch.deg2rad(x_from_deg[:, 1].to(dtype=torch.float64))
    lon2 = torch.deg2rad(x_to_deg[:, 0].to(dtype=torch.float64))
    lat2 = torch.deg2rad(x_to_deg[:, 1].to(dtype=torch.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    sin_dlat = torch.sin(dlat * 0.5)
    sin_dlon = torch.sin(dlon * 0.5)
    a = sin_dlat.square() + torch.cos(lat1) * torch.cos(lat2) * sin_dlon.square()
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=0.0)))
    return (torch.tensor(6_371_000.0, dtype=torch.float64) * c)


def _compute_routes_stats(
    *,
    Mb: torch.Tensor,
    xA: torch.Tensor,
    xB: torch.Tensor,
    tA: torch.Tensor,
    tB: torch.Tensor,
    y_max_meters: float | None,
    near_thresh_frac: float,
) -> dict[str, Any]:
    """Reconstruct routes from Mb mapping and compute validation + stats.

    Returns a dictionary suitable for JSON serialization.
    """
    # Ensure CPU tensors for Python-side indexing and light computations
    Mb_cpu = Mb.to(dtype=torch.int64).cpu()
    xA_cpu = xA.to(dtype=torch.float32).cpu()
    xB_cpu = xB.to(dtype=torch.float32).cpu()
    tA_cpu = tA.to(dtype=torch.int64).cpu()
    tB_cpu = tB.to(dtype=torch.int64).cpu()

    nA = int(xA_cpu.shape[0])
    nB = int(xB_cpu.shape[0])
    N = min(nA, nB)

    # Build inverse mapping Ma from Mb
    Ma_cpu = torch.full((nA,), -1, dtype=torch.int64)
    matched_mask = Mb_cpu != -1
    if bool(matched_mask.any()):
        rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
        cols = Mb_cpu.index_select(0, rows)
        Ma_cpu[cols] = rows

    # Precompute within-request distances for aligned B[i] -> A[i]
    # Use only the common prefix length N if shapes differ
    within_dists = _haversine_distance_pairs(xB_cpu[:N], xA_cpu[:N])  # [N]

    # Route reconstruction: starts are unmatched B
    starts = torch.nonzero(Mb_cpu[:N] == -1, as_tuple=False).squeeze(1).tolist()

    routes: list[list[int]] = []  # sequences of request indices i
    route_invalid_flags: list[bool] = []
    route_invalid_edges: list[int] = []
    ab_edges_a: list[int] = []  # collect all A->B edges for stats
    ab_edges_b: list[int] = []

    visited_b_global = torch.zeros((N,), dtype=torch.bool)

    for b0 in starts:
        route: list[int] = []
        invalid_route = False
        invalid_edges_in_route = 0
        visited_b_local = set()

        i = int(b0)
        while True:
            if i in visited_b_local:
                # Detected a cycle; mark invalid and stop this route
                invalid_route = True
                break
            visited_b_local.add(i)
            visited_b_global[i] = True
            route.append(i)

            a_idx = i
            if a_idx >= nA:
                # Out-of-range safety (should not happen if N == nA == nB)
                break
            b_next = int(Ma_cpu[a_idx].item())
            if b_next < 0 or b_next >= N:
                # Reached unmatched A (or out of analyzed range)
                break

            # Validate A->B transition (feasibility + time)
            xa = xA_cpu[a_idx].unsqueeze(0)  # [1,2]
            xb = xB_cpu[b_next].unsqueeze(0)
            dist_ab = float(_haversine_distance_pairs(xa, xb)[0].item())
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

    # Stats across routes
    num_routes = len(routes)
    route_lengths = torch.tensor([len(r) for r in routes], dtype=torch.int64)

    # Route distances
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

        # Reposition edges are between consecutive trips in the route
        if len(r) <= 1:
            reposition_total_per_route.append(0.0)
        else:
            a_idx_list = torch.tensor(r[:-1], dtype=torch.int64)
            b_idx_list = torch.tensor(r[1:], dtype=torch.int64)
            xa = xA_cpu.index_select(0, a_idx_list)
            xb = xB_cpu.index_select(0, b_idx_list)
            d = _haversine_distance_pairs(xa, xb).sum(dtype=torch.float64)
            reposition_total_per_route.append(float(d.item()))

    total_per_route = [w + r for w, r in zip(within_total_per_route, reposition_total_per_route)]

    def _summary_stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {k: None for k in ("min", "max", "mean", "median", "p10", "p90", "p99")}
        t = torch.tensor(vals, dtype=torch.float64)
        return {
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "median": float(t.median().item()),
            "p10": float(torch.quantile(t, 0.10).item()),
            "p90": float(torch.quantile(t, 0.90).item()),
            "p99": float(torch.quantile(t, 0.99).item()),
        }

    # Histograms
    def _histogram(vals: list[float], bins: int = 20) -> dict[str, Any]:
        if not vals:
            return {"bins": [], "counts": []}
        t = torch.tensor(vals, dtype=torch.float64)
        vmin = float(t.min().item())
        vmax = float(t.max().item())
        if vmax <= vmin:
            edges = torch.linspace(vmin, vmax + 1e-9, steps=bins + 1, dtype=torch.float64)
        else:
            edges = torch.linspace(vmin, vmax, steps=bins + 1, dtype=torch.float64)
        counts = torch.histc(t, bins=bins, min=vmin, max=vmax)
        return {
            "bins": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in counts.to(dtype=torch.int64).tolist()],
        }

    # A->B edges collected across all routes
    if ab_edges_a:
        a_idx_tensor = torch.tensor(ab_edges_a, dtype=torch.int64)
        b_idx_tensor = torch.tensor(ab_edges_b, dtype=torch.int64)
        xa_all = xA_cpu.index_select(0, a_idx_tensor)
        xb_all = xB_cpu.index_select(0, b_idx_tensor)
        ab_dists = _haversine_distance_pairs(xa_all, xb_all)
        ab_dists_list = [float(x) for x in ab_dists.tolist()]
    else:
        ab_dists_list = []

    # Near-threshold fraction among A->B edges
    near_frac = float(near_thresh_frac)
    near_threshold_fraction = None
    if y_max_meters is not None and y_max_meters > 0.0 and ab_dists_list:
        y_max = float(y_max_meters)
        lower = near_frac * y_max
        num_near = sum(1 for d in ab_dists_list if (d >= lower and d < y_max))
        near_threshold_fraction = num_near / max(1, len(ab_dists_list))

    # Requests unserved: both Mb[i] == -1 and Ma[i] == -1 on same request index
    requests_unserved = int(((Mb_cpu[:N] == -1) & (Ma_cpu[:N] == -1)).sum().item())

    summary = {
        "num_routes": int(num_routes),
        "invalid_routes": int(sum(1 for f in route_invalid_flags if f)),
        "invalid_edges": int(sum(route_invalid_edges)),
        "fraction_invalid_routes": (
            (sum(1 for f in route_invalid_flags if f) / num_routes) if num_routes > 0 else None
        ),
        "requests_unserved": requests_unserved,
        "route_length": {
            "stats": {
                "min": (int(route_lengths.min().item()) if num_routes > 0 else None),
                "max": (int(route_lengths.max().item()) if num_routes > 0 else None),
                "mean": (float(route_lengths.to(dtype=torch.float64).mean().item()) if num_routes > 0 else None),
                "median": (int(route_lengths.median().item()) if num_routes > 0 else None),
                "p10": (int(torch.quantile(route_lengths.to(dtype=torch.float64), 0.10).item()) if num_routes > 0 else None),
                "p90": (int(torch.quantile(route_lengths.to(dtype=torch.float64), 0.90).item()) if num_routes > 0 else None),
                "p99": (int(torch.quantile(route_lengths.to(dtype=torch.float64), 0.99).item()) if num_routes > 0 else None),
            },
            "hist": _histogram([int(x) for x in route_lengths.tolist()], bins=20),
        },
        "route_distance_m": {
            "within": _summary_stats(within_total_per_route),
            "reposition": _summary_stats(reposition_total_per_route),
            "total": _summary_stats(total_per_route),
        },
        "edge_distance_m": {
            "A_to_B": {
                "stats": _summary_stats(ab_dists_list),
                "hist": _histogram(ab_dists_list, bins=30),
                "near_y_max_fraction": near_threshold_fraction,
            }
        },
    }

    # Also include a compact text summary for quick reading
    compact = {
        "num_routes": summary["num_routes"],
        "invalid_routes": summary["invalid_routes"],
        "invalid_edges": summary["invalid_edges"],
        "requests_unserved": summary["requests_unserved"],
        "avg_route_len": summary["route_length"]["stats"]["mean"],
        "avg_total_dist_km": (
            (summary["route_distance_m"]["total"]["mean"] / 1000.0)
            if summary["route_distance_m"]["total"]["mean"] is not None
            else None
        ),
        "near_thresh_frac": near_threshold_fraction,
    }
    summary["compact"] = compact

    return summary


def main() -> None:
    parser = _build_parser()
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    config = _resolve_paths(_resolve_config(args))

    def log(message: str) -> None:
        print(message)

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log("=== NYC Taxi Haversine Speed Experiment ===")
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

    preview_n = max(0, config.preview_count)
    if preview_n > 0:
        log(f"\nPreviewing first {min(preview_n, len(df))} trips (pickup -> dropoff):")
        log("Idx | Pickup time           | Dropoff time          | Pickup (lon,lat)      | Dropoff (lon,lat)")
        for idx in range(min(preview_n, len(df))):
            pickup_time = df.iloc[idx][mapping.pickup_time]
            dropoff_time = df.iloc[idx][mapping.dropoff_time]
            pickup_lon = df.iloc[idx][mapping.pickup_lon]
            pickup_lat = df.iloc[idx][mapping.pickup_lat]
            dropoff_lon = df.iloc[idx][mapping.dropoff_lon]
            dropoff_lat = df.iloc[idx][mapping.dropoff_lat]
            log(
                f"{idx:3d} | {pickup_time} | {dropoff_time} | "
                f"({pickup_lon:.6f}, {pickup_lat:.6f}) | "
                f"({dropoff_lon:.6f}, {dropoff_lat:.6f})"
            )
        log("")

    xA, xB, tA, tB = prepare_tensors(df, mapping, device=device)
    log(
        "Prepared tensors: "
        f"xA{tuple(xA.shape)}[{xA.dtype}], "
        f"xB{tuple(xB.shape)}[{xB.dtype}], "
        f"tA{tuple(tA.shape)}[{tA.dtype}], "
        f"tB{tuple(tB.shape)}[{tB.dtype}] on device {device}"
    )

    # Post-prepare tensor preview
    if preview_n > 0:
        count = min(preview_n, xA.shape[0])
        log(f"\nTensor preview (first {count} in order):")
        log("Idx | xA(lon,lat)            | xB(lon,lat)            | tA (s)        | tB (s)")
        for i in range(count):
            xa_lon, xa_lat = float(xA[i, 0].item()), float(xA[i, 1].item())
            xb_lon, xb_lat = float(xB[i, 0].item()), float(xB[i, 1].item())
            ta = int(tA[i].item())
            tb = int(tB[i].item())
            log(
                f"{i:3d} | ({xa_lon:.6f}, {xa_lat:.6f}) | "
                f"({xb_lon:.6f}, {xb_lat:.6f}) | "
                f"{ta:12d} | {tb:12d}"
            )
        log("")

    if config.C is not None:
        if config.C <= 0:
            raise ValueError("C must be positive when provided")
        C = float(config.C)
        log(f"Using provided C value: C={C:.4f}")
    else:
        C = estimate_c(
            xA,
            xB,
            sample_size=config.c_sample,
            seed=config.seed,
        )
        log(
            "Estimated C value: "
            f"C={C:.4f} (sample_size={config.c_sample})"
        )

    log(
        "Solver parameters: "
        f"k={config.k}, delta={config.delta}, stopping_condition={config.stopping_condition}"
    )
    warmup_time = 0.0
    if not config.no_warmup:
        _, warmup_time = _run_solver(
            xA=xA,
            xB=xB,
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
        if not config.verbose:
            log("Per-iteration solver logging is disabled. Re-run with --verbose to stream progress details.")
    else:
        log("Skipping warm-up run (--no-warmup)")

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        if event == "iteration":
            log(
                f"[Iter {payload['iteration']}] "
                f"free_B={payload['free_b']} matched_B={payload['matched_b']} "
                f"f={payload['objective_gap']:.3f} threshold={payload['threshold']:.3f}"
            )
        elif event == "sentinel":
            try:
                it = int(payload.get("iteration", -1))
                S = int(payload.get("sentinel_count", 0))
                speed = float(payload.get("speed_mps", 0.0))
                N = int(payload.get("A_count", 0))
                log("")
                log(
                    f"Iter {it} — Sentinel Deep Dive (S={S}, speed={speed:.2f} m/s, A_count={N})"
                )
                sentinels = payload.get("sentinels", [])
                for s in sentinels:
                    b_idx = s.get("b_idx")
                    tB_iso = s.get("tB_iso")
                    tB_epoch = s.get("tB_epoch")
                    eligible = s.get("eligible_by_time")
                    future_invalid = s.get("future_invalid")
                    oracle_valid = s.get("oracle_valid")
                    kernel_allowed = s.get("kernel_allowed")
                    miss_K = s.get("miss_kernel")
                    miss_K_near = s.get("miss_kernel_near")
                    miss_O = s.get("false_positive")
                    q64 = s.get("margin64_quantiles")
                    q32 = s.get("margin32_quantiles")
                    zero_count = s.get("zero_slack_count")
                    top_slack = s.get("top_slack", [])
                    exK = s.get("miss_kernel_examples", [])
                    exO = s.get("false_positive_examples", [])

                    log(
                        f"Pickup b={b_idx} tB={tB_iso} (epoch={tB_epoch})"
                    )
                    log(
                        f"  Eligible drop-offs (tA ≤ tB): {eligible}/{N}  [future-invalid: {future_invalid}]"
                    )
                    log("  Feasibility by speed")
                    log(f"    - Oracle valid (float64): {oracle_valid}")
                    log(f"    - Kernel allowed (float32): {kernel_allowed}")
                    log(
                        f"    - Kernel missed valid (FN): {miss_K}  [near-boundary (±1.0s): {miss_K_near}]"
                    )
                    log(f"    - Kernel false positives (FP): {miss_O}")
                    log("  Speed margin (sec) = dt − dist/speed")
                    if q64 is not None:
                        log(
                            "    - Oracle64: "
                            f"min={q64['min']:.3f} p01={q64['p01']:.3f} p10={q64['p10']:.3f} "
                            f"p50={q64['p50']:.3f} p90={q64['p90']:.3f} p99={q64['p99']:.3f} max={q64['max']:.3f}"
                        )
                    if q32 is not None:
                        log(
                            "    - Kernel32: "
                            f"min={q32['min']:.3f} p01={q32['p01']:.3f} p10={q32['p10']:.3f} "
                            f"p50={q32['p50']:.3f} p90={q32['p90']:.3f} p99={q32['p99']:.3f} max={q32['max']:.3f}"
                        )
                    log(f"  Zero-slack edges (current duals): {zero_count}")
                    if top_slack:
                        log("  Smallest slack (top 5)")
                        for item in top_slack:
                            log(
                                f"    - a={item['a_idx']} slack={item['slack']} dt={item['dt_s']}s "
                                f"dist={item['dist_km']:.3f} km need_time={item['need_time_s']:.3f}s "
                                f"margin64={item['margin64_s']:.3f}s"
                            )
                    if exK:
                        log("  Mismatch examples — Kernel missed valid (FN)")
                        for e in exK:
                            log(
                                f"    • a={e['a_idx']} dt={e['dt_s']}s dist32={e['dist32_m']:.2f}m dist64={e['dist64_m']:.2f}m "
                                f"need={e['need_time_s']:.3f}s margin32={e['margin32_s']:.3f}s margin64={e['margin64_s']:.3f}s"
                            )
                    if exO:
                        log("  Mismatch examples — Kernel allowed invalid (FP)")
                        for e in exO:
                            log(
                                f"    • a={e['a_idx']} dt={e['dt_s']}s dist32={e['dist32_m']:.2f}m dist64={e['dist64_m']:.2f}m "
                                f"need={e['need_time_s']:.3f}s margin32={e['margin32_s']:.3f}s margin64={e['margin64_s']:.3f}s"
                            )
            except Exception:
                # Logging must not break the run
                pass
        # Temporarily disable tile-level logging
        # elif event == "tile":
        #     log(
        #         f"  Tile {payload['tile_index']} "
        #         f"size={payload['tile_size']} rows[{payload['tile_start']}:{payload['tile_end']})"
        #     )

    result, runtime = _run_solver(
        xA=xA,
        xB=xB,
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
    log(f"Measured run completed in {runtime:.4f} s over {int(result.iterations)} iterations")

    total_cost_m = float(result.matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))
    removed_by_future = float(result.metrics.get("removed_by_future", 0.0))
    removed_by_speed = float(result.metrics.get("removed_by_speed", 0.0))
    removed_by_ymax = float(result.metrics.get("removed_by_ymax", 0.0))

    avg_cost_m = total_cost_m / feasible_matches if feasible_matches > 0 else None
    avg_cost_km = avg_cost_m / 1000.0 if avg_cost_m is not None else None
    log("\n=== Match Summary ===")
    log(f"Feasible matches : {feasible_matches}")
    log(f"Free B nodes     : {free_B}")
    log(f"Removed by future: {removed_by_future}")
    log(f"Removed by speed : {removed_by_speed}")
    log(f"Removed by y_max : {removed_by_ymax}")
    log(f"Total cost (m)   : {total_cost_m:.4f}")
    log(f"Total cost (km)  : {total_cost_km:.6f}")
    if avg_cost_m is not None:
        log(f"Average cost (m) : {avg_cost_m:.4f}")
        log(f"Average cost (km): {avg_cost_km:.6f}")
    else:
        log("Average cost     : undefined (no feasible matches)")

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

    print(json.dumps(output, indent=2))

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        log(f"\nWrote results to {out_path}")
    else:
        log("\nNo output path provided; results printed to stdout.")

    # Optional routes post-processing
    if bool(config.routes):
        log("\n=== Routes Post-Processing (enabled) ===")
        try:
            routes_summary = _compute_routes_stats(
                Mb=result.Mb,
                xA=xA,
                xB=xB,
                tA=tA,
                tB=tB,
                y_max_meters=config.y_max_meters,
                near_thresh_frac=float(config.near_thresh_frac),
            )

            # Print a short compact summary
            comp = routes_summary.get("compact", {})
            log(
                "Routes: "
                f"num={comp.get('num_routes')} "
                f"invalid_routes={comp.get('invalid_routes')} "
                f"invalid_edges={comp.get('invalid_edges')} "
                f"unserved={comp.get('requests_unserved')} "
                f"avg_len={comp.get('avg_route_len')} "
                f"avg_dist_km={comp.get('avg_total_dist_km')} "
                f"near_thresh_frac={comp.get('near_thresh_frac')}"
            )

            # Attach to main JSON output structure for stdout consumers
            output.setdefault("routes", routes_summary)

            # Optionally write a separate JSON artifact
            if config.routes_json:
                rpath = Path(config.routes_json)
                rpath.parent.mkdir(parents=True, exist_ok=True)
                with rpath.open("w", encoding="utf-8") as f:
                    json.dump(routes_summary, f, indent=2)
                log(f"Routes stats written to {rpath}")
        except Exception as ex:
            log(f"Routes post-processing failed: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
