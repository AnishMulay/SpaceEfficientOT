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
    n: int | None = 10000
    random_sample: bool = True
    seed: int = 1
    device: str | None = "cuda"
    k: int = 512
    delta: float = 0.01
    stopping_condition: int | None = 1000
    c_sample: int = 1
    c_multiplier: float = 1.0
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 100000.0
    future_only: bool = True
    fill_policy: str = "none"
    preview_count: int = 5
    verbose: bool = True
    out: str | None = None
    no_warmup: bool = False


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
            "Example: python experiments/nyc_taxi_speed/run.py "
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
        help="Sample size for estimating C",
    )
    parser.add_argument(
        "--c-multiplier",
        dest="c_multiplier",
        type=float,
        default=None,
        help="Multiplier applied to max sampled distance",
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

    return replace(config, input=str(input_path), out=out_path)


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

    C = estimate_c(
        xA,
        xB,
        sample_size=config.c_sample,
        seed=config.seed,
        multiplier=config.c_multiplier,
    )
    log(
        "Estimated C value: "
        f"C={C:.4f} (sample_size={config.c_sample}, multiplier={config.c_multiplier})"
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
        elif event == "tile":
            log(
                f"  Tile {payload['tile_index']} "
                f"size={payload['tile_size']} rows[{payload['tile_start']}:{payload['tile_end']})"
            )

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
            "c_multiplier": config.c_multiplier,
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


if __name__ == "__main__":
    main()
