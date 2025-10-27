#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch

# Ensure local package imports work without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

torch.set_float32_matmul_precision("high")

from spef_ot import MatchResult, match  # noqa: E402

from estimate_c import estimate_c  # noqa: E402
from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402

@dataclass
class ExperimentConfig:
    input: str = "./data/nyc_taxi_day.parquet"
    date: str = "2014-14-10"
    n: int | None = 10000
    random_sample: bool = True
    seed: int = 1
    device: str | None = 'cuda'
    k: int = 512
    delta: float = 0.01
    cmax: int | None = 5
    stopping_condition: int | None = 1000
    c_sample: int = 64
    C: float | None = None
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
    updates: dict[str, object] = {}
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
    cmax_int: int | None,
    stopping_condition: int | None,
) -> tuple[MatchResult, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = match(
        xA,
        xB,
        kernel="haversine",
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        times_A=times_A,
        times_B=times_B,
        cmax_int=cmax_int,
        stopping_condition=stopping_condition,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return result, elapsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NYC taxi Haversine matching experiment")
    parser.add_argument("--config", type=str, help="Path to JSON config file", default=None)
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--k", type=int, default=None, help="Tile size (K)")
    parser.add_argument("--delta", type=float, default=None, help="Scaling delta")
    parser.add_argument("--cmax", type=int, default=None, help="Optional integerized cost clamp")
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
        input_path = (EXPERIMENT_DIR / input_path).resolve()

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
        # No CLI args – run with in-file defaults
        args = parser.parse_args([])

    config = _resolve_paths(_resolve_config(args))

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
    )

    xA, xB, tA, tB = prepare_tensors(df, mapping, device=device)

    if config.C is not None:
        if config.C <= 0:
            raise ValueError("C must be positive when provided")
        C = float(config.C)
    else:
        C = estimate_c(
            xA,
            xB,
            sample_size=config.c_sample,
            seed=config.seed,
        )

    # Warm-up run (amortize compilation/graph capture)
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
            cmax_int=config.cmax,
            stopping_condition=config.stopping_condition,
        )

    # Timed run with fresh state
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
        cmax_int=config.cmax,
        stopping_condition=config.stopping_condition,
    )

    total_cost_m = float(result.matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))

    avg_cost_m = total_cost_m / feasible_matches if feasible_matches > 0 else None
    avg_cost_km = avg_cost_m / 1000.0 if avg_cost_m is not None else None

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
            "cmax": config.cmax,
            "stopping_condition": config.stopping_condition,
            "C_estimate": C,
            "c_sample": config.c_sample,
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
        },
    }

    print(json.dumps(output, indent=2))

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()


def _load_config_from_json(path: Path, base: ExperimentConfig) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object at the top level")
    merged = asdict(base)
    merged.update({k: v for k, v in data.items() if k in merged})
    return ExperimentConfig(**merged)


def _apply_overrides(config: ExperimentConfig, namespace: argparse.Namespace) -> ExperimentConfig:
    updates = {}
    for field in fields(ExperimentConfig):
        name = field.name
        if not hasattr(namespace, name):
            continue
        value = getattr(namespace, name)
        if value is None:
            continue
        updates[name] = value
    if not updates:
        return config
    return replace(config, **updates)
