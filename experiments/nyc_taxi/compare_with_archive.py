#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

# Local imports without install
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

torch.set_float32_matmul_precision("high")

from spef_ot import match  # noqa: E402
from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402

from dataclasses import dataclass, asdict, replace, fields


@dataclass
class ExperimentConfig:
    input: str = "./data/nyc_taxi_day.parquet"
    date: str = "2014-01-01"
    n: int | None = None
    random_sample: bool = False
    seed: int = 1
    device: str | None = None
    k: int = 512
    delta: float = 0.01
    cmax: int | None = None
    stopping_condition: int | None = None


DEFAULT_CONFIG = ExperimentConfig()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare NYC Haversine: new vs archive")
    p.add_argument("--input", type=str, default=None, help="CSV/Parquet path")
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--cmax", type=int, default=None)
    p.add_argument("--stopping-condition", dest="stopping_condition", type=int, default=None)
    return p


def _apply_overrides(config: ExperimentConfig, ns: argparse.Namespace) -> ExperimentConfig:
    updates: dict[str, object] = {}
    for f in fields(ExperimentConfig):
        name = f.name
        if hasattr(ns, name):
            val = getattr(ns, name)
            if val is not None:
                updates[name] = val
    if not updates:
        return config
    return replace(config, **updates)


def _resolve_paths(config: ExperimentConfig) -> ExperimentConfig:
    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (EXPERIMENT_DIR / input_path).resolve()
    return replace(config, input=str(input_path))


def _load_archive_solver():
    arch = REPO_ROOT / "Archive"
    mod = arch / "spef_matching_nyc_2.py"
    if not mod.exists():
        raise FileNotFoundError(mod)
    if str(arch) not in sys.path:
        sys.path.append(str(arch))
    spec = importlib.util.spec_from_file_location("archive_spef_matching_nyc_2", mod)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.spef_matching_2


def _load_haversine_cpu():
    arch = REPO_ROOT / "Archive"
    mod = arch / "haversine_utils.py"
    if not mod.exists():
        raise FileNotFoundError(mod)
    if str(arch) not in sys.path:
        sys.path.append(str(arch))
    spec = importlib.util.spec_from_file_location("archive_haversine_utils", mod)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.haversine_distance_cpu


@torch.no_grad()
def estimate_c_single_pickup_cpu(
    xA: torch.Tensor,
    xB: torch.Tensor,
    *,
    seed: int,
    multiplier: float = 4.0,
) -> float:
    """Replicate C estimation from nyc_day_experiment_3: pick one B, loop on CPU, max dist * multiplier."""
    haversine_distance_cpu = _load_haversine_cpu()

    n = xB.shape[0]
    if n == 0:
        raise ValueError("xB is empty")
    rnd = random.Random(seed)
    idx = rnd.randrange(0, n)

    xb = xB[idx].detach().cpu()
    max_dist = 0.0
    for i in range(xA.shape[0]):
        xa = xA[i].detach().cpu()
        # Inputs are [lon, lat]
        lat_b, lon_b = xb[1], xb[0]
        lat_a, lon_a = xa[1], xa[0]
        dist = haversine_distance_cpu(lat_b, lon_b, lat_a, lon_a).item()
        if dist > max_dist:
            max_dist = dist
    return float(multiplier * max_dist)


def _run_new(
    xA, xB, tA, tB, C, k, delta, device, seed, cmax, stopping_condition
):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = match(
        xA,
        xB,
        kernel="haversine",
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        times_A=tA,
        times_B=tB,
        cmax_int=cmax,
        stopping_condition=stopping_condition,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return res, t1 - t0


def _run_old(
    solver, xA, xB, tA, tB, C, k, delta, device, seed, cmax, stopping_condition
):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Mb, yA, yB, cost, iters, metrics = solver(
        xA=xA,
        xB=xB,
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        tA=tA,
        tB=tB,
        cmax_int=cmax,
        stopping_condition=stopping_condition,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (Mb, yA, yB, cost, iters, metrics), t1 - t0


def main():
    p = _build_parser()
    if len(sys.argv) > 1:
        args = p.parse_args()
    else:
        args = p.parse_args([])

    config = _resolve_paths(_apply_overrides(DEFAULT_CONFIG, args))

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load and prepare identical data for both runs
    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
    )
    xA, xB, tA, tB = prepare_tensors(df, mapping, device=device)

    # Estimate C exactly like nyc_day_experiment_3
    C = estimate_c_single_pickup_cpu(xA, xB, seed=config.seed, multiplier=4.0)

    # Warm-up both
    solver_old = _load_archive_solver()
    _run_new(xA, xB, tA, tB, C, config.k, config.delta, device, config.seed, config.cmax, config.stopping_condition)
    _run_old(solver_old, xA, xB, tA, tB, C, config.k, config.delta, device, config.seed, config.cmax, config.stopping_condition)

    # Timed runs
    res_new, t_new = _run_new(
        xA, xB, tA, tB, C, config.k, config.delta, device, config.seed, config.cmax, config.stopping_condition
    )
    (Mb_old, yA_old, yB_old, cost_old, iters_old, metrics_old), t_old = _run_old(
        solver_old, xA, xB, tA, tB, C, config.k, config.delta, device, config.seed, config.cmax, config.stopping_condition
    )

    out = {
        "params": {
            "input": str(Path(config.input).resolve()),
            "date": config.date,
            "n": config.n,
            "random_sample": bool(config.random_sample),
            "seed": config.seed,
            "device": str(device),
            "k": config.k,
            "delta": config.delta,
            "cmax": config.cmax,
            "stopping_condition": config.stopping_condition,
            "C_estimate": C,
        },
        "new": {
            "runtime_sec": t_new,
            "cost": float(res_new.matching_cost),
            "iterations": int(res_new.iterations),
            "feasible_matches": float(res_new.metrics.get("feasible_matches", 0.0)),
            "free_B": float(res_new.metrics.get("free_B", 0.0)),
        },
        "archive": {
            "runtime_sec": t_old,
            "cost": float(cost_old.cpu()),
            "iterations": int(iters_old),
            "feasible_matches": float(metrics_old.get("feasible_matches", 0.0)),
            "free_B": float(metrics_old.get("free_B", 0.0)),
        },
        "parity": {
            "cost_equal": abs(float(res_new.matching_cost) - float(cost_old.cpu())) < 1e-6,
            "iters_equal": int(res_new.iterations) == int(iters_old),
        },
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
