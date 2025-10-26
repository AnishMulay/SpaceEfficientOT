#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
import sys
import time
from pathlib import Path

import torch

# Allow running without an install when invoked from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch.set_float32_matmul_precision("high")

from spef_ot import match  # noqa: E402
from spef_ot.kernels.euclidean_sq import SquaredEuclideanKernel  # noqa: E402


class InstrumentedEuclideanKernel(SquaredEuclideanKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tile_sizes = defaultdict(int)
        self._cpu_args = set()

    def compute_slack_tile(self, idxB, state, workspace, out=None):
        current_k = int(idxB.numel())
        self._tile_sizes[current_k] += 1

        device_map = {
            "idxB": idxB.device,
            "xA": workspace.xA.device,
            "xB": workspace.xB.device,
            "xAT": workspace.xAT.device,
            "xa2_cached": workspace.xa2_cached.device,
            "scale": workspace.scale.device,
            "yA": state.yA.device,
            "yB": state.yB.device,
        }
        for name, device in device_map.items():
            if device.type != "cuda":
                self._cpu_args.add(f"{name}:{device.type}")

        return super().compute_slack_tile(idxB, state, workspace, out=out)

    def finalize(self, problem, state, workspace):
        metrics = super().finalize(problem, state, workspace) or {}
        diagnostics = {
            "unique_tile_sizes": len(self._tile_sizes),
            "tile_sizes": {str(k): v for k, v in sorted(self._tile_sizes.items())},
            "cpu_args": sorted(self._cpu_args),
        }
        combined = dict(metrics)
        combined["debug"] = diagnostics
        return combined


def load_archive_solver():
    archive_dir = REPO_ROOT / "Archive"
    module_path = archive_dir / "spef_matching_v2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Archive solver not found at {module_path}")

    if str(archive_dir) not in sys.path:
        sys.path.append(str(archive_dir))

    spec = importlib.util.spec_from_file_location("archive_spef_matching_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.spef_matching_2


def generate_points(n: int, d: int, device: torch.device, seed: int, cache: bool = True):
    datasets_dir = Path(__file__).resolve().parent / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    cache_path = datasets_dir / f"points_n{n}_d{d}_seed{seed}.pt"
    if cache and cache_path.exists():
        data = torch.load(cache_path, map_location=device)
        return data["xa"].to(device), data["xb"].to(device)
    torch.manual_seed(seed)
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)
    if cache:
        torch.save({"xa": xa.cpu(), "xb": xb.cpu()}, cache_path)
    return xa, xb


def compute_C(xa: torch.Tensor, xb: torch.Tensor) -> float:
    n = xa.shape[0]
    rand_idx = torch.randint(0, n, (1,), device=xa.device)
    b_pt = xb[rand_idx]
    dists = torch.cdist(b_pt, xa)
    return float((dists.max() ** 2).item())


def _run_match(xa, xb, C, k, delta, seed, device, kernel):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = match(
        xa,
        xb,
        kernel=kernel,
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def _run_archive(xa, xb, C, k, delta, seed, device, solver):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Mb, yA, yB, cost, iterations = solver(
        xa.clone(), xb.clone(), C, k, delta, device=device, seed=seed
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (Mb, yA, yB, cost, iterations), t1 - t0


def main():
    p = argparse.ArgumentParser(description="Compare new experiment vs archived Euclidean solver")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p.add_argument("--debug", action="store_true", help="Collect tile/device diagnostics for new solver")
    args = p.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Comparing new vs archive (Euclidean)")
    print(f"Parameters: n={args.n}, d={args.d}, k={args.k}, delta={args.delta}, seed={args.seed}, device={device}")
    print("=" * 60)

    # Same data and C for both runs
    xa, xb = generate_points(args.n, args.d, device, args.seed, cache=True)
    C = compute_C(xa, xb)

    kernel_spec = InstrumentedEuclideanKernel if args.debug else "euclidean_sq"

    # Warm-up both solvers
    _, warmup_new = _run_match(xa, xb, C, args.k, args.delta, args.seed, device, kernel_spec)
    archive_solver = load_archive_solver()
    _, warmup_old = _run_archive(xa, xb, C, args.k, args.delta, args.seed, device, archive_solver)

    # Timed runs
    new_res, new_runtime = _run_match(
        xa,
        xb,
        C,
        args.k,
        args.delta,
        args.seed,
        device,
        kernel_spec,
    )

    archive_out, archive_runtime = _run_archive(
        xa,
        xb,
        C,
        args.k,
        args.delta,
        args.seed,
        device,
        archive_solver,
    )

    Mb_old, yA_old, yB_old, cost_old, iter_old = archive_out

    result = {
        "params": {
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "delta": args.delta,
            "seed": args.seed,
            "device": str(device),
            "C": C,
        },
        "new": {
            "warmup_runtime_sec": warmup_new,
            "runtime_sec": new_runtime,
            "cost": float(new_res.matching_cost),
            "iterations": int(new_res.iterations),
        },
        "archive": {
            "warmup_runtime_sec": warmup_old,
            "runtime_sec": archive_runtime,
            "cost": float(cost_old.cpu()),
            "iterations": int(iter_old),
        },
        "parity": {
            "cost_equal": abs(float(new_res.matching_cost) - float(cost_old.cpu())) < 1e-6,
            "iters_equal": int(new_res.iterations) == int(iter_old),
        },
    }

    if args.debug:
        result["new"]["debug"] = new_res.metrics.get("debug", {})

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
