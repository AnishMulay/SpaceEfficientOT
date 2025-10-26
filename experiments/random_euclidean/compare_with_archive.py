#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

# Allow running without an install when invoked from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spef_ot import match  # noqa: E402


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


def main():
    p = argparse.ArgumentParser(description="Compare new experiment vs archived Euclidean solver")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
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

    # New solver run
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    new_res = match(
        xa,
        xb,
        kernel="euclidean_sq",
        C=C,
        k=args.k,
        delta=args.delta,
        device=device,
        seed=args.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Archive solver run
    archive_solver = load_archive_solver()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    Mb_old, yA_old, yB_old, cost_old, iter_old = archive_solver(
        xa.clone(), xb.clone(), C, args.k, args.delta, device=device, seed=args.seed
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

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
            "runtime_sec": t1 - t0,
            "cost": float(new_res.matching_cost),
            "iterations": int(new_res.iterations),
        },
        "archive": {
            "runtime_sec": t3 - t2,
            "cost": float(cost_old.cpu()),
            "iterations": int(iter_old),
        },
        "parity": {
            "cost_equal": abs(float(new_res.matching_cost) - float(cost_old.cpu())) < 1e-6,
            "iters_equal": int(new_res.iterations) == int(iter_old),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

