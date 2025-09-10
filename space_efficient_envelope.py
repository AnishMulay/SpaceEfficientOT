#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from scipy.spatial.distance import cdist

# Import the provided algorithms
from matching import matching_torch_v1
from spef_matching import spef_matching_2

# ----------------------------
# Helpers
# ----------------------------

def now_ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def device_from_arg(arg):
    if arg is None or arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)

def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def reset_peak_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats(device)

def peak_mem_allocated(device):
    if device.type == "cuda":
        return torch.cuda.memory.max_memory_allocated(device)
    return 0

def gen_points(n, d, device, seed):
    # Keep parity with existing test: points in [0,1]^d
    torch.manual_seed(seed)
    np.random.seed(seed)
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)
    return xa, xb

def run_algorithm_A(n, d, delta, device, seed):
    """
    Baseline (matching_torch_v1) run:
    - Builds W on CPU via cdist (dense n x n).
    - Times only the algorithm call, mirroring the existing test.
    """
    xa, xb = gen_points(n, d, device, seed)
    xa_cpu = xa.detach().cpu().numpy()
    xb_cpu = xb.detach().cpu().numpy()

    # Build dense cost matrix W on CPU (float32)
    W = cdist(xb_cpu, xa_cpu, 'sqeuclidean').astype(np.float32)

    # Prepare tensors and constants
    W_tensor = torch.tensor(W, device=device, dtype=torch.float32)
    C = torch.tensor(float(W.max()), device=device, dtype=torch.float32)

    # Measure
    reset_peak_if_cuda(device)
    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    Mb, yA, yB, matching_cost, iteration = matching_torch_v1(W_tensor, C, delta, device)
    synchronize_if_cuda(device)
    t1 = time.perf_counter()

    runtime_s = t1 - t0
    peak_bytes = peak_mem_allocated(device)
    iterations = int(iteration) if not isinstance(iteration, int) else iteration
    throughput = n / runtime_s if runtime_s > 0 else float("inf")

    # Free references promptly
    del W_tensor, C, Mb, yA, yB, xa, xb

    return {
        "success": True,
        "runtime_s": runtime_s,
        "iterations": iterations,
        "throughput_pts_per_s": throughput,
        "peak_gpu_mem_bytes": peak_bytes,
        "error_type": "",
        "error_msg": ""
    }

def run_algorithm_B(n, d, k, delta, device, seed):
    """
    Space-efficient (spef_matching_2) run:
    - Does not build W; uses C = d (max squared distance in [0,1]^d is <= d).
    """
    xa, xb = gen_points(n, d, device, seed)
    C_val = float(d)  # analytical upper bound for squared Euclidean in unit hypercube
    C_tensor = torch.tensor(C_val, device=device, dtype=torch.float32)

    reset_peak_if_cuda(device)
    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    Mb, yA, yB, matching_cost, iteration = spef_matching_2(xa, xb, C_tensor, k, delta, device)
    synchronize_if_cuda(device)
    t1 = time.perf_counter()

    runtime_s = t1 - t0
    peak_bytes = peak_mem_allocated(device)
    iterations = int(iteration) if not isinstance(iteration, int) else iteration
    throughput = n / runtime_s if runtime_s > 0 else float("inf")

    del Mb, yA, yB, xa, xb, C_tensor
    return {
        "success": True,
        "runtime_s": runtime_s,
        "iterations": iterations,
        "throughput_pts_per_s": throughput,
        "peak_gpu_mem_bytes": peak_bytes,
        "error_type": "",
        "error_msg": ""
    }

def try_run_A(n, d, delta, device, seed):
    try:
        return n, run_algorithm_A(n, d, delta, device, seed)
    except Exception as e:
        return n, {
            "success": False,
            "runtime_s": None,
            "iterations": None,
            "throughput_pts_per_s": None,
            "peak_gpu_mem_bytes": peak_mem_allocated(device),
            "error_type": e.__class__.__name__,
            "error_msg": str(e)[:500],
        }

def try_run_B(n, d, k, delta, device, seed):
    try:
        return n, run_algorithm_B(n, d, k, delta, device, seed)
    except Exception as e:
        return n, {
            "success": False,
            "runtime_s": None,
            "iterations": None,
            "throughput_pts_per_s": None,
            "peak_gpu_mem_bytes": peak_mem_allocated(device),
            "error_type": e.__class__.__name__,
            "error_msg": str(e)[:500],
        }

def write_csv_header(path):
    must_write = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if must_write:
        w.writerow([
            "timestamp", "algo", "n", "d", "k", "delta", "device",
            "runtime_s", "iterations", "throughput_pts_per_s", "peak_gpu_mem_bytes",
            "success", "error_type", "error_msg"
        ])
    return f, w

def write_envelope_header(path):
    must_write = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if must_write:
        w.writerow(["timestamp", "algo", "d", "k", "max_n_success"])
    return f, w

def sweep_envelope_A(args, device, out_csv_writer):
    # Arithmetic sweep with refinement
    d = args.dims  # default single d
    last_success_n = 0
    first_failed_n = None

    # Coarse sweep
    n = args.start_A
    while n <= args.max_A:
        ts = now_ts()
        n_val, result = try_run_A(n, d, args.delta, device, args.seed)
        out_csv_writer.writerow([
            ts, "A", n_val, d, "", args.delta, device.type,
            result["runtime_s"], result["iterations"], result["throughput_pts_per_s"],
            result["peak_gpu_mem_bytes"], result["success"], result["error_type"], result["error_msg"]
        ])
        if result["success"]:
            last_success_n = n
            n += args.step_A
        else:
            first_failed_n = n
            break

    # Refinement sweep
    if first_failed_n is not None:
        for n2 in range(last_success_n + args.refine_A, first_failed_n, args.refine_A):
            ts = now_ts()
            n_val, result = try_run_A(n2, d, args.delta, device, args.seed)
            out_csv_writer.writerow([
                ts, "A", n_val, d, "", args.delta, device.type,
                result["runtime_s"], result["iterations"], result["throughput_pts_per_s"],
                result["peak_gpu_mem_bytes"], result["success"], result["error_type"], result["error_msg"]
            ])
            if result["success"]:
                last_success_n = n2
            else:
                break

    return d, last_success_n

def sweep_envelope_B(args, device, out_csv_writer):
    envelopes = []
    d = args.dims
    for k in args.ks:
        last_success_n = 0
        first_failed_n = None

        # Coarse sweep
        n = args.start_B
        while n <= args.max_B:
            ts = now_ts()
            n_val, result = try_run_B(n, d, k, args.delta, device, args.seed)
            out_csv_writer.writerow([
                ts, "B", n_val, d, k, args.delta, device.type,
                result["runtime_s"], result["iterations"], result["throughput_pts_per_s"],
                result["peak_gpu_mem_bytes"], result["success"], result["error_type"], result["error_msg"]
            ])
            if result["success"]:
                last_success_n = n
                n += args.step_B
            else:
                first_failed_n = n
                break

        # Refinement sweep
        if first_failed_n is not None:
            for n2 in range(last_success_n + args.refine_B, first_failed_n, args.refine_B):
                ts = now_ts()
                n_val, result = try_run_B(n2, d, k, args.delta, device, args.seed)
                out_csv_writer.writerow([
                    ts, "B", n_val, d, k, args.delta, device.type,
                    result["runtime_s"], result["iterations"], result["throughput_pts_per_s"],
                    result["peak_gpu_mem_bytes"], result["success"], result["error_type"], result["error_msg"]
                ])
                if result["success"]:
                    last_success_n = n2
                else:
                    break

        envelopes.append((d, k, last_success_n))
    return envelopes

def plot_results(output_dir, results_csv, envelope_csv):
    # Use only matplotlib to avoid extra deps
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Load results
    rows = []
    with open(results_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Group by algo/k
    def to_float(x):
        return None if x in (None, "", "None") else float(x)

    series = defaultdict(list)
    for row in rows:
        algo = row["algo"]
        d = row["d"]
        k = row["k"]
        key = (algo, d, k if k != "" else "NA")
        n = int(row["n"])
        success = row["success"] == "True"
        runtime = to_float(row["runtime_s"])
        peak = to_float(row["peak_gpu_mem_bytes"])
        thr = to_float(row["throughput_pts_per_s"])
        if success:
            series[key].append((n, runtime, peak, thr))

    # Sort each series by n
    for key in series.keys():
        series[key] = sorted(series[key], key=lambda t: t)

    # Runtime vs n
    plt.figure(figsize=(8,5))
    for key, vals in series.items():
        n_list = [v[0] for v in vals]
        rt_list = [v[1] for v in vals]
        label = f"Algo {key} (d={key[1]}, k={key[2]})"
        plt.plot(n_list, rt_list, marker='o', label=label)
    plt.xlabel("n")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs n")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_n.png"))
    plt.close()

    # Peak GPU memory vs n
    plt.figure(figsize=(8,5))
    for key, vals in series.items():
        n_list = [v[0] for v in vals]
        pk_list = [v[2]/(1024**2) for v in vals]  # MB
        label = f"Algo {key} (d={key[1]}, k={key[2]})"
        plt.plot(n_list, pk_list, marker='o', label=label)
    plt.xlabel("n")
    plt.ylabel("Peak GPU memory (MB)")
    plt.title("Peak GPU memory vs n")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peak_memory_vs_n.png"))
    plt.close()

    # Throughput vs n
    plt.figure(figsize=(8,5))
    for key, vals in series.items():
        n_list = [v[0] for v in vals]
        thr_list = [v[3] for v in vals]
        label = f"Algo {key} (d={key[1]}, k={key[2]})"
        plt.plot(n_list, thr_list, marker='o', label=label)
    plt.xlabel("n")
    plt.ylabel("Throughput (points/s)")
    plt.title("Throughput vs n")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_vs_n.png"))
    plt.close()

    # Envelope bar chart
    env = []
    with open(envelope_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            env.append(row)

    # Separate A and B
    A_vals = [int(r["max_n_success"]) for r in env if r["algo"] == "A"]
    B_vals = [(r["k"], int(r["max_n_success"])) for r in env if r["algo"] == "B"]
    plt.figure(figsize=(8,5))
    labels = []
    heights = []

    if len(A_vals) == 1:
        labels.append("A")
        heights.append(A_vals[0])

    for k, val in B_vals:
        labels.append(f"B(k={k})")
        heights.append(val)

    xpos = np.arange(len(labels))
    plt.bar(xpos, heights)
    plt.xticks(xpos, labels)
    plt.ylabel("Max n (success)")
    plt.title("Scalability envelope (max n without failure)")
    for x, h in zip(xpos, heights):
        plt.text(x, h, f"{h}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scalability_envelope.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive comparison: peak GPU memory, time, iterations, throughput, and scalability envelope.")
    parser.add_argument("--device", type=str, default="auto", help="cuda|cpu|auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.01, help="Scaling factor (kept fixed by default).")
    parser.add_argument("--dims", type=str, default="2", help="Comma-separated list of d; default '2' to mirror current test.")
    # Envelope parameters for A
    parser.add_argument("--start-A", dest="start_A", type=int, default=2000)
    parser.add_argument("--step-A", dest="step_A", type=int, default=2000)
    parser.add_argument("--refine-A", dest="refine_A", type=int, default=500)
    parser.add_argument("--max-A", dest="max_A", type=int, default=200000)
    # Envelope parameters for B
    parser.add_argument("--start-B", dest="start_B", type=int, default=5000)
    parser.add_argument("--step-B", dest="step_B", type=int, default=5000)
    parser.add_argument("--refine-B", dest="refine_B", type=int, default=1000)
    parser.add_argument("--max-B", dest="max_B", type=int, default=500000)
    parser.add_argument("--ks", type=str, default="250,500,750,1000", help="Tile sizes for B.")
    parser.add_argument("--out", type=str, default="comparison_results", help="Output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    # Parse lists
    args.dims = [int(x.strip()) for x in args.dims.split(",") if x.strip()]
    args.ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    device = device_from_arg(args.device)
    out_dir = ensure_dir(args.out)
    results_csv = os.path.join(out_dir, "results.csv")
    envelope_csv = os.path.join(out_dir, "envelope_summary.csv")

    # Open CSV writers
    f_results, w_results = write_csv_header(results_csv)
    f_env, w_env = write_envelope_header(envelope_csv)

    try:
        # The plan uses a single d by default; extend via --dims if needed.
        # Sweep A once (using the first d)
        d_A, max_n_A = sweep_envelope_A(args, device, w_results)
        w_env.writerow([now_ts(), "A", d_A, "", max_n_A])
        f_env.flush(); f_results.flush()

        # Sweep B for each k at the same d
        envelopes_B = sweep_envelope_B(args, device, w_results)
        for (d_B, k_B, max_n_B) in envelopes_B:
            w_env.writerow([now_ts(), "B", d_B, k_B, max_n_B])
        f_env.flush(); f_results.flush()

        # Plots
        if not args.no_plots:
            plot_results(out_dir, results_csv, envelope_csv)

        print(f"Done. CSV saved under: {out_dir}")
        if not args.no_plots:
            print("Plots saved:")
            print(" - runtime_vs_n.png")
            print(" - peak_memory_vs_n.png")
            print(" - throughput_vs_n.png")
            print(" - scalability_envelope.png")

    finally:
        f_results.close()
        f_env.close()

if __name__ == "__main__":
    main()
