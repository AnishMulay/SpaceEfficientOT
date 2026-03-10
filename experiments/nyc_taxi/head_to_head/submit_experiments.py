#!/usr/bin/env python3
"""
submit_experiments.py
=====================
Single entry point for the NYC Taxi head-to-head experiment.
Run this on the SLURM login node:

    python submit_experiments.py           # submits all jobs
    python submit_experiments.py --dry-run # prints scripts, submits nothing

One independent SLURM job is submitted per (N, method, trial) cell.
All jobs write a single row to a shared CSV file; file-locking ensures
concurrent writes from parallel jobs never corrupt the file.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

# =============================================================================
# >>>  EDIT THIS SECTION to match your cluster  <<<
# =============================================================================

SLURM = {
    "partition":      "gpu",           # partition name on ARC (e.g. "gpu", "volta_gpu")
    "gres":           "gpu:1",         # GPU resource string
    "mem":            "32G",           # RAM per job
    "cpus":           4,               # CPU cores per job
    "time_unscaled":  "06:00:00",      # wall-time for unscaled jobs (slowest)
    "time_scaled":    "02:00:00",      # wall-time for scaled jobs
    "time_exact":     "01:00:00",      # wall-time for exact (scipy) jobs — hard cap
    # Shell lines that activate your Python environment inside each job.
    # Replace 'spefot' with your actual conda env name.
    "env_setup": (
        "source ~/.bashrc\n"
        "conda activate spefot"
    ),
}

# =============================================================================
# >>>  EXPERIMENT GRID  — edit freely  <<<
# =============================================================================

N_VALUES = list(range(1000, 16000, 1000))   # [1000, 2000, ..., 15000] — 15 points
METHODS  = ["unscaled", "scaled", "exact"]
N_TRIALS = 3        # independent trials per (N, method) cell
SEED_BASE = 42      # trial i uses seed SEED_BASE + i

# Shared solver / data settings passed to every job
SOLVER_CFG = {
    "date":          "2014-10-14",
    "speed_mps":     8.0,
    "y_max_meters":  10000.0,
    "k":             512,             # tile size
    "delta":         0.001,           # target delta for unscaled
    "C":             10000.0,         # cost scale (= y_max_meters)
    "fill_policy":   "none",
    "future_only":   True,
}

# Exact jobs run for ALL N values — the 30-min SLURM wall-time is the natural cap.
# Jobs killed by the scheduler are recorded as status="timeout" in the CSV.
# Lower this only if a node cannot physically allocate the cost matrix.
MAX_EXACT_N = 15000   # effectively disabled

# =============================================================================
# Paths  (all relative to this file; nothing hard-coded)
# =============================================================================

HERE         = Path(__file__).resolve().parent
RESULTS_DIR  = HERE / "results"
RAW_CSV      = RESULTS_DIR / "raw_results.csv"
MANIFEST_CSV = RESULTS_DIR / "submitted_jobs.csv"
LOGS_DIR     = RESULTS_DIR / "logs"
RUN_SCRIPT   = HERE / "run_experiment.py"

# =============================================================================
# Internal helpers
# =============================================================================

def _wall_time(method: str) -> str:
    return {"unscaled": SLURM["time_unscaled"],
            "scaled":   SLURM["time_scaled"],
            "exact":    SLURM["time_exact"]}[method]


def _make_script(n: int, method: str, trial: int, seed: int) -> str:
    """Return a complete SLURM batch script as a string (no temp files needed)."""
    job_name = f"spefot_{method}_n{n}_t{trial}"
    out_log  = LOGS_DIR / f"{job_name}_%j.out"
    err_log  = LOGS_DIR / f"{job_name}_%j.err"
    python   = sys.executable   # same interpreter that runs this script

    args = (
        f"--n {n} "
        f"--method {method} "
        f"--trial {trial} "
        f"--seed {seed} "
        f"--results-csv {RAW_CSV} "
        f"--date {SOLVER_CFG['date']} "
        f"--speed-mps {SOLVER_CFG['speed_mps']} "
        f"--y-max-meters {SOLVER_CFG['y_max_meters']} "
        f"--k {SOLVER_CFG['k']} "
        f"--delta {SOLVER_CFG['delta']} "
        f"--C {SOLVER_CFG['C']} "
        f"--fill-policy {SOLVER_CFG['fill_policy']} "
        f"--max-exact-n {MAX_EXACT_N} "
        + ("--future-only" if SOLVER_CFG["future_only"] else "--no-future-only")
    )

    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={SLURM['partition']}\n"
        f"#SBATCH --gres={SLURM['gres']}\n"
        f"#SBATCH --mem={SLURM['mem']}\n"
        f"#SBATCH --cpus-per-task={SLURM['cpus']}\n"
        f"#SBATCH --time={_wall_time(method)}\n"
        f"#SBATCH --output={out_log}\n"
        f"#SBATCH --error={err_log}\n"
        f"\n"
        f"{SLURM['env_setup']}\n"
        f"\n"
        f"echo \"[job start] $(date)  node=$SLURMD_NODENAME  gpu=$CUDA_VISIBLE_DEVICES\"\n"
        f"\n"
        f"{python} {RUN_SCRIPT} {args}\n"
        f"\n"
        f"echo \"[job end]   $(date)  exit=$?\"\n"
    )


def _sbatch(script: str, dry_run: bool) -> str | None:
    """Pipe script to sbatch; return job_id string, or None on dry-run/failure."""
    if dry_run:
        print(script)
        print("─" * 60)
        return None
    r = subprocess.run(["sbatch", "--parsable"],
                       input=script, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] sbatch returned {r.returncode}:\n{r.stderr}", file=sys.stderr)
        return None
    return r.stdout.strip()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit all head-to-head NYC Taxi SLURM jobs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print job scripts without submitting anything")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    grid = [(n, m, t)
            for n in N_VALUES
            for m in METHODS
            for t in range(N_TRIALS)]

    total   = len(grid)
    print(f"Grid : {len(N_VALUES)} N-values × {len(METHODS)} methods × "
          f"{N_TRIALS} trials = {total} cells")
    print(f"CSV  : {RAW_CSV}")
    print(f"Logs : {LOGS_DIR}")
    print("Mode : DRY RUN\n" if args.dry_run else "")

    manifest: list[dict] = []
    submitted = skipped = failed = 0

    for n, method, trial in grid:
        seed = SEED_BASE + trial

        # Skip exact jobs that would OOM on the CPU
        if method == "exact" and n > MAX_EXACT_N:
            tag = f"SKIP (n>{MAX_EXACT_N})"
            print(f"  {tag:<14}  n={n:>6}  {method:<10}  trial={trial}")
            skipped += 1
            manifest.append({"submitted_at": dt.datetime.now().isoformat("T", "seconds"),
                              "job_id": "", "n": n, "method": method,
                              "trial": trial, "seed": seed, "status": "skipped"})
            continue

        script = _make_script(n, method, trial, seed)
        job_id = _sbatch(script, args.dry_run)

        if args.dry_run:
            status = "dry_run"
        elif job_id:
            status = "submitted"
            submitted += 1
        else:
            status = "failed"
            failed += 1

        if not args.dry_run:
            print(f"  {status:<14}  n={n:>6}  {method:<10}  "
                  f"trial={trial}  job_id={job_id or '—'}")
        manifest.append({"submitted_at": dt.datetime.now().isoformat("T", "seconds"),
                         "job_id": job_id or "", "n": n, "method": method,
                         "trial": trial, "seed": seed, "status": status})

    # Write manifest (append-safe)
    if manifest and not args.dry_run:
        write_hdr = not MANIFEST_CSV.exists()
        with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            if write_hdr:
                w.writeheader()
            w.writerows(manifest)
        print(f"\nManifest → {MANIFEST_CSV}")

    print(f"\nSubmitted={submitted}  Skipped={skipped}  "
          f"Failed={failed}  Total={total}")


if __name__ == "__main__":
    main()
