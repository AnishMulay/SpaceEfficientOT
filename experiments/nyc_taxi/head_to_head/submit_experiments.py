#!/usr/bin/env python3
"""
submit_experiments.py
=====================
Generates one .sh SLURM script per (N, method, trial) cell, writes them to
batch/scripts/, then submits each with sbatch.

Run on the login node:
    python submit_experiments.py           # generate + submit all jobs
    python submit_experiments.py --dry-run # generate scripts, do NOT submit
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# SLURM settings  (matched exactly to your working cluster config)
# =============================================================================

PARTITION = "rtx2060super"
CPUS      = 16

WALL_TIME = {
    "unscaled": "06:00:00",
    "scaled":   "02:00:00",
    "exact":    "01:00:00",
}

# =============================================================================
# Experiment grid
# =============================================================================

N_VALUES  = list(range(1000, 16000, 1000))   # [1000, 2000, ..., 15000]
METHODS   = ["unscaled", "scaled", "exact"]
N_TRIALS  = 3
SEED_BASE = 42    # trial i  ->  seed = SEED_BASE + i

# Shared solver settings
DATE         = "2014-10-14"
SPEED_MPS    = 8.0
Y_MAX_METERS = 10000.0
TILE_K       = 512
DELTA        = 0.01
C_VALUE      = 100000.0
FILL_POLICY  = "none"

# Exact jobs are attempted for ALL N -- the 1-hour wall-time is the natural cap.
# Set lower to skip large exact jobs if needed.
MAX_EXACT_N  = 15000

# =============================================================================
# Paths  (all relative to this file's location)
# =============================================================================

HERE        = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE / "batch" / "scripts"
RESULTS_DIR = HERE / "batch" / "results"
LOGS_DIR    = HERE / "batch" / "logs"
RUN_SCRIPT  = HERE / "run_experiment.py"
MANIFEST    = HERE / "batch" / "submitted_jobs.csv"

# =============================================================================
# Script template  (mirrors your working generate_batch_jobs.py exactly)
# =============================================================================

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH -J spefot_{method}_n{n}_t{trial}
#SBATCH -o {logs_dir}/spefot_{method}_n{n}_t{trial}-%j.out
#SBATCH -e {logs_dir}/spefot_{method}_n{n}_t{trial}-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task={cpus}
#SBATCH -t {wall_time}
#SBATCH -p {partition}

export PYTHONUNBUFFERED=1

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate spefenv

echo "Starting: method={method}  n={n}  trial={trial}"

python -u {run_script} \\
  --n {n} \\
  --method {method} \\
  --trial {trial} \\
  --seed {seed} \\
  --results-csv {results_csv} \\
  --date {date} \\
  --speed-mps {speed_mps} \\
  --y-max-meters {y_max_meters} \\
  --k {k} \\
  --delta {delta} \\
  --C {C} \\
  --fill-policy {fill_policy} \\
  --max-exact-n {max_exact_n} \\
  --future-only

echo "Done:  method={method}  n={n}  trial={trial}  exit=$?"

conda deactivate || true
"""

# =============================================================================
# Helpers
# =============================================================================

def _make_script(n: int, method: str, trial: int, seed: int) -> tuple[str, Path]:
    """Return (script_content, script_path). Each job writes its own CSV."""
    results_csv = RESULTS_DIR / f"results_{method}_n{n}_t{trial}.csv"
    script_path = SCRIPTS_DIR / f"job_{method}_n{n}_t{trial}.sh"

    content = SLURM_TEMPLATE.format(
        method       = method,
        n            = n,
        trial        = trial,
        seed         = seed,
        logs_dir     = LOGS_DIR,
        cpus         = CPUS,
        wall_time    = WALL_TIME[method],
        partition    = PARTITION,
        run_script   = RUN_SCRIPT,
        results_csv  = results_csv,
        date         = DATE,
        speed_mps    = SPEED_MPS,
        y_max_meters = Y_MAX_METERS,
        k            = TILE_K,
        delta        = DELTA,
        C            = C_VALUE,
        fill_policy  = FILL_POLICY,
        max_exact_n  = MAX_EXACT_N,
    )
    return content, script_path


def _submit(script_path: Path, dry_run: bool) -> str | None:
    """Submit script_path via sbatch; return job_id string or None."""
    if dry_run:
        return None
    r = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  [ERROR] sbatch failed: {r.stderr.strip()}", file=sys.stderr)
        return None
    # sbatch prints "Submitted batch job 12345"
    return r.stdout.strip().split()[-1]


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Write scripts but do not call sbatch")
    args = ap.parse_args()

    for d in (SCRIPTS_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    grid = [
        (n, method, trial)
        for n      in N_VALUES
        for method in METHODS
        for trial  in range(N_TRIALS)
    ]

    print(f"Grid : {len(N_VALUES)} N-values x {len(METHODS)} methods x "
          f"{N_TRIALS} trials = {len(grid)} jobs")
    print(f"Mode : {'DRY RUN -- scripts written, nothing submitted' if args.dry_run else 'SUBMITTING'}\n")

    manifest_rows = []
    submitted = failed = 0

    for n, method, trial in grid:
        seed    = SEED_BASE + trial
        content, script_path = _make_script(n, method, trial, seed)
        script_path.write_text(content)
        script_path.chmod(0o755)

        job_id = _submit(script_path, args.dry_run)

        if args.dry_run:
            status = "dry_run"
            print(f"  wrote  {script_path.name}")
        elif job_id:
            status = "submitted"
            submitted += 1
            print(f"  submitted  {script_path.name}  ->  job {job_id}")
        else:
            status = "failed"
            failed += 1
            print(f"  FAILED     {script_path.name}")

        manifest_rows.append({
            "submitted_at": dt.datetime.now().isoformat("T", "seconds"),
            "job_id":       job_id or "",
            "n":            n,
            "method":       method,
            "trial":        trial,
            "seed":         seed,
            "status":       status,
            "script":       str(script_path),
        })

    if not args.dry_run:
        write_hdr = not MANIFEST.exists()
        with MANIFEST.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            if write_hdr:
                w.writeheader()
            w.writerows(manifest_rows)
        print(f"\nManifest -> {MANIFEST}")

    print(f"\nSubmitted={submitted}  Failed={failed}  Total={len(grid)}")


if __name__ == "__main__":
    main()