#!/usr/bin/env python3
"""Generate all JSON config files and SLURM sbatch scripts for the exact-comparison experiment.

Run this locally (or on the login node) once to populate:
  batch/configs/   — one JSON per (solver, n, seed)
  batch/scripts/   — one .sbatch per config

The generated SLURM template intentionally mirrors the working
`nyc_taxi_speed_euclidean` jobs by default:
  - `--cpus-per-task=16`
  - no explicit `--mem` request
  - same partition / conda activation structure

Memory requests can still be added explicitly via CLI flags if your cluster
needs them.

Usage
-----
  python experiments/nyc_taxi_exact_comparison/batch/generate_configs.py \
      --input /home/amulay2/SpaceEfficientOT/experiments/nyc_taxi_speed_haversine/data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv \
      --partition rtx2060super \
      --conda-env spefenv

All other parameters (n grid, seeds, fixed hyperparams) are hard-coded below and
match the agreed experimental design. Edit the GRID / FIXED sections if needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Experimental grid (edit here to change the sweep)
# ---------------------------------------------------------------------------

N_VALUES = [1000, 2000, 5000, 10000, 15000, 20000]

SEEDS = [1, 2, 3]

SOLVERS = ["spef_unscaled", "spef_scaled", "ortools"]

# Stopping condition for SPEF solvers: ~5 % of n
STOPPING_CONDITION_MAP = {
    1000:  50,
    2000:  100,
    5000:  250,
    10000: 500,
    15000: 750,
    20000: 1000,
}

# Fixed hyperparameters shared by all runs
FIXED = {
    "date":         "2014-10-14",
    "random_sample": True,
    "speed_mps":    8.0,
    "y_max_meters": 10000.0,
    "future_only":  True,
    "fill_policy":  "none",
    "device":       "cuda",
    # SPEF-specific
    "delta":        0.001,
    "C":            100000.0,
    "k":            512,
}

# Per-solver SLURM resource spec.
# Defaults are aligned with the working Euclidean-speed batch scripts: 16 CPUs
# and no explicit --mem line unless the user asks for one.
SLURM_SPEC = {
    "spef_unscaled": {"time": "01:00:00", "cpus": 16, "mem": None},
    "spef_scaled":   {"time": "01:00:00", "cpus": 16, "mem": None},
    "ortools":       {"time": "02:00:00", "cpus": 16, "mem": None},
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BATCH_DIR = Path(__file__).resolve().parent
EXP_DIR   = BATCH_DIR.parent
REPO_ROOT = EXP_DIR.parents[1]

CONFIGS_DIR = BATCH_DIR / "configs"
SCRIPTS_DIR = BATCH_DIR / "scripts"
RESULTS_DIR = BATCH_DIR / "results"
LOGS_DIR    = BATCH_DIR / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_name(solver: str, n: int, seed: int) -> str:
    return f"config_{solver}_n{n}_seed{seed}"


def _script_name(solver: str, n: int, seed: int) -> str:
    return f"run_{solver}_n{n}_seed{seed}"


def _make_config(solver: str, n: int, seed: int, input_path: str) -> dict:
    cfg: dict = {
        "solver":  solver,
        "input":   input_path,
        "n":       n,
        "seed":    seed,
        **FIXED,
    }
    # SPEF-only keys: stopping_condition
    if solver in ("spef_unscaled", "spef_scaled"):
        cfg["stopping_condition"] = STOPPING_CONDITION_MAP[n]
    else:
        # Remove GPU device for ortools (CPU-only)
        cfg["device"] = "cpu"
        # Remove SPEF-only keys to keep the config clean
        for key in ("delta", "C", "k", "stopping_condition", "fill_policy"):
            cfg.pop(key, None)
    return cfg


def _make_sbatch(
    solver: str,
    n: int,
    seed: int,
    config_rel: str,
    partition: str,
    conda_env: str,
) -> str:
    spec = SLURM_SPEC[solver]
    job_name = f"ec_{solver}_n{n}_s{seed}"
    timeout_sec = {
        "spef_unscaled": 3600,
        "spef_scaled":   3600,
        "ortools":       7200,
    }[solver]

    # Relative paths from repo root (so they work regardless of submission dir)
    config_rel_path  = f"experiments/nyc_taxi_exact_comparison/batch/configs/{config_rel}.json"
    results_rel_path = "experiments/nyc_taxi_exact_comparison/batch/results"
    run_one_rel_path = "experiments/nyc_taxi_exact_comparison/batch/run_one.py"
    logs_rel_path    = "experiments/nyc_taxi_exact_comparison/batch/logs"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {logs_rel_path}/%x-%j.out",
        f"#SBATCH -e {logs_rel_path}/%x-%j.err",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --cpus-per-task={spec['cpus']}",
        f"#SBATCH -t {spec['time']}",
        f"#SBATCH -p {partition}",
        "",
        "export PYTHONUNBUFFERED=1",
        "",
        'cd "${SLURM_SUBMIT_DIR:-$PWD}"',
        "",
        'if [ -f "$HOME/.bashrc" ]; then',
        '  source "$HOME/.bashrc"',
        "fi",
        "",
        f"PATH=/usr/bin:/bin:$PATH conda activate {conda_env}",
        "",
        f"python -u {run_one_rel_path} \\",
        f"  --config {config_rel_path} \\",
        f"  --results-dir {results_rel_path} \\",
        f"  --timeout-sec {timeout_sec} \\",
        "  --print-cmd",
        "",
        "conda deactivate || true",
    ]
    if spec["mem"]:
        lines.insert(7, f"#SBATCH --mem={spec['mem']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate configs + sbatch scripts for the exact-comparison experiment"
    )
    ap.add_argument(
        "--input", required=True,
        help="Absolute path to the NYC taxi CSV on the cluster "
             "(e.g. /home/amulay2/SpaceEfficientOT/experiments/nyc_taxi_speed_haversine/data/...csv)"
    )
    ap.add_argument("--partition", default="rtx2060super", help="SLURM partition (default: rtx2060super)")
    ap.add_argument("--conda-env", default="spefenv", help="Conda environment name (default: spefenv)")
    ap.add_argument("--spef-mem", default=None,
                    help="Optional --mem value for spef_unscaled and spef_scaled jobs (default: omit)")
    ap.add_argument("--ortools-mem", default=None,
                    help="Optional --mem value for ortools jobs (default: omit)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    args = ap.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    SLURM_SPEC["spef_unscaled"]["mem"] = args.spef_mem
    SLURM_SPEC["spef_scaled"]["mem"] = args.spef_mem
    SLURM_SPEC["ortools"]["mem"] = args.ortools_mem

    total = 0
    for solver in SOLVERS:
        for n in N_VALUES:
            for seed in SEEDS:
                cname = _config_name(solver, n, seed)
                sname = _script_name(solver, n, seed)

                cfg = _make_config(solver, n, seed, args.input)
                sbatch = _make_sbatch(solver, n, seed, cname, args.partition, args.conda_env)

                cfg_path    = CONFIGS_DIR / f"{cname}.json"
                script_path = SCRIPTS_DIR / f"{sname}.sbatch"

                if args.dry_run:
                    print(f"[DRY] {cfg_path.name}  →  {script_path.name}")
                else:
                    with cfg_path.open("w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
                    with script_path.open("w", encoding="utf-8") as f:
                        f.write(sbatch)
                    script_path.chmod(0o755)

                total += 1

    if not args.dry_run:
        print(f"Generated {total} config + script pairs.")
        print(f"  configs  → {CONFIGS_DIR}")
        print(f"  scripts  → {SCRIPTS_DIR}")
    else:
        print(f"[DRY RUN] Would generate {total} config + script pairs.")


if __name__ == "__main__":
    main()
