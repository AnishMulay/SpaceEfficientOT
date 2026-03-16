#!/usr/bin/env python3
"""Generate configs + sbatch scripts for the combined NYC taxi comparison jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

N_VALUES = [1000, 2000, 5000, 10000, 15000, 20000]
SEEDS = [1, 2, 3]

STOPPING_CONDITION_MAP = {
    1000: 50,
    2000: 100,
    5000: 250,
    10000: 500,
    15000: 750,
    20000: 1000,
}

FIXED = {
    "solver": "combined",
    "date": "2014-10-14",
    "random_sample": True,
    "speed_mps": 8.0,
    "y_max_meters": 10000.0,
    "future_only": True,
    "fill_policy": "none",
    "device": "cuda",
    "delta": 0.0001,
    "C": 100000.0,
    "k": 512,
}

SLURM_SPEC = {
    "combined": {"time": "03:00:00", "cpus": 16, "mem": None},
}

BATCH_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BATCH_DIR / "configs"
SCRIPTS_DIR = BATCH_DIR / "scripts"
RESULTS_DIR = BATCH_DIR / "results"
LOGS_DIR = BATCH_DIR / "logs"


def _config_name(n: int, seed: int) -> str:
    return f"config_combined_n{n}_seed{seed}"


def _script_name(n: int, seed: int) -> str:
    return f"run_combined_n{n}_seed{seed}"


def _make_config(n: int, seed: int, input_path: str) -> dict[str, Any]:
    return {
        "input": input_path,
        "n": n,
        "seed": seed,
        "stopping_condition": STOPPING_CONDITION_MAP[n],
        **FIXED,
    }


def _make_sbatch(
    *,
    n: int,
    seed: int,
    config_rel: str,
    partition: str,
    conda_env: str,
) -> str:
    spec = SLURM_SPEC["combined"]
    timeout_sec = 10800
    job_name = f"pc_combined_n{n}_s{seed}"

    config_rel_path = f"experiments/nyc_taxi_pot_comparison/batch/configs/{config_rel}.json"
    results_rel_path = "experiments/nyc_taxi_pot_comparison/batch/results"
    run_one_rel_path = "experiments/nyc_taxi_pot_comparison/batch/run_one.py"
    logs_rel_path = "experiments/nyc_taxi_pot_comparison/batch/logs"

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate configs + sbatch scripts for combined jobs")
    ap.add_argument("--input", required=True, help="Absolute path to the NYC taxi CSV on the cluster")
    ap.add_argument("--partition", default="rtx2060super", help="SLURM partition (default: rtx2060super)")
    ap.add_argument("--conda-env", default="spefenv", help="Conda environment name (default: spefenv)")
    ap.add_argument("--combined-mem", default=None, help="Optional --mem value for combined jobs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    args = ap.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    SLURM_SPEC["combined"]["mem"] = args.combined_mem

    total = 0
    for n in N_VALUES:
        for seed in SEEDS:
            cname = _config_name(n, seed)
            sname = _script_name(n, seed)

            cfg = _make_config(n, seed, args.input)
            sbatch = _make_sbatch(
                n=n,
                seed=seed,
                config_rel=cname,
                partition=args.partition,
                conda_env=args.conda_env,
            )

            cfg_path = CONFIGS_DIR / f"{cname}.json"
            script_path = SCRIPTS_DIR / f"{sname}.sbatch"

            if args.dry_run:
                print(f"[DRY] {cfg_path.name} -> {script_path.name}")
            else:
                with cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                with script_path.open("w", encoding="utf-8") as f:
                    f.write(sbatch)
                script_path.chmod(0o755)

            total += 1

    if args.dry_run:
        print(f"[DRY RUN] Would generate {total} config + script pairs.")
    else:
        print(f"Generated {total} config + script pairs.")
        print(f"  configs -> {CONFIGS_DIR}")
        print(f"  scripts -> {SCRIPTS_DIR}")


if __name__ == "__main__":
    main()
