#!/usr/bin/env python3
"""Generate configs + sbatch scripts for the combined synthetic comparison jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

N_VALUES = [1000, 2000, 5000, 10000, 15000, 20000]
SEEDS = [1, 2, 3]

FIXED = {
    "solver": "combined_synthetic",
    "device": "cuda",
    "delta": 0.001,
    "k": 512,
    "dim": 2,
}

SLURM_SPEC = {
    "time": "02:00:00",
    "cpus": 16,
}

BATCH_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BATCH_DIR / "configs"
SCRIPTS_DIR = BATCH_DIR / "scripts"
RESULTS_DIR = BATCH_DIR / "results"
LOGS_DIR = BATCH_DIR / "logs"


def _config_name(n: int, seed: int) -> str:
    return f"config_combined_synthetic_n{n}_seed{seed}"


def _script_name(n: int, seed: int) -> str:
    return f"run_combined_synthetic_n{n}_seed{seed}"


def _make_config(n: int, seed: int) -> dict[str, Any]:
    return {
        "n": n,
        "seed": seed,
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
    timeout_sec = 7200
    job_name = f"pc_combined_synth_n{n}_s{seed}"

    config_rel_path = (
        f"experiments/synthetic_matching_comparison/batch/configs/{config_rel}.json"
    )
    results_rel_path = "experiments/synthetic_matching_comparison/batch/results"
    run_one_rel_path = "experiments/synthetic_matching_comparison/batch/run_one.py"
    logs_rel_path = "experiments/synthetic_matching_comparison/batch/logs"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {logs_rel_path}/%x-%j.out",
        f"#SBATCH -e {logs_rel_path}/%x-%j.err",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        f"#SBATCH --cpus-per-task={SLURM_SPEC['cpus']}",
        f"#SBATCH -t {SLURM_SPEC['time']}",
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
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate configs + sbatch scripts for combined synthetic jobs"
    )
    ap.add_argument(
        "--partition",
        default="rtx2060super",
        help="SLURM partition (default: rtx2060super)",
    )
    ap.add_argument(
        "--conda-env",
        default="spefenv",
        help="Conda environment name (default: spefenv)",
    )
    args = ap.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for n in N_VALUES:
        for seed in SEEDS:
            cname = _config_name(n, seed)
            sname = _script_name(n, seed)

            cfg = _make_config(n, seed)
            sbatch = _make_sbatch(
                n=n,
                seed=seed,
                config_rel=cname,
                partition=args.partition,
                conda_env=args.conda_env,
            )

            cfg_path = CONFIGS_DIR / f"{cname}.json"
            script_path = SCRIPTS_DIR / f"{sname}.sbatch"

            with cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle, indent=2)
            with script_path.open("w", encoding="utf-8") as handle:
                handle.write(sbatch)
            script_path.chmod(0o755)

            total += 1

    print(f"Generated {total} config + script pairs.")
    print(f"  configs -> {CONFIGS_DIR}")
    print(f"  scripts -> {SCRIPTS_DIR}")


if __name__ == "__main__":
    main()
