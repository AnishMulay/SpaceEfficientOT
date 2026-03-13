#!/usr/bin/env python3
"""Generate configs + sbatch scripts for the NYC taxi POT-comparison experiment.

Phase 1 (`--solver spef`) mirrors the exact-comparison SPEF sweep and writes
configs for `spef_unscaled` and `spef_scaled`.

Phase 2 (`--solver pot`) scans existing SPEF result JSONs in `batch/results/`,
extracts `feasible_matches`, and writes one `pot_partial` config per `(n, seed)`
with `top_k` set to that exact value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Experimental grid (edit here to change the sweep)
# ---------------------------------------------------------------------------

N_VALUES = [1000, 2000, 5000, 10000, 15000, 20000]

SEEDS = [1, 2, 3]

SPEF_SOLVERS = ["spef_unscaled", "spef_scaled"]
POT_SOLVERS = ["pot_partial"]

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
    "pot_partial":   {"time": "02:00:00", "cpus": 16, "mem": None},
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BATCH_DIR = Path(__file__).resolve().parent
EXP_DIR   = BATCH_DIR.parent
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


def _make_config(solver: str, n: int, seed: int, input_path: str, *, top_k: int | None = None) -> dict[str, Any]:
    cfg: dict = {
        "solver":  solver,
        "input":   input_path,
        "n":       n,
        "seed":    seed,
        **FIXED,
    }
    # SPEF-only keys: stopping_condition
    if solver in SPEF_SOLVERS:
        cfg["stopping_condition"] = STOPPING_CONDITION_MAP[n]
    else:
        for key in ("device", "delta", "C", "k", "stopping_condition", "fill_policy"):
            cfg.pop(key, None)
        cfg["top_k"] = top_k
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
    job_name = f"pc_{solver}_n{n}_s{seed}"
    timeout_sec = {
        "spef_unscaled": 3600,
        "spef_scaled":   3600,
        "pot_partial":   7200,
    }[solver]

    # Relative paths from repo root (so they work regardless of submission dir)
    config_rel_path  = f"experiments/nyc_taxi_pot_comparison/batch/configs/{config_rel}.json"
    results_rel_path = "experiments/nyc_taxi_pot_comparison/batch/results"
    run_one_rel_path = "experiments/nyc_taxi_pot_comparison/batch/run_one.py"
    logs_rel_path    = "experiments/nyc_taxi_pot_comparison/batch/logs"

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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_top_k(results_dir: Path) -> dict[tuple[int, int], int]:
    observed: dict[tuple[int, int], set[int]] = {}

    for result_path in sorted(results_dir.rglob("result.json")):
        try:
            data = _read_json(result_path)
        except Exception as ex:
            print(f"WARNING: failed to read {result_path}: {type(ex).__name__}: {ex}")
            continue

        if data.get("solver") not in SPEF_SOLVERS:
            continue

        params = data.get("params", {})
        metrics = data.get("metrics", {})
        n = params.get("n_requested")
        seed = params.get("seed")
        feasible_matches = metrics.get("feasible_matches")
        if n is None or seed is None or feasible_matches is None:
            print(f"WARNING: incomplete SPEF result in {result_path}; skipping")
            continue

        feasible_int = int(round(float(feasible_matches)))
        if abs(float(feasible_matches) - feasible_int) > 1e-6:
            print(f"WARNING: non-integer feasible_matches={feasible_matches} in {result_path}; skipping")
            continue

        key = (int(n), int(seed))
        observed.setdefault(key, set()).add(feasible_int)

    top_k_by_key: dict[tuple[int, int], int] = {}
    for key, values in observed.items():
        if len(values) != 1:
            n, seed = key
            values_str = ", ".join(str(v) for v in sorted(values))
            print(f"WARNING: conflicting feasible_matches for n={n}, seed={seed}: {values_str}; skipping POT config")
            continue
        top_k_by_key[key] = next(iter(values))
    return top_k_by_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate configs + sbatch scripts for the POT-comparison experiment"
    )
    ap.add_argument(
        "--solver", required=True, choices=["spef", "pot"],
        help="Generation phase: `spef` writes SPEF configs, `pot` derives POT configs from SPEF results"
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
    ap.add_argument("--pot-mem", default=None,
                    help="Optional --mem value for pot_partial jobs (default: omit)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be generated without writing")
    args = ap.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    SLURM_SPEC["spef_unscaled"]["mem"] = args.spef_mem
    SLURM_SPEC["spef_scaled"]["mem"] = args.spef_mem
    SLURM_SPEC["pot_partial"]["mem"] = args.pot_mem

    if args.solver == "spef":
        jobs: list[tuple[str, int, int, int | None]] = [
            (solver, n, seed, None)
            for solver in SPEF_SOLVERS
            for n in N_VALUES
            for seed in SEEDS
        ]
    else:
        top_k_by_key = _extract_top_k(RESULTS_DIR)
        jobs = []
        for n in N_VALUES:
            for seed in SEEDS:
                top_k = top_k_by_key.get((n, seed))
                if top_k is None:
                    print(f"WARNING: no SPEF result JSON found for n={n}, seed={seed}; skipping POT config")
                    continue
                jobs.append(("pot_partial", n, seed, top_k))

    total = 0
    for solver, n, seed, top_k in jobs:
        cname = _config_name(solver, n, seed)
        sname = _script_name(solver, n, seed)

        cfg = _make_config(solver, n, seed, args.input, top_k=top_k)
        sbatch = _make_sbatch(solver, n, seed, cname, args.partition, args.conda_env)

        cfg_path = CONFIGS_DIR / f"{cname}.json"
        script_path = SCRIPTS_DIR / f"{sname}.sbatch"

        if args.dry_run:
            extra = f"  top_k={top_k}" if top_k is not None else ""
            print(f"[DRY] {cfg_path.name}  →  {script_path.name}{extra}")
        else:
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            with script_path.open("w", encoding="utf-8") as f:
                f.write(sbatch)
            script_path.chmod(0o755)

        total += 1

    if not args.dry_run:
        print(f"Generated {total} config + script pairs for phase `{args.solver}`.")
        print(f"  configs  → {CONFIGS_DIR}")
        print(f"  scripts  → {SCRIPTS_DIR}")
    else:
        print(f"[DRY RUN] Would generate {total} config + script pairs for phase `{args.solver}`.")


if __name__ == "__main__":
    main()
