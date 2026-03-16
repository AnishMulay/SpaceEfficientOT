#!/usr/bin/env python3
"""Submit all (or a filtered subset of) sbatch scripts to the SLURM scheduler.

Run from the repo root on the cluster login node:
  python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py

Options let you filter by solver or n so you can stagger submissions.

Examples
--------
Submit everything:
  python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py

Submit only the combined jobs:
  python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py --solvers combined

Submit only n=1000,2000 as a quick sanity check:
  python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py --n 1000 2000

Dry-run (print sbatch commands without running them):
  python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BATCH_DIR   = Path(__file__).resolve().parent
SCRIPTS_DIR = BATCH_DIR / "scripts"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Submit POT-comparison sbatch jobs to SLURM")
    ap.add_argument(
        "--solvers", nargs="+",
        choices=["combined"],
        default=None,
        help="Restrict to these solvers (default: all available solvers)",
    )
    ap.add_argument(
        "--n", nargs="+", type=int, default=None,
        help="Restrict to these n values, e.g. --n 1000 2000 5000",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Restrict to these seeds (default: all seeds)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch commands without submitting",
    )
    ap.add_argument(
        "--scripts-dir", default=str(SCRIPTS_DIR),
        help=f"Directory containing .sbatch files (default: {SCRIPTS_DIR})",
    )
    return ap.parse_args()


def _matches_filter(
    script_name: str,
    solvers: list[str] | None,
    n_filter: list[int] | None,
    seed_filter: list[int] | None,
) -> bool:
    """Return True if the script filename satisfies all active filters.

    Expected name pattern: run_{solver}_n{N}_seed{S}.sbatch
    """
    # Parse solver
    if solvers is not None:
        if not any(f"run_{s}_" in script_name for s in solvers):
            return False

    # Parse n
    if n_filter is not None:
        matched_n = False
        for n in n_filter:
            if f"_n{n}_" in script_name:
                matched_n = True
                break
        if not matched_n:
            return False

    # Parse seed
    if seed_filter is not None:
        matched_seed = False
        for s in seed_filter:
            if script_name.endswith(f"_seed{s}.sbatch"):
                matched_seed = True
                break
        if not matched_seed:
            return False

    return True


def main() -> None:
    args = _parse_args()
    scripts_dir = Path(args.scripts_dir)

    if not scripts_dir.exists():
        print(
            f"ERROR: scripts directory not found: {scripts_dir}\n"
            "Run generate_configs.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_scripts = sorted(scripts_dir.glob("*.sbatch"))
    if not all_scripts:
        print(
            f"No .sbatch files found in {scripts_dir}.\n"
            "Run generate_configs.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    selected = [
        s for s in all_scripts
        if _matches_filter(s.name, args.solvers, args.n, args.seeds)
    ]

    if not selected:
        print("No scripts match the given filters.")
        sys.exit(0)

    submitted = 0
    failed    = 0

    for script in selected:
        cmd = ["sbatch", str(script)]
        if args.dry_run:
            print(f"[DRY] {' '.join(cmd)}")
            submitted += 1
            continue

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            job_line = result.stdout.strip()
            print(f"Submitted {script.name}  →  {job_line}")
            submitted += 1
        except subprocess.CalledProcessError as ex:
            print(f"FAILED  {script.name}: {ex.stderr.strip()}", file=sys.stderr)
            failed += 1
        except FileNotFoundError:
            print("ERROR: 'sbatch' not found. Are you on a SLURM login node?", file=sys.stderr)
            sys.exit(1)

    label = "[DRY] Would submit" if args.dry_run else "Submitted"
    print(f"\n{label} {submitted} jobs" + (f", {failed} failed" if failed else "") + ".")


if __name__ == "__main__":
    main()
