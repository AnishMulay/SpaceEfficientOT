#!/bin/bash
#SBATCH -J nyc_hav_cfg1             # Job name
#SBATCH -o batch/logs/%x-%j.out     # Stdout log
#SBATCH -e batch/logs/%x-%j.err     # Stderr log
#SBATCH -N 1                        # Nodes
#SBATCH -n 1                        # Tasks
#SBATCH --cpus-per-task=16          # CPU cores per task
#SBATCH -t 01:00:00                 # Max runtime (1 hour)
#SBATCH --partition=skylake         # Partition (CPU example)

set -euo pipefail
export PYTHONUNBUFFERED=1

# Use the directory where sbatch was invoked as the working directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure conda is available in non-interactive shells
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate your environment (edit if needed)
conda activate gpuenv

python -u batch/run_one.py \
  --config batch/configs/config1.json \
  --results-dir batch/results/nyc_haversine \
  --timeout-sec 3600 \
  --print-cmd

conda deactivate || true

