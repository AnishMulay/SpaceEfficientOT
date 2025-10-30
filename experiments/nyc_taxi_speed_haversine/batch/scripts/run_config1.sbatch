#!/bin/bash
#SBATCH -J nyc_hav_cfg1             # Job name
#SBATCH -o experiments/nyc_taxi_speed_haversine/batch/logs/%x-%j.out     # Stdout log
#SBATCH -e experiments/nyc_taxi_speed_haversine/batch/logs/%x-%j.err     # Stderr log
#SBATCH -N 1                        # Nodes
#SBATCH -n 1                        # Tasks
#SBATCH --cpus-per-task=16          # CPU cores per task
#SBATCH -t 02:00:00                 # Max runtime (1 hour)
#SBATCH -p rtx2060super             # GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU via GRES (cons_res)

set -euo pipefail
export PYTHONUNBUFFERED=1

# Use the directory where sbatch was invoked as the working directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure conda is available in non-interactive shells
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# Activate your environment (CUDA-friendly PATH prefix)
PATH=/usr/bin:/bin:$PATH conda activate spefenv

python -u experiments/nyc_taxi_speed_haversine/batch/run_one.py \
  --config experiments/nyc_taxi_speed_haversine/batch/configs/config1.json \
  --results-dir experiments/nyc_taxi_speed_haversine/batch/results/nyc_haversine \
  --timeout-sec 3600 \
  --print-cmd

conda deactivate || true
