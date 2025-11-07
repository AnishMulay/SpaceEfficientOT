#!/bin/bash
#SBATCH -J nyc_euc_n200k_d1e-4_sc4000
#SBATCH -o experiments/nyc_taxi_speed_euclidean/batch/logs/%x-%j.out
#SBATCH -e experiments/nyc_taxi_speed_euclidean/batch/logs/%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate spefenv

python -u experiments/nyc_taxi_speed_euclidean/batch/run_one.py \
  --config experiments/nyc_taxi_speed_euclidean/batch/configs/config_n200000_d0.0001_sc4000.json \
  --results-dir experiments/nyc_taxi_speed_euclidean/batch/results/nyc_euclidean \
  --timeout-sec 5400 \
  --print-cmd

conda deactivate || true

