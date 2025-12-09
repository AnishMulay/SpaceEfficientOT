#!/bin/bash
#SBATCH -J nyc_n400k_d1e-3_v10.0_sc20000
#SBATCH -o experiments/nyc_taxi_speed_euclidean/batch/logs/%x-%j.out
#SBATCH -e experiments/nyc_taxi_speed_euclidean/batch/logs/%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -p rtx2060super

export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

PATH=/usr/bin:/bin:$PATH conda activate spefenv

python -u experiments/nyc_taxi_speed_euclidean/batch/run_one.py \
  --config experiments/nyc_taxi_speed_euclidean/batch/configs/config_n400000_d1e-3_v10.0_sc20000.json \
  --results-dir experiments/nyc_taxi_speed_euclidean/batch/results \
  --timeout-sec 3600 \
  --print-cmd

conda deactivate || true
