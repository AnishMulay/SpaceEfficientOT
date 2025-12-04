#!/bin/bash
#SBATCH -J scaled_n400k_d1e-3_v12.0_sc8000
#SBATCH -o experiments/scaled_nyc_taxi_speed_euclidean/batch/logs/%x-%j.out
#SBATCH -e experiments/scaled_nyc_taxi_speed_euclidean/batch/logs/%x-%j.err
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

python -u experiments/scaled_nyc_taxi_speed_euclidean/batch/run_one.py \
  --config experiments/scaled_nyc_taxi_speed_euclidean/batch/configs/config_n400000_d1e-3_v12.0_sc8000.json \
  --results-dir experiments/scaled_nyc_taxi_speed_euclidean/batch/results \
  --timeout-sec 3600 \
  --print-cmd

conda deactivate || true
