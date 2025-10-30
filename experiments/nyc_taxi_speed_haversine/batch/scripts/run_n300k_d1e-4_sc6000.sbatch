#!/bin/bash
#SBATCH -J nyc_hav_n300k_d1e-4_sc6000
#SBATCH -o experiments/nyc_taxi_speed_haversine/batch/logs/%x-%j.out
#SBATCH -e experiments/nyc_taxi_speed_haversine/batch/logs/%x-%j.err
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

python -u experiments/nyc_taxi_speed_haversine/batch/run_one.py \
  --config experiments/nyc_taxi_speed_haversine/batch/configs/config_n300000_d0.0001_sc6000.json \
  --results-dir experiments/nyc_taxi_speed_haversine/batch/results/nyc_haversine \
  --timeout-sec 3600 \
  --print-cmd

conda deactivate || true

