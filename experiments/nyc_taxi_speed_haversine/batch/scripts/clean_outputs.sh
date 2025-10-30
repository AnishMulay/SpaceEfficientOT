#!/bin/bash
# Clean logs and results for NYC Haversine batch runs

set -e

# Resolve to experiment's batch directory regardless of current working dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"  # experiments/nyc_taxi_speed_haversine/batch

LOG_DIR="$BASE_DIR/logs"
RES_DIR="$BASE_DIR/results/nyc_haversine"

echo "Cleaning logs:    $LOG_DIR"
echo "Cleaning results: $RES_DIR"

# Keep .gitkeep if present; remove everything else (files and subdirs)
if [ -d "$LOG_DIR" ]; then
  find "$LOG_DIR" -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} +
fi

if [ -d "$RES_DIR" ]; then
  find "$RES_DIR" -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} +
fi

echo "Done."

