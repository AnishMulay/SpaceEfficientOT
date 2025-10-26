NYC Taxi Haversine Experiment
=============================

This experiment reproduces the one-day NYC taxi matching workflow using the
refactored `spef_ot` library and the Haversine slack kernel with time-window
masking. All inputs, helpers, and outputs are contained within this directory.

Directory layout:
- `data/`: Place the NYC taxi CSV/Parquet files here (latest schema with pickup
  and dropoff coordinates). These files are *not* committed; add them manually.
- `schemas.md`: Documents required column names and accepted aliases.
- `loader.py`: Reads data, validates columns, filters to a single day, handles
  sampling, and basic coordinate sanity checks.
- `prepare.py`: Converts filtered data into tensors (`xA`, `xB`, `tA`, `tB`).
- `estimate_c.py`: Helpers to estimate the scale `C` from Haversine distances.
- `run.py`: Main CLI that orchestrates loading, preparation, scale estimation,
  solver invocation, and result logging.
- `compare_with_archive.py` (optional): Harness to run both the new solver and
  the archived implementation on the same dataset for parity/performance checks.
- `outputs/`: JSON result logs.

The script exposes an in-file `DEFAULT_CONFIG` (see `run.py`). Adjust it to
reflect your preferred input path and parameters, then simply run:

```bash
python experiments/nyc_taxi/run.py
```

To override specific values on the command line (or when using a different
dataset), pass only the arguments you need:

```bash
python experiments/nyc_taxi/run.py \
  --input experiments/nyc_taxi/data/nyc_taxi_day.parquet \
  --date 2014-01-01 \
  --n 50000 --k 512 --delta 0.01 --cmax 10
```

You can also merge settings from a JSON file via `--config path/to/config.json`.

The script prints a JSON summary to stdout and optionally writes to
`experiments/nyc_taxi/outputs/` when `--out` (or the config's `out`) is set.

For detailed column requirements, see `schemas.md`.
