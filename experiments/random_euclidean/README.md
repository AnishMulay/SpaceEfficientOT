Random Euclidean Experiment
===========================

This experiment generates random points `xA`, `xB` in R^d and runs the
space-efficient matching solver from the `spef_ot` library using the
Squared Euclidean kernel.

Usage
-----

From the repo root:

  python experiments/random_euclidean/run.py --n 10000 --d 2 --k 1000 --delta 0.001 --seed 42 --device cpu --out experiments/random_euclidean/outputs/result.json

- Add `--debug` to collect tile-size/device diagnostics (helps spot dynamic-shape recompiles).
- Compare new vs archived implementation:

  python experiments/random_euclidean/compare_with_archive.py --device cuda --debug

Notes
-----
- C is computed as the square of the maximum Euclidean distance from a random
  B point to all A points, mirroring the archived experiments.
- Results are printed as JSON to stdout and optionally written to `--out`.
- The script adds `src/` to `sys.path` when run from the repository root,
  so no installation step is required during development.
