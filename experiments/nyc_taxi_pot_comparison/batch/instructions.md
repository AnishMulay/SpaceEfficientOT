# NYC Taxi POT Comparison Batch Run

Run these from the repo root on your interactive cluster node.

```bash
cd /Users/anish/Developer/NCSU/SpaceEfficientOT
conda activate spefenv
export CSV=/absolute/path/to/2014_Yellow_Taxi_Trip_Data_20141014-3.csv
export PARTITION=rtx2060super
```

If you want to clear old generated outputs first:

```bash
rm -rf experiments/nyc_taxi_pot_comparison/batch/configs experiments/nyc_taxi_pot_comparison/batch/scripts experiments/nyc_taxi_pot_comparison/batch/logs experiments/nyc_taxi_pot_comparison/batch/results && rm -f experiments/nyc_taxi_pot_comparison/batch/pot_comparison_results.csv
```

Generate configs and sbatch scripts:

```bash
python experiments/nyc_taxi_pot_comparison/batch/generate_configs.py --input "$CSV" --partition "$PARTITION" --conda-env spefenv
```

Submit all combined batch jobs:

```bash
python experiments/nyc_taxi_pot_comparison/batch/submit_experiments.py --solvers combined
```

Optional queue check:

```bash
squeue -u "$USER"
```

After all jobs finish, aggregate results:

```bash
python experiments/nyc_taxi_pot_comparison/batch/aggregate_results.py
```

Output CSV:

```bash
experiments/nyc_taxi_pot_comparison/batch/pot_comparison_results.csv
```

Each combined job produces three CSV rows:
- `spef_unscaled`
- `spef_scaled`
- `pot_emd`
