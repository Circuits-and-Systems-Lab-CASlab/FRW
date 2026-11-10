# `scripts`: batch and benchmark runners

Helper scripts that drive the `frw` command. Back to the [main README](../README.md).

## Replay the three Excel comparison cases

Requires the optional `excel` extra (`pip install -e ".[excel]"`).

The comparison workbook records the exact windows and boundary conditions used
for `W0.14_S0.14`, `W0.14_S0.70`, and `W0.14_S2.8`. To replay its stratified
jobs with a `0.01` convergence tolerance:

```bash
python scripts/run_excel_cases.py
```

The runner adds `Current FRW Jobs`, `Current FRW Matrices`, and
`Current Comparisons` sheets. It writes the workbook and a JSON checkpoint
after every job. Press `Ctrl+C` to stop the current FRW process group; running
the same command again skips completed jobs and resumes at the interrupted job.
The original workbook is copied once to
`frw_raphael_window_comparison.original.xlsx`.

Useful controls include:

```bash
# Inspect every command without running FRW.
python scripts/run_excel_cases.py --list

# Run or resume one case.
python scripts/run_excel_cases.py --case W0.14_S0.70

# Recalculate completed jobs instead of skipping them.
python scripts/run_excel_cases.py --rerun
```

By default the runner keeps the recorded seed, scale factor, 20,000-walk
minimum, 20,000,000-walk limit, and 5,000-hop limit. It uses four row workers.
The output JSON files, logs, and checkpoint are stored under
`output/current_excel_runs`.

## Benchmark worker counts

`benchmark_workers.py` runs the `frw` command once per worker count and
records the wall time and whether every run produced the same capacitance
results.

```bash
python scripts/benchmark_workers.py input/INPUTFILE_01.m --workers 1 2 4 --walks 2000
```

Without `--walks`, each run uses the normal adaptive-convergence defaults.
The default worker counts are `1 2 4 5`, with one run each (`--repeats`).
The adaptive options (`--tolerance`, `--min-walks`, `--max-walks`,
`--check-every`, `--min-hits`) and `--seed`, `--scale-factor`,
`--max-distance` are passed through to `frw`.

Results go to `output/worker_benchmark_<timestamp>/` unless `--output-dir`
is given: one JSON and log per run, plus `benchmark_runs.csv` and
`benchmark_summary.json`.
