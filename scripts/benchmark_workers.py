"""Benchmark FRW row parallelism with reproducible CLI runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKERS = (1, 2, 4, 5)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FRW CLI with different worker counts and record "
            "execution time and numerical-result consistency."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="MATLAB-style conductor input file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
        help="Worker counts to test. Default: 1 2 4 5.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs for each worker count. Default: 1.",
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=None,
        help=(
            "Use a fixed even number of walks per row. If omitted, the "
            "normal adaptive-convergence defaults are used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed passed to every run. Default: 1234.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Optional adaptive-convergence tolerance passed to the FRW CLI.",
    )
    parser.add_argument(
        "--min-walks",
        type=int,
        default=None,
        help="Optional adaptive minimum walks per row.",
    )
    parser.add_argument(
        "--max-walks",
        type=int,
        default=None,
        help="Optional adaptive maximum walks per row.",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=None,
        help="Optional adaptive convergence-check interval.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help="Optional minimum target hits for convergence.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Optional VGSS scale factor passed to the FRW CLI.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional VGSS maximum-distance fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for results. Default: "
            "output/worker_benchmark_<timestamp>."
        ),
    )
    return parser


def _unique_worker_counts(values: list[int]) -> list[int]:
    workers = []
    for value in values:
        if value <= 0:
            raise ValueError("Worker counts must be positive integers.")
        if value not in workers:
            workers.append(value)
    return workers


def _resolve_input(path: Path) -> Path:
    if path.is_absolute():
        return path

    from_current_directory = (Path.cwd() / path).resolve()
    if from_current_directory.exists():
        return from_current_directory

    return (PROJECT_ROOT / path).resolve()


def _add_optional_argument(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frw_summary(result_file: Path) -> dict:
    with result_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows = payload.get("net_results")
    if rows is None:
        rows = [payload]

    row_walks = {
        row["master_net"]: int(row["num_walks"])
        for row in rows
    }
    converged_rows = sum(
        1
        for row in rows
        if row.get("convergence", {}).get("converged") is True
    )
    return {
        "row_count": len(rows),
        "row_walks": row_walks,
        "total_walks": sum(row_walks.values()),
        "converged_rows": converged_rows,
    }


def _run_once(
    args,
    input_file: Path,
    output_dir: Path,
    workers: int,
    repeat: int,
) -> dict:
    run_name = f"workers_{workers}_run_{repeat}"
    result_file = output_dir / f"{run_name}.json"
    log_file = output_dir / f"{run_name}.log"

    command = [
        sys.executable,
        "-m",
        "FRW.cli",
        str(input_file),
        "--workers",
        str(workers),
        "--seed",
        str(args.seed),
        "--output",
        str(result_file),
    ]

    _add_optional_argument(command, "--walks", args.walks)
    _add_optional_argument(command, "--tolerance", args.tolerance)
    _add_optional_argument(command, "--min-walks", args.min_walks)
    _add_optional_argument(command, "--max-walks", args.max_walks)
    _add_optional_argument(command, "--check-every", args.check_every)
    _add_optional_argument(command, "--min-hits", args.min_hits)
    _add_optional_argument(command, "--scale-factor", args.scale_factor)
    _add_optional_argument(command, "--max-distance", args.max_distance)

    environment = os.environ.copy()
    source_directory = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_directory
        if not existing_pythonpath
        else os.pathsep.join([source_directory, existing_pythonpath])
    )

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_seconds = time.perf_counter() - start
    log_file.write_text(completed.stdout, encoding="utf-8")

    record = {
        "workers_requested": workers,
        "repeat": repeat,
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": completed.returncode,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "result_file": str(result_file),
        "log_file": str(log_file),
        "sha256": None,
        "matches_single_worker": None,
        "speedup_vs_single_worker": None,
        "row_count": None,
        "active_workers": None,
        "row_walks": None,
        "total_walks": None,
        "converged_rows": None,
    }

    if completed.returncode == 0 and result_file.exists():
        result_summary = _read_frw_summary(result_file)
        record.update(result_summary)
        record["active_workers"] = min(workers, record["row_count"])
        record["sha256"] = _sha256(result_file)

    return record


def _add_comparisons(records: list[dict]) -> None:
    baseline_by_repeat = {
        record["repeat"]: record
        for record in records
        if record["workers_requested"] == 1
        and record["status"] == "succeeded"
    }
    fallback_baseline = next(
        (record for record in records if record["status"] == "succeeded"),
        None,
    )

    for record in records:
        if record["status"] != "succeeded":
            continue

        baseline = baseline_by_repeat.get(record["repeat"], fallback_baseline)
        if baseline is None:
            continue

        record["matches_single_worker"] = (
            record["sha256"] == baseline["sha256"]
        )
        record["speedup_vs_single_worker"] = (
            baseline["elapsed_seconds"] / record["elapsed_seconds"]
        )


def _aggregate(records: list[dict], worker_counts: list[int]) -> list[dict]:
    summaries = []
    for workers in worker_counts:
        successful = [
            record
            for record in records
            if record["workers_requested"] == workers
            and record["status"] == "succeeded"
        ]
        if not successful:
            summaries.append(
                {
                    "workers_requested": workers,
                    "successful_runs": 0,
                    "mean_seconds": None,
                    "min_seconds": None,
                    "max_seconds": None,
                    "mean_speedup": None,
                    "all_results_match": False,
                }
            )
            continue

        times = [record["elapsed_seconds"] for record in successful]
        speedups = [
            record["speedup_vs_single_worker"]
            for record in successful
            if record["speedup_vs_single_worker"] is not None
        ]
        summaries.append(
            {
                "workers_requested": workers,
                "successful_runs": len(successful),
                "mean_seconds": sum(times) / len(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "mean_speedup": (
                    sum(speedups) / len(speedups) if speedups else None
                ),
                "all_results_match": all(
                    record["matches_single_worker"] for record in successful
                ),
            }
        )
    return summaries


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "workers_requested",
        "active_workers",
        "repeat",
        "elapsed_seconds",
        "speedup_vs_single_worker",
        "status",
        "return_code",
        "matches_single_worker",
        "row_count",
        "total_walks",
        "converged_rows",
        "row_walks",
        "sha256",
        "result_file",
        "log_file",
        "started_at_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = dict(record)
            csv_record["row_walks"] = json.dumps(
                record["row_walks"], sort_keys=True
            )
            writer.writerow(csv_record)


def _print_summary(summaries: list[dict]) -> None:
    print("\nBenchmark summary")
    print("workers   mean time   speedup   results match")
    for summary in summaries:
        if summary["successful_runs"] == 0:
            print(f"{summary['workers_requested']:>7}   failed")
            continue

        speedup = summary["mean_speedup"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.2f}x"
        match_text = "yes" if summary["all_results_match"] else "NO"
        print(
            f"{summary['workers_requested']:>7}   "
            f"{summary['mean_seconds']:>8.3f}s   "
            f"{speedup_text:>7}   {match_text}"
        )


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        worker_counts = _unique_worker_counts(args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    input_file = _resolve_input(args.input_file)
    if not input_file.is_file():
        parser.error(f"Input file not found: {input_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / f"worker_benchmark_{timestamp}"
    elif not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_file}")
    print(f"Workers: {', '.join(map(str, worker_counts))}")
    print(f"Repeats: {args.repeats}")
    print(f"Results: {output_dir}\n")

    records = []
    for repeat in range(1, args.repeats + 1):
        for workers in worker_counts:
            print(
                f"Running workers={workers}, repeat={repeat}... ",
                end="",
                flush=True,
            )
            record = _run_once(
                args,
                input_file,
                output_dir,
                workers,
                repeat,
            )
            records.append(record)
            print(
                f"{record['status']} in "
                f"{record['elapsed_seconds']:.3f} seconds"
            )

    _add_comparisons(records)
    summaries = _aggregate(records, worker_counts)

    csv_file = output_dir / "benchmark_runs.csv"
    json_file = output_dir / "benchmark_summary.json"
    _write_csv(csv_file, records)

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "input_file": str(input_file),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "seed": args.seed,
        "fixed_walks": args.walks,
        "worker_counts": worker_counts,
        "repeats": args.repeats,
        "runs": records,
        "worker_summary": summaries,
    }
    json_file.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    _print_summary(summaries)
    print(f"\nCSV:  {csv_file}")
    print(f"JSON: {json_file}")

    return 0 if all(record["status"] == "succeeded" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
