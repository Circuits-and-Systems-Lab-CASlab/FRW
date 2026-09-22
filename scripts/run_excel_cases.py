"""Replay the workbook's three FRW cases and record current results.

The source workbook remains a historical record.  This script adds separate
"Current FRW" sheets, saves after every job, and resumes completed jobs after
an interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = PROJECT_ROOT / "output"/ "frw_raphael_window_comparison.xlsx"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "input" / "FRW_MAT_test_cases"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "current_excel_runs"

CASE_INPUTS = {
    "W0.14_S0.14": "Testcase1_W0.14toS0.14/MATLABinputfile.m",
    "W0.14_S0.70": "TestCase8_W0.14toS0.70/MATLABinputfile.m",
    "W0.14_S2.8": "Testcase11_W0.14toS2.8/MATLABinputfile.m",
}

JOBS_SHEET = "Current FRW Jobs"
MATRICES_SHEET = "Current FRW Matrices"
COMPARISONS_SHEET = "Current Comparisons"

JOB_HEADERS = [
    "Job",
    "Case",
    "Window",
    "Boundary",
    "Seed",
    "Status",
    "Stopping status",
    "Walks",
    "Runtime (h)",
    "Return code",
    "Relative asymmetry",
    "Row-sum ratio",
    "Raw result",
    "Log",
    "Command",
    "Updated (UTC)",
]

COMPARISON_HEADERS = [
    "Job",
    "Case",
    "Window",
    "Boundary",
    "Raphael mode",
    "Walks",
    "Runtime (h)",
    "Relative Frobenius error",
    "Diagonal error",
    "Off-diagonal error",
    "Mean abs diff (pF)",
    "Max abs diff (pF)",
    "FRW asymmetry",
    "Raphael asymmetry",
    "FRW row-sum ratio",
    "Raphael row-sum ratio",
    "Within 1%",
]


@dataclass(frozen=True)
class Job:
    name: str
    case: str
    window_name: str
    boundary: str
    seed: int
    window: tuple[float, ...] | None
    input_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the stratified FRW jobs recorded in the comparison "
            "workbook and save progress after every job."
        )
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASE_INPUTS),
        help="Run only this case. Repeat the option to select several cases.",
    )
    parser.add_argument(
        "--job",
        action="append",
        help="Run only jobs whose names contain this text. May be repeated.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--min-walks", type=int, default=2000)
    parser.add_argument("--max-walks", type=int, default=1_000_000)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--min-hits", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=5_000)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run selected jobs again even when their result files exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected jobs and commands without running them.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def window_from(command_text: str) -> tuple[float, ...] | None:
    tokens = shlex.split(command_text)
    if "--window" not in tokens:
        return None
    start = tokens.index("--window") + 1
    values = tuple(float(value) for value in tokens[start : start + 6])
    if len(values) != 6:
        raise ValueError(f"Incomplete window in recorded command: {command_text}")
    return values


def read_jobs(workbook_path: Path, input_root: Path) -> list[Job]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheet = workbook["FRW Jobs"]
        headers = {
            str(cell.value): column
            for column, cell in enumerate(sheet[5], start=1)
            if cell.value is not None
        }
        required = {"Job", "Case", "Window", "Boundary", "Sampler", "Seed", "Command"}
        missing = sorted(required - headers.keys())
        if missing:
            raise ValueError(f"FRW Jobs is missing columns: {', '.join(missing)}")

        jobs = []
        for values in sheet.iter_rows(min_row=6, values_only=True):
            sampler = values[headers["Sampler"] - 1]
            if sampler != "stratified":
                continue
            case = str(values[headers["Case"] - 1])
            relative_input = CASE_INPUTS.get(case)
            if relative_input is None:
                continue
            command_text = str(values[headers["Command"] - 1])
            jobs.append(
                Job(
                    name=str(values[headers["Job"] - 1]),
                    case=case,
                    window_name=str(values[headers["Window"] - 1]),
                    boundary=str(values[headers["Boundary"] - 1]),
                    seed=int(values[headers["Seed"] - 1]),
                    window=window_from(command_text),
                    input_file=input_root / relative_input,
                )
            )
        return jobs
    finally:
        workbook.close()


def select_jobs(jobs: list[Job], cases: list[str] | None, names: list[str] | None) -> list[Job]:
    selected = jobs
    if cases:
        selected = [job for job in selected if job.case in cases]
    if names:
        selected = [job for job in selected if any(text in job.name for text in names)]
    return selected


def boundary_args(job: Job) -> list[str]:
    if job.boundary == "open":
        if job.window is not None:
            raise ValueError(f"Open job unexpectedly has a window: {job.name}")
        return []
    if job.window is None:
        raise ValueError(f"Finite-boundary job has no window: {job.name}")

    args = ["--window", *(str(value) for value in job.window)]
    if job.boundary == "dirichlet_all":
        return args
    if job.boundary == "mixed_xy_dirichlet_z_neumann":
        return [*args, "--neumann-axes", "z"]
    if job.boundary == "neumann_all":
        return [*args, "--neumann-axes", "x", "y", "z"]
    raise ValueError(f"Unknown boundary {job.boundary!r} in {job.name}")


def command_for(job: Job, args: argparse.Namespace, result_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "FRW.cli",
        str(job.input_file),
        "--workers",
        str(args.workers),
        "--seed",
        str(job.seed),
        "--tolerance",
        str(args.tolerance),
        "--min-walks",
        str(args.min_walks),
        "--max-walks",
        str(args.max_walks),
        "--check-every",
        str(args.check_every),
        "--min-hits",
        str(args.min_hits),
        "--max-hops",
        str(args.max_hops),
        "--scale-factor",
        "1.25",
        "--output",
        str(result_file),
        *boundary_args(job),
    ]


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"jobs": {}}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def stop_process(process: subprocess.Popen) -> None:
    """Stop the CLI and its worker-process group, then reap the parent."""

    if process.poll() is not None:
        process.wait()
        return

    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_command(command: list[str], log_file: Path) -> tuple[int, float]:
    environment = os.environ.copy()
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH")) if item
    )
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        except KeyboardInterrupt:
            stop_process(process)
            raise
        finally:
            if process.poll() is None:
                stop_process(process)
    return return_code, time.perf_counter() - start


def read_result(path: Path) -> tuple[list[str], np.ndarray, dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    matrix_payload = payload["capacitance_matrix"]
    labels = list(matrix_payload["net_order"])
    matrix = np.asarray(matrix_payload["matrix_F"], dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"Invalid capacitance matrix shape in {path}")
    return labels, matrix, payload


def result_summary(payload: dict, matrix: np.ndarray) -> dict:
    rows = payload["net_results"]
    converged = all(row["convergence"].get("converged") is True for row in rows)
    walks = sum(int(row["num_walks"]) for row in rows)
    stop_reasons = sorted({str(row["convergence"].get("stop_reason")) for row in rows})
    matrix_norm = float(np.linalg.norm(matrix))
    asymmetry = 0.0 if matrix_norm == 0.0 else float(np.linalg.norm(matrix - matrix.T) / matrix_norm)
    max_diagonal = float(np.max(np.abs(np.diag(matrix))))
    row_sum_ratio = 0.0 if max_diagonal == 0.0 else float(np.max(np.abs(matrix.sum(axis=1))) / max_diagonal)
    return {
        "status": "CONVERGED" if converged else "MAX_WALKS_OR_SAFETY_STOP",
        "stopping_status": ", ".join(stop_reasons),
        "walks": walks,
        "relative_asymmetry": asymmetry,
        "row_sum_ratio": row_sum_ratio,
    }


def read_reference_matrices(workbook) -> dict[str, tuple[list[str], np.ndarray]]:
    sheet = workbook["Raphael Matrices"]
    matrices = {}
    row = 1
    while row <= sheet.max_row:
        name = sheet.cell(row, 1).value
        if isinstance(name, str) and name.startswith("W0.14_"):
            labels = [sheet.cell(row + 1, column).value for column in range(2, 7)]
            values = [
                [sheet.cell(row + 2 + i, 2 + j).value for j in range(5)]
                for i in range(5)
            ]
            matrices[name] = (labels, np.asarray(values, dtype=float))
            row += 7
        else:
            row += 1
    return matrices


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)


def compare_matrices(frw: np.ndarray, reference: np.ndarray) -> dict:
    difference = frw - reference
    diagonal_mask = np.eye(frw.shape[0], dtype=bool)
    off_diagonal_mask = ~diagonal_mask
    reference_diagonal = reference[diagonal_mask]
    reference_off_diagonal = reference[off_diagonal_mask]
    frw_norm = float(np.linalg.norm(frw))
    reference_norm = float(np.linalg.norm(reference))
    frw_max_diagonal = float(np.max(np.abs(np.diag(frw))))
    reference_max_diagonal = float(np.max(np.abs(np.diag(reference))))
    return {
        "relative_frobenius_error": ratio(float(np.linalg.norm(difference)), reference_norm),
        "diagonal_error": ratio(
            float(np.linalg.norm(difference[diagonal_mask])),
            float(np.linalg.norm(reference_diagonal)),
        ),
        "off_diagonal_error": ratio(
            float(np.linalg.norm(difference[off_diagonal_mask])),
            float(np.linalg.norm(reference_off_diagonal)),
        ),
        "mean_abs_diff_pf": float(np.mean(np.abs(difference)) * 1e12),
        "max_abs_diff_pf": float(np.max(np.abs(difference)) * 1e12),
        "frw_asymmetry": ratio(float(np.linalg.norm(frw - frw.T)), frw_norm),
        "reference_asymmetry": ratio(
            float(np.linalg.norm(reference - reference.T)), reference_norm
        ),
        "frw_row_sum_ratio": ratio(float(np.max(np.abs(frw.sum(axis=1)))), frw_max_diagonal),
        "reference_row_sum_ratio": ratio(
            float(np.max(np.abs(reference.sum(axis=1)))), reference_max_diagonal
        ),
    }


def reference_names(job: Job) -> list[str]:
    if job.window_name in {"buffer_10x", "buffer_1x"}:
        modes = ("fdm",)
    elif job.window_name in {"tight_xyz", "touch_z"}:
        modes = ("fdm", "bem")
    else:
        modes = ()
    return [f"{job.case}__{job.window_name}__{mode}" for mode in modes]


def style_sheet(sheet, title: str, subtitle: str, headers: list[str]) -> None:
    sheet["A1"] = title
    sheet["A2"] = subtitle
    sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    sheet["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    for column, header in enumerate(headers, start=1):
        cell = sheet.cell(4, column, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.freeze_panes = "A5"
    sheet.auto_filter.ref = f"A4:{sheet.cell(4, len(headers)).coordinate}"


def ensure_sheets(workbook) -> None:
    if JOBS_SHEET not in workbook.sheetnames:
        sheet = workbook.create_sheet(JOBS_SHEET, 0)
        style_sheet(
            sheet,
            "Current FRW job status",
            "Stratified runs replaying the exact windows recorded in FRW Jobs.",
            JOB_HEADERS,
        )
    if COMPARISONS_SHEET not in workbook.sheetnames:
        sheet = workbook.create_sheet(COMPARISONS_SHEET, 1)
        style_sheet(
            sheet,
            "Current FRW comparisons",
            "Comparisons use only Raphael matrices with the same recorded window.",
            COMPARISON_HEADERS,
        )
    if MATRICES_SHEET not in workbook.sheetnames:
        sheet = workbook.create_sheet(MATRICES_SHEET, 2)
        sheet["A1"] = "Current FRW capacitance matrices"
        sheet["A2"] = "Values are in farads."
        sheet["A1"].font = Font(size=16, bold=True, color="FFFFFF")
        sheet["A1"].fill = PatternFill("solid", fgColor="1F4E78")


def clear_rows(sheet, first_row: int) -> None:
    if sheet.max_row >= first_row:
        sheet.delete_rows(first_row, sheet.max_row - first_row + 1)


def write_jobs(sheet, jobs: list[Job], state: dict) -> None:
    clear_rows(sheet, 5)
    for row, job in enumerate(jobs, start=5):
        record = state["jobs"].get(job.name, {})
        values = [
            job.name,
            job.case,
            job.window_name,
            job.boundary,
            job.seed,
            record.get("status", "PENDING"),
            record.get("stopping_status"),
            record.get("walks"),
            None if record.get("elapsed_seconds") is None else record["elapsed_seconds"] / 3600.0,
            record.get("return_code"),
            record.get("relative_asymmetry"),
            record.get("row_sum_ratio"),
            record.get("result_file"),
            record.get("log_file"),
            record.get("command"),
            record.get("updated_utc"),
        ]
        for column, value in enumerate(values, start=1):
            sheet.cell(row, column, value)
    sheet.column_dimensions["A"].width = 65
    sheet.column_dimensions["B"].width = 16
    sheet.column_dimensions["C"].width = 14
    sheet.column_dimensions["D"].width = 36
    sheet.column_dimensions["F"].width = 26
    sheet.column_dimensions["G"].width = 24
    sheet.column_dimensions["M"].width = 55
    sheet.column_dimensions["N"].width = 55
    sheet.column_dimensions["O"].width = 120
    sheet.column_dimensions["P"].width = 24
    for row in range(5, sheet.max_row + 1):
        sheet.cell(row, 9).number_format = "0.000"
        for column in (11, 12):
            sheet.cell(row, column).number_format = "0.000000"


def completed_results(jobs: list[Job], state: dict) -> list[tuple[Job, dict, list[str], np.ndarray]]:
    results = []
    for job in jobs:
        record = state["jobs"].get(job.name, {})
        if record.get("status") not in {"CONVERGED", "MAX_WALKS_OR_SAFETY_STOP"}:
            continue
        result_path = Path(record["result_file"])
        if not result_path.exists():
            continue
        labels, matrix, _ = read_result(result_path)
        results.append((job, record, labels, matrix))
    return results


def write_matrices(sheet, results: list[tuple[Job, dict, list[str], np.ndarray]]) -> None:
    clear_rows(sheet, 4)
    row = 4
    for job, _, labels, matrix in results:
        sheet.cell(row, 1, job.name)
        sheet.cell(row, 1).font = Font(bold=True, color="1F4E78")
        for column, label in enumerate(labels, start=2):
            sheet.cell(row + 1, column, label).font = Font(bold=True)
        for i, label in enumerate(labels):
            sheet.cell(row + 2 + i, 1, label).font = Font(bold=True)
            for j, value in enumerate(matrix[i]):
                cell = sheet.cell(row + 2 + i, 2 + j, float(value))
                cell.number_format = "0.000000E+00"
        row += len(labels) + 4
    sheet.column_dimensions["A"].width = 68
    for column in "BCDEF":
        sheet.column_dimensions[column].width = 16


def write_comparisons(sheet, results, references) -> None:
    clear_rows(sheet, 5)
    row = 5
    for job, record, _, matrix in results:
        for reference_name in reference_names(job):
            if reference_name not in references:
                continue
            _, reference = references[reference_name]
            metrics = compare_matrices(matrix, reference)
            mode = reference_name.rsplit("__", 1)[-1]
            values = [
                job.name,
                job.case,
                job.window_name,
                job.boundary,
                mode,
                record.get("walks"),
                record.get("elapsed_seconds", 0.0) / 3600.0,
                metrics["relative_frobenius_error"],
                metrics["diagonal_error"],
                metrics["off_diagonal_error"],
                metrics["mean_abs_diff_pf"],
                metrics["max_abs_diff_pf"],
                metrics["frw_asymmetry"],
                metrics["reference_asymmetry"],
                metrics["frw_row_sum_ratio"],
                metrics["reference_row_sum_ratio"],
                metrics["relative_frobenius_error"] <= 0.01,
            ]
            for column, value in enumerate(values, start=1):
                sheet.cell(row, column, value)
            row += 1
    sheet.column_dimensions["A"].width = 65
    sheet.column_dimensions["D"].width = 36
    for column in range(7, 17):
        for data_row in range(5, sheet.max_row + 1):
            sheet.cell(data_row, column).number_format = "0.000000"


def save_workbook(workbook_path: Path, jobs: list[Job], state: dict) -> None:
    workbook = load_workbook(workbook_path)
    ensure_sheets(workbook)
    references = read_reference_matrices(workbook)
    results = completed_results(jobs, state)
    write_jobs(workbook[JOBS_SHEET], jobs, state)
    write_matrices(workbook[MATRICES_SHEET], results)
    write_comparisons(workbook[COMPARISONS_SHEET], results, references)
    temporary = workbook_path.with_suffix(".saving.xlsx")
    workbook.save(temporary)
    workbook.close()
    os.replace(temporary, workbook_path)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--workers": args.workers,
        "--tolerance": args.tolerance,
        "--min-walks": args.min_walks,
        "--max-walks": args.max_walks,
        "--check-every": args.check_every,
        "--max-hops": args.max_hops,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in {
        "--min-walks": args.min_walks,
        "--max-walks": args.max_walks,
        "--check-every": args.check_every,
    }.items():
        if value % 2 != 0:
            raise ValueError(f"{name} must be even for paired stratified sampling")
    if args.min_walks > args.max_walks:
        raise ValueError("--min-walks cannot exceed --max-walks")
    if args.min_hits < 0:
        raise ValueError("--min-hits must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    workbook_path = resolve(args.workbook)
    input_root = resolve(args.input_root)
    output_dir = resolve(args.output_dir)
    if not workbook_path.exists():
        raise FileNotFoundError(workbook_path)

    all_jobs = read_jobs(workbook_path, input_root)
    selected_jobs = select_jobs(all_jobs, args.case, args.job)
    if not selected_jobs:
        raise ValueError("No jobs matched the requested filters")
    for job in selected_jobs:
        if not job.input_file.exists():
            raise FileNotFoundError(job.input_file)

    checkpoint = output_dir / "checkpoint.json"
    state = load_state(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = workbook_path.with_name(f"{workbook_path.stem}.original{workbook_path.suffix}")
    if not backup.exists() and not args.list:
        shutil.copy2(workbook_path, backup)

    print(f"Selected {len(selected_jobs)} stratified jobs from {workbook_path.name}.")
    for index, job in enumerate(selected_jobs, start=1):
        job_dir = output_dir / job.name
        command = command_for(job, args, job_dir / "frw_result.json")
        if args.list:
            print(f"[{index:02d}] {job.name}")
            print("     " + shlex.join(command))
    if args.list:
        return 0

    save_workbook(workbook_path, all_jobs, state)
    for index, job in enumerate(selected_jobs, start=1):
        job_dir = output_dir / job.name
        result_file = job_dir / "frw_result.json"
        log_file = job_dir / "frw.log"
        existing = state["jobs"].get(job.name, {})
        if (
            not args.rerun
            and existing.get("status") in {"CONVERGED", "MAX_WALKS_OR_SAFETY_STOP"}
            and result_file.exists()
        ):
            print(f"[{index}/{len(selected_jobs)}] Skipping completed job: {job.name}")
            continue

        command = command_for(job, args, result_file)
        record = {
            "status": "RUNNING",
            "started_utc": utc_now(),
            "updated_utc": utc_now(),
            "command": shlex.join(command),
            "result_file": str(result_file),
            "log_file": str(log_file),
        }
        state["jobs"][job.name] = record
        save_json(checkpoint, state)
        save_workbook(workbook_path, all_jobs, state)
        print(f"\n[{index}/{len(selected_jobs)}] Starting {job.name}")

        try:
            return_code, elapsed = run_command(command, log_file)
        except KeyboardInterrupt:
            record.update(
                status="INTERRUPTED",
                updated_utc=utc_now(),
            )
            save_json(checkpoint, state)
            save_workbook(workbook_path, all_jobs, state)
            print("\nInterrupted. Progress is saved; rerun the same command to resume.")
            return 130

        record.update(
            return_code=return_code,
            elapsed_seconds=elapsed,
            updated_utc=utc_now(),
        )
        if return_code == 0 and result_file.exists():
            _, matrix, payload = read_result(result_file)
            record.update(result_summary(payload, matrix))
        else:
            record["status"] = "FAILED"
        save_json(checkpoint, state)
        save_workbook(workbook_path, all_jobs, state)
        print(
            f"[{index}/{len(selected_jobs)}] "
            f"{record['status']} in {elapsed / 3600.0:.3f} h"
        )

    print(f"\nFinished. Updated workbook: {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
