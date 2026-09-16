import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bgs.matlab_parser import parse_matlab_geometry
from bgs.sampling import VGSSSamplingStats
from bgs.vgss import build_all_net_vgss
from GGFT.ggft import STRATIFIED_FIRST_TRANSITION, CubeGFT
from .capacitance import (
    CapacitanceRowAccumulator,
    assemble_capacitance_matrix,
    print_capacitance_matrix,
)
from .metrics import MetricsAccumulator
from .convergence import check_row_convergence
from .random_walk import get_system_bounds, run_one_walk
from .visualization import extract_walk_trace, visualize_3d_walk


def _select_master_vgss(all_net_vgss, master_name: str | None = None):
    if not all_net_vgss:
        raise ValueError("No VGSS nets were generated.")

    if master_name is None:
        return all_net_vgss[0]

    for net_vgss in all_net_vgss:
        if getattr(net_vgss.net, "name", None) == master_name:
            return net_vgss

    available_names = ", ".join(
        getattr(net_vgss.net, "name", "<unknown>")
        for net_vgss in all_net_vgss
    )
    raise ValueError(
        f"Master net {master_name!r} was not found. "
        f"Available nets: {available_names}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FRW on a MATLAB-style conductor input file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the MATLAB-style geometry input file.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.25,
        help="Scale factor for VGSS generation. Default: 1.25.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=10.0,
        help=(
            "Maximum-distance fallback for VGSS generation. "
            "Default: 10.0."
        ),
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=None,
        help=(
            "Run exactly this positive even number of walks per master net, "
            "disabling adaptive convergence."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Adaptive row-max relative 3SE tolerance. Default: 0.03.",
    )
    parser.add_argument(
        "--min-walks",
        type=int,
        default=2000,
        help="Minimum adaptive walks per row. Default: 2000.",
    )
    parser.add_argument(
        "--max-walks",
        type=int,
        default=1_000_000,
        help="Maximum adaptive walks per row. Default: 1000000.",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=200,
        help="Check convergence after this many additional walks. Default: 200.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=10,
        help="Minimum hits required for every target entry. Default: 10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility. Default: 1234.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "frw_exported_data.json",
        help=(
            "Path to export FRW and capacitance data. "
            "Default: output/frw_exported_data.json"
        ),
    )

    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help=(
            "Run only this VGSS net as master, for example Net1. "
            "If omitted, run every net and assemble the full matrix."
        ),
    )

    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Show a 3D visualization of one saved walk per master net.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Maximum number of master-net rows to run in parallel. "
            "The actual count is limited by the number of selected rows. "
            "Default: 4."
        ),
    )

    return parser


def _row_rng(seed: int, row_index: int) -> np.random.Generator:
    """Return the deterministic RNG stream assigned to one matrix row."""

    return np.random.default_rng(np.random.SeedSequence([seed, row_index]))


def _worker_count(requested_workers: int, row_count: int) -> int:
    """Do not create more worker processes than there are rows to run."""

    return min(requested_workers, row_count)


def _run_master_row(
    conductors,
    master_vgss,
    *,
    row_index: int,
    seed: int,
    adaptive: bool,
    walk_limit: int,
    minimum_walks: int | None,
    check_every: int | None,
    relative_tolerance: float | None,
    minimum_target_hits: int | None,
    record_visualization: bool,
) -> dict:
    """Calculate one independently sampled capacitance-matrix row."""

    rng = _row_rng(seed, row_index)
    gft = CubeGFT.load_default()
    system_center, system_size = get_system_bounds(conductors)
    math_kill_radius = system_size * 500_000.0
    first_transition_mode = STRATIFIED_FIRST_TRANSITION

    sampling_stats = VGSSSamplingStats()
    capacitance_accumulator = CapacitanceRowAccumulator(
        conductors,
        master_vgss,
        first_transition_mode=first_transition_mode,
    )
    metrics_accumulator = MetricsAccumulator(conductors, master_vgss)
    visualization_outcome = None
    convergence_result = None
    convergence_checks = 0
    stop_reason = "maximum_walks" if adaptive else "fixed_walk_count"

    for walk_index in range(walk_limit):
        record_path = record_visualization and walk_index == 0
        outcome = run_one_walk(
            conductors=conductors,
            master_vgss=master_vgss,
            gft=gft,
            rng=rng,
            walk_index=walk_index,
            sampling_stats=sampling_stats,
            system_center=system_center,
            math_kill_radius=math_kill_radius,
            first_transition_mode=first_transition_mode,
            record_path=record_path,
        )
        sample_completed = capacitance_accumulator.update(outcome)
        metrics_accumulator.update(outcome)

        if record_path:
            visualization_outcome = outcome

        if not adaptive or not sample_completed:
            continue

        completed_walks = capacitance_accumulator.walk_count
        
        # Determine whether to check convergence at this point.
        should_check = (completed_walks >= minimum_walks and ( completed_walks % check_every == 0 or completed_walks == walk_limit ) )
        if not should_check:
            continue

        convergence_result = check_row_convergence(
            capacitance_accumulator.statistics(),
            relative_tolerance=relative_tolerance,
            minimum_target_hits=minimum_target_hits,
        )
        convergence_checks += 1
        if convergence_result["converged"]:
            stop_reason = "converged"
            break

    if adaptive:
        if convergence_result is None:
            convergence_result = check_row_convergence(
                capacitance_accumulator.statistics(),
                relative_tolerance=relative_tolerance,
                minimum_target_hits=minimum_target_hits,
            )
        convergence = {
            "enabled": True,
            "stop_reason": stop_reason,
            "walks_used": capacitance_accumulator.walk_count,
            "pair_samples": capacitance_accumulator.sample_count,
            "minimum_walks": minimum_walks,
            "maximum_walks": walk_limit,
            "check_every": check_every,
            "checks_performed": convergence_checks,
            **convergence_result,
        }
    else:
        convergence = {
            "enabled": False,
            "converged": None,
            "stop_reason": stop_reason,
            "walks_used": capacitance_accumulator.walk_count,
            "pair_samples": capacitance_accumulator.sample_count,
        }

    context = master_vgss.sampling_context
    h_estimate = sampling_stats.estimate_H(context)
    capacitance_row = capacitance_accumulator.finalize(h_estimate)
    capacitance_row["convergence"] = convergence
    metrics = metrics_accumulator.finalize()

    export_data = {
        "master_net": master_vgss.net.name,
        "num_walks": capacitance_accumulator.walk_count,
        "Sg": sampling_stats.estimate_surface_area(context),
        "mean_p": sampling_stats.estimate_mean_p(),
        "H": h_estimate,
        "sampling_stats": sampling_stats.as_dict(context),
        "first_transition_mode": first_transition_mode,
        "convergence": convergence,
        "metrics": metrics,
        "capacitance": capacitance_row,
    }
    if visualization_outcome is not None:
        export_data["visualization_walk"] = visualization_outcome.as_dict()

    return {
        "row_index": row_index,
        "master_net": master_vgss.net.name,
        "capacitance_row": capacitance_row,
        "export_data": export_data,
    }


def _print_row_progress(completed: int, total: int, row_result: dict) -> None:
    convergence = row_result["export_data"]["convergence"]
    walks = convergence["walks_used"]
    master_net = row_result["master_net"]

    if not convergence["enabled"]:
        details = f"fixed count, {walks:,} walks"
    else:
        ratio = convergence["final_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.3e}"
        status = "converged" if convergence["converged"] else "maximum walks"
        details = f"{status}, {walks:,} walks, ratio={ratio_text}"

    print(f"[{completed}/{total}] {master_net}: {details}")


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    if args.walks is not None:
        if args.walks <= 0 or args.walks % 2 != 0:
            parser.error("--walks must be a positive even integer")
        adaptive = False
        walk_limit = args.walks
        minimum_walks = None
        check_every = None
        relative_tolerance = None
        minimum_target_hits = None
    else:
        even_positive = {
            "--min-walks": args.min_walks,
            "--max-walks": args.max_walks,
            "--check-every": args.check_every,
        }
        for name, value in even_positive.items():
            if value <= 0 or value % 2 != 0:
                parser.error(f"{name} must be a positive even integer")
        if args.min_walks > args.max_walks:
            parser.error("--min-walks cannot exceed --max-walks")
        if not np.isfinite(args.tolerance) or args.tolerance <= 0.0:
            parser.error("--tolerance must be finite and positive")
        if args.min_hits < 0:
            parser.error("--min-hits must be non-negative")

        adaptive = True
        walk_limit = args.max_walks
        minimum_walks = args.min_walks
        check_every = args.check_every
        relative_tolerance = args.tolerance
        minimum_target_hits = args.min_hits

    filename = args.input_file
    print(f"Reading geometry from {filename}...\n")

    try:
        system_conductors = parse_matlab_geometry(filename)
    except FileNotFoundError:
        print(f"ERROR: Could not find {filename!s}")
        raise SystemExit(1)

    if not system_conductors:
        print("No conductors found in the file.")
        raise SystemExit(1)

    all_net_vgss = build_all_net_vgss(
        system_conductors,
        scale_factor=args.scale_factor,
        max_distance=args.max_distance,
    )

    run_all_nets = args.master is None
    nets_to_run = (
        all_net_vgss
        if run_all_nets
        else [_select_master_vgss(all_net_vgss, args.master)]
    )

    net_indices = {
        net_vgss.net.name: index
        for index, net_vgss in enumerate(all_net_vgss)
    }
    row_count = len(nets_to_run)
    active_workers = _worker_count(args.workers, row_count)
    row_results = {}

    print("\nExecuting Online Geometric Walk Engine...")
    if active_workers == 1:
        print(f"Running {row_count} row(s) sequentially.\n")
        for completed, master_vgss in enumerate(nets_to_run, start=1):
            row_index = net_indices[master_vgss.net.name]
            result = _run_master_row(
                system_conductors,
                master_vgss,
                row_index=row_index,
                seed=args.seed,
                adaptive=adaptive,
                walk_limit=walk_limit,
                minimum_walks=minimum_walks,
                check_every=check_every,
                relative_tolerance=relative_tolerance,
                minimum_target_hits=minimum_target_hits,
                record_visualization=args.visualization,
            )
            row_results[row_index] = result
            _print_row_progress(completed, row_count, result)
    else:
        print(
            f"Running {row_count} rows with "
            f"{active_workers} worker processes.\n"
        )
        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            future_to_net = {}
            for master_vgss in nets_to_run:
                row_index = net_indices[master_vgss.net.name]
                future = executor.submit(
                    _run_master_row,
                    system_conductors,
                    master_vgss,
                    row_index=row_index,
                    seed=args.seed,
                    adaptive=adaptive,
                    walk_limit=walk_limit,
                    minimum_walks=minimum_walks,
                    check_every=check_every,
                    relative_tolerance=relative_tolerance,
                    minimum_target_hits=minimum_target_hits,
                    record_visualization=args.visualization,
                )
                future_to_net[future] = master_vgss.net.name

            for completed, future in enumerate(
                as_completed(future_to_net), start=1
            ):
                master_name = future_to_net[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"FRW row {master_name} failed in a worker process."
                    ) from exc
                row_results[result["row_index"]] = result
                _print_row_progress(completed, row_count, result)

    ordered_row_results = [
        row_results[net_indices[master_vgss.net.name]]
        for master_vgss in nets_to_run
    ]
    results = [item["export_data"] for item in ordered_row_results]
    capacitance_rows = [
        item["capacitance_row"] for item in ordered_row_results
    ]

    if args.visualization:
        vgss_by_name = {
            net_vgss.net.name: net_vgss for net_vgss in all_net_vgss
        }
        for result in results:
            visualization_walk = result.get("visualization_walk")
            if visualization_walk is None:
                continue
            path_points, maximal_cubes, terminal = extract_walk_trace(
                visualization_walk
            )
            visualize_3d_walk(
                system_conductors,
                vgss_by_name[result["master_net"]],
                path_points,
                maximal_cubes,
                terminal,
                all_net_vgss,
            )

    # Assemble the matrix only when every master-net row was requested.
    matrix_result = None
    if run_all_nets:
        nets = [net_vgss.net.name for net_vgss in all_net_vgss]
        matrix_result = assemble_capacitance_matrix(capacitance_rows, nets)
        print_capacitance_matrix(matrix_result)

    if run_all_nets:
        export_payload = {
            "capacitance_matrix": matrix_result,
            "net_results": results,
        }
    else:
        export_payload = results[0]

    output_filename = args.output
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting results to {output_filename}...")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(export_payload, outfile, indent=4)

    if run_all_nets:
        print(
            f"Success! Saved {len(results)} independently sampled "
            "master-net rows."
        )
    else:
        print(f"Success! Saved {results[0]['num_walks']} walks.")


if __name__ == "__main__":
    main()
