import argparse
import json
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
        default=5000,
        help=(
            "Positive even number of random walks per master net. "
            "Default: 5000."
        ),
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

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args.walks <= 0 or args.walks % 2 != 0:
        parser.error(
            "--walks must be a positive even integer when the CLI uses "
            "stratified first-transition sampling"
        )

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

    rng = np.random.default_rng(args.seed)
    gft = CubeGFT.load_default()
    system_center, system_size = get_system_bounds(system_conductors)
    math_kill_radius = system_size * 500_000.0
    first_transition_mode = STRATIFIED_FIRST_TRANSITION

    run_all_nets = args.master is None
    nets_to_run = (
        all_net_vgss
        if run_all_nets
        else [_select_master_vgss(all_net_vgss, args.master)]
    )

    results = []
    capacitance_rows = []

    for master_vgss in nets_to_run:
        print("-" * 60)
        print("\nExecuting Online Geometric Walk Engine...")
        print(f"Master Net: {master_vgss.net.name}")
        print("Conductors: " + ", ".join(master_vgss.net.conductor_names))
        print("-" * 30)

        sampling_stats = VGSSSamplingStats()
        capacitance_accumulator = CapacitanceRowAccumulator(
            system_conductors,
            master_vgss,
            first_transition_mode=first_transition_mode,
        )
        metrics_accumulator = MetricsAccumulator(
            system_conductors,
            master_vgss,
        )
        visualization_outcome = None

        for walk_index in range(args.walks):
            percent_done = (walk_index + 1) / args.walks * 100.0
            print(
                f"\rProgress: {percent_done:.1f}% "
                f"({walk_index + 1}/{args.walks})",
                end="",
                flush=True,
            )

            record_path = args.visualization and walk_index == 0 #record only the first walk for visualization
            outcome = run_one_walk(
                conductors=system_conductors,
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
            capacitance_accumulator.update(outcome)
            metrics_accumulator.update(outcome)

            if record_path:
                visualization_outcome = outcome

        print()

        context = master_vgss.sampling_context
        h_estimate = sampling_stats.estimate_H(context)
        capacitance_row = capacitance_accumulator.finalize(h_estimate)
        metrics = metrics_accumulator.finalize()

        export_data = {
            "master_net": master_vgss.net.name,
            "num_walks": args.walks,
            "Sg": sampling_stats.estimate_surface_area(context),
            "mean_p": sampling_stats.estimate_mean_p(),
            "H": h_estimate,
            "sampling_stats": sampling_stats.as_dict(context),
            "first_transition_mode": first_transition_mode,
            "metrics": metrics,
            "capacitance": capacitance_row,
        }

        if visualization_outcome is not None:
            visualization_walk = visualization_outcome.as_dict()
            export_data["visualization_walk"] = visualization_walk
            path_points, maximal_cubes, terminal = extract_walk_trace(
                visualization_walk
            )
            visualize_3d_walk(
                system_conductors,
                master_vgss,
                path_points,
                maximal_cubes,
                terminal,
                all_net_vgss,
            )
        capacitance_rows.append(capacitance_row)

        results.append(export_data)

    #-----------capacitance----------
    matrix_result = None
    if run_all_nets:
        nets=[net_vgss.net.name for net_vgss in all_net_vgss]
        matrix_result = assemble_capacitance_matrix( capacitance_rows, nets, )
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
            f"Success! Saved {args.walks} walks for "
            f"{len(results)} master nets."
        )
    else:
        print(f"Success! Saved {args.walks} walks.")
