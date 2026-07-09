import argparse
from pathlib import Path

import numpy as np

from bgs.matlab_parser import parse_matlab_geometry
from bgs.vgss import build_all_net_vgss
from .random_walk import run_frw_geometry
from .visualization import visualize_3d_walk
from .metrics import calculate_frw_metrics, print_metrics


def _select_master_vgss(all_net_vgss, master_name: str | None = None):
    if not all_net_vgss:
        raise ValueError("No VGSS nets were generated.")

    if master_name is None:
        return all_net_vgss[0]

    for net_vgss in all_net_vgss:
        if getattr(net_vgss.net, "name", None) == master_name:
            return net_vgss

    available_names = ", ".join(getattr(net_vgss.net, "name", "<unknown>") for net_vgss in all_net_vgss)
    raise ValueError(f"Master net '{master_name}' was not found. Available nets: {available_names}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FRW geometric sampler on a MATLAB-style conductor input file."
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
        help="Maximum distance fallback for VGSS used only when no finite distance to another non-touching conductor can be found. Default: 10.0.",
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=5000,
        help="Number of random walks to perform. Default: 5000.",
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
        help="Path to export the FRW walk data JSON. Default: output/frw_exported_data.json",
    )
    group = parser.add_mutually_exclusive_group() #(--master | --all)
    group.add_argument(
        "--master",
        type=str,
        default=None,
        help="Name of the VGSS net to use as the master (for example: C1). Default: the first net.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run FRW for every generated net instead of a single selected master net.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable 3D visualization of the geometry and VGSS.",
    )

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    filename = args.input_file
    print(f"Reading geometry from {filename}...\n")

    try:
        system_conductors = parse_matlab_geometry(filename)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filename}'")
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

    if args.all:
        nets_to_run = all_net_vgss
        export_target = []
    else:
        master_vgss = _select_master_vgss(all_net_vgss, args.master)
        nets_to_run = [master_vgss]
        export_target = None

    results = []
    for master_vgss in nets_to_run:
        print("-" * 60)
        print("\nExecuting Batch Geometric Walk Engine...")
        print(f"Master Net: {master_vgss.net.name}")
        print(f"Conductors: {', '.join(master_vgss.net.conductor_names)}")
        print("-" * 30)


        if not args.no_visualization:
            visualize_3d_walk(system_conductors, master_vgss, rng, all_net_vgss)

        #actual FRW
        export_data = run_frw_geometry(
            system_conductors,
            master_vgss,
            rng,
            num_walks=args.walks,
        )

        print(f"Total Walks Completed: {len(export_data['walks'])}")
        print(f"Estimated Sg: {export_data['Sg']:.6f}")

        #get metrics
        export_data["metrics"] = calculate_frw_metrics(export_data["walks"], system_conductors, master_vgss)
        print_metrics(export_data["metrics"])
        results.append(export_data)


    output_filename = args.output
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting all walk data to {output_filename}...")

    with open(output_filename, "w") as outfile:
        import json
        json.dump(results[0] if export_target is None else results, outfile, indent=4)

    if args.all:
        print(f"Success! Data for {args.walks} walks successfully saved for {len(results)} nets.")
    else:
        print(f"Success! Data for {args.walks} walks successfully saved.")

