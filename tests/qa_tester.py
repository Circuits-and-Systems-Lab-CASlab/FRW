import argparse
import csv
import itertools
import os
import sys
import concurrent.futures
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# ==========================================
# 1. DYNAMIC PATH & DIRECTORY FIX
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

# ==========================================
# 2. CORE IMPORTS
# ==========================================
from bgs.matlab_parser import parse_matlab_geometry
from bgs.vgss import build_all_net_vgss
from FRW.random_walk import run_frw_geometry
from FRW.metrics import calculate_frw_metrics
from FRW.visualization import visualize_3d_walk

# ==========================================
# 3. CONFIGURATION 
# ==========================================
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output" / "qa_results"
PLOTS_DIR = OUTPUT_DIR / "plots"

def build_parser():
    parser = argparse.ArgumentParser(description="Automated QA and Convergence Tester")
    
    parser.add_argument("--files", nargs="+", help="Specific .m files to run (e.g., INPUTFILE_01.m)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 9999], help="Override RNG seeds")
    parser.add_argument("--scale-factors", nargs="+", type=float, default=[1.0, 1.25], help="Override Scale Factors")
    parser.add_argument("--max-dists", nargs="+", type=float, default=[10.0], help="Override Max Distances")
    parser.add_argument("--walks", nargs="+", type=int, default=[100, 1000, 5000, 10000], help="Override Walk Counts (N)")
    
    parser.add_argument("--no-plots", action="store_true", help="Skip generating 3D visualizations to save time")
    parser.add_argument("--verbose", action="store_true", help="Show all underlying prints from the pipeline")
    
    safe_cores = max(1, os.cpu_count() // 2)
    parser.add_argument("--workers", type=int, default=safe_cores, help="Number of CPU cores to use")
    
    return parser

# ==========================================
# MULTIPROCESSING WORKER TASK
# ==========================================
def execute_frw_task(task_payload):
    os.chdir(PROJECT_ROOT)
    
    (input_filename, conductors, net_vgss, all_net_vgss_list, 
     sf, max_dist, seed, walks, no_plots, verbose) = task_payload

    devnull = open(os.devnull, 'w')
    
    with redirect_stdout(sys.stdout if verbose else devnull):
        rng = np.random.default_rng(seed)
        
        export_data = run_frw_geometry(conductors, net_vgss, rng, num_walks=walks)
        metrics = calculate_frw_metrics(export_data["walks"], conductors, net_vgss)

        if not no_plots:
            plot_filename = f"{Path(input_filename).stem}_{net_vgss.net.name}_SF{sf}_S{seed}_W{walks}.png"
            vis_rng = np.random.default_rng(seed) 
            visualize_3d_walk(
                conductors, net_vgss, vis_rng, all_net_vgss_list, 
                save_path=PLOTS_DIR / plot_filename, show=False
            )

    devnull.close()

    term_rates = metrics.get("termination_rates", {})
    hop_stats = metrics.get("hop_statistics", {})
    
    return {
        "Input File": input_filename,
        "Net Name": net_vgss.net.name,
        "Scale Factor": sf,
        "Max Distance": max_dist,
        "Seed": seed,
        "Walks (N)": walks,
        "Estimated Sg (Convergence)": export_data["Sg"],
        "Accepted VGSS Points": export_data["sampling_stats"]["accepted_count"],
        "Geometry Rejections": export_data["sampling_stats"]["geometry_rejections"],
        "Hit Conductor Rate": term_rates.get("hit_conductor", 0),
        "Self Net Hits": metrics.get("self_net_hits", 0),
        "Other Net Hits": metrics.get("other_net_hits", 0),
        "Escaped Infinity Rate": term_rates.get("escaped", 0),
        "Max Hops Hit Rate": term_rates.get("max_hops", 0),
        "Mean Hops": hop_stats.get("mean", 0),
        "Hop Std Dev": hop_stats.get("std", 0),
        "Max Hops": hop_stats.get("max", 0)
    }

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================
def main():
    args = build_parser().parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Optimized Parallel QA Pipeline (Using {args.workers} Cores)...\n" + "="*50)
    
    # [FIX 2] - Default to ALL files if none are provided
    if args.files:
        input_files = [INPUT_DIR / f for f in args.files if (INPUT_DIR / f).exists()]
    else:
        input_files = list(INPUT_DIR.glob("*.m"))
        # Optional: Sort the files alphabetically so the terminal output feels more organized
        input_files.sort()

    if not input_files:
        print(f"No valid input files found in {INPUT_DIR}.")
        return

    tasks = []
    devnull = open(os.devnull, 'w')
    
    for input_file in input_files:
        print(f"Preparing geometry: {input_file.name}")
        try:
            conductors = parse_matlab_geometry(input_file)
        except Exception as e:
            print(f" [!] Skipping {input_file.name} due to parser error: {e}")
            continue

        for sf, max_dist in itertools.product(args.scale_factors, args.max_dists):
            with redirect_stdout(sys.stdout if args.verbose else devnull):
                all_net_vgss = build_all_net_vgss(conductors, scale_factor=sf, max_distance=max_dist)

            for net_vgss in all_net_vgss:
                for seed, walks in itertools.product(args.seeds, args.walks):
                    tasks.append((
                        input_file.name, conductors, net_vgss, all_net_vgss, 
                        sf, max_dist, seed, walks, args.no_plots, args.verbose
                    ))
    
    devnull.close()
    
    total_tasks = len(tasks)
    print(f"\nSuccessfully built {total_tasks} simulation tasks.")
    print("Executing tests... (This may take a moment)\n")

    results_data = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(execute_frw_task, task): task for task in tasks}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            net_name, sf, seed, walks = task[1].name if hasattr(task[1], 'name') else task[2].net.name, task[4], task[6], task[7]
            
            try:
                result = future.result()
                results_data.append(result)
                # Keep the terminal output real-time, even if it prints out of order
                print(f"[{completed}/{total_tasks}] Finished | Net: {net_name:<6} | SF: {sf} | Seed: {seed:<5} | Walks: {walks:<6}")
            except Exception as exc:
                print(f"[{completed}/{total_tasks}] FAILED   | Net: {net_name:<6} | Error: {exc}")

    # [FIX 1] - Sort the final data before writing to CSV
    if results_data:
        results_data.sort(key=lambda row: (
            row["Input File"], 
            row["Net Name"], 
            row["Scale Factor"], 
            row["Walks (N)"], 
            row["Seed"]
        ))

        csv_file = OUTPUT_DIR / "qa_convergence_results.csv"
        headers = results_data[0].keys()
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results_data)
            
        print(f"\n" + "="*50)
        print(f"QA TESTING COMPLETE")
        print(f"Results exported to: {csv_file}")
    else:
        print("\nNo test data was generated.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()