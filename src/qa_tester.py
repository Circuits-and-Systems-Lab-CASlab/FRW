import csv
import itertools
from pathlib import Path

import numpy as np

# Import core functionalities directly from your bgs and FRW modules
from bgs.matlab_parser import parse_matlab_geometry
from bgs.vgss import build_all_net_vgss
from FRW.random_walk import run_frw_geometry
from FRW.metrics import calculate_frw_metrics
from FRW.visualization import visualize_3d_walk

# ==========================================
# CONFIGURATION & TEST MATRIX
# ==========================================
INPUT_DIR = Path("../input")
OUTPUT_DIR = Path("../output/qa_results")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create output directories if they don't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Select the input files you want to batch process. 
# (You can also use list(INPUT_DIR.glob("*.m")) to run everything)
TARGET_FILES = [
    "INPUTFILE_01.m",
    "INPUTFILE_06_Tilted.m",
    "INPUTFILE_TOUCHING_DIRECT.m"
]

# The Parameter Grid
SEEDS = [1234, 9999]                     # Different RNG seeds for robustness
SCALE_FACTORS = [1.0, 1.25]              # Algorithm 4.2 scaling
MAX_DISTANCES = [10.0]                   # Fallback extents
WALK_COUNTS = [100, 1000, 5000, 10000]   # Increasing N points to test H/Sg convergence


def main():
    print(f"Starting Automated QA Testing Pipeline...\n" + "="*40)
    results_data = []

    # Filter inputs based on TARGET_FILES
    # Grab every single .m file in the directory
    input_files = list(INPUT_DIR.glob("*.m"))

    if not input_files:
        print(f"No input files found in {INPUT_DIR} matching targets.")
        return

    for input_file in input_files:
        print(f"\nProcessing Geometry: {input_file.name}")
        
        try:
            conductors = parse_matlab_geometry(input_file)
        except Exception as e:
            print(f" ERROR: Skipping {input_file.name} due to parser error: {e}")
            continue

        # Iterate over VGSS generation parameters
        for sf, max_dist in itertools.product(SCALE_FACTORS, MAX_DISTANCES):
            print(f" -> Building VGSS (Scale Factor: {sf}, Max Dist: {max_dist})")
            all_net_vgss = build_all_net_vgss(conductors, scale_factor=sf, max_distance=max_dist)

            # Iterate over every detected net
            for net_vgss in all_net_vgss:
                
                # Iterate over random seeds and walk counts (N)
                for seed, walks in itertools.product(SEEDS, WALK_COUNTS):
                    print(f"    Net: {net_vgss.net.name:<6} | Seed: {seed:<5} | Walks (N): {walks}")
                    
                    # Initialize RNG
                    rng = np.random.default_rng(seed)

                    # Execute the Random Walk
                    export_data = run_frw_geometry(
                        conductors,
                        net_vgss,
                        rng,
                        num_walks=walks
                    )

                    # Calculate Metrics
                    metrics = calculate_frw_metrics(export_data["walks"], conductors, net_vgss)

                    # Generate and save the visualization silently
                    plot_filename = f"{input_file.stem}_{net_vgss.net.name}_SF{sf}_S{seed}_W{walks}.png"
                    plot_path = PLOTS_DIR / plot_filename
                    
                    # Create a fresh RNG with the same seed so the visual walk matches exactly
                    vis_rng = np.random.default_rng(seed) 
                    visualize_3d_walk(
                        conductors, 
                        net_vgss, 
                        vis_rng, 
                        all_net_vgss, 
                        save_path=plot_path, 
                        show=False # Prevent interactive windows from pausing the script
                    )

                    # Extract specific termination and metric values safely
                    term_rates = metrics.get("termination_rates", {})
                    hop_stats = metrics.get("hop_statistics", {})
                    self_hits = metrics.get("self_net_hits", 0)
                    other_hits = metrics.get("other_net_hits", 0)

                    # Append flattened data row for CSV export
                    results_data.append({
                        "Input File": input_file.name,
                        "Net Name": net_vgss.net.name,
                        "Scale Factor": sf,
                        "Max Distance": max_dist,
                        "Seed": seed,
                        "Walks (N)": walks,
                        "Estimated Sg (Convergence)": export_data["Sg"],
                        "Accepted VGSS Points": export_data["sampling_stats"]["accepted_count"],
                        "Geometry Rejections": export_data["sampling_stats"]["geometry_rejections"],
                        "Hit Conductor Rate": term_rates.get("hit_conductor", 0),
                        "Escaped Infinity Rate": term_rates.get("escaped", 0),
                        "Max Hops Hit Rate": term_rates.get("max_hops", 0),
                        "Self Net Hits": self_hits,          # <-- NEW METRIC
                        "Other Net Hits": other_hits,        # <-- NEW METRIC
                        "Mean Hops": hop_stats.get("mean", 0),
                        "Max Hops": hop_stats.get("max", 0)
                    })

    # Export Data to an Excel-readable CSV
    csv_file = OUTPUT_DIR / "qa_convergence_results.csv"
    if results_data:
        # Extract headers from dictionary keys
        headers = results_data[0].keys()
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results_data)
            
        print("\n" + "="*40)
        print("QA TESTING COMPLETE")
        print(f"Results exported to: {csv_file}")
        print(f"3D Visualizations saved in: {PLOTS_DIR}")
    else:
        print("No test data was generated.")

if __name__ == "__main__":
    main()