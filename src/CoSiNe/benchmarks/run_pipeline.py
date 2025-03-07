import os
import sys

import pandas as pd

# Import your benchmark and saving functions from your pipeline code.
from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import *


def run_pipeline_for_params(params, run_id, base_results_dir="results"):
    """
    Run the entire pipeline for one set of parameters.

    Parameters:
      params (dict): Parameter dictionary containing keys: n, tau1, tau2, mu, P_minus, P_plus, min_community, average_degree, seed, resolution.
      run_id (int): The index of this run (for naming).
      base_results_dir (str): Base directory to store all results.

    Returns:
      None. All output CSV files are saved in a subdirectory.
    """
    # Build a unique subdirectory name based on parameters
    subdir = f"run_{run_id}_n_{params['n']}_tau1_{params['tau1']}_tau2_{params['tau2']}_mu_{params['mu']}"
    subdir_path = os.path.join(base_results_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

    # Print configuration info
    print(f"\n🔹 Running pipeline for {subdir}...")

    # Generate the graphs.
    graphs = generate_signed_LFR_benchmark_graph(
        n=params["n"],
        tau1=params["tau1"],
        tau2=params["tau2"],
        mu=params["mu"],
        P_minus=params["P_minus"],
        P_plus=params["P_plus"],
        min_community=params["min_community"],
        average_degree=params["average_degree"],
        seed=params["seed"],
    )
    if graphs is None:
        print("❌ Graph generation failed. Skipping this configuration.")
        return
    G_signed, G_pos, G_neg = graphs

    resolution = params.get("resolution", 1.0)
    n_runs = params.get("n_runs", 20)
    raw_results, agg_results = benchmark(
        G_signed, G_pos, G_neg, resolution=resolution, n_runs=n_runs
    )

    # Save raw and aggregated results to CSV in the subdirectory.
    raw_filename = os.path.join(subdir_path, f"benchmark_raw_res_{resolution}.csv")
    agg_filename = os.path.join(subdir_path, f"benchmark_agg_res_{resolution}.csv")
    save_raw_results_to_csv(raw_results, raw_filename)
    save_aggregated_results_to_csv(agg_results, agg_filename)

    # (Optional) You can also generate plots and save them in the same subdirectory.
    # For example, call your plotting function with output_dir=subdir_path.
    print(f"✅ Pipeline for {subdir} completed.\n")


if __name__ == "__main__":
    # Read the external parameter table.
    param_file = "src/CoSiNe/benchmarks/params.csv"
    if not os.path.exists(param_file):
        print(f"Parameter file {param_file} not found.")
        sys.exit(1)

    params_df = pd.read_csv(param_file)
    # Convert DataFrame rows to dictionaries.
    param_list = params_df.to_dict(orient="records")

    # Run pipeline for each parameter configuration.
    for idx, params in enumerate(param_list, start=1):
        run_pipeline_for_params(params, run_id=idx, base_results_dir="results")

    print("✅ All pipeline runs completed.")
