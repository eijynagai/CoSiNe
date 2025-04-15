import logging
import os
import sys

import pandas as pd

# Import benchmark and saving functions from the pipeline module.
from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    benchmark,
    generate_signed_LFR_benchmark_graph,
    save_aggregated_results_to_csv,
    save_raw_results_to_csv,
)

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="pipeline.log",  # Log messages will also be written to this file.
    filemode="w",  # Overwrite the log file each run; use "a" to append.
)


def run_pipeline_for_params(params, run_id, base_results_dir="results"):
    """
    Run the full pipeline for a given set of parameters.

    Parameters:
      params (dict): Parameter dictionary containing keys: n, tau1, tau2, mu, P_minus,
                     P_plus, min_community, average_degree, seed, resolution, and optionally n_runs.
      run_id (int): The current run's ID (used for naming results).
      base_results_dir (str): Root directory where the results CSVs will be stored.

    Returns:
      None. Raw and aggregated results are saved as CSV files in a subdirectory.
    """
    # Build a unique subdirectory name based on parameters.
    subdir = (
        f"run_{run_id}_n_{params['n']}_tau1_{params['tau1']}_"
        f"tau2_{params['tau2']}_mu_{params['mu']}"
    )
    subdir_path = os.path.join(base_results_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

    logging.info(f"Running pipeline for {subdir}...")

    # Generate benchmark graphs.
    try:
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
    except Exception as e:
        logging.error(f"Graph generation raised an exception: {e}")
        return

    if graphs is None:
        logging.error("Graph generation failed. Skipping this configuration.")
        return

    G_signed, G_pos, G_neg = graphs

    # Get resolution and number of runs, using defaults if not provided.
    resolution = params.get("resolution", 1.0)
    n_runs = params.get("n_runs", 20)

    # Run the benchmark and collect results.
    try:
        raw_results, agg_results = benchmark(
            G_signed, G_pos, G_neg, resolution=resolution, n_runs=n_runs
        )
    except Exception as e:
        logging.error(f"Benchmarking failed: {e}")
        return

    # Save the raw and aggregated results to CSV.
    raw_filename = os.path.join(subdir_path, f"benchmark_raw_res_{resolution}.csv")
    agg_filename = os.path.join(subdir_path, f"benchmark_agg_res_{resolution}.csv")

    try:
        save_raw_results_to_csv(raw_results, raw_filename)
        save_aggregated_results_to_csv(agg_results, agg_filename)
    except Exception as e:
        logging.error(f"Saving results failed: {e}")
        return

    logging.info(f"Pipeline for {subdir} completed successfully.")


def main():
    # Define the parameter CSV file path.
    param_file = os.path.join("src", "CoSiNe", "benchmarks", "params.csv")
    if not os.path.exists(param_file):
        logging.error(f"Parameter file '{param_file}' not found.")
        sys.exit(1)

    # Read parameters from the CSV file into a list of dictionaries.
    try:
        params_df = pd.read_csv(param_file)
        param_list = params_df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Failed to read or parse parameter file: {e}")
        sys.exit(1)

    # Iterate over each parameter configuration and run the pipeline.
    for idx, params in enumerate(param_list, start=1):
        run_pipeline_for_params(params, run_id=idx, base_results_dir="results")

    logging.info("All pipeline runs completed.")


if __name__ == "__main__":
    main()
