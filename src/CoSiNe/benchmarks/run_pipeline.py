import logging
import os
import sys
import pandas as pd
from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    benchmark,
    generate_signed_LFR_benchmark_graph,
    plot_boxplots_for_metrics,
    save_aggregated_results_to_csv,
    save_raw_results_to_csv,
)

logging.getLogger().handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="pipeline.log",
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


def run_pipeline_for_params(params, run_id, base_results_dir="results"):
    """
    Run the pipeline for a given set of parameters, across multiple resolutions.

    For each set of parameters (each row in params.csv), we:
      1. Create a subdirectory unique to this parameter set.
      2. Generate G_signed, G_pos, G_neg (the signed graph).
      3. Loop over a list of resolution values. For each resolution:
         - Run the benchmark n_runs times
         - Save raw & aggregated results (with resolution in filename)
         - Optionally, plot from the in-memory raw_results

    Parameters:
      params (dict): Parameter dictionary (n, tau1, tau2, mu, etc.).
      run_id (int): Index for naming the subdirectory.
      base_results_dir (str): Root directory to store results.
    """
    # 1. Build the subdirectory name from parameters
    subdir = (
        f"run_{run_id}_n_{params['n']}_tau1_{params['tau1']}_"
        f"tau2_{params['tau2']}_mu_{params['mu']}_pp_{params['P_plus']}_"
        f"pm_{params['P_minus']}"
    )
    subdir_path = os.path.join(base_results_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

    logging.info(f"🔹 Running pipeline for {subdir}...")

    # 2. Generate the graphs
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
        logging.error(f"Graph generation error: {e}")
        return

    if graphs is None:
        logging.error("Graph generation returned None. Skipping.")
        return

    G_signed, G_pos, G_neg = graphs

    # 3. Determine how many runs & define the range of resolutions
    n_runs = params.get("n_runs", 20)
    # Example: you can store the resolution values in your params.csv OR define them here:
    resolution_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # 4. Loop over the resolution values
    for res in resolution_values:
        logging.info(f"Starting benchmark at resolution={res}, n_runs={n_runs}...")
        try:
            raw_results, agg_results = benchmark(
                G_signed, G_pos, G_neg, resolution=res, n_runs=n_runs
            )
        except Exception as e:
            logging.error(f"Benchmark failed at resolution={res}: {e}")
            continue

        # 4a. Save CSV files for raw & aggregated
        raw_filename = os.path.join(subdir_path, f"benchmark_raw_res_{res}.csv")
        agg_filename = os.path.join(subdir_path, f"benchmark_agg_res_{res}.csv")

        try:
            save_raw_results_to_csv(raw_results, raw_filename)
            save_aggregated_results_to_csv(agg_results, agg_filename)
        except Exception as e:
            logging.error(f"Saving CSV failed for resolution={res}: {e}")
            continue

        # 4b. (Optional) Plot from raw_results in memory, storing in the same subdir
        metrics_to_plot = [
            "NMI",
            "ARI",
            "F1",
            "Number of Communities",
            "Execution Time (s)",
        ]
        try:
            plot_boxplots_for_metrics(
                raw_results,
                metrics=metrics_to_plot,
                output_dir=subdir_path,
                resolution=res,
            )
        except Exception as e:
            logging.error(f"Plotting failed for resolution={res}: {e}")
            # Not a critical failure, so continue

    logging.info(f"Pipeline for {subdir} finished.\n")


def main():
    # A. Locate your parameter CSV
    param_file = os.path.join("src", "CoSiNe", "benchmarks", "params.csv")
    if not os.path.exists(param_file):
        logging.error(f"Parameter file '{param_file}' not found.")
        sys.exit(1)

    # B. Read the parameter table
    try:
        params_df = pd.read_csv(param_file)
        param_list = params_df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Failed to read params.csv: {e}")
        sys.exit(1)

    # C. Run pipeline for each set of parameters
    for idx, params in enumerate(param_list, start=1):
        run_pipeline_for_params(params, run_id=idx, base_results_dir="results")

    logging.info("All pipeline runs completed.")


if __name__ == "__main__":
    main()
