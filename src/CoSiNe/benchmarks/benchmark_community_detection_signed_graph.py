import csv
import logging
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from palettable.colorbrewer.qualitative import Set1_9
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from CoSiNe.community_detection.external.signedLFR import signed_LFR_benchmark_graph
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.spectral_clustering import run_spectral_clustering

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_ground_truth_communities(G):
    """
    Extract the ground-truth community labels from the 'community' attribute of each node.
    If a node has multiple memberships, the first one is taken; if no community is found, assign -1.
    """
    ground_truth = []
    for node in sorted(G.nodes()):
        comm_attr = G.nodes[node].get("community", None)
        if not comm_attr:
            ground_truth.append(-1)
        elif isinstance(comm_attr, (set, list)):
            ground_truth.append(list(comm_attr)[0])
        else:
            ground_truth.append(comm_attr)
    return ground_truth


def benchmark_signed_and_unsigned(G_signed, G_pos, G_neg, resolution=1.0):
    """
    Run both signed and unsigned community detection methods.

    For LouvainSigned, multiple runs are executed with alpha values (0.1 to 1.0).
    Other methods (Louvain and Leiden) use the provided resolution.
    Infomap, Greedy modularity, and Spectral clustering are run on the positive graph.

    Returns:
      A list of dictionaries with results for one benchmark run.
    """
    logging.info(f"Benchmarking started at resolution = {resolution}.")

    methods = {}
    # Prepare LouvainSigned variants (requires both positive and negative graphs)
    for alpha in [round(x * 0.1, 1) for x in range(1, 11)]:
        method_name = f"LouvainSigned_a{alpha}"
        methods[method_name] = (
            lambda G_pos, G_neg, alpha=alpha, resolution=resolution: run_louvain_signed(
                G_pos, G_neg, alpha=alpha, resolution=resolution
            ),
            "signed",
        )
    # Unsigned methods accepting resolution.
    methods["Louvain"] = (
        lambda G_pos, resolution=resolution: run_louvain(G_pos, resolution=resolution),
        "pos",
    )
    methods["Leiden"] = (
        lambda G_pos, resolution=resolution: run_leiden(G_pos, resolution=resolution),
        "pos",
    )
    # Methods that do not require the resolution parameter.
    methods["Infomap"] = (run_infomap, "pos")
    methods["Greedy modularity"] = (run_greedy_modularity, "pos")
    methods["Spectral clustering"] = (run_spectral_clustering, "pos")

    ground_truth = get_ground_truth_communities(G_signed)
    run_results = []

    for method_name, (func, graph_type) in methods.items():
        start_time = time.time()

        # Select the appropriate graph and compute communities.
        if graph_type == "signed":
            communities = func(G_pos, G_neg)
            node_list = sorted(G_signed.nodes())
        else:
            communities = (
                func(G_pos, resolution=resolution)
                if "resolution" in func.__code__.co_varnames
                else func(G_pos)
            )
            node_list = sorted(G_pos.nodes())

        execution_time = time.time() - start_time

        # Log missing community assignments if any.
        missing = [node for node in node_list if node not in communities]
        if missing:
            logging.warning(
                f"Nodes missing community assignments for {method_name}: {missing}"
            )

        if isinstance(communities, dict):
            predicted = [communities[node] for node in node_list]
        else:
            predicted = communities

        nmi = normalized_mutual_info_score(ground_truth, predicted)
        ari = adjusted_rand_score(ground_truth, predicted)
        f1 = f1_score(ground_truth, predicted, average="macro")
        num_communities = len(set(predicted))

        logging.info(
            f"Method {method_name}: {num_communities} communities in {execution_time:.2f}s "
            f"(NMI={nmi:.3f}, ARI={ari:.3f})"
        )

        run_results.append(
            {
                "Method": method_name,
                "Graph Used": graph_type,
                "Resolution": resolution,
                "Execution Time (s)": round(execution_time, 3),
                "Number of Communities": num_communities,
                "NMI": round(nmi, 3),
                "ARI": round(ari, 3),
                "F1": round(f1, 3),
            }
        )

    logging.info(f"Benchmarking finished at resolution = {resolution}.")
    return run_results


def benchmark(G_signed, G_pos, G_neg, resolution=1.0, n_runs=20):
    """
    Run benchmarks for a given resolution over n_runs iterations.

    Returns:
      raw_results: List of dictionaries with results for each run.
      agg_results: List of aggregated result dictionaries with mean and std of each metric.
    """
    raw_results = []
    for run in range(n_runs):
        logging.info(f"Run {run+1}/{n_runs} at resolution = {resolution}...")
        run_results = benchmark_signed_and_unsigned(
            G_signed, G_pos, G_neg, resolution=resolution
        )
        for row in run_results:
            row["Run"] = run + 1
            raw_results.append(row)

    # Aggregate results by method.
    aggregated = {}
    for row in raw_results:
        method = row["Method"]
        if method not in aggregated:
            aggregated[method] = {
                "Execution Time (s)": [],
                "Number of Communities": [],
                "NMI": [],
                "ARI": [],
                "F1": [],
            }
        aggregated[method]["Execution Time (s)"].append(row["Execution Time (s)"])
        aggregated[method]["Number of Communities"].append(row["Number of Communities"])
        aggregated[method]["NMI"].append(row["NMI"])
        aggregated[method]["ARI"].append(row["ARI"])
        aggregated[method]["F1"].append(row["F1"])

    agg_results = []
    for method, metrics in aggregated.items():
        agg_results.append(
            {
                "Method": method,
                "Resolution": resolution,
                "Graph Used": "signed" if "Signed" in method else "pos",
                "Execution Time (s) Avg": round(
                    np.mean(metrics["Execution Time (s)"]), 3
                ),
                "Execution Time (s) Std": round(
                    np.std(metrics["Execution Time (s)"]), 3
                ),
                "Number of Communities Avg": round(
                    np.mean(metrics["Number of Communities"]), 3
                ),
                "Number of Communities Std": round(
                    np.std(metrics["Number of Communities"]), 3
                ),
                "NMI Avg": round(np.mean(metrics["NMI"]), 3),
                "NMI Std": round(np.std(metrics["NMI"]), 3),
                "ARI Avg": round(np.mean(metrics["ARI"]), 3),
                "ARI Std": round(np.std(metrics["ARI"]), 3),
                "F1 Avg": round(np.mean(metrics["F1"]), 3),
                "F1 Std": round(np.std(metrics["F1"]), 3),
            }
        )
    return raw_results, agg_results


def generate_signed_LFR_benchmark_graph(
    n, tau1, tau2, mu, P_minus, P_plus, min_community, average_degree, seed
):
    """
    Generate an LFR benchmark graph and derive its positive/negative subgraphs.

    Logs graph generation parameters, and returns a tuple of graphs:
      (G_signed, G_pos, G_neg)
    """
    logging.info("Starting LFR graph generation with parameters:")
    logging.info(
        f"n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, "
        f"P_minus={P_minus}, P_plus={P_plus}, min_comm={min_community}, "
        f"avg_deg={average_degree}, seed={seed}"
    )

    start_time = time.time()
    try:
        G_signed = signed_LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            P_minus=P_minus,
            P_plus=P_plus,
            average_degree=average_degree,
            min_community=min_community,
            seed=seed,
        )
        logging.info(f"LFR Graph generated in {time.time() - start_time:.2f} sec")
    except nx.exception.ExceededMaxIterations:
        logging.error("LFR Generation failed due to max iterations.")
        return None

    # Construct positive subgraph.
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) > 0:
            G_pos.add_edge(u, v, **d)
    logging.info(
        f"Positive subgraph: {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges"
    )

    # Construct negative subgraph.
    G_neg = nx.Graph()
    G_neg.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) < 0:
            G_neg.add_edge(u, v, **d)

    return G_signed, G_pos, G_neg


def save_raw_results_to_csv(raw_results, filename):
    """
    Save raw benchmark results to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fieldnames = [
        "Run",
        "Method",
        "Resolution",
        "Graph Used",
        "Execution Time (s)",
        "Number of Communities",
        "NMI",
        "ARI",
        "F1",
    ]
    try:
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_results:
                writer.writerow(row)
        logging.info(f"Raw results saved as CSV to: {filename}")
    except Exception as e:
        logging.error(f"Failed to save raw results: {e}")


def save_aggregated_results_to_csv(agg_results, filename):
    """
    Save aggregated benchmark results to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fieldnames = [
        "Method",
        "Resolution",
        "Graph Used",
        "Execution Time (s) Avg",
        "Execution Time (s) Std",
        "Number of Communities Avg",
        "Number of Communities Std",
        "NMI Avg",
        "NMI Std",
        "ARI Avg",
        "ARI Std",
        "F1 Avg",
        "F1 Std",
    ]
    try:
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in agg_results:
                writer.writerow(row)
        logging.info(f"Aggregated results saved as CSV to: {filename}")
    except Exception as e:
        logging.error(f"Failed to save aggregated results: {e}")


def save_boxplots_for_metrics_by_resolution(resolution, metrics, raw_dir, output_dir):
    """
    Read raw CSV data for the given resolution and generate boxplots for each metric.

    Parameters:
      resolution (float): Resolution value to identify the CSV file.
      metrics (list of str): Metrics to plot.
      raw_dir (str): Directory of the raw CSV file.
      output_dir (str): Directory to save the boxplots.
    """
    raw_filename = os.path.join(raw_dir, f"benchmark_raw_res_{resolution}.csv")
    if not os.path.exists(raw_filename):
        logging.error(f"Raw CSV file {raw_filename} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_filename)
    sns.set_theme(style="whitegrid")
    palette = Set1_9.mpl_colors

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="Method", y=metric, data=df, palette=palette)
        ax.set_title(
            f"Distribution of {metric} at resolution {resolution}", fontsize=16
        )
        ax.set_xlabel("Method", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_file = os.path.join(
            output_dir, f"benchmark_raw_res_{resolution}_{metric}_boxplot.png"
        )
        try:
            plt.savefig(out_file, dpi=300)
            logging.info(
                f"Saved boxplot for {metric} at resolution {resolution} to: {out_file}"
            )
        except Exception as e:
            logging.error(f"Failed to save boxplot for {metric}: {e}")
        plt.close()


# Entry point for running the full benchmark pipeline.
if __name__ == "__main__":
    logging.info("Starting full benchmark pipeline...")
    start_time = time.time()

    # Generate graph with hardcoded parameters or modify as needed.
    # (Ensure you pass all required parameters.)
    try:
        graphs = generate_signed_LFR_benchmark_graph(
            n=250,
            tau1=3.0,
            tau2=1.5,
            mu=0.1,
            P_minus=0.5,
            P_plus=0.8,
            min_community=20,
            average_degree=5,
            seed=10,
        )
    except Exception as e:
        logging.error(f"Graph generation failed with exception: {e}")
        exit(1)

    if graphs is None:
        logging.error("LFR Graph Generation Failed. Exiting...")
        exit(1)

    G_signed, G_pos, G_neg = graphs
    logging.info(
        f"Graph ready: {G_signed.number_of_nodes()} nodes, {G_signed.number_of_edges()} edges."
    )
    logging.info(f"LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    # List of resolution values to benchmark.
    resolution_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    n_runs = 20  # Number of runs per resolution

    for res in resolution_values:
        logging.info(f"Running benchmark for resolution = {res} over {n_runs} runs...")
        res_start = time.time()
        raw_results, agg_results = benchmark(
            G_signed, G_pos, G_neg, resolution=res, n_runs=n_runs
        )
        exec_time = time.time() - res_start
        logging.info(
            f"Benchmark for resolution = {res} finished in {exec_time:.2f} sec."
        )

        raw_filename = os.path.join("results", f"benchmark_raw_res_{res}.csv")
        agg_filename = os.path.join("results", f"benchmark_agg_res_{res}.csv")
        save_raw_results_to_csv(raw_results, raw_filename)
        save_aggregated_results_to_csv(agg_results, agg_filename)

    logging.info("All benchmarks complete.")

    # Generate boxplots.
    metrics_to_plot = [
        "NMI",
        "ARI",
        "F1",
        "Number of Communities",
        "Execution Time (s)",
    ]
    for res in resolution_values:
        save_boxplots_for_metrics_by_resolution(
            res, metrics_to_plot, raw_dir="results", output_dir="plots"
        )

    logging.info("Benchmark pipeline finished.")
