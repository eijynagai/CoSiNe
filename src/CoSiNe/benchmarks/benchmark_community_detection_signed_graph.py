import csv
import time

import networkx as nx
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from CoSiNe.community_detection.external.signedLFR import signed_LFR_benchmark_graph
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.spectral_clustering import run_spectral_clustering


def get_ground_truth_communities(G):
    """
    Extract the ground-truth community label for each node from
    the 'community' node attribute in the LFR graph.
    If a node has multiple memberships, pick the first.
    """
    ground_truth = []
    for node in sorted(G.nodes()):
        comm_attr = G.nodes[node].get("community", None)
        if not comm_attr:
            ground_truth.append(-1)
            continue
        if isinstance(comm_attr, (set, list)):
            ground_truth.append(list(comm_attr)[0])
        else:
            ground_truth.append(comm_attr)
    return ground_truth


def benchmark_signed_and_unsigned(G_signed, G_pos, G_neg, resolution=1.0):
    """
    Run both signed and unsigned community detection algorithms:
      - For LouvainSigned, run multiple times with alpha values from 0.1 to 1.0 (ascending)
        and pass the given resolution.
      - Other methods (Louvain, Leiden) receive the resolution parameter.
      - Infomap, Greedy modularity, and Spectral clustering are run on G_pos.

    Returns a list of result dictionaries for one run.
    """
    print(f"🔹 Benchmarking started at resolution={resolution}.")

    methods = {}
    # LouvainSigned variants (require two graphs)
    for alpha in [round(x * 0.1, 1) for x in range(1, 11)]:
        method_name = f"LouvainSigned_a{alpha}"
        methods[method_name] = (
            lambda G_pos, G_neg, alpha=alpha, resolution=resolution: run_louvain_signed(
                G_pos, G_neg, alpha=alpha, resolution=resolution
            ),
            "signed",
        )
    # Unsigned methods that support resolution.
    methods["Louvain"] = (
        lambda G_pos, resolution=resolution: run_louvain(G_pos, resolution=resolution),
        "pos",
    )
    methods["Leiden"] = (
        lambda G_pos, resolution=resolution: run_leiden(G_pos, resolution=resolution),
        "pos",
    )
    # Methods that do not take resolution.
    methods["Infomap"] = (run_infomap, "pos")
    methods["Greedy modularity"] = (run_greedy_modularity, "pos")
    methods["Spectral clustering"] = (run_spectral_clustering, "pos")

    ground_truth = get_ground_truth_communities(G_signed)
    run_results = []
    for method_name, (func, graph_type) in methods.items():
        start_time = time.time()

        if graph_type == "signed":
            communities = func(G_pos, G_neg)
            node_list = sorted(G_signed.nodes())
        else:
            communities = func(G_pos)
            node_list = sorted(G_pos.nodes())

        execution_time = time.time() - start_time

        if isinstance(communities, dict):
            predicted = [communities[node] for node in node_list]
        else:
            predicted = communities

        nmi = normalized_mutual_info_score(ground_truth, predicted)
        ari = adjusted_rand_score(ground_truth, predicted)
        f1 = f1_score(ground_truth, predicted, average="macro")
        num_communities = len(set(predicted))

        print(
            f"✅ {method_name} => {num_communities} communities in {execution_time:.2f}s "
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

    print(f"✅ Benchmarking finished at resolution={resolution}.")
    return run_results


def benchmark(G_signed, G_pos, G_neg, resolution=1.0, n_runs=20):
    """
    Run the benchmark n_runs times for a given resolution, and store all raw values.

    Returns:
        raw_results: list of dicts, one per run (each dict includes a "Run" key)
        agg_results: list of aggregated dicts (averages and std dev for each method)
    """
    raw_results = []
    for run in range(n_runs):
        print(f"🔹 Run {run+1}/{n_runs} at resolution={resolution}...")
        run_results = benchmark_signed_and_unsigned(
            G_signed, G_pos, G_neg, resolution=resolution
        )
        # Annotate each result with the run number.
        for row in run_results:
            row["Run"] = run + 1
            raw_results.append(row)

    # Aggregate raw results by method.
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


import time

import networkx as nx

from CoSiNe.community_detection.external.signedLFR import signed_LFR_benchmark_graph


def generate_signed_LFR_benchmark_graph(
    n, tau1, tau2, mu, P_minus, P_plus, min_community, average_degree, seed
):
    """Generate an LFR benchmark graph and its positive/negative subgraphs using provided parameters."""
    print("🔹 Starting LFR graph generation with parameters:")
    print(
        f"n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, P_minus={P_minus}, P_plus={P_plus}, min_comm={min_community}, avg_deg={average_degree}, seed={seed}"
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
        print(f"✅ LFR Graph generated in {time.time()-start_time:.2f} sec")
    except nx.exception.ExceededMaxIterations:
        print("❌ LFR Generation failed due to max iterations.")
        return None

    # Create positive-only subgraph.
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) > 0:
            G_pos.add_edge(u, v, **d)
    print(
        f"✅ Positive subgraph: {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges"
    )

    # Create negative-only subgraph.
    G_neg = nx.Graph()
    G_neg.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) < 0:
            G_neg.add_edge(u, v, **d)

    return G_signed, G_pos, G_neg


def save_raw_results_to_csv(raw_results, filename):
    """
    Save the raw results (each run) to a CSV file.
    """
    import os

    os.makedirs("results", exist_ok=True)
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
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_results:
            writer.writerow(row)
    print(f"✅ Raw results saved as CSV to: {filename}")


def save_aggregated_results_to_csv(agg_results, filename):
    """
    Save the aggregated (averaged and std) results to a CSV file.
    """
    import os

    os.makedirs("results", exist_ok=True)
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
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_results:
            writer.writerow(row)
    print(f"✅ Aggregated results saved as CSV to: {filename}")


import os

import matplotlib.pyplot as plt

# Plotting
import pandas as pd
import seaborn as sns
from palettable.colorbrewer.qualitative import Set1_9  # example palette


def save_boxplots_for_metrics_by_resolution(resolution, metrics, raw_dir, output_dir):
    """
    For a given resolution, read the corresponding raw CSV file from raw_dir
    (e.g., 'raw_dir/benchmark_raw_res_{resolution}.csv') and create & save a boxplot
    for each metric. The output files are saved in the same directory (output_dir),
    which should be the subdirectory where you also store the CSV tables.

    Parameters:
      resolution (float): The resolution value (used to build the filename).
      metrics (list of str): List of metric column names to plot (e.g.,
                             ["NMI", "ARI", "Number of Communities", "Execution Time (s)"]).
      raw_dir (str): Directory where the raw CSV file is stored.
      output_dir (str): Directory where the plots will be saved.
    """
    raw_filename = os.path.join(raw_dir, f"benchmark_raw_res_{resolution}.csv")
    if not os.path.exists(raw_filename):
        print(f"Raw file {raw_filename} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_filename)

    # Set Seaborn theme.
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
        plt.savefig(out_file, dpi=300)
        print(
            f"✅ Saved boxplot for {metric} at resolution {resolution} to: {out_file}"
        )
        plt.close()


if __name__ == "__main__":
    print("🔹 Running full benchmark pipeline...")

    start_time = time.time()
    graphs = generate_signed_LFR_benchmark_graph()
    if graphs is None:
        print("❌ LFR Graph Generation Failed. Exiting...")
        import sys

        sys.exit(1)
    G_signed, G_pos, G_neg = graphs

    print(
        f"✅ Graph Ready: {G_signed.number_of_nodes()} nodes, {G_signed.number_of_edges()} edges."
    )
    print(f"🔹 LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    # List of resolutions to test.
    resolution_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    n_runs = 20  # number of runs per resolution

    for res in resolution_values:
        print(f"🔹 Running benchmark for resolution = {res} over {n_runs} runs...")
        res_start = time.time()
        raw_results, agg_results = benchmark(
            G_signed, G_pos, G_neg, resolution=res, n_runs=n_runs
        )
        exec_time = time.time() - res_start
        print(f"🔹 Benchmark for resolution = {res} finished in {exec_time:.2f} sec.")

        raw_filename = f"results/benchmark_raw_res_{res}.csv"
        agg_filename = f"results/benchmark_agg_res_{res}.csv"
        save_raw_results_to_csv(raw_results, raw_filename)
        save_aggregated_results_to_csv(agg_results, agg_filename)

    print("✅ All benchmarks complete.")

    # Plotting
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
