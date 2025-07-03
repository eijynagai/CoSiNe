import csv
import logging
import os
import time

import networkx as nx
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from signedLFR.signedLFR import signed_LFR_benchmark_graph
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.spectral_clustering import run_spectral_clustering

# Configure logging (if desired at the library level).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_ground_truth_communities(G):
    """
    Extract the ground-truth community labels from the 'community' attribute of each node.
    If multiple memberships, pick the first. If no community is found, assign -1.
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

    LouvainSigned: uses the passed 'resolution' argument.
    All other methods: forced to resolution=1.0 (or no resolution if not required).
    """
    logging.info(
        f"Benchmarking started. LouvainSigned uses resolution={resolution}; others use resolution=1.0 if applicable."
    )

    methods = {}

    # LouvainSigned variants => each alpha uses the user-provided 'resolution'
    for alpha in [round(x * 0.1, 1) for x in range(1, 11)]:
        method_name = f"LouvainSigned_a{alpha}"
        methods[method_name] = (
            # Pass the user-chosen 'resolution'
            lambda G_pos, G_neg, alpha=alpha, r=resolution: run_louvain_signed(
                G_pos, G_neg, alpha=alpha, resolution=r
            ),
            "signed",
        )

    # Now for the methods that must remain at resolution=1.0
    # (If their code normally accepts a resolution, we override it to 1.0)
    methods["Louvain"] = (
        lambda G_pos: run_louvain(G_pos, resolution=1.0),  # forcibly 1.0
        "pos",
    )
    methods["Leiden"] = (
        lambda G_pos: run_leiden(G_pos, resolution=1.0),  # forcibly 1.0
        "pos",
    )

    # Methods that either do not support a resolution or do not need it:
    methods["Infomap"] = (run_infomap, "pos")
    methods["Greedy modularity"] = (run_greedy_modularity, "pos")
    methods["Spectral clustering"] = (run_spectral_clustering, "pos")

    # Retrieve ground truth
    ground_truth = get_ground_truth_communities(G_signed)
    run_results = []

    # Iterate over methods
    for method_name, (func, graph_type) in methods.items():
        start_time = time.time()

        if graph_type == "signed":
            # Pass (G_pos, G_neg) for signed
            communities = func(G_pos, G_neg)
            node_list = sorted(G_signed.nodes())
        else:
            # For "pos" methods, just pass G_pos
            communities = func(G_pos)
            node_list = sorted(G_pos.nodes())

        execution_time = time.time() - start_time

        # Check missing
        missing = [node for node in node_list if node not in communities]
        if missing:
            logging.warning(f"Nodes missing for {method_name}: {missing}")

        # Convert dictionary-based communities to list if needed
        if isinstance(communities, dict):
            predicted = [communities[node] for node in node_list]
        else:
            predicted = communities

        # Metrics
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
                "Resolution": resolution if "Signed" in method_name else 1.0,
                "Execution Time (s)": round(execution_time, 3),
                "Number of Communities": num_communities,
                "NMI": round(nmi, 3),
                "ARI": round(ari, 3),
                "F1": round(f1, 3),
            }
        )

    logging.info("Benchmarking finished.")
    return run_results


def benchmark(G_signed, G_pos, G_neg, resolution=1.0, n_runs=20):
    """
    Repeats benchmark_signed_and_unsigned n_runs times for a given resolution.

    Returns:
      raw_results: list of dicts (each is one run's data)
      agg_results: aggregated stats (mean, std) for each method
    """
    raw_results = []
    for i in range(n_runs):
        logging.info(f"Run {i + 1}/{n_runs} at resolution={resolution}...")
        run_data = benchmark_signed_and_unsigned(
            G_signed, G_pos, G_neg, resolution=resolution
        )
        # Tag each result with run index
        for row in run_data:
            row["Run"] = i + 1
            raw_results.append(row)

    # Aggregate
    aggregated = {}
    for row in raw_results:
        m = row["Method"]
        if m not in aggregated:
            aggregated[m] = {
                "Execution Time (s)": [],
                "Number of Communities": [],
                "NMI": [],
                "ARI": [],
                "F1": [],
            }
        aggregated[m]["Execution Time (s)"].append(row["Execution Time (s)"])
        aggregated[m]["Number of Communities"].append(row["Number of Communities"])
        aggregated[m]["NMI"].append(row["NMI"])
        aggregated[m]["ARI"].append(row["ARI"])
        aggregated[m]["F1"].append(row["F1"])

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
    Create an LFR benchmark graph (signed) and produce G_signed, G_pos, G_neg.
    """
    logging.info("Starting LFR graph generation with parameters:")
    logging.info(
        f"n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, "
        f"P_minus={P_minus}, P_plus={P_plus}, min_comm={min_community}, "
        f"avg_deg={average_degree}, seed={seed}"
    )

    start_t = time.time()
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
        logging.info(f"LFR graph generated in {time.time() - start_t:.2f} sec")
    except nx.exception.ExceededMaxIterations:
        logging.error("LFR generation failed due to max iterations.")
        return None

    # Build positive subgraph
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) > 0:
            G_pos.add_edge(u, v, **d)

    logging.info(
        f"Positive subgraph: {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges"
    )

    # Build negative subgraph
    G_neg = nx.Graph()
    G_neg.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) < 0:
            G_neg.add_edge(u, v, **d)

    return G_signed, G_pos, G_neg


def save_raw_results_to_csv(raw_results, filename):
    """
    Save raw benchmark results to CSV.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directories for {filename}: {e}")
        return

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
        logging.error(f"Error writing raw results to {filename}: {e}")


def save_aggregated_results_to_csv(agg_results, filename):
    """
    Save aggregated benchmark results to CSV.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directories for {filename}: {e}")
        return

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
        logging.error(f"Error writing aggregated results to {filename}: {e}")


def plot_boxplots_for_metrics(raw_results, metrics, output_dir, resolution):
    """
    Create and save boxplots for each specified metric directly from in-memory raw_results,
    labeling plots with the current resolution value.
    """
    import logging
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from palettable.colorbrewer.qualitative import Set1_9

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(raw_results)

    logging.debug(f"Plotting from DataFrame with columns: {df.columns.tolist()}")
    sns.set_theme(style="whitegrid")
    palette = Set1_9.mpl_colors

    for metric in metrics:
        if metric not in df.columns:
            logging.warning(f"Metric '{metric}' not found in raw_results.")
            continue

        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="Method", y=metric, data=df, palette=palette)

        # Title can include the resolution
        ax.set_title(
            f"Distribution of {metric} at resolution={resolution}", fontsize=16
        )

        ax.set_xlabel("Method", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Incorporate the resolution into the output filename as well
        out_file = os.path.join(output_dir, f"{metric}_boxplot_res_{resolution}.png")

        try:
            plt.savefig(out_file, dpi=300)
            logging.info(
                f"Saved boxplot for '{metric}' at resolution {resolution} to: {out_file}"
            )
        except Exception as e:
            logging.error(f"Failed to save boxplot for '{metric}': {e}")
        finally:
            plt.close()
