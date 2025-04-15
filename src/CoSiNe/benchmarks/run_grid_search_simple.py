import os
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from CoSiNe.community_detection.louvain_signed import run_louvain_signed

# --- Synthetic Graph Generation and Helper Functions ---


def generate_sbm_graph():
    """
    Generate a synthetic graph with two communities using a stochastic block model.
    Returns a graph with clear community structure and a "block" attribute.
    """
    sizes = [100, 100]  # two communities of 100 nodes each
    p_in = 0.8
    p_out = 0.05
    p_matrix = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
    # Set positive edge weights.
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G


def add_negative_edges(G, num_edges, community_labels):
    """
    Add a specified number of negative edges between communities.
    community_labels: dict mapping node -> community id.
    """
    inter_pairs = []
    communities = {}
    for node, comm in community_labels.items():
        communities.setdefault(comm, []).append(node)
    comm_ids = list(communities.keys())
    for i in range(len(comm_ids)):
        for j in range(i + 1, len(comm_ids)):
            for u in communities[comm_ids[i]]:
                for v in communities[comm_ids[j]]:
                    inter_pairs.append((u, v))
    random.shuffle(inter_pairs)
    count = 0
    for u, v in inter_pairs:
        if count >= num_edges:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=-1)
            count += 1
    return G


def compute_signed_modularity(G, communities):
    """
    Compute a simple signed modularity:
      Q_total = Q_pos - Q_neg,
    where Q_pos is computed on the positive subgraph and Q_neg on the negative subgraph.
    """
    G_pos = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0
    )
    G_neg = nx.Graph(
        (u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) < 0
    )
    try:
        Q_pos = nx.algorithms.community.quality.modularity(G_pos, communities)
    except Exception:
        Q_pos = 0
    try:
        Q_neg = nx.algorithms.community.quality.modularity(G_neg, communities)
    except Exception:
        Q_neg = 0
    return Q_pos - Q_neg


def partition_to_communities(partition):
    """Convert a partition dict {node: community} into a list of sets."""
    communities = []
    for comm in set(partition.values()):
        communities.append({node for node, c in partition.items() if c == comm})
    return communities


def get_ground_truth_communities(G):
    """
    Extract ground truth from the 'block' attribute.
    """
    ground_truth = []
    for node in sorted(G.nodes()):
        ground_truth.append(G.nodes[node].get("block", -1))
    return ground_truth


# --- Grid Search Over Alpha and Resolution ---


def grid_search_parameters(alpha_values, resolution_values, n_runs=10, neg_edges=200):
    """
    Perform a grid search over alpha and resolution for LouvainSigned.
    For each (alpha, resolution) pair, run n_runs iterations on a base SBM network
    (with a fixed number of negative edges added), and compute:
      - Signed modularity,
      - NMI and ARI against ground truth (using the "block" attribute).
    Returns a DataFrame with the aggregated (mean and std) results for each parameter pair.
    """
    # Generate base graph.
    G_base = generate_sbm_graph()
    ground_truth = get_ground_truth_communities(G_base)
    community_labels = nx.get_node_attributes(G_base, "block")

    records = []
    for alpha in alpha_values:
        for res in resolution_values:
            mods, nmis, aris = [], [], []
            for run in range(n_runs):
                G = G_base.copy()
                G = add_negative_edges(G, neg_edges, community_labels)
                if neg_edges == 0:
                    nodes = list(G.nodes())
                    if len(nodes) >= 2:
                        G.add_edge(nodes[0], nodes[1], weight=-1e-6)
                G_pos = nx.Graph(
                    (u, v, d)
                    for u, v, d in G.edges(data=True)
                    if d.get("weight", 0) > 0
                )
                G_neg = nx.Graph(
                    (u, v, d)
                    for u, v, d in G.edges(data=True)
                    if d.get("weight", 0) < 0
                )
                # Run LouvainSigned.
                partition = run_louvain_signed(
                    G_pos, G_neg, alpha=alpha, resolution=res
                )
                comm_ls = partition_to_communities(partition)
                Q_total = compute_signed_modularity(G, comm_ls)
                mods.append(Q_total)
                # Get predicted communities as list.
                predicted = [partition[node] for node in sorted(G.nodes())]
                nmi = normalized_mutual_info_score(ground_truth, predicted)
                ari = adjusted_rand_score(ground_truth, predicted)
                nmis.append(nmi)
                aris.append(ari)
            record = {
                "alpha": round(alpha, 1),
                "resolution": round(res, 1),
                "modularity_avg": np.mean(mods),
                "modularity_std": np.std(mods),
                "nmi_avg": np.mean(nmis),
                "nmi_std": np.std(nmis),
                "ari_avg": np.mean(aris),
                "ari_std": np.std(aris),
            }
            records.append(record)
            print(
                f"alpha: {alpha:.1f}"
                f"res: {res:.1f}"
                f"Q: {record['modularity_avg']:.3f} ± {record['modularity_std']:.3f}"
                f"NMI: {record['nmi_avg']:.3f} ± {record['nmi_std']:.3f}"
                f"ARI: {record['ari_avg']:.3f} ± {record['ari_std']:.3f}"
            )
    results_df = pd.DataFrame(records)
    return results_df


# --- Plotting Heatmaps ---


def plot_heatmaps(results_df, metric, output_file):
    """
    Generate and save a heatmap for the specified metric from grid search results.

    The tick labels are formatted to one decimal place.

    Parameters:
      results_df (DataFrame): Must contain columns "alpha", "resolution", and the metric.
      metric (str): One of "modularity_avg", "nmi_avg", or "ari_avg".
      output_file (str): File path to save the heatmap.
    """
    pivot = results_df.pivot(index="alpha", columns="resolution", values=metric)
    sns.set_theme(style="white")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="viridis", cbar_kws={"label": metric}
    )
    ax.set_ylabel("Alpha", fontsize=12)
    ax.set_xlabel("Resolution", fontsize=12)
    ax.set_title(f"Heatmap of {metric}", fontsize=14)
    # Format tick labels
    ax.set_yticklabels([f"{x:.1f}" for x in pivot.index])
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved heatmap to {output_file}")
    plt.show()


# --- Network Summary Functions ---


def network_summary(G):
    """
    Compute summary statistics for a NetworkX graph.
    Returns a dictionary of statistics.
    """
    summary = {}
    summary["Number of nodes"] = G.number_of_nodes()
    summary["Number of edges"] = G.number_of_edges()
    degrees = [d for n, d in G.degree()]
    summary["Average degree"] = np.mean(degrees)
    summary["Min degree"] = np.min(degrees)
    summary["Max degree"] = np.max(degrees)
    summary["Density"] = nx.density(G)
    if nx.is_connected(G):
        summary["Largest connected component size"] = G.number_of_nodes()
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        summary["Largest connected component size"] = len(largest_cc)
    summary["Average clustering coefficient"] = nx.average_clustering(G)
    return summary


def save_network_summary(summary, filename):
    with open(filename, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"Network summary saved to {filename}")


# --- Main Script ---

if __name__ == "__main__":
    # Define grid search ranges.
    alpha_values = np.linspace(0.1, 1.0, 10)
    resolution_values = np.linspace(0.5, 1.5, 11)
    n_runs = 10  # Number of iterations per parameter pair.
    neg_edges = 200  # Fixed number of negative edges.

    # Run grid search.
    results_df = grid_search_parameters(
        alpha_values, resolution_values, n_runs=n_runs, neg_edges=neg_edges
    )
    os.makedirs("results", exist_ok=True)
    results_csv = "results/grid_search_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved grid search results to {results_csv}")

    # Generate and save heatmaps for each metric.
    plot_heatmaps(results_df, "modularity_avg", "results/heatmap_modularity.png")
    plot_heatmaps(results_df, "nmi_avg", "results/heatmap_nmi.png")
    plot_heatmaps(results_df, "ari_avg", "results/heatmap_ari.png")

    # Also compute and save summary statistics of the base graph.
    G_base = generate_sbm_graph()
    summary_stats = network_summary(G_base)
    summary_file = "results/network_summary.txt"
    save_network_summary(summary_stats, summary_file)
