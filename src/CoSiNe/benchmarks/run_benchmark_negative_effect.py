import os

# Removed unused import
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed

# --- For demonstration, here's a simple SBM generator and related functions ---


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


# --- Robust benchmark function for negative-edge effect for LouvainSigned and Louvain ---


def run_robust_benchmark_negative_effect(n_runs=20):
    """
    For a range of negative edge counts (from 0 to 1000, in steps of 100),
    run the benchmark n_runs times for each configuration using only:
      - LouvainSigned (on both positive and negative subgraphs)
      - Original Louvain (on the positive subgraph only)
    Compute the signed modularity (for LouvainSigned) and standard modularity (for Louvain)
    for each run, then aggregate the results.

    Returns:
      neg_counts: array of negative edge counts tested.
      results_dict: dict mapping method to list of average modularity values per neg_count.
      std_dict: dict mapping method to list of standard deviations.
    """
    # Generate a base graph.
    G_base = generate_sbm_graph()
    community_labels = nx.get_node_attributes(G_base, "block")

    neg_counts = np.linspace(0, 1000, 11, dtype=int)
    results_dict = {"LouvainSigned": [], "Louvain": []}
    std_dict = {"LouvainSigned": [], "Louvain": []}

    for neg in neg_counts:
        mod_vals = {"LouvainSigned": [], "Louvain": []}
        for run in range(n_runs):
            G = G_base.copy()
            G = add_negative_edges(G, neg, community_labels)
            if neg == 0:
                # Add a dummy negative edge to avoid division by zero.
                nodes = list(G.nodes())
                if len(nodes) >= 2:
                    G.add_edge(nodes[0], nodes[1], weight=-1e-6)
            G_pos = nx.Graph(
                (u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) > 0
            )
            G_neg = nx.Graph(
                (u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0) < 0
            )

            # Run LouvainSigned.
            part_ls = run_louvain_signed(G_pos, G_neg, alpha=1.0, resolution=1.0)
            comm_ls = partition_to_communities(part_ls)
            Q_ls = compute_signed_modularity(G, comm_ls)
            mod_vals["LouvainSigned"].append(Q_ls)

            # Run original Louvain on positive subgraph.
            part_lv = run_louvain(G_pos, resolution=1.0)
            comm_lv = partition_to_communities(part_lv)
            try:
                Q_lv = nx.algorithms.community.quality.modularity(G_pos, comm_lv)
            except Exception:
                Q_lv = 0
            mod_vals["Louvain"].append(Q_lv)

        # Aggregate
        for method in mod_vals:
            results_dict[method].append(np.mean(mod_vals[method]))
            std_dict[method].append(np.std(mod_vals[method]))
        print(f"Negative edges: {neg}")
        for method in mod_vals:
            avg = np.mean(mod_vals[method])
            std = np.std(mod_vals[method])
            print(f"  {method}: {avg:.3f} ± {std:.3f}")

    return neg_counts, results_dict, std_dict


# --- Improved plotting with Seaborn ---


def plot_modularity_comparison(neg_counts, results_dict, std_dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Build a DataFrame from the results.
    data = []
    for method in results_dict.keys():
        for i, neg in enumerate(neg_counts):
            data.append(
                {
                    "NegativeEdges": neg,
                    "Method": method,
                    "ModularityAvg": results_dict[method][i],
                    "ModularityStd": std_dict[method][i],
                }
            )
    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.figure(figsize=(10, 6))
    # Create a line plot with hue by method.
    ax = sns.lineplot(
        data=df,
        x="NegativeEdges",
        y="ModularityAvg",
        hue="Method",
        marker="o",
        err_style="bars",
        ci=None,
    )
    # Manually add error bars without adding new legend entries.
    for method in df["Method"].unique():
        subset = df[df["Method"] == method]
        ax.errorbar(
            subset["NegativeEdges"],
            subset["ModularityAvg"],
            yerr=subset["ModularityStd"],
            fmt="none",
            capsize=5,
        )

    ax.set_xlabel("Number of inter-community negative edges", fontsize=14)
    ax.set_ylabel("Modularity", fontsize=14)
    ax.set_title(
        "Modularity vs. Negative Edges: LouvainSigned vs. Louvain", fontsize=16
    )
    plt.legend(title="Method")
    plt.tight_layout()

    out_file = os.path.join(output_dir, "modularity_comparison.png")
    plt.savefig(out_file, dpi=300)
    print(f"✅ Plot saved to: {out_file}")
    plt.show()


# --- Main script ---
if __name__ == "__main__":
    neg_counts, results_dict, std_dict = run_robust_benchmark_negative_effect(n_runs=20)
    plot_modularity_comparison(neg_counts, results_dict, std_dict, output_dir="plots")
