import sys
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath("src"))
from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)
from CoSiNe.community_detection.louvain_signed import run_louvain_signed
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.greedy_modularity import run_greedy_modularity
from CoSiNe.community_detection.spectral_clustering import run_spectral_clustering

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score

###############################################################################
# 1. Load best (alpha, gamma) from Optuna results
################################################################################
best_params_path = "src/CoSiNe/benchmarks/hyperparam_tuning/results/best_params_nmi.json"
with open(best_params_path, "r") as f:
    best = json.load(f)
best_alpha = best["alpha"]
best_gamma = best["gamma"]

################################################################################ 2. Define the Methods for Comparison (Infomap removed)
################################################################################
methods = {
    "LouvainSigned_default":  ("signed", {"alpha": 0.6, "resolution": 1.0}),
    "LouvainSigned_best":     ("signed", {"alpha": best_alpha, "resolution": best_gamma}),
    "Louvain":                ("pos",    {"resolution": 1.0}),
    "Leiden":                 ("pos",    {"resolution": 1.0}),
    "GreedyModularity":       ("pos",    {}),
    "SpectralClustering":     ("pos",    {}),
}

################################################################################ 3. Choose a Set of Seeds
###############################################################################
# 3.1 Iterate over seeds
seeds = [42, 43, 44]
results = []

for seed in seeds:
    G_signed, G_pos, G_neg = generate_signed_LFR_benchmark_graph(
        n=1000,          # ~1000 cells/nodes
        tau1=3.0,          # power-law exponent for node degree (controls hub-iness)
        tau2=1.5,        # power-law exponent for community sizes
        mu=0.1667,          # 10% of each node’s edges go outside its “ground-truth” community
        P_minus=0.1,     # 10% of inter-community links carry a negative weight
        P_plus=0.01,     # 1% of same-community edges are flipped to negative
        min_community=20,# allow small communities (nested)
        average_degree=6,# ~6 edges/node on average
        seed=seed,
    )

    # 3.2. Extract ground-truth labels from G_signed
    node_list = sorted(G_signed.nodes())
    ground_truth = [G_signed.nodes[n]["community"] for n in node_list]

    # 3.3. Run each method on this graph
    for method_name, (graph_type, params) in methods.items():
        t_start = time.perf_counter()

        if graph_type == "signed":
            comm_mapping = run_louvain_signed(
                G_pos,
                G_neg,
                alpha=params["alpha"],
                resolution=params["resolution"],
            )
            pred_nodes = sorted(G_signed.nodes())
        else:
            if method_name == "Louvain":
                comm_mapping = run_louvain(G_pos, resolution=params["resolution"])
            elif method_name == "Leiden":
                comm_mapping = run_leiden(G_pos, resolution=params["resolution"])
            elif method_name == "GreedyModularity":
                comm_mapping = run_greedy_modularity(G_pos)
            elif method_name == "SpectralClustering":
                comm_mapping = run_spectral_clustering(G_pos)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            pred_nodes = sorted(G_pos.nodes())

        t_elapsed = time.perf_counter() - t_start

        # 3.4. Build the predicted-labels list, filling missing nodes with –1 if needed
        if isinstance(comm_mapping, dict):
            predicted = [comm_mapping.get(n, -1) for n in pred_nodes]
        else:
            predicted = comm_mapping

        # 3.5. Compute metrics
        nmi = normalized_mutual_info_score(ground_truth, predicted)
        ari = adjusted_rand_score(ground_truth, predicted)
        f1 = f1_score(ground_truth, predicted, average="macro")
        num_comms = len(set(predicted))

        results.append({
            "Method": method_name,
            "Seed": seed,
            "Execution Time (s)": t_elapsed,
            "NMI": nmi,
            "ARI": ari,
            "F1": f1,
            "Num Communities": num_comms,
        })

###############################################################################
# 4. Aggregate & Display Results
################################################################################
df = pd.DataFrame(results)
print("\n=== Raw Results ===")
print(df)

# Save to CSV
output_csv = "results/comparison_results.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"\nSaved full results to: {output_csv}\n")

###############################################################################
# 5. Plot Boxplots for Each Metric and Save to results/plots/
################################################################################
sns.set_theme(style="whitegrid")

plot_dir = "results/plots"
os.makedirs(plot_dir, exist_ok=True)

for metric in ["NMI", "ARI", "F1", "Execution Time (s)"]:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Method", y=metric, data=df)

    # For quality metrics, zoom into [0.9, 1.0]
    if metric in {"NMI", "ARI", "F1"}:
        ax.set_ylim(0.5, 1.0)

    # If execution times vary by orders of magnitude, use log scale
    if metric == "Execution Time (s)":
        ax.set_yscale("log")

    ax.set_title(f"{metric} Comparison Across Methods", fontsize=16)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(plot_dir, f"{metric.replace(' ', '_')}_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot: {plot_path}")

    plt.show()