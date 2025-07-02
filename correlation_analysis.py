#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1) Load performance and topology data
df_perf = pd.read_csv("results/swept_benchmark.csv")
df_topo = pd.read_csv("results/scenario_topology.csv")

# 2) Merge performance and topology on common keys
merge_keys = ["scenario"]
if "Seed" in df_perf.columns and "Seed" in df_topo.columns:
    merge_keys.append("Seed")

df = pd.merge(df_perf, df_topo, on=merge_keys, how="inner")

# 3) Spearman correlation: topology vs performance
topo_cols = [
    "density",
    "avg_deg",
    "avg_deg_pos",
    "avg_deg_neg",
    "comm_size_mean",
    "comm_size_std",
    "modularity",
    "avg_clustering",
    "transitivity",
    "num_cc",
    "diameter",
    "avg_path_length",
    "assortativity",
    "fiedler_gap",
    "signed_laplacian_gap",
]
perf_cols = ["NMI", "ARI", "F1", "AUPRC", "Time(s)"]

# Subset and drop missing
df_corr = df[topo_cols + perf_cols].dropna()

# Compute Spearman correlation matrix
corr_spearman = df_corr.corr(method="spearman").loc[topo_cols, perf_cols]

# Plot and save Spearman heatmap
os.makedirs("results/plots", exist_ok=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Spearman Correlation: Topology vs Performance")
plt.tight_layout()
plt.savefig("results/plots/corr_topo_perf.png", dpi=300)
plt.close()

print(
    "Spearman topology-vs-performance correlation heatmap saved to results/plots/corr_topo_perf.png"
)
