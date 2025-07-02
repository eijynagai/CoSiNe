#!/usr/bin/env python
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Fixed order of methods for consistent coloring
methods_order = ["LouvainSigned", "Louvain", "Leiden", "Greedy", "Infomap", "LPA"]
palette_colors = sns.color_palette("tab10", n_colors=len(methods_order))
method_colors = OrderedDict(zip(methods_order, palette_colors))

# 1) Load the data
df = pd.read_csv("results/runtime_benchmark.csv")

# 2) Set up style
sns.set_theme(style="whitegrid")

# 3) Log–log runtime vs n
plt.figure(figsize=(8, 6))
ax = sns.lineplot(
    data=df,
    x="n",
    y="runtime_s",
    hue="method",
    hue_order=methods_order,
    style="avg_deg",
    markers=True,
    dashes=True,
    estimator="mean",
    errorbar="sd",
    palette=method_colors,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Runtime vs. Network Size (log–log)")
ax.set_xlabel("Number of Nodes (n)")
ax.set_ylabel("Runtime (s)")
plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/runtime_vs_n_loglog.png", dpi=300)
plt.close()

# 4) Boxplot: runtime vs density per n (manual loop)
for n_val in sorted(df["n"].unique()):
    df_n = df[df["n"] == n_val]
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(
        x="avg_deg",
        y="runtime_s",
        hue="method",
        hue_order=methods_order,
        data=df_n,
        palette=method_colors,
    )
    ax.set_yscale("log")
    ax.set_title(f"Runtime vs. Avg Degree (n={n_val})")
    ax.set_xlabel("Average Degree ⟨k⟩")
    ax.set_ylabel("Runtime (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/runtime_vs_density_n_{n_val}.png", dpi=300)
    plt.close()

# 5) (Optional) Facet by mixing μ
# g2 = sns.FacetGrid(
#     df, col="mu", row="method",
#     hue="method", hue_order=methods_order,
#     margin_titles=True, sharey=False
# )
# g2.map_dataframe(sns.boxplot, x="avg_deg", y="runtime_s", palette=method_colors)
# for ax in g2.axes.flatten():
#     ax.set_yscale("log")
# g2.set_axis_labels("⟨k⟩", "Runtime (s)")
# plt.tight_layout()
# g2.savefig("results/plots/runtime_vs_density_mu_facet.png", dpi=300)
