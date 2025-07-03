#!/usr/bin/env python
import logging
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

# Locate the scenarios CSV or raise error
scenarios_path = os.path.join(project_root, "config", "batch", "scenarios_param.csv")
if not os.path.exists(scenarios_path):
    raise FileNotFoundError(f"Could not find simulation settings CSV at {scenarios_path}")
    
import networkx as nx
import numpy as np
import pandas as pd
from networkx.linalg.spectrum import laplacian_spectrum
from tqdm import tqdm

from CoSiNe.benchmarks.benchmark_community_detection_signed_graph import (
    generate_signed_LFR_benchmark_graph,
)

sys.path.append(os.path.abspath("src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 1) Read scenarios
scenarios = pd.read_csv(scenarios_path, comment="#")

records = []
for _, row in tqdm(
    scenarios.iterrows(), total=len(scenarios), desc="Profiling scenarios"
):
    scenario = row["scenario"]
    # Generate the signed LFR graph; skip if it fails
    res = generate_signed_LFR_benchmark_graph(
        n=int(row["n"]),
        tau1=float(row["tau1"]),
        tau2=float(row["tau2"]),
        mu=float(row["mu"]),
        P_minus=float(row["P_minus"]),
        P_plus=float(row["P_plus"]),
        average_degree=float(row["average_degree"]),
        min_community=int(row["min_community"]),
        seed=0,
    )
    if res is None:
        logging.warning(f"Scenario {scenario}: LFR generation failed, skipping.")
        continue
    G_signed, G_pos, G_neg = res
    # Basic counts
    n = G_signed.number_of_nodes()
    m = G_signed.number_of_edges()
    m_pos = G_pos.number_of_edges()
    m_neg = G_neg.number_of_edges()
    density = 2 * m / (n * (n - 1))
    avg_deg = 2 * m / n
    avg_deg_pos = 2 * m_pos / n
    avg_deg_neg = 2 * m_neg / n

    # Planted communities
    gt = nx.get_node_attributes(G_signed, "community")
    inv: dict[int, list[int]] = {}
    for u, c in gt.items():
        inv.setdefault(c, []).append(u)
    planted_sets = list(inv.values())
    num_planted = len(planted_sets)
    comm_sizes = [len(c) for c in planted_sets]
    comm_size_mean = np.mean(comm_sizes)
    comm_size_std = np.std(comm_sizes)

    # Ground-truth modularity on positive backbone
    modularity = nx.community.modularity(G_pos, planted_sets)

    # Clustering & connectivity
    avg_clust = nx.average_clustering(G_pos)
    trans = nx.transitivity(G_pos)
    num_cc = nx.number_connected_components(G_pos)
    # Largest component stats
    comps = list(nx.connected_components(G_pos))
    largest = max(comps, key=len)
    H = G_pos.subgraph(largest)
    try:
        diameter = nx.diameter(H)
        avg_path = nx.average_shortest_path_length(H)
    except nx.NetworkXError:
        diameter = np.nan
        avg_path = np.nan
    assort = nx.degree_assortativity_coefficient(G_pos)

    # Spectral gaps
    # 1) Unsigned Laplacian (Fiedler gap)
    eigs = laplacian_spectrum(G_pos)
    eigs_sorted = sorted(eigs)
    fiedler_gap = float(eigs_sorted[1]) if len(eigs_sorted) > 1 else np.nan

    # 2) Signed Laplacian
    nodes = list(G_signed.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    size = len(nodes)
    W = np.zeros((size, size))
    # Positive edges
    for u, v, d in G_pos.edges(data=True):
        i, j = idx[u], idx[v]
        w = d.get("weight", 1)
        W[i, j] += w
        W[j, i] += w
    # Negative edges
    for u, v, d in G_neg.edges(data=True):
        i, j = idx[u], idx[v]
        w = abs(d.get("weight", -1))
        W[i, j] -= w
        W[j, i] -= w
    D = np.diag(np.sum(np.abs(W), axis=1))
    L_signed = D - W
    eig_signed = np.linalg.eigvalsh(L_signed)
    eig_signed_sorted = np.sort(eig_signed)
    signed_gap = float(eig_signed_sorted[1]) if len(eig_signed_sorted) > 1 else np.nan

    records.append(
        {
            "scenario": scenario,
            "n": n,
            "m": m,
            "m_pos": m_pos,
            "m_neg": m_neg,
            "density": density,
            "avg_deg": avg_deg,
            "avg_deg_pos": avg_deg_pos,
            "avg_deg_neg": avg_deg_neg,
            "num_planted": num_planted,
            "comm_size_mean": comm_size_mean,
            "comm_size_std": comm_size_std,
            "modularity": modularity,
            "avg_clustering": avg_clust,
            "transitivity": trans,
            "num_cc": num_cc,
            "diameter": diameter,
            "avg_path_length": avg_path,
            "assortativity": assort,
            "fiedler_gap": fiedler_gap,
            "signed_laplacian_gap": signed_gap,
        }
    )

# 2) Save to CSV
df = pd.DataFrame(records)
out_dir = os.path.join(project_root, "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "scenario_topology.csv")
df.to_csv(out, index=False)
print(f"Topology profile saved to: {out}")
print(df)