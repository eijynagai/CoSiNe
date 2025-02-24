import csv
import time

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from CoSiNe.community_detection.external.signedLFR import signed_LFR_benchmark_graph
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed


def get_ground_truth_communities(G):
    """
    Extract the ground-truth community label for each node from
    the 'community' node attribute in the LFR graph.
    If a node has multiple community memberships, we just
    pick the first for standard metrics like NMI or ARI.
    """
    ground_truth = []
    for node in sorted(G.nodes()):
        # The LFR attribute is often a set of community IDs
        comm_attr = G.nodes[node].get("community", None)
        if not comm_attr:
            ground_truth.append(-1)  # or some placeholder
            continue

        # If it's a set/list, pick the first label
        # (Overlapping communities would require special handling.)
        if isinstance(comm_attr, (set, list)):
            ground_truth.append(list(comm_attr)[0])
        else:
            ground_truth.append(comm_attr)
    return ground_truth


def benchmark_signed_and_unsigned(G_signed, G_pos, G_neg):
    """
    Run both signed and unsigned algorithms:
      - 'LouvainSigned' uses G_pos + G_neg
      - 'Louvain', 'Leiden', 'Infomap' use just G_pos
    """
    print("🔹 Benchmarking started.")

    methods = {
        "LouvainSigned": (run_louvain_signed, "signed"),
        "Louvain": (run_louvain, "pos"),
        "Leiden": (run_leiden, "pos"),
        "Infomap": (run_infomap, "pos"),
    }

    # We'll always use ground truth from the same place (both graphs
    # have the same node attributes). G_signed or G_pos will do.
    ground_truth = get_ground_truth_communities(G_signed)

    results = []

    for method_name, (func, graph_type) in methods.items():
        start_time = time.time()

        if graph_type == "signed":
            # 'run_louvain_signed' typically wants G_pos and G_neg as TWO arguments
            communities = func(G_pos, G_neg)
            # For consistent node ordering, let’s use the same node list as G_signed
            node_list = sorted(G_signed.nodes())
        else:
            # 'run_louvain', 'run_leiden', 'run_infomap' want a single graph => G_pos
            communities = func(G_pos)
            # We'll use G_pos's node ordering (which should match G_signed anyway)
            node_list = sorted(G_pos.nodes())

        execution_time = time.time() - start_time

        # Convert the returned partition to a list aligned with node ordering
        if isinstance(communities, dict):
            predicted = [communities[node] for node in node_list]
        else:
            # If it's already a list or similar structure
            predicted = communities

        # Compute NMI, ARI
        nmi = normalized_mutual_info_score(ground_truth, predicted)
        ari = adjusted_rand_score(ground_truth, predicted)
        num_communities = len(set(predicted))

        print(
            f"✅ {method_name} => {num_communities} communities in {execution_time:.2f}s "
            f"(NMI={nmi:.3f}, ARI={ari:.3f})"
        )

        results.append(
            {
                "Method": method_name,
                "Graph Used": graph_type,
                "Execution Time (s)": round(execution_time, 3),
                "Number of Communities": num_communities,
                "NMI": round(nmi, 3),
                "ARI": round(ari, 3),
            }
        )

    print("✅ Benchmarking finished.")
    return results


def generate_signed_LFR_benchmark_graph():
    """Generate an LFR benchmark graph with debug prints."""
    print("🔹 Starting LFR graph generation...")

    # Parameters
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    P_minus = 0.1
    P_plus = 0.8
    min_community = 20
    average_degree = 5
    seed = 10

    print(
        f"🔹 Parameters: n={n}, tau1={tau1}, tau2={tau2}, "
        f"mu={mu}, p_minus={P_minus}, p_plus={P_plus}, "
        f"min_community={min_community}, av_degree={average_degree} "
    )

    start_time = time.time()

    try:
        print("🔹 Generating signed LFR graph...")
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
        print(f"✅ LFR Graph generated in {time.time() - start_time:.2f} seconds")
    except nx.exception.ExceededMaxIterations:
        print("❌ LFR Generation failed due to max iterations.")
        return None

    # Create positive-only subgraph for algorithms that don't handle negative edges
    G_pos = nx.Graph()
    G_pos.add_nodes_from(
        G_signed.nodes(data=True)
    )  # copy node attributes (including 'community')

    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) > 0:  # keep only positive edges
            G_pos.add_edge(u, v, **d)

    print(
        f"✅ Positive subgraph: {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges"
    )

    # Also build G_neg
    G_neg = nx.Graph()
    G_neg.add_nodes_from(G_signed.nodes(data=True))

    for u, v, d in G_signed.edges(data=True):
        if d["weight"] > 0:
            G_pos.add_edge(u, v, **d)
        else:
            G_neg.add_edge(u, v, **d)

    return G_signed, G_pos, G_neg


def save_results_to_csv(results, filename="results/benchmark_results.csv"):
    import os

    os.makedirs("results", exist_ok=True)

    # Decide on columns
    fieldnames = [
        "Method",
        "Graph Used",
        "Execution Time (s)",
        "Number of Communities",
        "NMI",
        "ARI",
    ]

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Results saved as CSV to: {filename}")


if __name__ == "__main__":
    print("🔹 Running full benchmark pipeline...")

    start_time = time.time()
    G_signed, G_pos, G_neg = generate_signed_LFR_benchmark_graph()
    if G_signed is None:
        print("❌ LFR Graph Generation Failed. Exiting...")
        import sys

        sys.exit(1)  # Stop if generation fails

    print(
        f"✅ Graph Ready: {G_signed.number_of_nodes()} nodes, {G_signed.number_of_edges()} edges."
    )
    print(f"🔹 LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    start_time = time.time()
    results = benchmark_signed_and_unsigned(G_signed, G_pos, G_neg)
    print(f"🔹 Community Detection Time: {time.time() - start_time:.2f} sec")

    print("✅ Benchmarking complete:", results)

    # Save results to CSV
    save_results_to_csv(results)
