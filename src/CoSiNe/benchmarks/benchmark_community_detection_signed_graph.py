import csv
import time

import networkx as nx
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


def benchmark_signed_and_unsigned(G_signed, G_pos, G_neg, resolution=1.0):
    """
    Run both signed and unsigned community detection algorithms:
      - For LouvainSigned, we run multiple times with alpha values ranging from 0.1 to 1.0 (ascending)
        and pass the given resolution.
      - Other methods (Louvain, Leiden) receive the resolution parameter.
      - Infomap, Greedy modularity, and Spectral clustering are run on G_pos without a resolution parameter.

    Parameters:
        G_signed (networkx.Graph): The full signed graph (for ground truth extraction).
        G_pos (networkx.Graph): Positive-only subgraph.
        G_neg (networkx.Graph): Negative-only subgraph.
        resolution (float): Resolution parameter for methods that support it.

    Returns:
        list of dict: Benchmark results.
    """
    print("🔹 Benchmarking started.")

    methods = {}
    # Add LouvainSigned runs (these methods require two graphs, and support resolution).
    for alpha in [round(x * 0.1, 1) for x in range(1, 11)]:
        method_name = f"LouvainSigned_a{alpha}"
        # Use lambda with default arguments so each alpha is captured correctly.
        methods[method_name] = (
            lambda G_pos, G_neg, alpha=alpha, resolution=resolution: run_louvain_signed(
                G_pos, G_neg, alpha=alpha, resolution=resolution
            ),
            "signed",
        )

    # For unsigned methods that support resolution, pass it via a lambda.
    methods["Louvain"] = (
        lambda G_pos, resolution=resolution: run_louvain(G_pos, resolution=resolution),
        "pos",
    )
    methods["Leiden"] = (
        lambda G_pos, resolution=resolution: run_leiden(G_pos, resolution=resolution),
        "pos",
    )

    # For methods that do not accept a resolution parameter, call them directly.
    methods["Infomap"] = (run_infomap, "pos")
    methods["Greedy modularity"] = (run_greedy_modularity, "pos")
    methods["Spectral clustering"] = (run_spectral_clustering, "pos")

    # Use ground truth from G_signed (nodes in both graphs share the same 'community' attribute)
    ground_truth = get_ground_truth_communities(G_signed)

    results = []
    for method_name, (func, graph_type) in methods.items():
        start_time = time.time()

        if graph_type == "signed":
            # For signed methods, pass both positive and negative graphs.
            communities = func(G_pos, G_neg)
            # Use node ordering from G_signed for consistency.
            node_list = sorted(G_signed.nodes())
        else:
            # For unsigned methods, pass only the positive subgraph.
            communities = func(G_pos)
            node_list = sorted(G_pos.nodes())

        execution_time = time.time() - start_time

        # Convert the returned partition to a list ordered by node.
        if isinstance(communities, dict):
            predicted = [communities[node] for node in node_list]
        else:
            predicted = communities

        # Compute evaluation metrics.
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
    P_minus = 0.5
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


def save_results_to_csv(results, filename=None):
    import os

    os.makedirs("results", exist_ok=True)

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

    # Define a list of resolutions to test.
    resolution_values = [0.5, 0.75, 1.0, 1.25, 1.5]

    for res in resolution_values:
        print(f"🔹 Running benchmark for resolution = {res}...")
        res_start = time.time()
        results = benchmark_signed_and_unsigned(G_signed, G_pos, G_neg, resolution=res)
        exec_time = time.time() - res_start
        print(f"🔹 Benchmark for resolution = {res} finished in {exec_time:.2f} sec.")

        # Save results to CSV with the resolution in the filename.
        filename = f"results/benchmark_results_res_{res}.csv"
        save_results_to_csv(results, filename=filename)

    print("✅ All benchmarks complete.")
