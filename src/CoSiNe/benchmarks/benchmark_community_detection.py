import csv
import time

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain


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


def benchmark_community_detection(graph):
    print("🔹 Benchmarking started.")
    methods = {"Louvain": run_louvain, "Leiden": run_leiden, "Infomap": run_infomap}

    # Get ground-truth from the LFR graph
    ground_truth = get_ground_truth_communities(graph)

    results = []

    for method_name, func in methods.items():
        print(f"🔹 Running {method_name}...")
        start_time = time.time()
        communities = func(graph)  # your detection function
        execution_time = time.time() - start_time

        # Convert the returned partition to a list aligned with node ordering
        # If 'communities' is a dict {node: community_id}, make a list
        # sorted by node index. If it's already a list, ensure the order is correct.
        if isinstance(communities, dict):
            predicted = [communities[node] for node in sorted(graph.nodes())]
        else:
            # Assume it's a list indexed by node
            predicted = communities

        # Calculate NMI, ARI
        # If there's no overlap in ground-truth, these metrics are straightforward
        nmi = normalized_mutual_info_score(ground_truth, predicted)
        ari = adjusted_rand_score(ground_truth, predicted)

        num_communities = len(set(predicted))

        print(
            f"✅ {method_name} detected {num_communities} communities in {execution_time:.2f}s "
            f"(NMI={nmi:.3f}, ARI={ari:.3f})"
        )

        results.append(
            {
                "Method": method_name,
                "Execution Time (s)": execution_time,
                "Number of Communities": num_communities,
                "NMI": round(nmi, 3),
                "ARI": round(ari, 3),
            }
        )

    print("✅ Benchmarking finished.")
    return results


def generate_lfr_graph():
    """Generate an LFR benchmark graph with debug prints."""
    print("🔹 Starting LFR graph generation...")

    # Parameters
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    min_community = 20
    average_degree = 5
    max_iters = 500
    seed = 10

    print(
        f"🔹 Parameters: n={n}, tau1={tau1}, tau2={tau2}, "
        f"mu={mu}, min_community={min_community}"
    )

    start_time = time.time()

    try:
        print("🔹 Generating LFR graph...")
        G = nx.LFR_benchmark_graph(
            n,
            tau1,
            tau2,
            mu,
            average_degree=average_degree,
            min_community=min_community,
            max_iters=max_iters,
            seed=seed,
        )
        print(f"✅ LFR Graph generated in {time.time() - start_time:.2f} seconds")
    except nx.exception.ExceededMaxIterations:
        print("❌ LFR Generation failed due to max iterations.")
        return None

    # Convert node labels to integers
    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(G)

    print(f"✅ Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# NEW: Utility to save results to CSV
def save_results_to_csv(results, filename="results/benchmark_results.csv"):
    import os

    os.makedirs("results", exist_ok=True)

    # Decide on columns
    fieldnames = [
        "Method",
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
    test_graph = generate_lfr_graph()
    if test_graph is None:
        print("❌ LFR Graph Generation Failed. Exiting...")
        import sys

        sys.exit(1)  # Stop if generation fails

    print(
        f"✅ Graph Ready: {test_graph.number_of_nodes()} nodes, {test_graph.number_of_edges()} edges."
    )
    print(f"🔹 LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    start_time = time.time()
    results = benchmark_community_detection(test_graph)
    print(f"🔹 Community Detection Time: {time.time() - start_time:.2f} sec")

    print("✅ Benchmarking complete:", results)

    # Save results to CSV
    save_results_to_csv(results)
