import time

import networkx as nx

from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain


def benchmark_community_detection(graph):
    print("🔹 Benchmarking started.")
    methods = {"Louvain": run_louvain, "Leiden": run_leiden, "Infomap": run_infomap}

    results = []

    for method, func in methods.items():
        print(f"🔹 Running {method}...")  # Debug print
        start_time = time.time()
        communities = func(graph)  # Runs the algorithm
        execution_time = time.time() - start_time

        print(
            f"✅ {method} detected {len(set(communities))} communities in {execution_time:.2f} sec."
        )  # Debug print

        results.append(
            {
                "Method": method,
                "Execution Time (s)": execution_time,
                "Number of Communities": len(set(communities)),
            }
        )

    print("✅ Benchmarking finished.")
    return results


def generate_lfr_graph():
    """Generate an LFR benchmark graph with debug prints."""
    print("🔹 Starting LFR graph generation...")

    # Parameters
    n = 100  # Reduce size for debugging
    tau1 = 3  # Degree distribution exponent
    tau2 = 1.5  # Community size distribution exponent
    mu = 0.10  # Mixing parameter
    min_community = 20  # Ensure valid community formation
    average_degree = 3  # Lower to avoid excessive computation

    print(
        f"🔹 Parameters: n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, min_community={min_community}"
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
            seed=42,
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


if __name__ == "__main__":
    print("🔹 Running full benchmark pipeline...")

    start_time = time.time()
    test_graph = generate_lfr_graph()
    if test_graph is None:
        print("❌ LFR Graph Generation Failed. Exiting...")
    else:
        print(
            f"✅ Graph Ready: {test_graph.number_of_nodes()} nodes, {test_graph.number_of_edges()} edges."
        )

    print(f"🔹 LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    start_time = time.time()
    results = benchmark_community_detection(test_graph)
    print(f"🔹 Community Detection Time: {time.time() - start_time:.2f} sec")

    print("✅ Benchmarking complete:", results)
