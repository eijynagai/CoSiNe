import csv
import time

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

# Example placeholders for your actual detection functions
from CoSiNe.community_detection.infomap import run_infomap
from CoSiNe.community_detection.leiden import run_leiden
from CoSiNe.community_detection.louvain import run_louvain
from CoSiNe.community_detection.louvain_signed import run_louvain_signed


def get_ground_truth_communities(G):
    """Extract ground-truth from node['community']."""
    ground_truth = []
    for node in sorted(G.nodes()):
        comm_attr = G.nodes[node].get("community", None)
        if not comm_attr:
            ground_truth.append(-1)
            continue

        if isinstance(comm_attr, (set, list)):
            ground_truth.append(list(comm_attr)[0])
        else:
            ground_truth.append(comm_attr)
    return ground_truth


def benchmark_signed_and_unsigned(G_signed, G_pos):
    """
    We pass in both the signed graph (G_signed) and
    the positive-only subgraph (G_pos).

    Then we define which methods run on which graph.
    """
    print("🔹 Benchmarking started.")

    # NOTE: The second item in each tuple is the graph to use:
    # 'signed' => G_signed
    # 'pos'    => G_pos
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
        # Decide which graph to pass
        if graph_type == "signed":
            graph = G_signed
        else:
            graph = G_pos

        print(f"🔹 Running {method_name} on {graph_type} graph...")

        start_time = time.time()
        communities = func(graph)  # your detection function
        execution_time = time.time() - start_time

        # Convert partition to list sorted by node ID
        if isinstance(communities, dict):
            predicted = [communities[node] for node in sorted(graph.nodes())]
        else:
            predicted = communities

        # Calculate NMI, ARI
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


def save_results_to_csv(results, filename="results/benchmark_results.csv"):
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


def generate_signed_LFR_benchmark_graph():
    """Generate a signed LFR graph and a positive-only subgraph."""
    from CoSiNe.community_detection.external.signedLFR import signed_LFR_benchmark_graph

    print("🔹 Starting signed LFR graph generation...")

    # Example parameters
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    P_minus = 0.1
    P_plus = 0.8
    min_community = 20
    average_degree = 5
    max_iters = 500
    seed = 10

    print(
        f"🔹 Parameters: n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, p_minus={P_minus}, p_plus={P_plus}, "
        f"min_community={min_community}, av_degree={average_degree} "
    )

    try:
        G_signed = signed_LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            P_minus=P_minus,
            P_plus=P_plus,
            average_degree=average_degree,
            min_community=min_community,
            max_iters=max_iters,
            seed=seed,
        )
        print("✅ Signed LFR graph generated.")
    except nx.exception.ExceededMaxIterations:
        print("❌ LFR Generation failed due to max iterations.")
        return None, None

    # Create a positive-only subgraph (for unsigned methods)
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G_signed.nodes(data=True))
    for u, v, d in G_signed.edges(data=True):
        if d.get("weight", 0) > 0:
            G_pos.add_edge(u, v, **d)

    print(
        f"✅ G_signed: {G_signed.number_of_nodes()} nodes, {G_signed.number_of_edges()} edges"
    )
    print(
        f"✅ G_pos:    {G_pos.number_of_nodes()} nodes, {G_pos.number_of_edges()} edges"
    )

    return G_signed, G_pos


if __name__ == "__main__":
    print("🔹 Running full signed+unsigned benchmark pipeline...")

    start_time = time.time()
    G_signed, G_pos = generate_signed_LFR_benchmark_graph()
    if G_signed is None or G_pos is None:
        print("❌ Signed LFR Graph Generation Failed. Exiting...")
        import sys

        sys.exit(1)

    print(f"🔹 LFR Graph Generation Time: {time.time() - start_time:.2f} sec")

    # Benchmark both
    start_time = time.time()
    results = benchmark_signed_and_unsigned(G_signed, G_pos)
    print(f"🔹 Community Detection Time: {time.time() - start_time:.2f} sec")

    print("✅ Benchmarking complete:", results)
    save_results_to_csv(results)
