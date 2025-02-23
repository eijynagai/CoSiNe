import community as community_louvain  # python-louvain
import networkx as nx


def run_louvain(G, resolution=1.0):
    """
    Runs the Louvain community detection algorithm.

    Parameters:
        G (networkx.Graph): Input graph.
        resolution (float): Resolution parameter for modularity optimization.

    Returns:
        dict: Node-to-community mapping.
    """
    partition = community_louvain.best_partition(G, resolution=resolution)
    return partition


# Example usage
if __name__ == "__main__":
    G = nx.erdos_renyi_graph(10, 0.3)  # Example graph
    communities = run_louvain(G)
    print("Louvain Communities:", communities)
