import igraph as ig
import leidenalg as la
import networkx as nx


def run_leiden(G, partition_type=la.ModularityVertexPartition):
    """
    Runs the Leiden community detection algorithm.

    Parameters:
        G (networkx.Graph): Input graph.
        partition_type: The type of partitioning to use (default: ModularityVertexPartition).

    Returns:
        dict: Node-to-community mapping.
    """
    # Convert NetworkX graph to iGraph
    node_list = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=node_list)
    G_ig = ig.Graph.Adjacency((adj_matrix > 0).tolist())

    # Run Leiden algorithm
    partition = la.find_partition(G_ig, partition_type)

    # Convert results to node dictionary
    node_to_cluster = {
        node_list[i]: partition.membership[i] for i in range(len(node_list))
    }

    return node_to_cluster


# Example usage
if __name__ == "__main__":
    G = nx.erdos_renyi_graph(10, 0.3)  # Example graph
    communities = run_leiden(G)
    print("Leiden Communities:", communities)
