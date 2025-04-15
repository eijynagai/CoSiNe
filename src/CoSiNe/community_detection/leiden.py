import igraph as ig
import leidenalg as la
import networkx as nx


def run_leiden(G, partition_type=la.ModularityVertexPartition, resolution=1.0):
    """
    Runs the Leiden community detection algorithm on the positive graph G.

    Converts the NetworkX graph to an igraph, ensuring node names are preserved and edge weights are set.

    Parameters:
      G (networkx.Graph): Input positive graph.
      partition_type: Type of partitioning (default: la.ModularityVertexPartition).
      resolution (float): Resolution parameter.

    Returns:
      dict: Mapping from original node labels to community IDs.
    """
    # Remove the edge_attrs keyword.
    G_ig = ig.Graph.from_networkx(G)

    # Ensure every vertex has a "name" attribute.
    if "name" not in G_ig.vs.attribute_names():
        G_ig.vs["name"] = [str(node) for node in G.nodes()]

    # If weight is not preserved, assign a default weight of 1.
    if "weight" not in G_ig.edge_attributes():
        G_ig.es["weight"] = 1

    # Attempt to run Leiden with resolution.
    try:
        partition = la.find_partition(G_ig, partition_type, resolution=resolution)
    except Exception as e:
        print("Leiden partitioning with resolution failed, falling back. Error:", e)
        partition = la.find_partition(G_ig, partition_type)

    # Map igraph vertex indices back to original node labels.
    node_to_cluster = {v["name"]: partition.membership[v.index] for v in G_ig.vs}
    return node_to_cluster


# Example usage:
if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.3, seed=42)
    # Ensure edges have weights:
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    communities = run_leiden(G, resolution=1.0)
    print("Leiden Communities:", communities)
