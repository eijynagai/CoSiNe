import igraph as ig
import leidenalg as la
import networkx as nx


def run_leiden(G, partition_type=la.RBConfigurationVertexPartition, resolution=1.0):
    """
    Runs the Leiden community detection algorithm on the positive graph G.

    Converts the NetworkX graph to an igraph, ensuring node names are preserved and edge weights are set.

    Parameters:
      G (networkx.Graph): Input positive graph.
      partition_type: Type of partitioning (default: RBConfigurationVertexPartition for resolution adjustments).
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

    # Determine if the partition type accepts a resolution_parameter.
    supports_resolution = (
        "resolution_parameter" in partition_type.__init__.__code__.co_varnames
    )

    try:
        if supports_resolution:
            partition = la.find_partition(
                G_ig, partition_type, resolution_parameter=resolution
            )
        else:
            partition = la.find_partition(G_ig, partition_type)
    except Exception as e:
        print("Leiden partitioning with resolution failed, falling back. Error:", e)
        partition = la.find_partition(G_ig, partition_type)

    node_to_cluster = {int(v["name"]): partition.membership[v.index] for v in G_ig.vs}
    # print("igraph vertex names:", G_ig.vs["name"])
    # print("Calculated membership:", partition.membership)
    return node_to_cluster


# Example usage:
if __name__ == "__main__":

    G = nx.erdos_renyi_graph(10, 0.3, seed=42)
    # Ensure edges have weights:
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    communities = run_leiden(G, resolution=1.0)
    print("Leiden Communities:", communities)
